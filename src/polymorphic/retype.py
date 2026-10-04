"""
Explicit retyping of polymorphic model instances.

This module implements :meth:`polymorphic.models.PolymorphicModel.retype`,
which changes the concrete type of an existing database row inside a
polymorphic inheritance tree while preserving the base row (primary key)
and all shared fields.

The operation:

* finds the common concrete ancestor of the current and target type,
* inserts rows for the target inheritance chain below that ancestor
  (subclass specific fields are supplied by the caller),
* safely removes the old child rows that are no longer needed - without
  ever cascading into the kept ancestor rows,
* switches the ``polymorphic_ctype`` in the same transaction.

Relations pointing at the removed child tables are inspected up front;
relations that cannot be preserved (``PROTECT``/``RESTRICT``/``DO_NOTHING``
with existing objects) are reported before anything is executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from django.contrib.contenttypes.fields import GenericRelation
from django.contrib.contenttypes.models import ContentType
from django.db import DEFAULT_DB_ALIAS, models, transaction
from django.db.models.deletion import CASCADE, DO_NOTHING, PROTECT, RESTRICT, SET_DEFAULT, SET_NULL

from .utils import concrete_descendants, get_base_polymorphic_model

if TYPE_CHECKING:
    from .models import PolymorphicModel


class RetypeError(RuntimeError):
    """Base class for all retype failures."""


class RetypeNotAllowedError(RetypeError):
    """The requested target type is not in the allowed set of types."""


class RetypeValidationError(RetypeError):
    """The supplied field values are missing or invalid."""


class RetypeProtectedError(RetypeError):
    """
    Relations pointing to rows that must be removed block the retype.

    The offending relations are available on :attr:`effects`.
    """

    def __init__(self, message: str, effects: Iterable["RetypeRelationEffect"]) -> None:
        super().__init__(message)
        self.effects = list(effects)


#: relation effect actions
ACTION_BLOCK = "block"
ACTION_CASCADE = "cascade"
ACTION_UPDATE = "update"

_ON_DELETE_NAMES = {
    PROTECT: "PROTECT",
    RESTRICT: "RESTRICT",
    CASCADE: "CASCADE",
    DO_NOTHING: "DO_NOTHING",
    SET_NULL: "SET_NULL",
    SET_DEFAULT: "SET_DEFAULT",
}


@dataclass(frozen=True)
class RetypeRelationEffect:
    """The effect a retype has on one relation pointing to a removed child table."""

    removed_model: type[models.Model]
    model: type[models.Model]
    field_name: str
    on_delete: str
    count: int
    action: str

    def describe(self) -> str:
        return (
            f"{self.on_delete} {self.model._meta.label}.{self.field_name} -> "
            f"{self.removed_model._meta.label} ({self.count} object(s)) [{self.action}]"
        )


@dataclass
class RetypePlan:
    """A (possibly not yet executed) plan to retype one polymorphic instance."""

    instance: "PolymorphicModel"
    source_model: type[models.Model]
    target_model: type[models.Model]
    base_model: type[models.Model]
    common_ancestor: type[models.Model]
    removed_models: list[type[models.Model]] = field(default_factory=list)
    created_models: list[type[models.Model]] = field(default_factory=list)
    field_values: dict[str, Any] = field(default_factory=dict)
    relation_effects: list[RetypeRelationEffect] = field(default_factory=list)
    proxy_only: bool = False
    using: str = DEFAULT_DB_ALIAS
    executed: bool = False

    @property
    def target_concrete_model(self) -> type[models.Model]:
        return self.target_model._meta.concrete_model

    @property
    def blocking_effects(self) -> list[RetypeRelationEffect]:
        return [effect for effect in self.relation_effects if effect.action == ACTION_BLOCK]

    @property
    def blocked(self) -> bool:
        return bool(self.blocking_effects)

    def describe(self) -> str:
        """Return a human readable description of this plan."""
        lines = [
            f"Retype plan: {self.source_model._meta.label}#{self.instance.pk} -> "
            f"{self.target_model._meta.label}"
        ]
        if self.proxy_only:
            lines.append("  proxy-only switch (no table changes)")
        else:
            lines.append(f"  common ancestor: {self.common_ancestor._meta.label}")
            for model in self.removed_models:
                lines.append(f"  delete row: {model._meta.label} ({model._meta.db_table})")
            for model in self.created_models:
                lines.append(f"  create row: {model._meta.label} ({model._meta.db_table})")
        lines.append(f"  content type -> {self.target_model._meta.label}")
        for effect in self.relation_effects:
            lines.append(f"  relation: {effect.describe()}")
        lines.append(f"  status: {'executed' if self.executed else 'not executed (dry-run)'}")
        return "\n".join(lines)

    def get_instance(self) -> "PolymorphicModel":
        """Fetch the retyped instance as its new real type."""
        if not self.executed:
            raise RetypeError("This retype plan has not been executed yet.")
        return self.base_model._default_manager.db_manager(self.using).get(pk=self.instance.pk)


def _unwrap_on_delete(on_delete: Any) -> Any:
    # on_delete callables may be wrapped in a PolymorphicGuard
    return getattr(on_delete, "action", on_delete)


def _classify_on_delete(on_delete: Any) -> str:
    action = _unwrap_on_delete(on_delete)
    if action in (PROTECT, RESTRICT, DO_NOTHING):
        return ACTION_BLOCK
    if action is CASCADE:
        return ACTION_CASCADE
    return ACTION_UPDATE


def _on_delete_name(on_delete: Any) -> str:
    action = _unwrap_on_delete(on_delete)
    return _ON_DELETE_NAMES.get(action, getattr(action, "__name__", repr(action)))


def _mti_chain(model: type[models.Model]) -> list[type[models.Model]]:
    """Concrete multi-table inheritance chain, root first, down to ``model``."""
    chain = []
    current: type[models.Model] | None = model._meta.concrete_model
    while current is not None:
        chain.append(current)
        current = next(
            (parent for parent, link in current._meta.parents.items() if link is not None),
            None,
        )
    chain.reverse()
    return chain


def _common_ancestor(
    source_chain: list[type[models.Model]], target_chain: list[type[models.Model]]
) -> type[models.Model] | None:
    common = None
    for source_model, target_model in zip(source_chain, target_chain):
        if source_model is not target_model:
            break
        common = source_model
    return common


def _collect_relation_effects(
    removed_models: list[type[models.Model]], pk: Any, using: str
) -> list[RetypeRelationEffect]:
    """Inspect relations pointing at the rows that will be removed."""
    removed = {model._meta.concrete_model for model in removed_models}
    effects: list[RetypeRelationEffect] = []
    for model in removed_models:
        for rel in model._meta.related_objects:
            related_model = rel.related_model._meta.concrete_model
            if related_model in removed:
                # rows within the removed subtree are deleted anyway
                continue
            related_field = rel.field
            count = (
                related_model._base_manager.db_manager(using)
                .filter(**{related_field.name: pk})
                .count()
            )
            if not count:
                continue
            on_delete = related_field.remote_field.on_delete
            effects.append(
                RetypeRelationEffect(
                    removed_model=model,
                    model=rel.related_model,
                    field_name=related_field.name,
                    on_delete=_on_delete_name(on_delete),
                    count=count,
                    action=_classify_on_delete(on_delete),
                )
            )
        for private in model._meta.private_fields:
            if not isinstance(private, GenericRelation):
                continue
            ctypes = ContentType.objects.db_manager(using).get_for_models(
                *removed_models, for_concrete_model=False
            )
            count = (
                private.related_model._base_manager.db_manager(using)
                .filter(
                    **{
                        f"{private.content_type_field_name}__in": [
                            ct.pk for ct in ctypes.values()
                        ],
                        private.object_id_field_name: pk,
                    }
                )
                .count()
            )
            if not count:
                continue
            effects.append(
                RetypeRelationEffect(
                    removed_model=model,
                    model=private.related_model,
                    field_name=private.name,
                    on_delete="CASCADE",
                    count=count,
                    action=ACTION_CASCADE,
                )
            )
    return effects


def _validate_field_values(
    created_models: list[type[models.Model]], field_values: dict[str, Any]
) -> None:
    """Ensure caller supplied values exactly cover the required subclass fields."""
    known: dict[str, models.Field[Any, Any]] = {}
    missing = []
    for model in created_models:
        for local_field in model._meta.local_concrete_fields:
            if local_field.primary_key or local_field.auto_created:
                continue
            known[local_field.name] = local_field
            known.setdefault(local_field.attname, local_field)
            provided = local_field.name in field_values or local_field.attname in field_values
            if (
                not provided
                and not local_field.null
                and not local_field.has_default()
                and not local_field.has_db_default()
                and not getattr(local_field, "auto_now", False)
                and not getattr(local_field, "auto_now_add", False)
            ):
                missing.append(f"{model._meta.label}.{local_field.name}")
    unknown = sorted(key for key in field_values if key not in known)
    errors = []
    if missing:
        errors.append("missing required field(s): " + ", ".join(missing))
    if unknown:
        errors.append("unknown field(s) for the target type: " + ", ".join(unknown))
    if errors:
        raise RetypeValidationError("Invalid retype field values: " + "; ".join(errors))


def _resolve_allowed_types(
    instance: "PolymorphicModel",
    base_model: type[models.Model],
    allowed_types: Iterable[type[models.Model]] | None,
) -> frozenset[type[models.Model]]:
    if allowed_types is None:
        allowed_types = getattr(instance, "polymorphic_retype_allowed_types", None)
    if allowed_types is None:
        return frozenset([base_model, *concrete_descendants(base_model, include_proxy=True)])
    return frozenset(allowed_types)


def build_retype_plan(
    instance: "PolymorphicModel",
    target_model: type[models.Model],
    *,
    allowed_types: Iterable[type[models.Model]] | None = None,
    field_values: dict[str, Any] | None = None,
    using: str = DEFAULT_DB_ALIAS,
) -> RetypePlan:
    """Build (but do not execute) a :class:`RetypePlan` for ``instance``."""
    from .models import PolymorphicModel

    if instance.pk is None:
        raise RetypeError("Cannot retype an unsaved instance.")
    if (
        not isinstance(target_model, models.base.ModelBase)
        or not issubclass(target_model, PolymorphicModel)
        or target_model._meta.abstract
    ):
        raise RetypeError(
            "target_model must be a concrete (or proxy) PolymorphicModel subclass, "
            f"got {target_model!r}."
        )

    field_values = dict(field_values or {})

    source_model = instance.get_real_concrete_instance_class()
    if source_model is None:
        raise RetypeError(
            f"The content type of {instance.__class__.__name__}#{instance.pk} is stale; "
            "cannot determine the current concrete type."
        )
    base_model = get_base_polymorphic_model(source_model)
    assert base_model is not None
    target_concrete = target_model._meta.concrete_model
    if get_base_polymorphic_model(target_concrete) is not base_model:
        raise RetypeError(
            f"{target_model._meta.label} does not belong to the "
            f"{base_model._meta.label} polymorphic tree."
        )

    allowed = _resolve_allowed_types(instance, base_model, allowed_types)
    if target_model not in allowed:
        raise RetypeNotAllowedError(
            f"Retyping {source_model._meta.label}#{instance.pk} to "
            f"{target_model._meta.label} is not allowed. Allowed types: "
            + ", ".join(sorted(model._meta.label for model in allowed))
        )

    if target_model is instance.get_real_instance_class():
        raise RetypeError(
            f"Instance #{instance.pk} is already of type {target_model._meta.label}."
        )

    proxy_only = target_concrete is source_model
    removed_models: list[type[models.Model]] = []
    created_models: list[type[models.Model]] = []
    if proxy_only:
        common_ancestor = source_model
    else:
        source_chain = _mti_chain(source_model)
        target_chain = _mti_chain(target_concrete)
        common_ancestor = _common_ancestor(source_chain, target_chain)
        if common_ancestor is None:  # pragma: no cover - guarded by the tree check above
            raise RetypeError(
                f"{source_model._meta.label} and {target_model._meta.label} have no "
                "common concrete ancestor."
            )
        split = source_chain.index(common_ancestor) + 1
        removed_models = list(reversed(source_chain[split:]))  # leaf first
        created_models = target_chain[target_chain.index(common_ancestor) + 1 :]

    _validate_field_values(created_models, field_values)
    relation_effects = _collect_relation_effects(removed_models, instance.pk, using)

    return RetypePlan(
        instance=instance,
        source_model=source_model,
        target_model=target_model,
        base_model=base_model,
        common_ancestor=common_ancestor,
        removed_models=removed_models,
        created_models=created_models,
        field_values=field_values,
        relation_effects=relation_effects,
        proxy_only=proxy_only,
        using=using,
    )


def _insert_chain_row(
    model: type[models.Model], pk: Any, field_values: dict[str, Any], using: str
) -> None:
    """Insert a single inheritance-chain row without touching any parent table."""
    obj = model()
    setattr(obj, model._meta.pk.attname, pk)
    for local_field in model._meta.local_concrete_fields:
        if local_field.primary_key:
            continue
        if local_field.name in field_values:
            setattr(obj, local_field.name, field_values[local_field.name])
        elif local_field.attname in field_values:
            setattr(obj, local_field.attname, field_values[local_field.attname])
    obj._state.adding = True
    obj._state.db = using
    obj._save_table(raw=True, cls=model, force_insert=True, using=using)


def _invalidate_instance_cache(instance: "PolymorphicModel") -> None:
    """Explicitly invalidate cached state on the stale Python instance."""
    fields_cache = getattr(instance._state, "fields_cache", None)
    if fields_cache is None:  # Django < 5.0
        fields_cache = getattr(instance, "_fields_cache", None)
    if fields_cache is not None:
        fields_cache.clear()
    prefetched = getattr(instance, "_prefetched_objects_cache", None)
    if prefetched is not None:
        prefetched.clear()


def execute_retype(plan: RetypePlan) -> RetypePlan:
    """Execute a previously built :class:`RetypePlan` in a single transaction."""
    if plan.blocked:
        effects = plan.blocking_effects
        raise RetypeProtectedError(
            f"Cannot retype {plan.source_model._meta.label}#{plan.instance.pk} to "
            f"{plan.target_model._meta.label}: relations prevent removing the old "
            "child row(s): " + "; ".join(effect.describe() for effect in effects),
            effects,
        )

    instance = plan.instance
    using = plan.using
    old_ctype_id = instance.polymorphic_ctype_id
    try:
        with transaction.atomic(using=using):
            # remove the old child rows leaf-first, keeping all ancestors;
            # keep_parents=True guarantees the base rows are never cascaded into
            for model in plan.removed_models:
                obj = (
                    model._default_manager.db_manager(using)
                    .non_polymorphic()
                    .get(pk=instance.pk)
                )
                obj.delete(using=using, keep_parents=True)
            # build the target inheritance chain below the common ancestor
            for model in plan.created_models:
                _insert_chain_row(model, instance.pk, plan.field_values, using)
            # switch the content type together with the table changes
            new_ctype = ContentType.objects.db_manager(using).get_for_model(
                plan.target_model, for_concrete_model=False
            )
            plan.base_model._default_manager.db_manager(using).non_polymorphic().filter(
                pk=instance.pk
            ).update(polymorphic_ctype_id=new_ctype.pk)
            instance.polymorphic_ctype_id = new_ctype.pk
    except Exception:
        # the transaction rolls back; restore the original concrete type in memory
        instance.polymorphic_ctype_id = old_ctype_id
        raise

    _invalidate_instance_cache(instance)
    plan.executed = True
    return plan
