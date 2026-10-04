"""
Explicit retyping of polymorphic model instances.

This module implements :meth:`polymorphic.models.PolymorphicModel.retype`,
which changes an existing database object from one concrete type in a
polymorphic hierarchy to another (for example, turning a draft of one
document type into another document type) while preserving the primary key
and all fields stored on the shared ancestor tables.

The retype operation:

* requires the target type to be a member of an explicit allow-list,
* finds the deepest common concrete ancestor of the current and target type,
* inserts rows for the target inheritance chain below that ancestor
  (subclass specific field values must be supplied by the caller),
* removes the now obsolete child rows of the old type **without** using a
  normal ``delete()`` so the shared base row can never be cascade deleted,
* checks relations pointing at the removed child tables up front and
  reports the ones that cannot be preserved before any change is made,
* switches the ``polymorphic_ctype`` content type in the same transaction
  as the table changes, restoring the original concrete type on failure,
* handles proxy-only switches (content type change only) separately from
  real multi-table switches,
* explicitly invalidates the cached state of the old Python instance, and
* supports a dry-run mode that returns the :class:`RetypePlan` without
  touching the database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import FieldDoesNotExist
from django.db import DEFAULT_DB_ALIAS, models, transaction
from django.db.models.deletion import CASCADE, PROTECT, SET_DEFAULT, SET_NULL

from .base import PolymorphicModelBase

if TYPE_CHECKING:
    from .models import PolymorphicModel


class RetypeError(RuntimeError):
    """Base class for all retype failures."""


class RetypeNotAllowed(RetypeError):
    """The requested target type is not in the allowed set of target types."""


class RetypeValidationError(RetypeError):
    """The supplied field values are invalid for the target type."""


class RetypeBlocked(RetypeError):
    """
    Relations pointing at rows that retyping would remove cannot be preserved.

    The offending relations are available as :attr:`issues`.
    """

    def __init__(self, issues: list[RetypeRelationIssue]):
        self.issues = issues
        details = "; ".join(str(issue) for issue in issues)
        super().__init__(
            f"Cannot retype: {len(issues)} relation(s) point at rows that would be "
            f"removed and cannot be preserved: {details}"
        )


@dataclass(frozen=True)
class RetypeRelationIssue:
    """
    A relation pointing at a child table row that retyping would remove.

    ``blocking`` is True for relations whose ``on_delete`` behavior
    (``PROTECT``/``CASCADE``) makes the retype unsafe; non-blocking issues are
    reported as warnings only.
    """

    related_model: type[models.Model]
    field_name: str
    on_delete: Any
    count: int
    blocking: bool

    def __str__(self) -> str:
        on_delete = getattr(self.on_delete, "__name__", self.on_delete)
        return (
            f"{self.related_model._meta.label}.{self.field_name} "
            f"(on_delete={on_delete}, {self.count} referencing row(s))"
        )


@dataclass(frozen=True)
class RetypeRelationFixup:
    """
    A referential integrity fixup applied while executing a retype plan.

    Relations with ``on_delete=SET_NULL``/``SET_DEFAULT`` pointing at removed
    child tables cannot be preserved, but their declared behavior is applied
    to the referencing rows so no dangling references are left behind.
    """

    related_model: type[models.Model]
    field_name: str
    value: Any


def _concrete_chain(model: type[models.Model]) -> list[type[models.Model]]:
    """
    Return the concrete polymorphic inheritance chain of ``model``,
    ordered from the polymorphic root down to (and including) ``model``.
    """
    from .models import PolymorphicModel

    chain = [
        cls
        for cls in reversed(model.mro())
        if isinstance(cls, PolymorphicModelBase)
        and cls is not PolymorphicModel
        and not cls._meta.abstract
        and not cls._meta.proxy
    ]
    for cls in chain:
        if len(cls._meta.parents) > 1:
            raise RetypeError(
                f"Retyping is not supported for models with multiple concrete "
                f"parents ({cls._meta.label})."
            )
    return chain


def _plain_queryset(model: type[models.Model], using: str):
    """Return a non-polymorphic queryset for raw table level operations."""
    qs = model._base_manager.db_manager(using).all()
    if hasattr(qs, "non_polymorphic"):
        qs = qs.non_polymorphic()
    return qs


@dataclass
class RetypePlan:
    """
    A fully validated plan describing how an instance will be retyped.

    Instances are created by :func:`plan_retype` (or
    :meth:`polymorphic.models.PolymorphicModel.retype` with
    ``dry_run=True``). Call :meth:`execute` to perform the retype.
    """

    #: The (possibly stale-typed) Python instance being retyped.
    instance: PolymorphicModel
    #: The current concrete model of the instance.
    source_model: type[models.Model]
    #: The requested target model (may be a proxy).
    target_model: type[models.Model]
    #: The deepest concrete model shared by the source and target chains.
    common_ancestor: type[models.Model]
    #: The polymorphic root model that carries ``polymorphic_ctype``.
    root_model: type[models.Model]
    #: Models whose rows must be inserted, ordered parent first.
    create_models: list[type[models.Model]]
    #: Models whose rows must be removed, ordered leaf first.
    delete_models: list[type[models.Model]]
    #: Field values for the newly created rows, keyed by owning model.
    new_row_values: dict[type[models.Model], dict[str, Any]]
    #: Field updates applied to rows that are kept, keyed by owning model.
    ancestor_updates: dict[type[models.Model], dict[str, Any]]
    #: Relations that prevent the retype from executing.
    blockers: list[RetypeRelationIssue]
    #: Relations that cannot be preserved but do not prevent the retype.
    warnings: list[RetypeRelationIssue]
    #: Integrity fixups applied for non-blocking relations during execution.
    fixups: list[RetypeRelationFixup]
    #: True when only the content type changes (no table changes).
    is_proxy_switch: bool
    #: The database alias the plan was built for.
    using: str = DEFAULT_DB_ALIAS

    @property
    def changes_tables(self) -> bool:
        """True when executing this plan inserts or removes table rows."""
        return bool(self.create_models or self.delete_models)

    def execute(self) -> PolymorphicModel:
        """
        Execute the plan in a single transaction and return the re-fetched
        instance in its new real type.

        Raises :class:`RetypeBlocked` if :attr:`blockers` is not empty.
        On any failure the transaction is rolled back and the in-memory
        ``polymorphic_ctype`` of the old instance is restored to the
        original concrete type.
        """
        if self.blockers:
            raise RetypeBlocked(self.blockers)

        instance = self.instance
        using = self.using
        pk = instance.pk
        original_ctype_id = instance.polymorphic_ctype_id
        new_ctype = ContentType.objects.db_manager(using).get_for_model(
            self.target_model, for_concrete_model=False
        )
        try:
            with transaction.atomic(using=using):
                # 1. apply caller supplied updates to rows that are kept
                for model, values in self.ancestor_updates.items():
                    _plain_queryset(model, using).filter(pk=pk).update(**values)

                # 2. insert the rows of the target chain, parent first
                for model in self.create_models:
                    new_row = model()
                    new_row._state.adding = True
                    setattr(new_row, model._meta.pk.attname, pk)
                    for name, value in self.new_row_values.get(model, {}).items():
                        setattr(new_row, name, value)
                    new_row._save_table(raw=True, cls=model, using=using)

                # 3. switch the content type on the root row
                _plain_queryset(self.root_model, using).filter(pk=pk).update(
                    polymorphic_ctype=new_ctype
                )

                # 4. apply integrity fixups for relations that cannot be
                #    preserved (SET_NULL/SET_DEFAULT behavior)
                for fixup in self.fixups:
                    fixup.related_model._base_manager.db_manager(using).filter(
                        **{fixup.field_name: pk}
                    ).update(**{fixup.field_name: fixup.value})

                # 5. remove obsolete child rows, leaf first, bypassing the
                #    deletion collector so the shared base row is never
                #    cascade deleted
                for model in self.delete_models:
                    _plain_queryset(model, using).filter(pk=pk)._raw_delete(using)
        except Exception:
            # the transaction rolled back - restore the original concrete
            # type on the in-memory instance as well
            instance.polymorphic_ctype_id = original_ctype_id
            raise

        # explicitly invalidate the cached state of the old Python instance:
        # its class no longer matches the database row
        instance.polymorphic_ctype = new_ctype
        instance._state.fields_cache.clear()

        # return a fresh instance of the new real type
        manager = self.root_model.objects  # type: ignore[attr-defined]
        return manager.db_manager(using).get(pk=pk)  # type: ignore[no-any-return]


def _resolve_field_owners(
    chain: list[type[models.Model]], fields: dict[str, Any]
) -> dict[type[models.Model], dict[str, Any]]:
    """
    Map supplied field names to the chain model that owns them.

    Raises :class:`RetypeValidationError` for unknown or non-concrete fields.
    """
    owners: dict[type[models.Model], dict[str, Any]] = {}
    for name, value in fields.items():
        owner = None
        for model in chain:
            try:
                field_obj = model._meta.get_field(name)
            except FieldDoesNotExist:
                continue
            if not field_obj.concrete or field_obj.many_to_many:
                raise RetypeValidationError(
                    f"Cannot set non-concrete field {name!r} during retype."
                )
            owner = field_obj.model
            break
        if owner is None or owner not in chain:
            raise RetypeValidationError(
                f"Unknown field {name!r} for retype target chain "
                f"{' -> '.join(m.__name__ for m in chain)}."
            )
        owners.setdefault(owner, {})[name] = value
    return owners


def _check_required_fields(
    create_models: list[type[models.Model]],
    new_row_values: dict[type[models.Model], dict[str, Any]],
) -> None:
    """
    Ensure the caller supplied values for all subclass specific fields of the
    new rows that have neither a default nor are nullable.
    """
    missing: list[str] = []
    for model in create_models:
        supplied = new_row_values.get(model, {})
        for field_obj in model._meta.local_concrete_fields:
            if field_obj.primary_key or getattr(
                field_obj.remote_field, "parent_link", False
            ):
                continue
            if field_obj.name in supplied:
                continue
            if field_obj.has_default() or field_obj.null or field_obj.blank:
                continue
            missing.append(f"{model._meta.label}.{field_obj.name}")
    if missing:
        raise RetypeValidationError(
            f"Missing required field value(s) for retype target: "
            f"{', '.join(missing)}. Subclass specific fields must be supplied "
            f"by the caller."
        )


def _check_relations(
    instance: PolymorphicModel,
    delete_models: list[type[models.Model]],
    chain_models: set[type[models.Model]],
    using: str,
) -> tuple[
    list[RetypeRelationIssue], list[RetypeRelationIssue], list[RetypeRelationFixup]
]:
    """
    Find relations pointing at the rows that retyping would remove.

    Returns ``(blockers, warnings, fixups)``. Relations with
    ``on_delete=PROTECT`` or ``on_delete=CASCADE`` that actually reference the
    instance are blockers; ``SET_NULL``/``SET_DEFAULT`` relations produce
    fixups that are applied on execution; other behaviors (``SET``,
    ``DO_NOTHING``) are reported as warnings only.
    """
    blockers: list[RetypeRelationIssue] = []
    warnings: list[RetypeRelationIssue] = []
    fixups: list[RetypeRelationFixup] = []
    delete_set = set(delete_models)
    for model in delete_models:
        for rel in model._meta.related_objects:
            if getattr(rel, "parent_link", False):
                continue
            if rel.model is not model:
                # inherited reverse relation pointing at a kept ancestor table
                continue
            related_model = rel.related_model
            if related_model in delete_set or related_model in chain_models:
                # inheritance internal relation
                continue
            count = (
                related_model._base_manager.db_manager(using)
                .filter(**{rel.field.name: instance.pk})
                .count()
            )
            if not count:
                continue
            on_delete = rel.on_delete if rel.on_delete is not None else CASCADE
            issue = RetypeRelationIssue(
                related_model=related_model,
                field_name=rel.field.name,
                on_delete=on_delete,
                count=count,
                blocking=on_delete in (PROTECT, CASCADE),
            )
            (blockers if issue.blocking else warnings).append(issue)
            if on_delete is SET_NULL:
                fixups.append(RetypeRelationFixup(related_model, rel.field.name, None))
            elif on_delete is SET_DEFAULT:
                fixups.append(
                    RetypeRelationFixup(
                        related_model, rel.field.name, rel.field.get_default()
                    )
                )
    return blockers, warnings, fixups


def plan_retype(
    instance: PolymorphicModel,
    target: type[models.Model],
    *,
    allowed,
    fields: dict[str, Any] | None = None,
    using: str | None = None,
) -> RetypePlan:
    """
    Build and validate a :class:`RetypePlan` for retyping ``instance`` to
    ``target``.

    :param instance: The saved polymorphic instance to retype.
    :param target: The target model class (may be a proxy model).
    :param allowed: The explicit set of target types that are permitted.
    :param fields: Values for subclass specific fields of the target type
        (and optionally updates for fields on kept ancestor rows).
    :param using: The database alias to plan for.
    """
    from .models import PolymorphicModel

    if not isinstance(instance, PolymorphicModel):
        raise RetypeError("Can only retype PolymorphicModel instances.")
    if instance.pk is None:
        raise RetypeError("Cannot retype an unsaved instance.")
    if not (isinstance(target, type) and issubclass(target, models.Model)):
        raise RetypeError(f"Retype target must be a model class, got {target!r}.")
    if target._meta.abstract:
        raise RetypeError(f"Cannot retype to abstract model {target.__name__}.")
    if allowed is None:
        raise RetypeError("An explicit set of allowed target types is required.")
    if not any(target is allowed_model for allowed_model in allowed):
        raise RetypeNotAllowed(
            f"Retyping to {target._meta.label} is not allowed. "
            f"Allowed targets: "
            f"{', '.join(sorted(m._meta.label for m in allowed)) or '(none)'}."
        )

    using = using or instance._state.db or DEFAULT_DB_ALIAS

    source_real = instance.get_real_instance_class()
    if source_real is None:
        raise RetypeError(
            f"Cannot determine the current concrete type of "
            f"{instance.__class__.__name__}#{instance.pk}."
        )
    source_model = source_real._meta.concrete_model
    target_concrete = target._meta.concrete_model

    source_chain = _concrete_chain(source_model)
    target_chain = _concrete_chain(target_concrete)

    common: list[type[models.Model]] = []
    for source_cls, target_cls in zip(source_chain, target_chain):
        if source_cls is not target_cls:
            break
        common.append(source_cls)
    if not common:
        raise RetypeError(
            f"{target._meta.label} does not share a polymorphic ancestor with "
            f"{source_model._meta.label}; retyping across hierarchies is not "
            f"possible."
        )
    common_ancestor = common[-1]
    root_model = source_chain[0]

    create_models = target_chain[len(common) :]
    delete_models = list(reversed(source_chain[len(common) :]))

    supplied = dict(fields or {})
    owners = _resolve_field_owners(target_chain, supplied)
    create_set = set(create_models)
    new_row_values = {
        model: values for model, values in owners.items() if model in create_set
    }
    ancestor_updates = {
        model: values for model, values in owners.items() if model not in create_set
    }

    _check_required_fields(create_models, new_row_values)

    blockers, warnings, fixups = _check_relations(
        instance, delete_models, set(source_chain), using
    )

    return RetypePlan(
        instance=instance,
        source_model=source_model,
        target_model=target,
        common_ancestor=common_ancestor,
        root_model=root_model,
        create_models=create_models,
        delete_models=delete_models,
        new_row_values=new_row_values,
        ancestor_updates=ancestor_updates,
        blockers=blockers,
        warnings=warnings,
        fixups=fixups,
        is_proxy_switch=not create_models and not delete_models,
        using=using,
    )
