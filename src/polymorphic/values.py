"""
Polymorphic ``values()`` projection support.

Unlike a regular polymorphic query, a projection never constructs real subclass
instances and never performs the additional per-subclass "downcast" queries.
Instead one single SQL query is issued which LEFT OUTER JOINs the relevant
subclass tables.

Fields that are specific to a subclass are requested with the familiar
``Model___field`` syntax (or the ``applabel__Model___field`` variant). Every such
projected field gets an accompanying boolean "presence" column, named
``<alias>__present``, which distinguishes rows where the field does not belong
to the row's concrete type (``present is False``) from rows where the column
exists but is SQL ``NULL`` (``present is True`` and the value is ``None``).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING, Any

from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import models
from django.db.models import Exists, F, OuterRef
from django.db.models.expressions import BaseExpression
from django.db.models.fields.related import ForeignObjectRel

from .query_translate import translate_polymorphic_field_path
from .utils import route_to_ancestor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .query import PolymorphicQuerySet

# Columns that always describe the row and may not be used as projection aliases
# or presence columns.
IDENTITY_ALIASES = frozenset({"polymorphic_ctype_id", "polymorphic_label"})
PRESENCE_SUFFIX = "__present"


@dataclass(frozen=True)
class _PolymorphicSpec:
    """A single ``Model___field`` projection specification."""

    alias: str
    model: type[models.Model]
    field_name: str
    field: models.Field[Any, Any]
    translated_path: str
    depth: int


@dataclass
class PolymorphicValuesProjection:
    """
    Resolved description of a :meth:`PolymorphicQuerySet.polymorphic_values`
    query.

    The queryset carries one of these while the query is built; the custom
    iterable uses it to assemble the final dictionaries.
    """

    base_model: type[models.Model]
    common_fields: tuple[str, ...]
    specs: tuple[_PolymorphicSpec, ...] = dataclass_field(default=())
    # Concrete model -> name of the (hidden) annotation holding its presence bit
    presence_annotations: dict[type[models.Model], str] = dataclass_field(
        default_factory=dict
    )
    # Alias -> ordered concrete models contributing to the (coalesced) value
    alias_models: dict[str, list[type[models.Model]]] = dataclass_field(default_factory=dict)
    # Output column order, excluding the two hidden identity annotations.
    value_fields: tuple[str, ...] = ()

    def get_presence(self, alias: str, row: dict[str, Any]) -> bool:
        """OR together the presence bits of every model contributing to alias."""
        return any(row[self.presence_annotations[model]] for model in self.alias_models[alias])


def _resolve_polymorphic_spec(
    base_model: type[models.Model],
    token: str,
    explicit_alias: str | None,
    explicit_model: type[models.Model] | None = None,
) -> _PolymorphicSpec:
    """
    Resolve a ``Model___field`` (possibly ``applabel__Model___field``) token to
    a concrete subclass field. Raises FieldError for anything illegal.
    """
    if "___" not in token:
        raise FieldError(
            f"Polymorphic projection field {token!r} is not a polymorphic field path; "
            f"use the Model___field syntax or pass it as a common field."
        )

    classname, sep, pure_field_path = token.partition("___")
    if not classname or not pure_field_path or "___" in pure_field_path:
        raise FieldError(f"Invalid polymorphic field path {token!r}.")

    if not sep:  # pragma: no cover - guarded by the '___' check above
        raise FieldError(f"Invalid polymorphic field path {token!r}.")

    # Resolve the model first so unknown models/non-descendants raise a clear
    # FieldError rather than the internal AssertionError used by the generic
    # field-path translator.
    model = explicit_model or _model_from_token(base_model, token)
    if not issubclass(model, base_model):
        raise FieldError(
            f"{model._meta.label} is not derived from {base_model._meta.label}"
        )
    # Translate the join path following exactly the rules of filters/order_by.
    translated = translate_polymorphic_field_path(base_model, token)

    if model is base_model:
        raise FieldError(
            f"Polymorphic projection field {token!r} targets the base model "
            f"{base_model._meta.label}; request it as a common field instead."
        )
    if model._meta.proxy:
        raise FieldError(
            f"Polymorphic projection field {token!r} targets proxy model "
            f"{model._meta.label}; project fields from its concrete model. Proxy "
            f"identity is still reported via 'polymorphic_ctype_id'."
        )
    if model._meta.abstract:  # pragma: no cover - abstract models cannot be referenced
        raise FieldError(f"Polymorphic projection field {token!r} targets an abstract model.")

    # Only a single, local, concrete column is allowed: no relations, no
    # traversal, no parent-link pointers, no inherited columns (those must be
    # requested from the table that actually holds them).
    if "__" in pure_field_path:
        raise FieldError(
            f"Polymorphic projection field {token!r} must name a single local field; "
            f"related field paths are not supported."
        )
    try:
        django_field = model._meta.get_field(pure_field_path)
    except FieldDoesNotExist as e:
        raise FieldError(
            f"Polymorphic projection field {token!r}: {model._meta.label} has no field "
            f"named {pure_field_path!r}."
        ) from e
    if isinstance(django_field, ForeignObjectRel) or getattr(django_field, "one_to_many", False):
        raise FieldError(
            f"Polymorphic projection field {token!r}: reverse and many-to-many relations "
            f"are not supported."
        )
    if django_field.remote_field is not None and getattr(
        django_field.remote_field, "parent_link", False
    ):
        raise FieldError(
            f"Polymorphic projection field {token!r}: parent pointer fields cannot be "
            f"projected; use the common primary key instead."
        )
    if django_field not in model._meta.local_concrete_fields:
        raise FieldError(
            f"Polymorphic projection field {token!r}: {pure_field_path!r} is inherited by "
            f"{model._meta.label}; project it from the model whose table stores it."
        )

    alias = explicit_alias or pure_field_path
    depth = len(route_to_ancestor(model, base_model))
    return _PolymorphicSpec(
        alias=alias,
        model=model,
        field_name=pure_field_path,
        field=django_field,
        translated_path=translated,
        depth=depth,
    )


def _model_from_token(
    base_model: type[models.Model], token: str
) -> type[models.Model]:
    """Return the model class addressed by a ``[applabel__]Model___field`` token."""
    classname = token.partition("___")[0].lstrip("-")
    if "__" in classname:
        app_label, _, model_name = classname.partition("__")
        from django.apps import apps

        try:
            return apps.get_model(app_label, model_name)
        except LookupError as e:
            raise FieldError(f"Model {app_label}.{model_name} does not exist") from e

    # Mirror translate_polymorphic_field_path: a real relation on the base
    # model sharing the name is not a class lookup.
    try:
        candidate = base_model._meta.get_field(classname)
    except FieldDoesNotExist:
        candidate = None
    if isinstance(candidate, (models.ForeignKey,)) and not getattr(
        getattr(candidate, "remote_field", None), "parent_link", False
    ):
        raise FieldError(f"{classname!r} is not a polymorphic subclass of {base_model.__name__}.")

    from .utils import _map_queryname_to_class

    try:
        return _map_queryname_to_class(base_model, classname)
    except AssertionError as e:
        raise FieldError(str(e)) from e


def _coerce_spec(
    base_model: type[models.Model], token: Any, explicit_alias: str | None
) -> _PolymorphicSpec:
    """Accept ``Model___field`` strings or ``(Model, field_name)`` tuples."""
    if isinstance(token, tuple) and len(token) == 2:
        model_cls, field_name = token
        _validate_tuple_model(base_model, model_cls)
        if not isinstance(field_name, str):
            raise FieldError("(model, field) polymorphic_values() field names must be strings.")
        qualified = f"{model_cls._meta.app_label}__{model_cls.__name__}___{field_name}"
        return _resolve_polymorphic_spec(
            base_model, qualified, explicit_alias, explicit_model=model_cls
        )
    if isinstance(token, str):
        return _resolve_polymorphic_spec(base_model, token, explicit_alias)
    raise FieldError(
        "Polymorphic projection specs must be Model___field strings or "
        "(Model, field_name) tuples."
    )


def _is_single_model_field_pair(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], type)
        and issubclass(value[0], models.Model)
        and isinstance(value[1], str)
    )


def build_projection(
    base_model: type[models.Model],
    positional: "Sequence[Any]",
    expressions: "dict[str, Any]",
) -> PolymorphicValuesProjection:
    """
    Parse and validate every argument of ``polymorphic_values()`` eagerly, i.e.
    before any SQL is executed.
    """
    used_aliases: set[str] = set(IDENTITY_ALIASES)
    # User-visible output order: identity columns first, then every explicit
    # field in the order it was declared (presence right after its value).
    value_fields: list[str] = ["id", "polymorphic_ctype_id", "polymorphic_label"]

    resolved_common: list[str] = []
    specs: list[_PolymorphicSpec] = []

    def reserve(alias: str) -> None:
        if alias in used_aliases:
            raise FieldError(f"Duplicate projection field {alias!r}.")
        used_aliases.add(alias)

    def add_common(name: str) -> None:
        if not isinstance(name, str) or not name:
            raise FieldError("Common projection field names must be non-empty strings.")
        if name.endswith(PRESENCE_SUFFIX):
            raise FieldError(
                f"Common field name {name!r} conflicts with the reserved presence suffix "
                f"{PRESENCE_SUFFIX!r}."
            )
        reserve(name)
        # Validate eagerly that the field exists on the base model (pk and the
        # polymorphic content type are common columns too).
        lookup_name = base_model._meta.pk.name if name == "pk" else name
        try:
            found = base_model._meta.get_field(lookup_name)
        except FieldDoesNotExist:
            found = None
        if found is None and name not in {"pk", "polymorphic_ctype_id"}:
            raise FieldError(
                f"Cannot resolve keyword {name!r} into field on {base_model.__name__}."
            )
        resolved_common.append(name)
        value_fields.append(name)

    def add_spec(token: Any, explicit_alias: str | None) -> _PolymorphicSpec:
        spec = _coerce_spec(base_model, token, explicit_alias)
        if explicit_alias is None:
            reserve(spec.alias)
        specs.append(spec)
        return spec

    for item in positional:
        if isinstance(item, str):
            if "___" in item:
                spec = add_spec(item, None)
                value_fields.append(spec.alias)
                value_fields.append(f"{spec.alias}{PRESENCE_SUFFIX}")
            else:
                add_common(item)
        elif isinstance(item, BaseExpression):
            raise FieldError(
                "Bare expressions are not accepted positionally; pass them as a "
                "keyword alias (alias=Expression(...)) or annotate() first."
            )
        else:
            spec = add_spec(item, None)
            value_fields.append(spec.alias)
            value_fields.append(f"{spec.alias}{PRESENCE_SUFFIX}")

    kwarg_specs: dict[str, list[Any]] = {}
    for alias, value in expressions.items():
        if not isinstance(alias, str) or not alias:
            raise FieldError("Projection aliases must be non-empty strings.")
        if isinstance(value, BaseExpression):
            reserve(alias)
            value_fields.append(alias)
            continue
        if (
            isinstance(value, str)
            and "___" in value
            or _is_single_model_field_pair(value)
        ):
            # Single Model___field token / (Model, field) pair
            reserve(alias)
            kwarg_specs[alias] = [value]
        elif isinstance(value, (tuple, list)) and not isinstance(value, str):
            # Multi-spec alias: sequence of tokens / (Model, field) pairs
            if not value:
                raise FieldError(f"Alias {alias!r} has no projection specs.")
            reserve(alias)
            kwarg_specs[alias] = list(value)
        else:
            raise FieldError(
                f"Value for alias {alias!r} must be a query expression, a Model___field "
                f"token, a (Model, field_name) pair, or a sequence of those."
            )
        value_fields.append(alias)
        value_fields.append(f"{alias}{PRESENCE_SUFFIX}")

    # Resolve keyword specs (possibly multiple models per alias).
    for alias, tokens in kwarg_specs.items():
        for token in tokens:
            add_spec(token, alias)

    # Cross-subclass checks: identical aliases must have compatible output
    # types, and COALESCE ordering must be deterministic.
    by_alias: dict[str, list[_PolymorphicSpec]] = {}
    for spec in specs:
        by_alias.setdefault(spec.alias, []).append(spec)

    for alias, group in by_alias.items():
        types = {type(spec.field) for spec in group}
        if len(types) > 1:
            labels = ", ".join(
                f"{spec.model.__name__}.{spec.field_name} ({type(spec.field).__name__})"
                for spec in group
            )
            raise FieldError(
                f"Alias {alias!r} combines fields with incompatible types: {labels}."
            )
        models_in_group = {spec.model for spec in group}
        if len(models_in_group) != len(group):
            raise FieldError(
                f"Alias {alias!r} is provided more than once for the same model."
            )

    ordered_specs = tuple(sorted(specs, key=lambda s: (s.alias, -s.depth)))

    projection = PolymorphicValuesProjection(
        base_model=base_model,
        common_fields=tuple(resolved_common),
        specs=ordered_specs,
        value_fields=tuple(value_fields),
    )
    for spec in ordered_specs:
        projection.alias_models.setdefault(spec.alias, []).append(spec.model)
    return projection


def _validate_tuple_model(
    base_model: type[models.Model], model_cls: Any
) -> None:
    if not isinstance(model_cls, type) or not issubclass(model_cls, models.Model):
        raise FieldError("(model, field) polymorphic_values() specs require a model class.")
    if not issubclass(model_cls, base_model):
        raise FieldError(
            f"{model_cls._meta.label} is not derived from {base_model._meta.label}"
        )



def annotate_projection(
    queryset: "PolymorphicQuerySet", projection: PolymorphicValuesProjection
) -> "PolymorphicQuerySet":
    """
    Add hidden annotations to *queryset*:

    * one presence bit per concrete subclass table that is referenced,
    * one coalesced value per polymorphic alias,
    * the content-type app label/model used to build ``polymorphic_label``.
    """
    annotations: dict[str, Any] = {}

    for spec in projection.specs:
        if spec.model not in projection.presence_annotations:
            annotation_name = f"_polymorphic_present_{spec.model._meta.model_name}"
            annotations[annotation_name] = _presence_expression(
                projection.base_model, spec.model
            )
            projection.presence_annotations[spec.model] = annotation_name

    for alias in projection.alias_models:
        group = [spec for spec in projection.specs if spec.alias == alias]
        expressions_ = [F(spec.translated_path) for spec in group]
        if len(expressions_) == 1:
            annotations[alias] = expressions_[0]
        else:
            from django.db.models.functions import Coalesce

            annotations[alias] = Coalesce(*expressions_)

    annotations["_polymorphic_app_label"] = F("polymorphic_ctype__app_label")
    annotations["_polymorphic_model"] = F("polymorphic_ctype__model")

    # All F() paths here are already translated plain Django paths, so call the
    # vanilla annotate to avoid Model___field re-translation.
    return queryset._polymorphic_raw_annotate(**annotations)


def _presence_expression(
    base_model: type[models.Model], model: type[models.Model]
) -> Exists:
    """
    EXISTS(SELECT 1 FROM <model table> WHERE <parent ptr> = <outer path>).

    The outer reference walks the LEFT OUTER JOIN chain from the base table to
    the immediate concrete parent of *model*.
    """
    route = route_to_ancestor(model, base_model)
    # route is [ParentLinkInfo(immediate parent, link), ...] depth-first.
    parent_model, link = route[0].model, route[0].link

    if parent_model is base_model:
        outer_ref: BaseExpression | F = OuterRef("pk")
    else:
        base_path = translate_polymorphic_field_path(
            base_model, f"{parent_model.__name__}___{parent_model._meta.pk.name}"
        )
        outer_ref = OuterRef(base_path)

    manager = model._base_objects  # type: ignore[attr-defined]
    return Exists(manager.filter(**{link.attname: outer_ref}))


class PolymorphicValuesIterable(models.query.ValuesIterable):
    """
    ValuesIterable that post-processes raw rows:

    * adds ``<alias>__present`` booleans by OR-ing per-model presence bits,
    * removes the hidden presence annotations,
    * builds ``polymorphic_label`` from the resolved content type while the
      integer ``polymorphic_ctype_id`` keeps proxy identity intact.
    """

    def __iter__(self):
        projection: PolymorphicValuesProjection = self.queryset._polymorphic_values_projection  # type: ignore[attr-defined]
        for raw_row in super().__iter__():
            app_label = raw_row.get("_polymorphic_app_label")
            model_name = raw_row.get("_polymorphic_model")
            presence = {
                alias: projection.get_presence(alias, dict(raw_row))
                for alias in projection.alias_models
            }

            row: dict[str, Any] = {}
            for key in projection.value_fields:
                if key == "polymorphic_label":
                    row[key] = (
                        f"{app_label}.{model_name}" if app_label and model_name else None
                    )
                elif key.endswith(PRESENCE_SUFFIX):
                    row[key] = presence[key[: -len(PRESENCE_SUFFIX)]]
                else:
                    row[key] = raw_row.get(key)
            yield row
