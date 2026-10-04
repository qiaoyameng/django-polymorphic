"""
Explicit polymorphic ``values()`` projection.

Unlike :meth:`~django.db.models.query.QuerySet.values`, which only sees the
fields of the base table, :meth:`PolymorphicQuerySet.polymorphic_values` can
project fields from specific subclasses (using the ``ModelX___field`` syntax)
while keeping the result lightweight: no real subclass instances are ever
constructed. Filtering, ordering, slicing/pagination and the ``SELECT`` all
happen in a single database query.

Each returned row exposes:

* ``polymorphic_ctype_id`` - a stable identifier of the row's model (proxy
  model identity is preserved),
* plain values for common (base model) fields,
* subclass specific fields wrapped so that a field that does not apply to a
  row can be distinguished from a field that exists but is ``NULL``:

  * the value maps to :data:`NOT_PRESENT` when the row is not an instance of
    the subclass that owns the field,
  * an additional ``"<alias>__present"`` boolean is included,
  * :meth:`PolymorphicValuesRow.is_present` and
    :meth:`PolymorphicValuesRow.present_fields` provide the same information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db.models import Case, F, Q, Value, When
from django.db.models.functions import Coalesce
from django.db.models.query import ValuesIterable

from .query_translate import (
    _get_query_related_name,
    translate_polymorphic_field_path,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from django.db.models.fields import Field as DjangoField

    from .models import PolymorphicModel

#: Annotation prefix for internal presence bits (kept out of the public row).
_PRESENT_PREFIX = "__poly_present__"


class _NotPresent:
    """Marker for a subclass field that does not apply to a row."""

    _instance: _NotPresent | None = None

    def __new__(cls) -> _NotPresent:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "NOT_PRESENT"

    def __copy__(self) -> _NotPresent:
        return self

    def __deepcopy__(self, memo: Any) -> _NotPresent:
        return self


NOT_PRESENT = _NotPresent()
"""
Sentinel used for projected subclass fields that do not apply to a row.
Compare with ``None``, which means the column exists on the row but is
``NULL`` in the database.
"""


class PolymorphicValuesRow(dict):
    """
    A single row returned by :meth:`PolymorphicQuerySet.polymorphic_values`.

    Behaves like the regular ``dict`` returned by
    :meth:`~django.db.models.query.QuerySet.values`, but subclass specific
    fields are either ``None`` (the column exists but is ``NULL``) or
    :data:`NOT_PRESENT` (the row is not an instance of the owning subclass).
    Presence metadata is available through the ``"<alias>__present"`` keys
    and the :meth:`is_present` helper.
    """

    def is_present(self, alias: str) -> bool:
        """Return whether ``alias`` belongs to this row's model type."""
        return bool(self.get(f"{alias}__present", True))

    def present_fields(self) -> set[str]:
        """Return the projected subclass aliases present on this row."""
        return {
            key[: -len("__present")]
            for key, present in self.items()
            if key.endswith("__present") and present
        }


class PolymorphicValuesIterable(ValuesIterable):
    """ValuesIterable that rewrites presence bits into the returned rows."""

    def __iter__(self):
        queryset = self.queryset
        query = queryset.query
        compiler = query.get_compiler(queryset.db)

        if query.selected:
            names = list(query.selected)
        else:
            names = [
                *query.extra_select,
                *query.values_select,
                *query.annotation_select,
            ]

        presence_map: dict[str, str] = getattr(queryset, "_polymorphic_presence_map", {})
        annotation_names = set(presence_map.values())
        ctype_alias = "polymorphic_ctype_id"

        for row in compiler.results_iter(
            chunked_fetch=self.chunked_fetch, chunk_size=self.chunk_size
        ):
            data = {names[i]: row[i] for i in range(len(names))}
            result = PolymorphicValuesRow()
            result[ctype_alias] = data.get(ctype_alias)

            is_present = {
                public_name: bool(data.get(annotation_name))
                for public_name, annotation_name in presence_map.items()
            }

            for name, value in data.items():
                if name == ctype_alias or name in annotation_names:
                    continue
                if name in is_present and not is_present[name]:
                    value = NOT_PRESENT
                result[name] = value

            for public_name, present in is_present.items():
                result[f"{public_name}__present"] = present

            yield result


@dataclass(frozen=True)
class _PolySpec:
    alias: str
    expression: Any
    target_model: type[PolymorphicModel] | None
    field_path: str
    field: DjangoField | None
    translated_path: str | None
    bare_positional: bool = False


def _resolve_field(model: type[PolymorphicModel], field_path: str) -> DjangoField:
    """Resolve a (possibly relation spanning) field path on a model."""
    current = model
    field = None
    for part in field_path.split("__"):
        if part == "pk":
            field = current._meta.pk
        else:
            try:
                field = current._meta.get_field(part)
            except FieldDoesNotExist as exc:
                choices = ", ".join(sorted(f.name for f in current._meta.get_fields()))
                raise FieldError(
                    f"Cannot resolve keyword '{part}' into field for model "
                    f"{current._meta.label} (in '{field_path}'). Choices are: {choices}"
                ) from exc
        related_model = getattr(field, "related_model", None)
        if related_model is not None:
            current = related_model
    assert field is not None
    return field


def _gate_path(base_model: type[PolymorphicModel], target_model: type[PolymorphicModel]) -> str:
    """
    Parent-link path from the base table to the concrete table backing
    ``target_model`` (empty string when that table is the base table itself).
    The result is usable directly as ``"<path>__isnull"``.
    """
    from .utils import route_to_ancestor

    concrete = target_model._meta.concrete_model
    base_concrete = base_model._meta.concrete_model
    if concrete is base_concrete:
        return ""

    route = route_to_ancestor(concrete, base_concrete)
    if not route:
        raise FieldError(
            f"Cannot determine inheritance path from {base_model._meta.label} "
            f"to {target_model._meta.label}."
        )
    # route runs child -> parent; the ORM join path runs parent -> child and
    # each step uses the child table's reverse parent-link name.
    parts = [_get_query_related_name(info.link.model) for info in reversed(route)]
    return "__".join(parts)


def _gate_condition(
    base_model: type[PolymorphicModel],
    target_model: type[PolymorphicModel],
    using: str,
) -> Q:
    """
    Condition that is true exactly for rows whose most derived concrete type
    is the table backing ``target_model`` (or a proxy descendant of it).

    For a real (multi-table) child the existence of the parent-link row is
    sufficient. For a proxy of an ancestor table the physical row is shared,
    so the polymorphic content type is used to keep the proxy identity.
    """
    from .utils import concrete_descendants

    gate_path = _gate_path(base_model, target_model)
    if gate_path:
        return Q(**{f"{gate_path}__isnull": False})

    # The concrete table is the base table itself. A plain base-model target
    # matches every row; only proxy targets need a content-type restriction.
    if not target_model._meta.proxy:
        return Q(pk__isnull=False)

    concrete = target_model._meta.concrete_model
    allowed = [target_model, *concrete_descendants(target_model, include_proxy=True)]
    allowed = [m for m in allowed if m._meta.concrete_model is concrete]
    ctype_ids = set()
    for model in allowed:
        ctype = ContentType.objects.db_manager(using).get_for_model(
            model, for_concrete_model=False
        )
        ctype_ids.add(ctype.pk)
    return Q(polymorphic_ctype_id__in=ctype_ids)


def _column_path(
    base_model: type[PolymorphicModel],
    target: type[PolymorphicModel],
    field_path: str,
) -> str:
    """
    Full ORM path selecting ``field_path`` on ``target`` from the base query,
    e.g. ``model2b__model2c__field3``.
    """
    if target._meta.proxy and target._meta.concrete_model is base_model._meta.concrete_model:
        # A proxy of the base table: the column physically lives in the base.
        return field_path
    return translate_polymorphic_field_path(base_model, f"{target.__name__}___{field_path}")


def _resolve_poly_spec(
    base_model: type[PolymorphicModel],
    alias: str,
    expression: Any,
    bare_positional: bool = False,
) -> _PolySpec:
    if not isinstance(expression, (str, F)):
        # Free-form expression (Value, Func, ...): used as-is.
        return _PolySpec(alias, expression, None, "", None, None)

    field_path = expression.name if isinstance(expression, F) else expression
    classname, sep, pure_path = field_path.partition("___")

    if not sep:
        field = _resolve_field(base_model, field_path)
        return _PolySpec(
            alias, expression, None, field_path, field, None, bare_positional
        )

    if not classname:
        raise FieldError(f"Invalid polymorphic field path '{field_path}'.")

    from .query_translate import _map_queryname_to_class

    try:
        target_model = _map_queryname_to_class(base_model, classname)
    except AssertionError as exc:
        raise FieldError(
            f"'{classname}' is not a subclass of {base_model._meta.label}"
        ) from exc

    if not pure_path:
        raise FieldError(
            f"Polymorphic field path '{field_path}' is missing a field name "
            f"(expected '{classname}___fieldname')."
        )

    field = _resolve_field(target_model, pure_path)
    translated = _column_path(base_model, target_model, pure_path)
    return _PolySpec(alias, expression, target_model, pure_path, field, translated)


def _validate_specs(specs: Sequence[_PolySpec], base_model: type[PolymorphicModel]) -> None:
    """Raise before SQL is emitted for conflicting projections."""
    base_local_fields = {f.name for f in base_model._meta.local_fields}

    by_alias: dict[str, list[_PolySpec]] = {}
    for spec in specs:
        by_alias.setdefault(spec.alias, []).append(spec)

    for alias, group in by_alias.items():
        has_common = any(s.target_model is None for s in group)
        has_subclass = any(s.target_model is not None for s in group)
        if has_common and has_subclass:
            raise FieldError(
                f"Field name '{alias}' is ambiguous: it is requested both as a "
                "common base field and as a subclass field. Use distinct aliases "
                "(alias='ModelX___field') to disambiguate."
            )

        subclass_group = [s for s in group if s.target_model is not None]
        if len(subclass_group) <= 1:
            continue
        type_keys = {
            s.field.get_internal_type() for s in subclass_group if s.field is not None
        }
        if len(type_keys) > 1:
            details = ", ".join(
                f"{s.target_model._meta.label}.{s.field_path} "
                f"({s.field.get_internal_type()})"
                for s in subclass_group
            )
            raise FieldError(
                f"Alias '{alias}' combines subclass fields with incompatible "
                f"data types: {details}."
            )

    # Inherited field conflict: a subclass-specific positional field
    # (alias == 'ModelX___field') whose leaf name collides with a *positional*
    # common field of the same bare name. Keyword-aliased projections are
    # unambiguous and are allowed.
    common_positional_names = {
        s.alias for s in specs if s.target_model is None and s.bare_positional
    }
    for spec in specs:
        if spec.target_model is None:
            continue
        leaf = spec.field_path.split("__")[-1]
        # Conflict requires the *output alias* to equal the bare common name;
        # a keyword-aliased subclass projection (e.g. red="Proxy___name") is
        # unambiguous.
        if (
            leaf in base_local_fields
            and leaf in common_positional_names
            and spec.alias == leaf
        ):
            raise FieldError(
                f"Field '{spec.expression}' inherits the base-model field "
                f"'{leaf}', which is also projected directly. The two cannot "
                "share an output name; use a keyword alias for one of them."
            )


def build_polymorphic_values_queryset(
    queryset: Any, fields: Iterable[Any], expressions: dict[str, Any]
) -> Any:
    """Build (without evaluating) the queryset for a polymorphic projection."""
    base_model = queryset.model

    specs: list[_PolySpec] = []
    for field in fields:
        if isinstance(field, tuple) and len(field) == 2:
            alias, expression = field
            if not isinstance(alias, str):
                raise FieldError("polymorphic_values() aliases must be strings.")
            specs.append(_resolve_poly_spec(base_model, alias, expression))
        elif isinstance(field, (str, F)):
            specs.append(
                _resolve_poly_spec(base_model, field, field, bare_positional=True)
            )
        else:
            raise FieldError(
                "polymorphic_values() positional arguments must be field names, "
                "F() expressions, or (alias, expression) tuples."
            )

    for alias, expression in expressions.items():
        if alias.startswith("__"):
            raise FieldError(
                f"Invalid alias '{alias}': names starting with '__' are reserved "
                "for polymorphic projection metadata."
            )
        specs.append(_resolve_poly_spec(base_model, alias, expression))

    _validate_specs(specs, base_model)

    clone = queryset._chain()
    clone.polymorphic_disabled = True

    value_fields: list[Any] = ["polymorphic_ctype_id"]
    value_expressions: dict[str, Any] = {}
    presence_names: list[str] = []
    presence_map: dict[str, str] = {}

    by_alias: dict[str, list[_PolySpec]] = {}
    for spec in specs:
        if spec.target_model is None:
            if isinstance(spec.expression, str):
                value_fields.append(spec.expression)
            else:
                # F()/Func/... expression: register it under the alias so
                # set_values() receives a plain string key.
                value_expressions[spec.alias] = spec.expression
                value_fields.append(spec.alias)
        else:
            by_alias.setdefault(spec.alias, []).append(spec)

    for index, (alias, group) in enumerate(by_alias.items()):
        # Per-alias presence: the row is "present" when *any* contributing
        # concrete table joins in.
        presence_name = f"{_PRESENT_PREFIX}alias_{index}"
        conditions: Q | None = None
        for spec in group:
            condition = _gate_condition(base_model, spec.target_model, clone.db)
            conditions = condition if conditions is None else (conditions | condition)
        assert conditions is not None
        clone.query.add_annotation(
            Case(When(conditions, then=Value(1)), default=Value(0)),
            presence_name,
        )
        presence_names.append(presence_name)
        presence_map[alias] = presence_name

        expression = F(group[0].translated_path)
        for spec in group[1:]:
            expression = Coalesce(expression, F(spec.translated_path))
        clone.query.add_annotation(expression, alias)
        value_fields.append(alias)

    value_fields.extend(presence_names)

    clone = super(type(queryset), clone)._values(*value_fields, **value_expressions)
    clone._iterable_class = PolymorphicValuesIterable
    clone._polymorphic_presence_map = presence_map
    return clone
