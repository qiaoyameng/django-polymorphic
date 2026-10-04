"""
Bulk updating polymorphic objects.

This module adds support for updating a heterogeneous list of polymorphic objects
inside a single database transaction. Every item carries its own set of fields to
update; fields may be plain names (resolved against the object's concrete class)
or use the ``ModelName___field`` syntax to address a field on an ancestor or
descendant model explicitly.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

from django.apps import apps
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import DEFAULT_DB_ALIAS, connections, models, router, transaction
from django.db.models import Case, F, Value, When
from django.db.models.fields.related import ForeignObjectRel, RelatedField

from .models import PolymorphicModel
from .utils import concrete_descendants, get_base_polymorphic_model

__all__ = [
    "ObjectUpdate",
    "UpdateStatus",
    "ObjectUpdateResult",
    "BulkUpdateReport",
    "BulkUpdateError",
    "BulkUpdateValidationError",
    "ObjectNotFound",
    "StaleObjectError",
    "StrictBulkUpdateError",
    "polymorphic_bulk_update",
]


class UpdateStatus(str, Enum):
    """Per-object outcome of a bulk update."""

    UPDATED = "updated"
    NOT_FOUND = "not_found"
    STALE = "stale"
    ERROR = "error"
    DUPLICATE = "duplicate"


class BulkUpdateError(Exception):
    """Base class for bulk update related errors."""


class BulkUpdateValidationError(BulkUpdateError):
    """An individual update item is invalid (bad field or value)."""


class ObjectNotFound(BulkUpdateError):
    """A row of an object to update no longer exists."""


class StaleObjectError(BulkUpdateError):
    """An input object conflicts with the database content type (deleted/changed)."""


class StrictBulkUpdateError(BulkUpdateError):
    """Raised in strict mode when one or more items could not be updated."""


@dataclass
class ObjectUpdate:
    """
    A single object with the fields to update on it.

    :param obj: The polymorphic model instance (its primary key must be set).
    :param fields: Mapping of field names (plain or ``ModelName___field``)
        to the new value. Values may be plain Python values or
        :class:`~django.db.models.F` expressions.
    """

    obj: PolymorphicModel
    fields: dict[str, Any]


@dataclass
class ObjectUpdateResult:
    """The result of updating a single :class:`ObjectUpdate`."""

    object: ObjectUpdate
    status: UpdateStatus
    error: Exception | None = None
    #: Concrete tables whose row was missing (for ``NOT_FOUND`` results).
    missing_tables: list[str] = field(default_factory=list)

    @property
    def updated(self) -> bool:
        return self.status is UpdateStatus.UPDATED

    @property
    def ok(self) -> bool:
        return self.status is UpdateStatus.UPDATED


@dataclass
class BulkUpdateReport:
    """Per-object results of a bulk update, in input order."""

    results: list[ObjectUpdateResult] = field(default_factory=list)

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> ObjectUpdateResult:
        return self.results[index]

    @property
    def updated_objects(self) -> list[ObjectUpdate]:
        return [r.object for r in self.results if r.status is UpdateStatus.UPDATED]

    @property
    def failed_results(self) -> list[ObjectUpdateResult]:
        return [r for r in self.results if r.status is not UpdateStatus.UPDATED]

    @property
    def all_succeeded(self) -> bool:
        return not self.failed_results


class _ResolvedField(NamedTuple):
    field_name: str
    target_model: type[models.Model]
    owner_model: type[models.Model]
    field: models.Field[Any, Any]
    value: Any


class _PreparedItem(NamedTuple):
    index: int
    item: ObjectUpdate
    obj: PolymorphicModel
    concrete_model: type[models.Model]
    root_model: type[models.Model]
    db: str
    resolved: list[_ResolvedField]


# Polymorphic-managed fields which can never be changed via this API.
_PROTECTED_FIELD_NAMES = frozenset({"polymorphic_ctype", "polymorphic_ctype_id"})


def _plain_queryset(model: type[models.Model], using: str) -> models.QuerySet:
    """A plain (non-polymorphic, unfiltered) queryset for raw table operations."""
    return model._base_manager.db_manager(using).non_polymorphic()


def _resolve_named_model(
    root_model: type[models.Model], name: str
) -> type[models.Model]:
    """Resolve a ``ModelName`` or ``app_label__ModelName`` component."""
    if "__" in name:
        app_label, model_name = name.split("__", 1)
        try:
            model = apps.get_model(app_label, model_name)
        except LookupError as exc:
            raise FieldError(f"Model {app_label}.{model_name} does not exist") from exc
        if not issubclass(model, root_model):
            raise FieldError(
                f"{model._meta.label} is not derived from {root_model._meta.label}"
            )
        return model

    # A relation-style field takes precedence (mirrors query_translate behaviour).
    try:
        related = root_model._meta.get_field(name)
    except FieldDoesNotExist:
        pass
    else:
        if isinstance(related, (RelatedField, ForeignObjectRel)):
            raise BulkUpdateValidationError(
                f"Cannot update relation path '{name}'; bulk update only supports "
                "model-local fields (use 'ModelName___field' for subclass fields)."
            )

    name_map: dict[str, list[type[models.Model]]] = defaultdict(list)
    candidates: list[type[models.Model]] = [root_model]
    # Search the complete hierarchy, proxy models included (they map to a table).
    candidates.extend(concrete_descendants(root_model, include_proxy=True))
    for cls in candidates:
        name_map[cls.__name__.lower()].append(cls)

    matches = name_map.get(name.lower(), [])
    if len(matches) > 1:
        labels = ", ".join(sorted(m._meta.label for m in matches))
        raise FieldError(
            f"{name} could refer to any of [{labels}]. "
            f"Use the syntax applabel__{name}___field to disambiguate."
        )
    if not matches:
        raise FieldError(f"{name} is not a subclass of {root_model._meta.label}")
    return matches[0]


def _protected_pk_fields(model: type[models.Model]) -> set[models.Field[Any, Any]]:
    """All primary key fields along the multi-table inheritance chain."""
    pks: set[models.Field[Any, Any]] = set(model._meta.pk_fields)
    for parent in model._meta.all_parents:
        pks.update(parent._meta.pk_fields)
    return pks


def _concrete_chain(model: type[models.Model]) -> list[type[models.Model]]:
    """Concrete models from ``model`` up to (and including) its concrete root."""
    chain: list[type[models.Model]] = []
    current: type[models.Model] | None = model
    while current is not None:
        if not current._meta.abstract and not current._meta.proxy:
            chain.append(current)
        if current._meta.proxy:
            # Proxy models do not introduce tables; continue on proxied model.
            current = current._meta.proxy_for_model
        else:
            parents = [
                parent
                for parent in current._meta.parents
                if not parent._meta.abstract and parent is not current
            ]
            if not parents:
                break
            # MTI chains used by polymorphic models are single inheritance.
            current = parents[0]
    return chain


def _owner_of_field(
    model: type[models.Model], field: models.Field[Any, Any]
) -> type[models.Model] | None:
    """Return the concrete model/table that owns ``field`` in ``model``'s chain."""
    for candidate in _concrete_chain(model):
        if any(local is field for local in candidate._meta.local_fields):
            return candidate
    return None


def _resolve_field(
    obj: PolymorphicModel,
    obj_class: type[models.Model],
    concrete_model: type[models.Model],
    root_model: type[models.Model],
    field_path: str,
    value: Any,
) -> _ResolvedField:
    """Resolve and validate a single (field_path, value) pair for one object."""
    prefix, sep, field_name = field_path.partition("___")
    if sep and prefix:
        target_model = _resolve_named_model(root_model, prefix)
        # Proxy models map to their concrete table; the type relationship is
        # still checked against the (possibly proxy) class named by the user.
        if not issubclass(concrete_model, target_model):
            raise BulkUpdateValidationError(
                f"Field '{field_path}' belongs to {target_model._meta.label}, "
                f"which is not an ancestor of {concrete_model._meta.label}: "
                "sibling/descendant subclass fields cannot be written to this object."
            )
        lookup_model = target_model
    else:
        target_model = concrete_model
        lookup_model = obj_class
        field_name = field_path

    if "__" in field_name:
        raise BulkUpdateValidationError(
            f"Cannot update field path '{field_path}': expressions may not cross "
            "relations. Only simple fields of the object's own tables are supported."
        )

    if field_name in _PROTECTED_FIELD_NAMES:
        raise BulkUpdateValidationError(
            f"'{field_name}' is managed by django-polymorphic and cannot be modified."
        )

    try:
        field = lookup_model._meta.get_field(field_name)
    except FieldDoesNotExist as exc:
        raise BulkUpdateValidationError(
            f"{lookup_model._meta.label} has no field '{field_name}'"
        ) from exc

    if not field.concrete or field.many_to_many:
        raise BulkUpdateValidationError(
            f"Only concrete fields can be bulk updated ('{field_name}' is not)."
        )

    if field in _protected_pk_fields(concrete_model):
        raise BulkUpdateValidationError(
            f"Primary key field '{field_name}' cannot be modified."
        )

    # Parent links hold the shared primary keys of the inheritance chain.
    remote_field = getattr(field, "remote_field", None)
    if remote_field is not None and getattr(remote_field, "parent_link", False):
        raise BulkUpdateValidationError(
            f"Parent link field '{field_name}' cannot be modified."
        )

    owner_model = _owner_of_field(target_model, field)
    if owner_model is None or not issubclass(concrete_model, owner_model):
        raise BulkUpdateValidationError(
            f"Field '{field_path}' is not stored on a table of "
            f"{concrete_model._meta.label}."
        )

    if field.is_relation and isinstance(value, models.Model):
        # Assign via the descriptor so the ``*_id`` attribute and relation cache
        # stay consistent; _prepare_related_fields_for_save then rejects unsaved
        # related instances. A wrong model type raises ValueError here.
        try:
            setattr(obj, field.name, value)
        except ValueError as exc:
            raise BulkUpdateValidationError(str(exc)) from exc

    # Validate values: plain values or restricted F expressions only.
    if isinstance(value, F):
        f_name = value.name
        if not isinstance(f_name, str) or "__" in f_name:
            raise BulkUpdateValidationError(
                f"F expression F({f_name!r}) for field '{field_path}' may only "
                "reference a simple field of the same table; cross-table or "
                "relation paths are not supported."
            )
        if f_name in _PROTECTED_FIELD_NAMES:
            raise BulkUpdateValidationError(
                f"F expression for '{field_path}' references protected field "
                f"'{f_name}'."
            )
        try:
            f_field = target_model._meta.get_field(f_name)
        except FieldDoesNotExist as exc:
            raise BulkUpdateValidationError(
                f"{target_model._meta.label} has no field '{f_name}' for an "
                f"F expression assigned to '{field_path}'."
            ) from exc
        if not f_field.concrete or f_field.many_to_many:
            raise BulkUpdateValidationError(
                f"F expression F({f_name!r}) for field '{field_path}' must "
                "reference a concrete field."
            )
        # The referenced column must live on the very same table being updated.
        f_owner = _owner_of_field(target_model, f_field)
        if f_owner is not owner_model:
            raise BulkUpdateValidationError(
                f"F expression for '{field_path}' cannot cross tables: "
                f"'{f_name}' lives on {f_owner._meta.db_table if f_owner else '?'} "
                f"but the target column lives on {owner_model._meta.db_table}."
            )

    return _ResolvedField(field_name, target_model, owner_model, field, value)


def _coerce_item(item: Any) -> ObjectUpdate:
    if isinstance(item, ObjectUpdate):
        if not isinstance(item.fields, Mapping):
            raise TypeError(
                "ObjectUpdate.fields must be a mapping of field name to value."
            )
        return item
    if isinstance(item, tuple) and len(item) == 2:
        obj, fields = item
        if isinstance(fields, Mapping):
            return ObjectUpdate(obj=obj, fields=dict(fields))
        if isinstance(fields, Iterable) and not isinstance(fields, (str, bytes)):
            # Django bulk_update style: names are read from the instance.
            return ObjectUpdate(obj=obj, fields={name: getattr(obj, name) for name in fields})
    raise TypeError(
        "Each bulk update item must be an ObjectUpdate or a "
        "(instance, {field: value}) / (instance, [field, ...]) tuple."
    )


def _prepare_item(
    index: int,
    item: ObjectUpdate,
    *,
    allowed_base: type[models.Model] | None,
) -> _PreparedItem:
    obj = item.obj
    if not isinstance(obj, PolymorphicModel):
        raise BulkUpdateValidationError(
            f"{obj!r} is not a PolymorphicModel; only polymorphic objects are supported."
        )
    if not obj._is_pk_set():
        raise BulkUpdateValidationError(
            "All bulk updated objects must have a primary key set."
        )

    root_model = get_base_polymorphic_model(type(obj))
    assert root_model is not None  # guaranteed by isinstance check above
    concrete_model = type(obj)._meta.concrete_model
    obj_class = type(obj)

    if allowed_base is not None and not issubclass(concrete_model, allowed_base):
        raise BulkUpdateValidationError(
            f"{root_model._meta.label} is not part of the {allowed_base._meta.label} "
            "polymorphic tree used for this update."
        )

    using = router.db_for_write(root_model, instance=obj) or obj._state.db or DEFAULT_DB_ALIAS

    resolved = [
        _resolve_field(obj, obj_class, concrete_model, root_model, name, value)
        for name, value in item.fields.items()
    ]
    return _PreparedItem(index, item, obj, concrete_model, root_model, using, resolved)


def _preflight(
    prepared: list[_PreparedItem],
) -> tuple[
    dict[int, ObjectUpdateResult],
    list[_PreparedItem],
    dict[tuple[str, type[models.Model]], list[_PreparedItem]],
]:
    """
    Validate that every row still exists and matches the input object's type.

    Returns (results_by_index, survivors, items_by_table) where results_by_index
    contains results for rejected items and items_by_table maps each concrete
    table to the surviving items updating it.
    """
    rejected: dict[int, ObjectUpdateResult] = {}

    # 1) Root-table rows: existence + polymorphic_ctype vs. input type.
    root_groups: dict[tuple[str, type[models.Model]], list[_PreparedItem]] = defaultdict(list)
    for prepared_item in prepared:
        root_groups[(prepared_item.db, prepared_item.root_model)].append(prepared_item)

    for (using, root_model), group in root_groups.items():
        pks = {item.obj.pk for item in group}
        stored_ct_ids: dict[Any, int | None] = dict(
            _plain_queryset(root_model, using)
            .filter(pk__in=pks)
            .values_list("pk", "polymorphic_ctype_id")
        )
        # Resolve stored (possibly proxy) content types to their concrete model.
        ct_models: dict[int, type[models.Model] | None] = {}
        known_ids = {ct_id for ct_id in stored_ct_ids.values() if ct_id is not None}
        if known_ids:
            from django.contrib.contenttypes.models import ContentType

            for ct_id, app_label, model_name in ContentType.objects.db_manager(using).filter(
                pk__in=known_ids
            ).values_list("pk", "app_label", "model"):
                resolved_ct = apps.get_model(app_label, model_name)
                ct_models[ct_id] = (
                    resolved_ct._meta.concrete_model if resolved_ct is not None else None
                )
        for item in group:
            stored_ct_id = stored_ct_ids.get(item.obj.pk, _MISSING)
            if stored_ct_id is _MISSING:
                rejected[item.index] = ObjectUpdateResult(
                    item.item,
                    UpdateStatus.NOT_FOUND,
                    ObjectNotFound(
                        f"{root_model._meta.label} with pk {item.obj.pk} does not exist."
                    ),
                    [root_model._meta.db_table],
                )
                continue
            stored_concrete = ct_models.get(stored_ct_id) if stored_ct_id is not None else None
            if stored_concrete is not item.concrete_model:
                rejected[item.index] = ObjectUpdateResult(
                    item.item,
                    UpdateStatus.STALE,
                    StaleObjectError(
                        f"Input object {item.concrete_model._meta.label}#{item.obj.pk} "
                        "does not match the polymorphic type stored in the database."
                    ),
                )

    # 2) Every concrete table updated by a surviving item must have a row.
    table_groups: dict[tuple[str, type[models.Model]], list[_PreparedItem]] = defaultdict(list)
    survivors: list[_PreparedItem] = []
    for prepared_item in prepared:
        if prepared_item.index in rejected:
            continue
        owners = {resolved.owner_model for resolved in prepared_item.resolved}
        if not owners:
            # Nothing to write; a successful no-op.
            rejected[prepared_item.index] = ObjectUpdateResult(
                prepared_item.item, UpdateStatus.UPDATED
            )
            continue
        survivors.append(prepared_item)
        for owner in owners:
            table_groups[(prepared_item.db, owner)].append(prepared_item)

    missing_by_index: dict[int, list[str]] = defaultdict(list)
    for (using, table_model), group in table_groups.items():
        pks = {item.obj.pk for item in group}
        found = set(
            _plain_queryset(table_model, using)
            .filter(pk__in=pks)
            .values_list("pk", flat=True)
        )
        for item in group:
            if item.obj.pk not in found:
                missing_by_index[item.index].append(table_model._meta.db_table)

    final_survivors: list[_PreparedItem] = []
    for item in survivors:
        if item.index in missing_by_index:
            rejected[item.index] = ObjectUpdateResult(
                item.item,
                UpdateStatus.NOT_FOUND,
                ObjectNotFound(
                    f"Row(s) missing for {item.concrete_model._meta.label}#{item.obj.pk}: "
                    + ", ".join(missing_by_index[item.index])
                ),
                missing_by_index[item.index],
            )
        else:
            final_survivors.append(item)

    surviving_table_groups: dict[
        tuple[str, type[models.Model]], list[_PreparedItem]
    ] = defaultdict(list)
    for (using, table_model), group in table_groups.items():
        live = [item for item in group if item.index not in rejected]
        if live:
            surviving_table_groups[(using, table_model)] = live

    return rejected, final_survivors, surviving_table_groups





_MISSING = object()


def _chunks(items: list[_PreparedItem], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _fields_for_table(
    item: _PreparedItem, table_model: type[models.Model]
) -> list[_ResolvedField]:
    return [resolved for resolved in item.resolved if resolved.owner_model is table_model]


def _plain_column_value(item: _PreparedItem, resolved: _ResolvedField) -> Any:
    """Effective scalar value for a column (resolves related instances to pks)."""
    value = resolved.value
    if resolved.field.is_relation and isinstance(value, models.Model):
        # After _prepare_related_fields_for_save the *_id attribute holds the pk.
        value = getattr(item.obj, resolved.field.attname)
    return value


def _then_value(item: _PreparedItem, resolved: _ResolvedField) -> Any:
    if isinstance(resolved.value, F):
        return resolved.value
    return Value(_plain_column_value(item, resolved), output_field=resolved.field)


def _update_single_row(
    item: _PreparedItem, table_model: type[models.Model], *, use_savepoint: bool
) -> None:
    kwargs: dict[str, Any] = {}
    for resolved in _fields_for_table(item, table_model):
        try:
            item.obj._prepare_related_fields_for_save(
                operation_name="polymorphic_bulk_update", fields=[resolved.field]
            )
        except ValueError as exc:
            raise BulkUpdateValidationError(str(exc)) from exc
        kwargs[resolved.field.attname] = _plain_column_value(item, resolved)
    with transaction.atomic(using=item.db, savepoint=use_savepoint):
        rows = _plain_queryset(table_model, item.db).filter(pk=item.obj.pk).update(**kwargs)
    if rows == 0:
        raise ObjectNotFound(
            f"{table_model._meta.label} row {item.obj.pk} disappeared during update."
        )


def _update_table(
    table_model: type[models.Model],
    using: str,
    items: list[_PreparedItem],
    batch_size: int | None,
    *,
    strict: bool,
    excluded_indexes: set[int],
) -> dict[int, tuple[UpdateStatus, Exception | None]]:
    """
    Update one concrete table in batches.

    Per-item validation errors (e.g. unsaved related objects) are reported
    without issuing SQL for that item. Database failures are isolated per batch
    with savepoints in non-strict mode and retried row by row, so an individual
    constraint error can never be reported as a batch-wide success. In strict
    mode any failure propagates to roll back the whole transaction.
    """
    failures: dict[int, tuple[UpdateStatus, Exception | None]] = {}
    if not items:
        return failures

    # Items that failed cross-table validation are excluded entirely.
    valid_items = [item for item in items if item.index not in excluded_indexes]
    if not valid_items:
        return failures

    fields = list(
        {
            resolved.field
            for item in valid_items
            for resolved in _fields_for_table(item, table_model)
        }
    )
    connection = connections[using]
    max_batch = connection.ops.bulk_batch_size(
        [table_model._meta.pk, table_model._meta.pk] + fields,
        [item.obj for item in valid_items],
    )
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        effective_size = min(batch_size, max_batch)
    else:
        effective_size = max_batch

    requires_casting = connection.features.requires_casted_case_in_updates

    for batch in _chunks(valid_items, effective_size):
        update_kwargs: dict[str, Any] = {}
        for db_field in fields:
            whens = []
            for item in batch:
                matching = next(
                    (
                        resolved
                        for resolved in _fields_for_table(item, table_model)
                        if resolved.field is db_field
                    ),
                    None,
                )
                then = (
                    _then_value(item, matching)
                    if matching is not None
                    else F(db_field.attname)
                )
                whens.append(When(pk=item.obj.pk, then=then))
            case = Case(*whens, output_field=db_field)
            if requires_casting:
                from django.db.models.functions import Cast

                case = Cast(case, output_field=db_field)
            update_kwargs[db_field.attname] = case

        pks = [item.obj.pk for item in batch]
        try:
            with transaction.atomic(using=using, savepoint=not strict):
                rows_updated = (
                    _plain_queryset(table_model, using)
                    .filter(pk__in=pks)
                    .update(**update_kwargs)
                )
            if rows_updated != len(batch):
                raise ObjectNotFound(
                    f"expected {len(batch)} updated row(s) on "
                    f"{table_model._meta.db_table}, got {rows_updated}"
                )
        except Exception as batch_error:
            if strict:
                raise StrictBulkUpdateError(
                    f"Failed to bulk update {table_model._meta.label}: {batch_error}"
                ) from batch_error

            # The savepoint was rolled back; isolate offending rows.
            for item in batch:
                try:
                    _update_single_row(item, table_model, use_savepoint=True)
                except ObjectNotFound as exc:
                    failures[item.index] = (UpdateStatus.NOT_FOUND, exc)
                except Exception as exc:
                    failures[item.index] = (UpdateStatus.ERROR, exc)

    return failures


def _ordered_tables(
    table_keys: Iterable[tuple[str, type[models.Model]]],
) -> list[tuple[str, type[models.Model]]]:
    """Order table keys leaf-first (children before parents) per database."""
    by_db: dict[str, list[type[models.Model]]] = defaultdict(list)
    for using, model in table_keys:
        by_db[using].append(model)

    ordered: list[tuple[str, type[models.Model]]] = []
    for using, models_in_db in by_db.items():
        models_in_db.sort(
            key=lambda model: -len(
                [p for p in model._meta.get_parent_list() if not p._meta.abstract]
            )
        )
        ordered.extend((using, model) for model in models_in_db)
    return ordered


def _sync_instances(
    items_by_table: dict[tuple[str, type[models.Model]], list[_PreparedItem]],
    success_indexes: set[int],
) -> None:
    """Reflect plain (non-F) new values on the in-memory instances."""
    for items in items_by_table.values():
        for item in items:
            if item.index not in success_indexes:
                continue
            for resolved in item.resolved:
                if isinstance(resolved.value, F):
                    continue
                setattr(item.obj, resolved.field.attname, resolved.value)
                if resolved.field.is_relation and resolved.field.is_cached(item.obj):
                    resolved.field.delete_cached_value(item.obj)


def _validate_related_values(
    survivors: list[_PreparedItem],
    items_by_table: dict[tuple[str, type[models.Model]], list[_PreparedItem]],
    *,
    strict: bool,
) -> tuple[set[int], dict[int, ObjectUpdateResult]]:
    """
    Validate related objects for every surviving item across all of its tables
    *before* any UPDATE is issued, so an invalid item is never partially written.
    """
    rejected_indexes: set[int] = set()
    results: dict[int, ObjectUpdateResult] = {}
    by_index = {item.index: item for item in survivors}
    for (using, _table_model), group in items_by_table.items():
        for item in group:
            if item.index in rejected_indexes:
                continue
            table_fields = [
                resolved.field for resolved in _fields_for_table(item, _table_model)
            ]
            try:
                item.obj._prepare_related_fields_for_save(
                    operation_name="polymorphic_bulk_update", fields=table_fields
                )
            except ValueError as exc:
                validation_error = BulkUpdateValidationError(str(exc))
                if strict:
                    raise StrictBulkUpdateError(str(validation_error)) from exc
                rejected_indexes.add(item.index)
                results[item.index] = ObjectUpdateResult(
                    by_index[item.index].item, UpdateStatus.ERROR, validation_error
                )
    return rejected_indexes, results


def polymorphic_bulk_update(
    objects: Iterable[ObjectUpdate | tuple[PolymorphicModel, Any]],
    *,
    batch_size: int | None = None,
    strict: bool = False,
    using: str | None = None,
    queryset_model: type[models.Model] | None = None,
) -> BulkUpdateReport:
    """
    Bulk update a heterogeneous list of polymorphic objects.

    :param objects: :class:`ObjectUpdate` instances, ``(instance, {field: value})``
        mappings or ``(instance, [field, ...])`` tuples. The same object (same root
        table and primary key) appearing more than once is rejected.
    :param batch_size: Maximum number of objects per UPDATE statement per table.
    :param strict: When true any failure rolls back the entire batch and raises
        :class:`StrictBulkUpdateError`. When false (default) valid updates are
        committed and a per-object :class:`BulkUpdateReport` is returned.
    :param using: Optional database alias; all objects must route to it.
    :param queryset_model: Internal constraint used by the queryset method to limit
        accepted objects to one polymorphic tree.
    :return: A report with one :class:`ObjectUpdateResult` per input object.
    """
    raw_items = list(objects)

    # Phase 1: normalization and validation (no database writes yet).
    results_by_index: dict[int, ObjectUpdateResult] = {}
    prepared_by_index: dict[int, _PreparedItem] = {}
    seen: dict[tuple[str, str, Any], int] = {}
    dbs: set[str] = set()

    for index, raw in enumerate(raw_items):
        # Malformed input shapes are programmer errors and raise immediately.
        item = _coerce_item(raw)
        try:
            prepared = _prepare_item(index, item, allowed_base=queryset_model)
        except (BulkUpdateError, FieldError) as exc:
            results_by_index[index] = ObjectUpdateResult(
                item, UpdateStatus.ERROR, exc
            )
            continue

        if using is not None and prepared.db != using:
            results_by_index[index] = ObjectUpdateResult(
                item,
                UpdateStatus.ERROR,
                BulkUpdateValidationError(
                    f"Object is routed to database '{prepared.db}', not '{using}'."
                ),
            )
            continue

        duplicate_key = (prepared.db, prepared.root_model._meta.label, prepared.obj.pk)
        if duplicate_key in seen:
            results_by_index[index] = ObjectUpdateResult(
                item,
                UpdateStatus.DUPLICATE,
                BulkUpdateValidationError(
                    f"{prepared.concrete_model._meta.label}#{prepared.obj.pk} appears "
                    f"more than once (first at index {seen[duplicate_key]}); duplicate "
                    "objects are rejected."
                ),
            )
            continue
        seen[duplicate_key] = index
        prepared_by_index[index] = prepared
        dbs.add(prepared.db)

    if using is None and len(dbs) > 1:
        # A single transaction cannot span multiple routed databases here.
        for index, prepared in prepared_by_index.items():
            results_by_index[index] = ObjectUpdateResult(
                prepared.item,
                UpdateStatus.ERROR,
                BulkUpdateValidationError(
                    "polymorphic_bulk_update() cannot update objects routed to "
                    f"multiple databases in one call (found: {sorted(dbs)})."
                ),
            )
        prepared_by_index = {}

    validation_failures = [
        result
        for result in results_by_index.values()
        if result.status is not UpdateStatus.UPDATED
    ]
    if strict and validation_failures:
        raise StrictBulkUpdateError(
            "; ".join(
                f"{result.status.value}: {result.error}" for result in validation_failures
            )
        )

    prepared = [prepared_by_index[index] for index in sorted(prepared_by_index)]

    def build_report() -> BulkUpdateReport:
        report = BulkUpdateReport()
        for index in range(len(raw_items)):
            result = results_by_index.get(index)
            if result is None:
                result = ObjectUpdateResult(
                    prepared_by_index[index].item, UpdateStatus.UPDATED
                )
                results_by_index[index] = result
            report.results.append(result)
        return report

    if not prepared:
        return build_report()

    write_db = using or next(iter(dbs))

    # Phases 2 and 3: preflight and per-table updates in one transaction.
    with transaction.atomic(using=write_db):
        preflight_results, survivors, items_by_table = _preflight(prepared)
        # Successful no-ops fill in otherwise missing results; failures overwrite.
        for index, result in preflight_results.items():
            if result.status is UpdateStatus.UPDATED:
                results_by_index.setdefault(index, result)
            else:
                results_by_index[index] = result

        if strict and any(
            result.status is not UpdateStatus.UPDATED
            for result in preflight_results.values()
        ):
            blocking = [
                result
                for result in preflight_results.values()
                if result.status is not UpdateStatus.UPDATED
            ]
            raise StrictBulkUpdateError(
                "; ".join(f"{result.status.value}: {result.error}" for result in blocking)
            )

        excluded_indexes, validation_results = _validate_related_values(
            survivors, items_by_table, strict=strict
        )
        for index, result in validation_results.items():
            results_by_index[index] = result
        survivor_indexes = {item.index for item in survivors} - excluded_indexes

        # Drop rejected items from the table groups.
        if excluded_indexes:
            items_by_table = {
                key: [item for item in group if item.index not in excluded_indexes]
                for key, group in items_by_table.items()
            }
            items_by_table = {key: group for key, group in items_by_table.items() if group}

        table_failures: dict[int, tuple[UpdateStatus, Exception | None]] = {}

        for table_using, table_model in _ordered_tables(items_by_table.keys()):
            table_failures.update(
                _update_table(
                    table_model,
                    table_using,
                    items_by_table[(table_using, table_model)],
                    batch_size,
                    strict=strict,
                    excluded_indexes=excluded_indexes,
                )
            )

        # NOT_FOUND wins over ERROR if multiple tables report for one object.
        status_priority = {
            UpdateStatus.ERROR: 1,
            UpdateStatus.NOT_FOUND: 2,
        }
        for index, (status, exc) in table_failures.items():
            current = results_by_index.get(index)
            if current is None or status_priority.get(status, 0) > status_priority.get(
                current.status, 0
            ):
                results_by_index[index] = ObjectUpdateResult(
                    prepared_by_index[index].item, status, exc
                )

        success_indexes = survivor_indexes - set(table_failures)
        _sync_instances(items_by_table, success_indexes)

    return build_report()
