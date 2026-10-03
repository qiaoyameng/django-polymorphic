"""
QuerySet for PolymorphicModel
"""

from __future__ import annotations

import copy
import heapq
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Generic, cast, overload

from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import FieldDoesNotExist
from django.db import connections, models, transaction
from django.db.models import AutoField
from django.db.models import FilteredRelation, Q
from django.db.models.expressions import Combinable
from django.db.models.query import ModelIterable, QuerySet
from typing_extensions import Self, TypeVar

from .query_translate import (
    translate_polymorphic_field_path,
    translate_polymorphic_filter_definitions_in_args,
    translate_polymorphic_filter_definitions_in_kwargs,
    translate_polymorphic_Q_object,
)
from .utils import concrete_descendants, get_base_polymorphic_model, route_to_ancestor

if TYPE_CHECKING:
    from .models import PolymorphicBulkCreateError, PolymorphicModel  # noqa: F401

_All = TypeVar("_All", bound="PolymorphicModel", covariant=True)
_Base = TypeVar("_Base", bound="PolymorphicModel", default="PolymorphicModel", covariant=True)

_A = TypeVar("_A", bound="PolymorphicModel")
_B = TypeVar("_B", bound="PolymorphicModel")
_C = TypeVar("_C", bound="PolymorphicModel")
_D = TypeVar("_D", bound="PolymorphicModel")

Polymorphic_QuerySet_objects_per_request: int = 2000
"""
The maximum number of objects requested per db-request by the polymorphic
queryset.iterator() implementation
"""

if TYPE_CHECKING:

    class BasePolymorphicModelIterable(ModelIterable[_All]):
        pass
else:

    class BasePolymorphicModelIterable(ModelIterable):
        pass


class _Inconsistent:
    """
    A marker class indicating that there is a mismatch between the content type
    and the actual class of an object retrieved from the database.
    """

    ...


class PolymorphicModelIterable(BasePolymorphicModelIterable, Generic[_All, _Base]):
    """
    ModelIterable for PolymorphicModel

    Yields real instances if qs.polymorphic_disabled is False,
    otherwise acts like a regular ModelIterable.
    """

    queryset: "PolymorphicQuerySet[_All, _Base]"

    def __iter__(self) -> Iterator[_All]:
        base_iter = super().__iter__()
        if self.queryset.polymorphic_disabled:
            return base_iter
        return self._polymorphic_iterator(base_iter)

    def _polymorphic_iterator(self, base_iter: Iterator[_All]) -> Iterator[_All]:
        """
        Here we do the same as::

            real_results = queryset._get_real_instances(list(base_iter))
            for o in real_results: yield o

        but it requests the objects in chunks from the database,
        with QuerySet.iterator(chunk_size) per chunk
        """

        # some databases have a limit on the number of query parameters, we must
        # respect this for generating get_real_instances queries because those
        # queries do a large WHERE IN clause with primary keys
        max_chunk = connections[self.queryset.db].features.max_query_params
        sql_chunk = self.chunk_size if self.chunked_fetch else None
        if max_chunk:
            sql_chunk = (
                max_chunk
                if not self.chunked_fetch  # chunk_size was not provided
                else min(max_chunk, self.chunk_size or max_chunk)
            )

        sql_chunk = sql_chunk or Polymorphic_QuerySet_objects_per_request

        while True:
            base_result_objects = []
            reached_end = False

            # Fetch in chunks
            for _ in range(sql_chunk):
                try:
                    o = next(base_iter)
                    base_result_objects.append(o)
                except StopIteration:
                    reached_end = True
                    break

            yield from self.queryset._get_real_instances(base_result_objects)

            if reached_end:
                return


def transmogrify(cls: type[_All], obj: models.Model) -> _All:
    """
    Upcast a class to a different type without asking questions.
    """
    if "__init__" not in obj.__class__.__dict__:
        # Just assign __class__ to a different value.
        new = obj
        new.__class__ = cls
        return cast(_All, new)
    else:
        # Run constructor, reassign values
        new = cls()
        for k, v in obj.__dict__.items():
            new.__dict__[k] = v  # pyright: ignore[reportIndexIssue]
        return new


def _mixed_polymorphic_chain(
    model: type["PolymorphicModel"],
) -> list[type["PolymorphicModel"]]:
    """
    Return the concrete polymorphic models that ``model`` is stored across,
    ordered from the root polymorphic table to the concrete model of ``model``
    (a proxy model is folded into its concrete model).

    Only single, linear multi-table inheritance chains and proxy models are
    supported. Anything else raises :class:`PolymorphicBulkCreateError` before
    any database write happens.
    """
    from .models import PolymorphicBulkCreateError, PolymorphicModel

    concrete_model = model._meta.concrete_model

    # A concrete model may have at most one concrete parent reached through a
    # parent-link field. Several linked parents mean multiple (diamond or
    # unrelated) inheritance and the correct insertion order is ambiguous.
    def check_model(mdl: type[models.Model]) -> None:
        linked_parents = [
            parent for parent, link in mdl._meta.parents.items() if link is not None
        ]
        if len(linked_parents) > 1:
            raise PolymorphicBulkCreateError(
                f"bulk_create_mixed() does not support {mdl.__name__}: it has "
                "multiple concrete parent tables "
                f"({', '.join(p.__name__ for p in linked_parents)}). Only "
                "single (linear) multi-table inheritance chains are supported."
            )
        for parent in linked_parents:
            if not issubclass(parent, PolymorphicModel):
                raise PolymorphicBulkCreateError(
                    f"bulk_create_mixed() does not support {mdl.__name__}: its "
                    f"concrete parent {parent.__name__} is not a "
                    "PolymorphicModel."
                )
            check_model(parent)

    check_model(concrete_model)

    root = get_base_polymorphic_model(concrete_model)
    assert root is not None, "Polymorphic model must have a polymorphic root"

    chain = [
        cls
        for cls in reversed(concrete_model.__mro__)
        if (
            issubclass(cls, PolymorphicModel)
            and not cls._meta.abstract
            and not cls._meta.proxy
            and cls is not PolymorphicModel
            and issubclass(cls, root)
        )
    ]
    return cast(list[type["PolymorphicModel"]], chain)


def _row_pk_value(obj: models.Model, model: type[models.Model]) -> Any:
    """
    Return the primary key value that the row of ``model`` would use in the
    inheritance chain of ``obj``, or ``None`` when it is unset.

    Multi-table models that share the ancestor primary key do not store it
    under their own (parent-link) ``pk.attname``; the value lives on an
    ancestor field. For example ``Middle(pk=6)`` assigns the value to the
    ``top_ptr_id`` link while the local ``pk.attname`` (also ``id``) stays
    unset, yet the Top row must be created with ``id=6``. Walk the parent
    links to find the field the value was actually assigned to.
    """
    attname = model._meta.pk.attname
    if attname not in obj.__dict__:
        return None
    value = obj.__dict__[attname]
    if value is not None:
        return value
    # The local pk field is unset: find the value on a parent link that points
    # at this model's primary key (shared ancestor pk case).
    for concrete in type(obj).__mro__:
        opts = getattr(concrete, "_meta", None)
        if opts is None:
            continue
        for parent, link in opts.parents.items():
            if link is None:
                continue
            if parent is model or issubclass(parent, model):
                link_value = obj.__dict__.get(link.attname)
                if link_value is not None:
                    return link_value
    return None


def _check_unsaved_relations(
    obj: "PolymorphicModel", chain: list[type["PolymorphicModel"]]
) -> None:
    """
    Mirror Django's unsaved-related-object protection (see
    ``Model._prepare_related_fields_for_save``) for every table the object will
    be written to, skipping parent links (wired automatically by this API) and
    the internal polymorphic content type. GenericForeignKeys are checked too.
    """
    from .models import PolymorphicBulkCreateError

    for model in chain:
        for field in model._meta.concrete_fields:
            if not field.is_relation or not field.is_cached(obj):
                continue
            if getattr(field.remote_field, "parent_link", False):
                continue
            related = getattr(obj, field.name, None)
            if not related:
                continue
            if not related._is_pk_set():
                raise PolymorphicBulkCreateError(
                    "bulk_create_mixed() prohibited to prevent data loss due "
                    f"to unsaved related object '{field.name}' on "
                    f"{obj.__class__.__name__}. Save related objects first."
                )
            if getattr(obj, field.attname) in field.empty_values:
                setattr(obj, field.name, related)
        for field in model._meta.private_fields:
            if (
                field.is_relation
                and field.is_cached(obj)
                and hasattr(field, "fk_field")
            ):
                related = field.get_cached_value(obj, default=None)
                if related and not related._is_pk_set():
                    raise PolymorphicBulkCreateError(
                        "bulk_create_mixed() prohibited to prevent data loss "
                        f"due to unsaved related object '{field.name}' on "
                        f"{obj.__class__.__name__}. Save related objects first."
                    )


###################################################################################
# PolymorphicQuerySet


class PolymorphicQuerySet(QuerySet[_All], Generic[_All, _Base]):
    """
    QuerySet for PolymorphicModel

    Contains the core functionality for PolymorphicModel

    Usually not explicitly needed, except if a custom queryset class
    is to be used.
    """

    polymorphic_disabled: bool
    polymorphic_deferred_loading: tuple[set[str], bool]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._iterable_class = PolymorphicModelIterable

        self.polymorphic_disabled = False
        # A parallel structure to django.db.models.query.Query.deferred_loading,
        # which we maintain with the untranslated field names passed to
        # .defer() and .only() in order to be able to retranslate them when
        # retrieving the real instance (so that the deferred fields apply
        # to that queryset as well).
        self.polymorphic_deferred_loading = (set(), True)

    def _clone(self, *args: Any, **kwargs: Any) -> Self:
        # Django's _clone only copies its own variables, so we need to copy ours here
        new = cast(Self, super()._clone(*args, **kwargs))  # type: ignore[misc]
        new.polymorphic_disabled = self.polymorphic_disabled
        new.polymorphic_deferred_loading = (
            copy.copy(self.polymorphic_deferred_loading[0]),
            self.polymorphic_deferred_loading[1],
        )
        return new

    @classmethod
    def as_manager(cls) -> models.Manager[_All]:
        """
        Override base :meth:`~django.db.models.query.QuerySet.as_manager` to return
        a manager extended from :class:`polymorphic.managers.PolymorphicManager`.
        """

        from .managers import PolymorphicManager

        manager = PolymorphicManager[_All, _Base].from_queryset(cls)()
        setattr(manager, "_built_with_as_manager", True)
        return manager

    as_manager.queryset_only = True  # type: ignore[attr-defined]

    def bulk_create(
        self,
        objs: Iterable[_All],
        batch_size: int | None = None,
        ignore_conflicts: bool = False,
        update_conflicts: bool = False,
        update_fields: Collection[str] | None = None,
        unique_fields: Collection[str] | None = None,
    ) -> list[_All]:
        objs = list(objs)
        for obj in objs:
            obj.pre_save_polymorphic()
        return super().bulk_create(objs, batch_size, ignore_conflicts=ignore_conflicts)

    def bulk_create_mixed(self, objs: Iterable[_All], batch_size: int | None = None) -> list[_All]:
        """
        Bulk create a batch of objects of different types that share a common
        polymorphic base class, retaining the real (polymorphic) type of every
        object.

        Unlike :meth:`bulk_create` (which only supports a single concrete table
        and therefore cannot be used with multi-table inherited models), this
        method supports a *mixed* batch: proxy types and models that use
        multi-table inheritance. Parent rows are inserted before child rows and
        the generated primary keys are wired into the correct parent-link
        fields. Objects may mix explicitly assigned primary keys with
        automatically generated primary keys, and shared ancestor tables are
        never written twice for the same object.

        :param objs: New (unsaved) objects, all instances of a concrete model
            in the same polymorphic inheritance chain as this queryset's model.
        :param batch_size: Maximum number of rows per individual ``INSERT``
            statement.
        :return: The created objects in the same order as ``objs``.
        :raises PolymorphicBulkCreateError: If validation fails. All checks
            happen before any row is written, and a single transaction makes
            sure a failure never leaves behind partially created objects.

        .. note::

            As with Django's :meth:`~django.db.models.query.QuerySet.bulk_create`,
            ``save()`` is not called and ``pre_save``/``post_save`` signals are
            not sent. Field-level ``pre_save`` hooks (such as ``auto_now`` and
            ``auto_now_add``) *are* executed, and many-to-many relations are
            not saved. Validations, custom ``save()`` logic and signal
            receivers on the involved models do not run.
        """
        if batch_size is not None and batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")

        from .models import PolymorphicBulkCreateError

        objs = list(objs)
        if not objs:
            return []

        using = self.db
        # Type-check against the queryset model (a proxy queryset such as
        # RedheadDuck.objects must only accept RedheadDuck instances, while
        # the concrete table used for writes is resolved per object).
        queryset_model = self.model

        # ------------------------------------------------------------------
        # Validation phase: every error below is raised before any INSERT is
        # issued so a rejected batch never touches the database.
        # ------------------------------------------------------------------
        plans: list[list[type[PolymorphicModel]]] = []
        seen_instances: set[int] = set()
        explicit_identities: set[tuple[type[PolymorphicModel], Any]] = set()

        for obj in objs:
            if not isinstance(obj, queryset_model):
                raise PolymorphicBulkCreateError(
                    f"bulk_create_mixed() only accepts instances of "
                    f"{queryset_model.__name__} or its subclasses, got "
                    f"{obj.__class__.__name__}."
                )
            if not obj._state.adding:
                raise PolymorphicBulkCreateError(
                    f"bulk_create_mixed() can only create new objects; "
                    f"{obj.__class__.__name__} (pk={obj.pk!r}) already exists."
                )
            if id(obj) in seen_instances:
                raise PolymorphicBulkCreateError(
                    "bulk_create_mixed() received the same object instance "
                    "twice; every object in the batch must be unique."
                )
            seen_instances.add(id(obj))

            chain = _mixed_polymorphic_chain(obj.__class__)
            _check_unsaved_relations(obj, chain)

            for model in chain:
                pk_value = _row_pk_value(obj, model)
                if pk_value is not None:
                    identity = (model, pk_value)
                    if identity in explicit_identities:
                        raise PolymorphicBulkCreateError(
                            f"bulk_create_mixed() received multiple objects "
                            f"that would create the same {model.__name__} row "
                            f"with primary key {pk_value!r}."
                        )
                    # Add after the check: a single object legitimately shares
                    # its root pk across the tables of its own chain, but a
                    # later object must not target the same (table, pk).
                    explicit_identities.add(identity)

            plans.append(chain)

        connection = connections[using]

        # Resolve one ContentType per distinct real class through the target
        # database alias (content type ids may differ across databases).
        ctype_ids: dict[type[PolymorphicModel], int] = {}
        ctype_manager = ContentType.objects.db_manager(using)
        for obj in objs:
            if obj.__class__ not in ctype_ids:
                ctype_ids[obj.__class__] = ctype_manager.get_for_model(
                    obj, for_concrete_model=False
                ).pk

        # ------------------------------------------------------------------
        # Planning phase: build one row per (object, concrete table) and order
        # the insert groups ancestor-first. The first-seen order of concrete
        # chains is a valid topological order: a parent always appears before
        # its children in every chain.
        # ------------------------------------------------------------------
        level_order: list[type[PolymorphicModel]] = []
        for chain in plans:
            for model in chain:
                if model not in level_order:
                    level_order.append(model)

        # Each row keeps the position (index) of its source object so child
        # rows can locate the already inserted parent row of the same object.
        rows_by_level: dict[
            type[PolymorphicModel], list[tuple[int, PolymorphicModel]]
        ] = {model: [] for model in level_order}
        for obj_index, (obj, chain) in enumerate(zip(objs, plans)):
            for model in chain:
                row = copy.copy(obj)
                transmogrify(model, row)
                # When the level shares the root primary key (the normal
                # multi-table case, e.g. Middle(pk=200) stores 200 on the
                # top_ptr_id link), mirror that value onto the root pk column
                # so the parent row gets the requested value. Models with
                # their own primary key (e.g. CustomPkInherit.pk -> custom_id)
                # are left untouched.
                if model._meta.pk is chain[0]._meta.pk:
                    root_pk_value = _row_pk_value(obj, model)
                    if root_pk_value is not None:
                        setattr(row, chain[0]._meta.pk.attname, root_pk_value)
                # polymorphic_ctype is a column on the root polymorphic table;
                # the root row must record the *real* (leaf) type so that
                # polymorphic reads can downcast the object.
                if model is chain[0]:
                    row.polymorphic_ctype_id = ctype_ids[obj.__class__]
                rows_by_level[model].append((obj_index, row))

        # ------------------------------------------------------------------
        # Write phase: a single transaction ensures a failure anywhere rolls
        # the whole batch back, so no object is ever left as a parent-only
        # row.
        # ------------------------------------------------------------------
        # obj_index -> pk of the inserted row, per already flushed level;
        # used to wire parent links before inserting child tables.
        inserted_pks_by_level: dict[type[PolymorphicModel], dict[int, Any]] = {
            model: {} for model in level_order
        }

        def flush_level(model: type[PolymorphicModel]) -> None:
            indexed_rows = rows_by_level[model]
            rows = [row for _, row in indexed_rows]
            opts = model._meta
            manager = cast(
                Any,
                model._base_manager.db_manager(using).non_polymorphic(),
            )
            manager._for_write = True
            manager._prepare_for_bulk_create(rows)

            # Wire this table's direct-parent links to the inserted parent rows.
            for obj_index, row in indexed_rows:
                for parent, link in model._meta.parents.items():
                    if link is not None and parent in inserted_pks_by_level:
                        setattr(
                            row,
                            link.attname,
                            inserted_pks_by_level[parent][obj_index],
                        )

            with_pk = [row for row in rows if row._is_pk_set()]
            without_pk = [row for row in rows if not row._is_pk_set()]

            if without_pk and not connection.features.can_return_rows_from_bulk_insert:
                raise PolymorphicBulkCreateError(
                    "bulk_create_mixed() requires the database backend to be "
                    "able to return primary keys from bulk inserts (needed to "
                    "wire up parent-link fields), but the "
                    f"{connection.vendor} backend cannot do this."
                )

            def insert_rows(
                rows_to_insert: Sequence[PolymorphicModel], include_auto_pk: bool
            ) -> None:
                fields = [
                    field
                    for field in opts.local_concrete_fields
                    if not field.generated
                    and (include_auto_pk or not isinstance(field, AutoField))
                ]
                returned_columns = manager._batched_insert(
                    list(rows_to_insert), fields, batch_size
                )
                for row, results in zip(rows_to_insert, returned_columns):
                    for result, field in zip(results, opts.db_returning_fields):
                        setattr(row, field.attname, result)

            # Match Django's bulk_create() ordering: explicit-pk rows first.
            insert_rows(with_pk, include_auto_pk=True)
            insert_rows(without_pk, include_auto_pk=False)

            for row in rows:
                row._state.adding = False
                row._state.db = using
            inserted_pks_by_level[model] = {
                obj_index: row.pk for obj_index, row in indexed_rows
            }

        with transaction.atomic(using=using, savepoint=False):
            for model in level_order:
                flush_level(model)

            # Copy the persisted values (generated primary keys, parent-link
            # pointers and every concrete field value) from each table's own
            # row back onto the original object. Reading every level row
            # (rather than just the leaf row) preserves values that live on an
            # ancestor table, such as a shared root primary key.
            for obj_index, obj in enumerate(objs):
                chain = plans[obj_index]
                for model in chain:
                    level_row = dict(rows_by_level[model])[obj_index]
                    for field in model._meta.local_concrete_fields:
                        attname = field.attname
                        if attname in level_row.__dict__:
                            obj.__dict__[attname] = level_row.__dict__[attname]
                obj._state.adding = False
                obj._state.db = using

        return objs

    def non_polymorphic(self) -> PolymorphicQuerySet[_Base, _Base]:
        """switch off polymorphic behaviour for this query.
        When the queryset is evaluated, only objects of the type of the
        base class used for this query are returned."""
        qs = self._clone()
        qs.polymorphic_disabled = True
        if issubclass(qs._iterable_class, PolymorphicModelIterable):
            qs._iterable_class = ModelIterable
        return cast(PolymorphicQuerySet[_Base, _Base], qs)

    @overload
    def instance_of(self, __a: type[_A], /) -> PolymorphicQuerySet[_A, _Base]: ...

    @overload
    def instance_of(
        self, __a: type[_A], __b: type[_B], /
    ) -> PolymorphicQuerySet[_A | _B, _Base]: ...

    @overload
    def instance_of(
        self, __a: type[_A], __b: type[_B], __c: type[_C], /
    ) -> PolymorphicQuerySet[_A | _B | _C, _Base]: ...

    @overload
    def instance_of(
        self, __a: type[_A], __b: type[_B], __c: type[_C], __d: type[_D], /
    ) -> PolymorphicQuerySet[_A | _B | _C | _D, _Base]: ...

    @overload
    def instance_of(self, *args: type["PolymorphicModel"]) -> PolymorphicQuerySet[_All, _Base]: ...

    def instance_of(
        self, *args: type["PolymorphicModel"]
    ) -> PolymorphicQuerySet["PolymorphicModel", _Base]:
        """Filter the queryset to only include the classes in args (and their subclasses)."""
        # Implementation in _translate_polymorphic_filter_defnition.
        return self.filter(instance_of=args)

    def not_instance_of(self, *args: type["PolymorphicModel"]) -> Self:
        """Filter the queryset to exclude the classes in args (and their subclasses)."""
        # Implementation in _translate_polymorphic_filter_defnition."""
        return self.filter(not_instance_of=args)

    def _filter_or_exclude(self, negate: bool, args: Any, kwargs: Any) -> Self:
        # We override this internal Django function as it is used for all filter member functions.
        q_objects = translate_polymorphic_filter_definitions_in_args(
            queryset_model=self.model, args=args, using=self.db
        )
        # filter_field='data'
        additional_args = translate_polymorphic_filter_definitions_in_kwargs(
            queryset_model=self.model, kwargs=kwargs, using=self.db
        )
        args = list(q_objects) + additional_args
        return cast(Self, super()._filter_or_exclude(negate=negate, args=args, kwargs=kwargs))  # type: ignore[misc]

    def order_by(self, *field_names: str | Combinable) -> Self:
        """translate the field paths in the args, then call vanilla order_by."""
        translated_fields = [
            translate_polymorphic_field_path(self.model, a)
            if isinstance(a, str)
            else a  # allow expressions to pass unchanged
            for a in field_names
        ]
        return super().order_by(*translated_fields)

    @overload
    def defer(self, field: None, /) -> Self: ...
    @overload
    def defer(self, *fields: str) -> Self: ...
    def defer(self, *fields: str | None) -> Self:
        """
        Translate the field paths in the args, then call vanilla defer.

        Also retain a copy of the original fields passed, which we'll need
        when we're retrieving the real instance (since we'll need to translate
        them again, as the model will have changed).
        """
        # Filter out None and translate fields
        str_fields = tuple(f for f in fields if f is not None)
        if str_fields:
            new_fields = tuple(translate_polymorphic_field_path(self.model, a) for a in str_fields)
            clone = super().defer(*new_fields)
            clone._polymorphic_add_deferred_loading(str_fields)
        else:
            # Handle defer(None) case
            clone = super().defer(None)
        return clone

    def only(self, *fields: str) -> Self:
        """
        Translate the field paths in the args, then call vanilla only.

        Also retain a copy of the original fields passed, which we'll need
        when we're retrieving the real instance (since we'll need to translate
        them again, as the model will have changed).
        """
        new_fields = {translate_polymorphic_field_path(self.model, a) for a in fields}
        clone = super().only(*new_fields)
        only_fields, deferred = clone.query.deferred_loading
        if only_fields and not deferred:
            clone.query.deferred_loading = (only_fields | {"polymorphic_ctype_id"}, False)
        clone._polymorphic_add_immediate_loading(fields)
        return clone

    def _polymorphic_add_deferred_loading(self, field_names: Iterable[str]) -> None:
        """
        Follows the logic of django.db.models.query.Query.add_deferred_loading(),
        but for the non-translated field names that were passed to self.defer().
        """
        existing, defer = self.polymorphic_deferred_loading
        if defer:
            # Add to existing deferred names.
            self.polymorphic_deferred_loading = existing.union(field_names), True
        else:
            # Remove names from the set of any existing "immediate load" names.
            self.polymorphic_deferred_loading = existing.difference(field_names), False

    def _polymorphic_add_immediate_loading(self, field_names: Iterable[str]) -> None:
        """
        Follows the logic of django.db.models.query.Query.add_immediate_loading(),
        but for the non-translated field names that were passed to self.only()
        """
        existing, defer = self.polymorphic_deferred_loading
        field_names = set(field_names)
        if "pk" in field_names:
            field_names.remove("pk")
            field_names.add(self.model._meta.pk.name)

        if defer:
            # Remove any existing deferred names from the current set before
            # setting the new names.
            self.polymorphic_deferred_loading = field_names.difference(existing), False
        else:
            # Replace any existing "immediate load" field names.
            self.polymorphic_deferred_loading = field_names, False

    def _process_aggregate_args(self, args: Sequence[Any], kwargs: dict[str, Any]) -> None:
        """for aggregate and annotate kwargs: allow ModelX___field syntax for kwargs, forbid it for args.
        Modifies kwargs if needed (these are Aggregate objects, we translate the lookup member variable)
        """
        ___lookup_assert_msg = "PolymorphicModel: annotate()/aggregate(): ___ model lookup supported for keyword arguments only"

        def patch_lookup(a):
            # The field on which the aggregate operates is
            # stored inside a complex query expression.
            if isinstance(a, Q):
                # modify in place - this should be fixed if we want immutable
                # aggregate/annotation expressions
                a.children = translate_polymorphic_Q_object(self.model, a).children
            elif isinstance(a, FilteredRelation):
                patch_lookup(a.condition)
            elif isinstance(a, models.F):
                a.name = translate_polymorphic_field_path(self.model, a.name)  # type: ignore[attr-defined]
            elif hasattr(a, "get_source_expressions"):
                for source_expression in a.get_source_expressions():
                    if source_expression is not None:
                        patch_lookup(source_expression)
            else:
                a.name = translate_polymorphic_field_path(self.model, a.name)

        def test___lookup(a):
            """*args might be complex expressions too in django 1.8 so
            the testing for a '___' is rather complex on this one"""
            if isinstance(a, Q):

                def tree_node_test___lookup(my_model, node):
                    "process all children of this Q node"
                    for i in range(len(node.children)):
                        child = node.children[i]

                        if type(child) is tuple:
                            # this Q object child is a tuple => a kwarg like Q( instance_of=ModelB )
                            assert "___" not in child[0], ___lookup_assert_msg
                        else:
                            # this Q object child is another Q object, recursively process this as well
                            tree_node_test___lookup(my_model, child)

                tree_node_test___lookup(self.model, a)
            elif hasattr(a, "get_source_expressions"):
                for source_expression in a.get_source_expressions():
                    if source_expression is not None:
                        test___lookup(source_expression)
            else:
                assert "___" not in a.name, ___lookup_assert_msg

        for a in args:
            test___lookup(a)
        for a in kwargs.values():
            patch_lookup(a)

    def annotate(self, *args: Any, **kwargs: Any) -> Self:
        """translate the polymorphic field paths in the kwargs, then call vanilla annotate.
        _get_real_instances will do the rest of the job after executing the query."""
        self._process_aggregate_args(args, kwargs)
        return super().annotate(*args, **kwargs)

    def aggregate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """translate the polymorphic field paths in the kwargs, then call vanilla aggregate.
        We need no polymorphic object retrieval for aggregate => switch it off."""
        self._process_aggregate_args(args, kwargs)
        qs = self.non_polymorphic()
        return super(PolymorphicQuerySet, qs).aggregate(*args, **kwargs)

    # Starting with Django 1.9, the copy returned by 'qs.values(...)' has the
    # same class as 'qs', so our polymorphic modifications would apply.
    # We want to leave values queries untouched, so we set 'polymorphic_disabled'.
    def _values(self, *args: Any, **kwargs: Any) -> Self:
        clone = cast(Self, super()._values(*args, **kwargs))  # type: ignore[misc]
        clone.polymorphic_disabled = True
        return clone

    # Since django_polymorphic 'V1.0 beta2', extra() always returns polymorphic results.
    # The resulting objects are required to have a unique primary key within the result set
    # (otherwise an error is thrown).
    # The "polymorphic" keyword argument is not supported anymore.
    # def extra(self, *args, **kwargs):

    def _get_real_instances(self, base_result_objects: Sequence[_All]) -> list[_All]:
        """
        Polymorphic object loader

        Does the same as:

            return [ o.get_real_instance() for o in base_result_objects ]

        but more efficiently.

        The list base_result_objects contains the objects from the executed
        base class query. The class of all of them is self.model (our base model).

        Some, many or all of these objects were not created and stored as
        class self.model, but as a class derived from self.model. We want to re-fetch
        these objects from the db as their original class so we can return them
        just as they were created/saved.

        We identify these objects by looking at o.polymorphic_ctype, which specifies
        the real class of these objects (the class at the time they were saved).

        First, we sort the result objects in base_result_objects for their
        subclass (from o.polymorphic_ctype), and then we execute one db query per
        subclass of objects. Here, we handle any annotations from annotate().

        Finally we re-sort the resulting objects into the correct order and
        return them as a list.
        """
        resultlist: list[Any] = []  # polymorphic list of result-objects

        # dict contains one entry per unique model type occurring in result,
        # in the format idlist_per_model[modelclass]=[list-of-object-ids]
        idlist_per_model: defaultdict[Any, list[Any]] = defaultdict(list)
        indexlist_per_model: defaultdict[Any, list[tuple[int, int]]] = defaultdict(list)

        # priority queue holding the query order of concrete classes
        # we need this so we can retry fetching objects further up the hierarchy with
        # stale content types - this can happen in some legitimate cases (deletion)
        # we build up and pop classes off this queue to ensure we process in tree
        # traversal order (leaves first) - this allows us to retry fetching as parent
        # classes if child class retrieval fails
        classes_to_query: list[tuple[int, Any]] = []

        # use the pk attribute for the base model type used in the query to identify
        # objects
        pk_name = self.model._meta.pk.attname

        # - sort base_result_object ids into idlist_per_model lists, depending on their real class;
        # - store objects that already have the correct class into "results"
        content_type_manager = ContentType.objects.db_manager(self.db)
        self_model_class_id = content_type_manager.get_for_model(
            self.model, for_concrete_model=False
        ).pk
        self_concrete_model_class_id = content_type_manager.get_for_model(
            self.model, for_concrete_model=True
        ).pk

        class_priorities = {
            mdl: idx + 1
            for idx, mdl in enumerate((*reversed(concrete_descendants(self.model)), self.model))
        }

        for i, base_object in enumerate(base_result_objects):
            if base_object.polymorphic_ctype_id == self_model_class_id:
                # Real class is exactly the same as base class, go straight to results
                resultlist.append(base_object)
            else:
                real_concrete_class = base_object.get_real_instance_class()
                real_concrete_class_id = base_object.get_real_concrete_instance_class_id()

                if real_concrete_class_id is None:
                    # Dealing with a stale content type
                    continue
                elif real_concrete_class_id == self_concrete_model_class_id:
                    # Real and base classes share the same concrete ancestor,
                    # upcast it and put it in the results
                    # real_concrete_class is guaranteed to be a PolymorphicModel subclass
                    resultlist.append(
                        transmogrify(
                            cast("type[PolymorphicModel]", real_concrete_class), base_object
                        )
                    )
                else:
                    # This model has a concrete derived class, track it for bulk retrieval.
                    real_concrete_class = cast(
                        "type[_All] | None",
                        content_type_manager.get_for_id(real_concrete_class_id).model_class(),
                    )
                    if real_concrete_class is not None:
                        if real_concrete_class not in idlist_per_model:
                            # Maintain a priority queue to process the model classes
                            # in order of their occurrence in the inheritance tree - leafs
                            # first
                            heapq.heappush(
                                classes_to_query,
                                (
                                    class_priorities.get(real_concrete_class, 0),
                                    real_concrete_class,
                                ),
                            )
                        idlist_per_model[real_concrete_class].append(getattr(base_object, pk_name))
                        indexlist_per_model[real_concrete_class].append((i, len(resultlist)))
                    resultlist.append(None)

        # For each model in "idlist_per_model" request its objects (the real model)
        # from the db and store them in results[].
        # Then we copy the annotate fields from the base objects to the real objects.
        # Then we copy the extra() select fields from the base objects to the real objects.
        # TODO: defer(), only(): support for these would be around here

        while classes_to_query:
            _, real_concrete_class = heapq.heappop(classes_to_query)
            assert real_concrete_class is not None  # Ensured by guard at line 467
            idlist = idlist_per_model.pop(real_concrete_class)
            indices = indexlist_per_model.pop(real_concrete_class)
            real_objects = real_concrete_class._base_objects.db_manager(self.db).filter(
                **{(f"{pk_name}__in"): idlist}
            )
            # copy select related configuration to new qs
            real_objects.query.select_related = self.query.select_related

            # Copy deferred fields configuration to the new queryset
            deferred_loading_fields = []
            existing_fields = self.polymorphic_deferred_loading[0]
            for field in existing_fields:
                try:
                    translated_field_name = translate_polymorphic_field_path(
                        real_concrete_class, field
                    )
                except AssertionError:
                    if "___" in field:
                        # The originally passed argument to .defer() or .only()
                        # was in the form Model2B___field2, where Model2B is
                        # now a superclass of real_concrete_class. Thus it's
                        # sufficient to just use the field name.
                        translated_field_name = field.rpartition("___")[-1]

                        # Check if the field does exist.
                        # Ignore deferred fields that don't exist in this subclass type.
                        try:
                            real_concrete_class._meta.get_field(translated_field_name)
                        except FieldDoesNotExist:
                            continue
                    else:
                        raise

                deferred_loading_fields.append(translated_field_name)
            real_objects.query.deferred_loading = (
                set(deferred_loading_fields),
                self.query.deferred_loading[1],
            )

            real_objects_dict = {
                getattr(real_object, pk_name): real_object for real_object in real_objects
            }

            for base_idx, result_idx in indices:
                base_object = base_result_objects[base_idx]
                o_pk = getattr(base_object, pk_name)
                real_object = real_objects_dict.get(o_pk)
                if real_object is None:
                    # Our content type is pointing to a row that does not exist anymore
                    # We try to find the next best available parent row
                    inheritance_path = route_to_ancestor(real_concrete_class, self.model)
                    if not inheritance_path or inheritance_path[0].model is self.model:
                        resultlist[result_idx] = base_object
                    else:
                        next_best_class = inheritance_path[0].model
                        if next_best_class not in idlist_per_model:
                            # add this class to the priority try queue
                            heapq.heappush(
                                classes_to_query,
                                (class_priorities.get(next_best_class, 0), next_best_class),
                            )
                        idlist_per_model[next_best_class].append(o_pk)
                        indexlist_per_model[next_best_class].append((base_idx, result_idx))
                        resultlist[result_idx] = _Inconsistent
                    continue

                # need shallow copy to avoid duplication in caches (see PR #353)
                real_object = copy.copy(real_object)
                real_class = (
                    real_concrete_class
                    if resultlist[result_idx] is _Inconsistent
                    else real_object.get_real_instance_class()
                )

                # If the real class is a proxy, upcast it
                if real_class != real_concrete_class:
                    real_object = transmogrify(
                        cast("type[PolymorphicModel]", real_class), real_object
                    )

                if self.query.annotations:
                    # New in Django 3.2+: annotation_select contains only the selected annotations
                    # (excluding aliases). Fallback for older Django versions if needed.
                    annotation_select = getattr(
                        self.query, "annotation_select", self.query.annotations
                    )
                    for anno_field_name in annotation_select.keys():
                        if hasattr(base_object, anno_field_name):
                            attr = getattr(base_object, anno_field_name)
                            setattr(real_object, anno_field_name, attr)

                if self.query.extra_select:
                    for select_field_name in self.query.extra_select.keys():
                        attr = getattr(base_object, select_field_name)
                        setattr(real_object, select_field_name, attr)

                resultlist[result_idx] = real_object

        resultlist = [i for i in resultlist if i and i is not _Inconsistent]

        # set polymorphic_annotate_names in all objects (currently just used for debugging/printing)
        if self.query.annotations:
            # get annotate field list
            annotation_select = getattr(self.query, "annotation_select", self.query.annotations)
            annotate_names = list(annotation_select.keys())
            for real_object in resultlist:
                real_object.polymorphic_annotate_names = annotate_names

        # set polymorphic_extra_select_names in all objects (currently just used for debugging/printing)
        if self.query.extra_select:
            # get extra select field list
            extra_select_names = list(self.query.extra_select.keys())
            for real_object in resultlist:
                real_object.polymorphic_extra_select_names = extra_select_names

        return resultlist

    def __repr__(self, *args, **kwargs):
        if self.model.polymorphic_query_multiline_output:
            result = ",\n  ".join(repr(o) for o in self.all())
            return f"[ {result} ]"
        else:
            return super().__repr__(*args, **kwargs)

    class _p_list_class(list[Any]):
        def __repr__(self, *args: Any, **kwargs: Any) -> str:
            result = ",\n  ".join(repr(o) for o in self)
            return f"[ {result} ]"

    def get_real_instances(self, base_result_objects: Iterable[_All] | None = None) -> list[_All]:
        """
        Cast a list of objects to their actual classes.

        This does roughly the same as::

            return [ o.get_real_instance() for o in base_result_objects ]

        but more efficiently.

        :rtype: PolymorphicQuerySet
        """
        "same as _get_real_instances, but make sure that __repr__ for ShowField... creates correct output"
        if base_result_objects is None:
            base_result_objects = cast(Iterable[_All], self)
        # Convert to list if needed for indexing
        base_result_list: Sequence[_All]
        if not isinstance(base_result_objects, list):
            base_result_list = list(base_result_objects)
        else:
            base_result_list = base_result_objects
        olist = self._get_real_instances(base_result_list)
        if not self.model.polymorphic_query_multiline_output:
            return olist
        clist = PolymorphicQuerySet._p_list_class(olist)
        return clist

    def delete(self) -> tuple[int, dict[str, int]]:
        """
        Deletion will be done non-polymorphically because Django's multi-table deletion
        mechanism is already walking the class hierarchy and producing a correct
        deletion graph. Introducing polymorphic querysets into the deletion process
        disrupts the model hierarchy/relationship traversal.
        """
        return QuerySet.delete(self.non_polymorphic())
