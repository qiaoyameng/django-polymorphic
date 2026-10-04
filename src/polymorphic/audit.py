"""
Audit and repair ``polymorphic_ctype`` pointers against real table structure.

Historical data can end up with ``polymorphic_ctype`` values that no longer
match reality: NULL pointers, content types of deleted or unrelated models,
pointers to leaf models whose table row is gone, or rows that exist in
several incompatible leaf tables at once.

:func:`audit_polymorphic_ctype` scans the base table in primary-key batches
and checks, for every row, which tables of the inheritance tree actually
contain the primary key. The result is a reviewable :class:`AuditReport`
that explains the evidence for every finding and only proposes an automatic
fix when the table structure uniquely determines the type. Ambiguous cases
(for example proxy identities, which leave no trace in the table structure)
are reported for manual decision instead.

:meth:`AuditReport.apply` re-verifies the original pointer and the table
existence relations as preconditions inside a per-batch transaction before
writing, so a finding is rejected when the data changed after the audit. It
never deletes business rows and never fabricates subclass rows; it only
rewrites the ``polymorphic_ctype`` pointer on the base table, reusing
:func:`~polymorphic.utils.reset_polymorphic_ctype`.
"""

from __future__ import annotations

import enum
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from django.contrib.contenttypes.models import ContentType
from django.db import DEFAULT_DB_ALIAS, models, transaction

from polymorphic.utils import (
    concrete_descendants,
    get_base_polymorphic_model,
    reset_polymorphic_ctype,
)

__all__ = [
    "ApplyResult",
    "AuditCategory",
    "AuditItem",
    "AuditReport",
    "audit_polymorphic_ctype",
]

_MISSING = object()


class AuditCategory(enum.Enum):
    """
    Classification of a ``polymorphic_ctype`` finding.
    """

    #: Stored pointer and table evidence agree.
    OK = "ok"
    #: ``polymorphic_ctype`` is NULL.
    POINTER_MISSING = "pointer_missing"
    #: Stored content type is deleted or not part of the inheritance tree.
    POINTER_NOT_SUBCLASS = "pointer_not_subclass"
    #: Stored content type points to a model whose table row does not exist.
    LEAF_ROW_MISSING = "leaf_row_missing"
    #: Stored content type points to an ancestor of the type the table
    #: evidence actually shows.
    POINTER_NOT_LEAF = "pointer_not_leaf"
    #: Rows exist in several sibling (mutually incompatible) leaf tables.
    MULTIPLE_INCOMPATIBLE_LEAVES = "multiple_incompatible_leaves"
    #: The concrete type is determined, but proxy models exist for it and the
    #: proxy identity cannot be inferred from the table structure.
    PROXY_UNRESOLVABLE = "proxy_unresolvable"


@dataclass(frozen=True)
class AuditItem:
    """
    A single finding of an audit.

    ``reason`` always states the evidence (which tables contain the primary
    key) and, for findings without an automatic fix, why no fix is proposed.
    """

    pk: Any
    category: AuditCategory
    #: The ``polymorphic_ctype_id`` found on the row during the audit.
    stored_ctype_id: int | None
    #: Human readable description of the stored pointer.
    stored: str
    #: Concrete models whose table contains the primary key, base first.
    existing_models: tuple[type[models.Model], ...]
    #: Deepest concrete model the table evidence determines, if unambiguous.
    inferred_model: type[models.Model] | None
    #: Model whose content type ``apply()`` would write, or None when the
    #: finding needs a manual decision.
    fix_model: type[models.Model] | None
    reason: str

    @property
    def existing(self) -> tuple[str, ...]:
        """Labels of the models whose table contains the primary key."""
        return tuple(model._meta.label for model in self.existing_models)

    @property
    def inferred(self) -> str | None:
        """Label of the inferred concrete model."""
        return self.inferred_model._meta.label if self.inferred_model else None

    @property
    def fix_to(self) -> str | None:
        """Label of the model ``apply()`` would point the row to."""
        return self.fix_model._meta.label if self.fix_model else None


@dataclass
class ApplyResult:
    """
    Outcome statistics of :meth:`AuditReport.apply`.
    """

    #: Findings whose pointer was rewritten.
    fixed: int = 0
    #: Findings rejected because the data changed after the audit.
    rejected: int = 0
    #: Findings without an automatic fix (manual decision required).
    manual: int = 0
    #: Number of per-batch transactions executed.
    batches: int = 0
    #: Number of fixes per target model label.
    fixed_by_model: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    #: Findings rejected because their preconditions no longer held.
    rejected_items: list[AuditItem] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return the outcome as a plain, JSON-serializable mapping."""
        return {
            "fixed": self.fixed,
            "rejected": self.rejected,
            "manual": self.manual,
            "batches": self.batches,
            "fixed_by_model": dict(self.fixed_by_model),
        }


@dataclass
class AuditReport:
    """
    Result of :func:`audit_polymorphic_ctype`.

    ``items`` contains one :class:`AuditItem` per inconsistent row; rows
    where the stored pointer matches the table evidence are only counted in
    ``ok``.
    """

    #: The audited base polymorphic model.
    model: type[models.Model]
    using: str = DEFAULT_DB_ALIAS
    #: Total number of base table rows inspected.
    scanned: int = 0
    #: Number of rows where pointer and table evidence agree.
    ok: int = 0
    items: list[AuditItem] = field(default_factory=list)

    @property
    def fixable(self) -> list[AuditItem]:
        """Findings with a uniquely determined automatic fix."""
        return [item for item in self.items if item.fix_model is not None]

    @property
    def manual_items(self) -> list[AuditItem]:
        """Findings that need a manual decision."""
        return [item for item in self.items if item.fix_model is None]

    def by_category(self) -> dict[AuditCategory, list[AuditItem]]:
        """Group the findings by category."""
        grouped: dict[AuditCategory, list[AuditItem]] = defaultdict(list)
        for item in self.items:
            grouped[item.category].append(item)
        return dict(grouped)

    def summary(self) -> dict[str, Any]:
        """Return audit statistics as a plain, JSON-serializable mapping."""
        counts = Counter(item.category for item in self.items)
        return {
            "model": self.model._meta.label,
            "using": self.using,
            "scanned": self.scanned,
            "ok": self.ok,
            "findings": len(self.items),
            "auto_fixable": len(self.fixable),
            "manual": len(self.manual_items),
            "by_category": {
                category.value: counts.get(category, 0)
                for category in AuditCategory
                if category is not AuditCategory.OK
            },
        }

    def fix_plan(self) -> list[dict[str, Any]]:
        """
        Return the reviewable repair plan: one entry per automatically
        fixable finding, describing the pointer change and its evidence.
        """
        return [
            {
                "pk": item.pk,
                "action": "set_polymorphic_ctype",
                "from": item.stored,
                "to": item.fix_to,
                "evidence": list(item.existing),
                "reason": item.reason,
            }
            for item in self.fixable
        ]

    def apply(self, *, batch_size: int = 500) -> ApplyResult:
        """
        Apply the automatic fixes of this report.

        Every batch runs in its own transaction. Before writing, each
        finding's preconditions are re-verified under a row lock:

        * the row's current ``polymorphic_ctype_id`` must still equal the
          value seen during the audit, and
        * the set of inheritance tables containing the primary key must be
          unchanged.

        Findings whose preconditions no longer hold are rejected (counted in
        :attr:`ApplyResult.rejected`) and left untouched. The actual pointer
        update reuses :func:`~polymorphic.utils.reset_polymorphic_ctype`
        with the original pointer value as a SQL-level precondition. No
        business rows are deleted and no subclass rows are created.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        result = ApplyResult(manual=len(self.manual_items))
        fixable = self.fixable
        if not fixable:
            return result

        context = _build_tree_context(self.model, self.using)
        ctype_manager = ContentType.objects.db_manager(self.using)

        for start in range(0, len(fixable), batch_size):
            batch = fixable[start : start + batch_size]
            result.batches += 1
            pks = [item.pk for item in batch]
            with transaction.atomic(using=self.using):
                current = dict(
                    _table_queryset(self.model, self.using)
                    .select_for_update()
                    .filter(pk__in=pks)
                    .values_list("pk", "polymorphic_ctype_id")
                )
                existence = _fetch_existence(context, pks, self.using)

                groups: dict[type[models.Model], list[tuple[Any, int | None]]] = defaultdict(list)
                for item in batch:
                    if current.get(item.pk, _MISSING) != item.stored_ctype_id:
                        result.rejected += 1
                        result.rejected_items.append(item)
                        continue
                    now_existing = frozenset(
                        model for model in context.concrete_models if item.pk in existence[model]
                    )
                    if now_existing != frozenset(item.existing_models):
                        result.rejected += 1
                        result.rejected_items.append(item)
                        continue
                    assert item.fix_model is not None
                    groups[item.fix_model].append((item.pk, item.stored_ctype_id))

                for fix_model, entries in groups.items():
                    target_ctype = ctype_manager.get_for_model(fix_model, for_concrete_model=False)
                    by_stored: dict[int | None, list[Any]] = defaultdict(list)
                    for pk, stored_ctype_id in entries:
                        by_stored[stored_ctype_id].append(pk)
                    for stored_ctype_id, group_pks in by_stored.items():
                        reset_polymorphic_ctype(
                            fix_model,
                            using=self.using,
                            pk__in=group_pks,
                            polymorphic_ctype_id=stored_ctype_id,
                        )
                        # Count honestly: a filtered custom default manager
                        # could have hidden rows from the update.
                        verified = set(
                            _table_queryset(self.model, self.using)
                            .filter(pk__in=group_pks, polymorphic_ctype_id=target_ctype.pk)
                            .values_list("pk", flat=True)
                        )
                        result.fixed += len(verified)
                        result.fixed_by_model[fix_model._meta.label] += len(verified)
                        for item in batch:
                            if item.pk in set(group_pks) - verified:
                                result.rejected += 1
                                result.rejected_items.append(item)
        return result


@dataclass(frozen=True)
class _TreeContext:
    """Inheritance tree metadata used to evaluate table evidence."""

    base: type[models.Model]
    #: Base model plus all concrete descendants, ancestors first.
    concrete_models: tuple[type[models.Model], ...]
    #: Content type id -> concrete or proxy model of the tree.
    ctype_to_model: dict[int, type[models.Model]]
    #: Concrete model -> proxy models of the tree pointing at it.
    proxies_by_concrete: dict[type[models.Model], tuple[type[models.Model], ...]]


def _table_queryset(model: type[models.Model], using: str) -> models.QuerySet:
    """
    Unfiltered queryset over the model's own table.

    Custom default managers (e.g. filtered ones) must not hide corruption,
    so the audit bypasses them entirely.
    """
    return models.QuerySet(model=model, using=using)


def _build_tree_context(base: type[models.Model], using: str) -> _TreeContext:
    ctype_manager = ContentType.objects.db_manager(using)
    ctype_to_model: dict[int, type[models.Model]] = {}
    proxies: dict[type[models.Model], list[type[models.Model]]] = defaultdict(list)
    for model in (base, *concrete_descendants(base, include_proxy=True)):
        ctype = ctype_manager.get_for_model(model, for_concrete_model=False)
        ctype_to_model[ctype.pk] = model
        if model._meta.proxy:
            proxies[model._meta.concrete_model].append(model)
    return _TreeContext(
        base=base,
        concrete_models=(base, *concrete_descendants(base)),
        ctype_to_model=ctype_to_model,
        proxies_by_concrete={model: tuple(found) for model, found in proxies.items()},
    )


def _fetch_existence(
    context: _TreeContext, pks: list[Any], using: str
) -> dict[type[models.Model], set[Any]]:
    """Return, per concrete model of the tree, which of ``pks`` its table holds."""
    existence = {}
    for model in context.concrete_models:
        existence[model] = set(
            _table_queryset(model, using).filter(pk__in=pks).values_list("pk", flat=True)
        )
    return existence


def _chain_deepest(
    existing: tuple[type[models.Model], ...],
) -> type[models.Model] | None:
    """
    Return the deepest model when ``existing`` forms a single inheritance
    chain, or None when incompatible (sibling) tables are involved.
    """
    deepest = existing[0]
    for model in existing[1:]:
        if issubclass(model, deepest):
            deepest = model
        elif not issubclass(deepest, model):
            return None
    return deepest


def _foreign_ctype_labels(
    context: _TreeContext, ctype_ids: set[int], using: str
) -> dict[int, str]:
    """Describe stored content type ids that are not part of the tree."""
    unknown = [ctype_id for ctype_id in ctype_ids if ctype_id not in context.ctype_to_model]
    labels: dict[int, str] = {}
    if unknown:
        found = {}
        for ctype in ContentType.objects.db_manager(using).filter(pk__in=unknown):
            try:
                found[ctype.pk] = ctype.model_class()._meta.label
            except (AttributeError, LookupError):
                found[ctype.pk] = f"{ctype.app_label}.{ctype.model}"
        for ctype_id in unknown:
            labels[ctype_id] = found.get(ctype_id, f"<deleted content type #{ctype_id}>")
    return labels


def _classify(
    pk: Any,
    stored_ctype_id: int | None,
    existing: tuple[type[models.Model], ...],
    context: _TreeContext,
    foreign_labels: dict[int, str],
) -> AuditItem | None:
    """Classify one row; return None when pointer and evidence agree."""
    if stored_ctype_id is None:
        stored_label = "NULL"
        stored_model = None
    else:
        stored_model = context.ctype_to_model.get(stored_ctype_id)
        if stored_model is not None:
            stored_label = stored_model._meta.label
        else:
            stored_label = foreign_labels[stored_ctype_id]

    labels = tuple(model._meta.label for model in existing)
    evidence = f"existing tables: {', '.join(labels)}"
    deepest = _chain_deepest(existing)

    if deepest is None:
        return AuditItem(
            pk=pk,
            category=AuditCategory.MULTIPLE_INCOMPATIBLE_LEAVES,
            stored_ctype_id=stored_ctype_id,
            stored=stored_label,
            existing_models=existing,
            inferred_model=None,
            fix_model=None,
            reason=(
                f"rows for this primary key exist in incompatible tables ({evidence}); "
                "no single type fits the evidence and picking the deepest model could "
                "silently choose the wrong branch, so a manual decision is required"
            ),
        )

    if stored_ctype_id is None:
        diagnosis = AuditCategory.POINTER_MISSING
        problem = "polymorphic_ctype is NULL"
    elif stored_model is None:
        diagnosis = AuditCategory.POINTER_NOT_SUBCLASS
        problem = (
            f"stored content type ({stored_label}) is not a model of the "
            f"{context.base._meta.label} inheritance tree"
        )
    else:
        stored_concrete = stored_model._meta.concrete_model
        if stored_concrete not in existing:
            diagnosis = AuditCategory.LEAF_ROW_MISSING
            problem = (
                f"stored content type points to {stored_model._meta.label} but no row "
                f"exists in table {stored_concrete._meta.db_table}"
            )
        elif stored_concrete is deepest:
            return None
        else:
            diagnosis = AuditCategory.POINTER_NOT_LEAF
            problem = (
                f"stored content type points to {stored_model._meta.label} but the "
                f"table evidence shows the more derived {deepest._meta.label}"
            )

    proxies = context.proxies_by_concrete.get(deepest, ())
    if proxies:
        proxy_labels = ", ".join(proxy._meta.label for proxy in proxies)
        return AuditItem(
            pk=pk,
            category=AuditCategory.PROXY_UNRESOLVABLE,
            stored_ctype_id=stored_ctype_id,
            stored=stored_label,
            existing_models=existing,
            inferred_model=deepest,
            fix_model=None,
            reason=(
                f"{problem}; table evidence ({evidence}) determines the concrete type "
                f"{deepest._meta.label}, but proxy models exist for it ({proxy_labels}) "
                "and the proxy identity cannot be inferred from the table structure, "
                "so a manual decision is required"
            ),
        )

    return AuditItem(
        pk=pk,
        category=diagnosis,
        stored_ctype_id=stored_ctype_id,
        stored=stored_label,
        existing_models=existing,
        inferred_model=deepest,
        fix_model=deepest,
        reason=(
            f"{problem}; table evidence ({evidence}) uniquely determines "
            f"{deepest._meta.label}"
        ),
    )


def audit_polymorphic_ctype(
    model: type[models.Model],
    *,
    using: str = DEFAULT_DB_ALIAS,
    batch_size: int = 1000,
) -> AuditReport:
    """
    Audit ``polymorphic_ctype`` pointers of a polymorphic inheritance tree
    against the actual table structure.

    The base table is scanned in primary-key batches; for every row the audit
    checks which tables of the inheritance tree actually contain the primary
    key and classifies the finding (see :class:`AuditCategory`). An automatic
    fix is only proposed when the table structure uniquely determines the
    type - the deepest existing model is never picked blindly, and proxy
    identities (which leave no trace in the table structure) are left for
    manual decision.

    :param model: Any model of the polymorphic inheritance tree; the audit
        always runs from the tree's base model.
    :param using: Database alias to audit.
    :param batch_size: Number of base rows inspected per batch.
    :return: An :class:`AuditReport` with reviewable findings and a fix plan.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    base = get_base_polymorphic_model(model)
    if base is None:
        raise TypeError(f"{model!r} is not a polymorphic model")

    context = _build_tree_context(base, using)
    report = AuditReport(model=base, using=using)

    last_pk = None
    while True:
        queryset = _table_queryset(base, using).order_by("pk")
        if last_pk is not None:
            queryset = queryset.filter(pk__gt=last_pk)
        rows = list(queryset.values_list("pk", "polymorphic_ctype_id")[:batch_size])
        if not rows:
            break
        report.scanned += len(rows)
        pks = [pk for pk, _ in rows]
        existence = _fetch_existence(context, pks, using)
        foreign_labels = _foreign_ctype_labels(
            context,
            {stored for _, stored in rows if stored is not None},
            using,
        )
        for pk, stored_ctype_id in rows:
            existing = tuple(
                tree_model
                for tree_model in context.concrete_models
                if pk in existence[tree_model]
            )
            item = _classify(pk, stored_ctype_id, existing, context, foreign_labels)
            if item is None:
                report.ok += 1
            else:
                report.items.append(item)
        last_pk = pks[-1]

    return report
