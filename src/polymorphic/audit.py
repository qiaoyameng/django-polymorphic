from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from django.apps import apps
from django.contrib.contenttypes.models import ContentType
from django.db import DEFAULT_DB_ALIAS, connections, models, transaction
from django.db.models import Exists, OuterRef

from .utils import (
    _get_polymorphic_ctype,
    concrete_descendants,
    get_base_polymorphic_model,
    route_to_ancestor,
)


class PolymorphicCtypeIssue(str, Enum):
    NULL_POINTER = "null_pointer"
    MISSING_CONTENT_TYPE = "missing_content_type"
    STALE_CONTENT_TYPE = "stale_content_type"
    POINTER_NOT_SUBCLASS = "pointer_not_subclass"
    POINTER_NOT_LEAF = "pointer_not_leaf"
    LEAF_ROW_MISSING = "leaf_row_missing"
    INCOMPATIBLE_LEAVES = "incompatible_leaves"
    PROXY_TYPE_UNRESOLVABLE = "proxy_type_unresolvable"
    BROKEN_INHERITANCE_CHAIN = "broken_inheritance_chain"


class PolymorphicCtypeApplyStatus(str, Enum):
    APPLIED = "applied"
    NOT_REPAIRABLE = "not_repairable"
    STALE = "stale"
    DRY_RUN = "dry_run"


@dataclass(frozen=True)
class PolymorphicCtypeModelEvidence:
    root_label: str
    concrete_models: tuple[str, ...]
    proxy_models: tuple[str, ...]
    table_schema: tuple[tuple[str, str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class PolymorphicCtypeFinding:
    root_label: str
    pk: Any
    issues: tuple[str, ...]
    current_ctype_id: int | None
    current_model_label: str | None
    actual_tables: tuple[str, ...]
    actual_leaves: tuple[str, ...]
    broken_tables: tuple[str, ...]
    proxy_options: tuple[str, ...]
    proposed_model_label: str | None
    evidence: str = ""
    reason_not_auto_repaired: str | None = None
    batch: int = 0


@dataclass(frozen=True)
class PolymorphicCtypeBatch:
    root_label: str
    number: int
    first_pk: Any | None
    last_pk: Any | None
    scanned: int
    findings: int
    auto_repairable: int


@dataclass(frozen=True)
class PolymorphicCtypeReport:
    using: str
    batch_size: int
    roots: tuple[str, ...]
    model_evidence: tuple[PolymorphicCtypeModelEvidence, ...] = ()
    findings: tuple[PolymorphicCtypeFinding, ...] = ()
    batches: tuple[PolymorphicCtypeBatch, ...] = ()

    @property
    def scanned(self) -> int:
        return sum(batch.scanned for batch in self.batches)

    @property
    def auto_repairable(self) -> int:
        return sum(batch.auto_repairable for batch in self.batches)

    @property
    def issue_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for finding in self.findings:
            counts.update(finding.issues)
        return dict(sorted(counts.items()))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolymorphicCtypeApplyItem:
    root_label: str
    pk: Any
    status: str
    issues: tuple[str, ...]
    proposed_model_label: str | None
    detail: str


@dataclass(frozen=True)
class PolymorphicCtypeApplyResult:
    using: str
    dry_run: bool
    roots: tuple[str, ...]
    items: tuple[PolymorphicCtypeApplyItem, ...]
    batches: int

    @property
    def applied(self) -> int:
        return sum(
            item.status == PolymorphicCtypeApplyStatus.APPLIED.value for item in self.items
        )

    @property
    def stale(self) -> int:
        return sum(item.status == PolymorphicCtypeApplyStatus.STALE.value for item in self.items)

    @property
    def not_repairable(self) -> int:
        return sum(
            item.status == PolymorphicCtypeApplyStatus.NOT_REPAIRABLE.value
            for item in self.items
        )

    @property
    def dry_run_verified(self) -> int:
        return sum(item.status == PolymorphicCtypeApplyStatus.DRY_RUN.value for item in self.items)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Snapshot:
    root: type[models.Model]
    concrete: tuple[type[models.Model], ...]
    proxies: tuple[type[models.Model], ...]
    concrete_labels: tuple[str, ...]
    proxy_labels: tuple[str, ...]
    routes: dict[str, tuple[type[models.Model], ...]]
    root_pointer_fields: dict[str, str]
    proxy_options: dict[str, tuple[str, ...]]
    table_schema: tuple[tuple[str, str, tuple[str, ...]], ...]


def _plain_manager(model: type[models.Model], using: str) -> models.Manager:
    manager = models.Manager()
    manager.model = model
    return manager.db_manager(using)


def _root_for(model: type[models.Model]) -> type[models.Model]:
    root = get_base_polymorphic_model(model)
    if root is None:
        raise ValueError(f"{model._meta.label} is not a polymorphic model")
    return root


def _snapshot_for(root: type[models.Model]) -> _Snapshot:
    root = _root_for(root)
    concrete = tuple(dict.fromkeys((root, *concrete_descendants(root, include_proxy=False))))
    proxies = tuple(
        model
        for model in dict.fromkeys(concrete_descendants(root, include_proxy=True))
        if model._meta.proxy
    )
    routes: dict[str, tuple[type[models.Model], ...]] = {root._meta.label: ()}
    root_pointer_fields: dict[str, str] = {}
    proxy_options: dict[str, tuple[str, ...]] = {}

    for model in concrete[1:]:
        routes[model._meta.label] = (
            model,
            *(link.model for link in route_to_ancestor(model, root)),
        )
        root_field = next(
            (
                field.attname
                for field in model._meta.fields
                if getattr(field, "one_to_one", False) and field.related_model is root
            ),
            None,
        )
        if root_field is None:
            raise ValueError(
                f"{model._meta.label} has no materialized pointer to {root._meta.label}"
            )
        root_pointer_fields[model._meta.label] = root_field

    for model in concrete:
        proxy_options[model._meta.label] = tuple(
            proxy._meta.label
            for proxy in proxies
            if proxy._meta.concrete_model is model and proxy is not model
        )

    table_schema = tuple(
        (
            model._meta.label,
            model._meta.db_table,
            tuple(
                (link.model._meta.label, link.link.attname)
                for link in route_to_ancestor(model, root)
            ),
        )
        for model in concrete
    )

    return _Snapshot(
        root=root,
        concrete=concrete,
        proxies=proxies,
        concrete_labels=tuple(model._meta.label for model in concrete),
        proxy_labels=tuple(model._meta.label for model in proxies),
        routes=routes,
        root_pointer_fields=root_pointer_fields,
        proxy_options=proxy_options,
        table_schema=table_schema,
    )


def _normalize_roots(models_: Iterable[type[models.Model]]) -> tuple[type[models.Model], ...]:
    roots: dict[str, type[models.Model]] = {}
    for model in models_:
        root = _root_for(model)
        roots[root._meta.label] = root
    return tuple(roots.values())


def _table_rows(
    snapshot: _Snapshot, pks: tuple[Any, ...], using: str
) -> tuple[dict[str, set[Any]], dict[str, dict[Any, dict[str, Any]]]]:
    physical: dict[str, set[Any]] = {
        snapshot.root._meta.label: set(
            _plain_manager(snapshot.root, using).filter(pk__in=pks).values_list("pk", flat=True)
        )
    }
    rows_by_pk: dict[str, dict[Any, dict[str, Any]]] = {
        snapshot.root._meta.label: {pk: {"pk": pk} for pk in physical[snapshot.root._meta.label]}
    }
    for model in snapshot.concrete[1:]:
        root_field = snapshot.root_pointer_fields[model._meta.label]
        link_fields = tuple(link.link.attname for link in route_to_ancestor(model, snapshot.root))
        field_names = (model._meta.pk.attname, root_field, *link_fields)
        values = (
            _plain_manager(model, using)
            .filter(**{f"{root_field}__in": pks})
            .values(*dict.fromkeys(field_names))
        )
        rows = tuple(values)
        physical[model._meta.label] = {row[root_field] for row in rows}
        rows_by_pk[model._meta.label] = {row[model._meta.pk.attname]: row for row in rows}
    return physical, rows_by_pk


def _evidence_for(
    snapshot: _Snapshot,
    pk: Any,
    physical: dict[str, set[Any]],
    rows_by_pk: dict[str, dict[Any, dict[str, Any]]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    reachable = {snapshot.root._meta.label}

    def has_route(
        model: type[models.Model], row_pk: Any, expected_root_pk: Any
    ) -> bool:
        label = model._meta.label
        row = rows_by_pk.get(label, {}).get(row_pk)
        if row is None:
            return False
        if model is snapshot.root:
            return row_pk == expected_root_pk
        root_field = snapshot.root_pointer_fields[label]
        if row.get(root_field) != expected_root_pk:
            return False
        route = route_to_ancestor(model, snapshot.root)
        first_link = route[0]
        return has_route(first_link.model, row.get(first_link.link.attname), expected_root_pk)

    for model in snapshot.concrete[1:]:
        label = model._meta.label
        root_field = snapshot.root_pointer_fields[label]
        row = next(
            (
                row
                for row in rows_by_pk.get(label, {}).values()
                if row.get(root_field) == pk
            ),
            None,
        )
        if row is not None and has_route(model, row[model._meta.pk.attname], pk):
            reachable.add(label)

    actual = tuple(
        model._meta.label
        for model in snapshot.concrete
        if model._meta.label in reachable
    )
    leaves = tuple(
        label
        for label in actual
        if not any(
            other != label
            and label in {ancestor._meta.label for ancestor in snapshot.routes[other]}
            for other in actual
        )
    )
    broken = tuple(
        label
        for label, values in physical.items()
        if pk in values and label not in reachable
    )
    return actual, leaves, broken


def _ctype_rows(
    ctype_ids: tuple[Any, ...], using: str
) -> dict[int, tuple[ContentType | None, type[models.Model] | None]]:
    rows: dict[int, tuple[ContentType | None, type[models.Model] | None]] = {}
    existing = ContentType.objects.db_manager(using).in_bulk(
        ctype_id for ctype_id in ctype_ids if ctype_id is not None
    )
    for ctype_id in ctype_ids:
        if ctype_id is None:
            continue
        ctype = existing.get(ctype_id)
        rows[ctype_id] = (ctype, ctype.model_class() if ctype is not None else None)
    return rows


def _make_finding(
    snapshot: _Snapshot,
    pk: Any,
    current_ctype_id: int | None,
    ctype_info: tuple[ContentType | None, type[models.Model] | None] | None,
    actual: tuple[str, ...],
    leaves: tuple[str, ...],
    broken: tuple[str, ...],
    batch: int,
) -> PolymorphicCtypeFinding | None:
    root_label = snapshot.root._meta.label
    issues: list[PolymorphicCtypeIssue] = []
    current_label = ctype_info[1]._meta.label if ctype_info and ctype_info[1] else None
    current_is_subclass = bool(
        ctype_info
        and ctype_info[1] is not None
        and issubclass(ctype_info[1], snapshot.root)
    )
    current_is_proxy = bool(ctype_info and ctype_info[1] and ctype_info[1]._meta.proxy)

    unique_leaf = leaves[0] if len(leaves) == 1 else None
    concrete_label = (
        ctype_info[1]._meta.concrete_model._meta.label
        if current_is_subclass
        else None
    )
    proxy_options = snapshot.proxy_options.get(unique_leaf or root_label, ())

    if len(leaves) > 1:
        issues.append(PolymorphicCtypeIssue.INCOMPATIBLE_LEAVES)
    if broken:
        issues.append(PolymorphicCtypeIssue.BROKEN_INHERITANCE_CHAIN)

    if current_ctype_id is None:
        issues.append(PolymorphicCtypeIssue.NULL_POINTER)
    elif ctype_info is None or ctype_info[0] is None:
        issues.append(PolymorphicCtypeIssue.MISSING_CONTENT_TYPE)
    elif ctype_info[1] is None:
        issues.append(PolymorphicCtypeIssue.STALE_CONTENT_TYPE)
    elif not current_is_subclass:
        issues.append(PolymorphicCtypeIssue.POINTER_NOT_SUBCLASS)
    elif current_is_proxy:
        if concrete_label not in actual:
            issues.append(PolymorphicCtypeIssue.LEAF_ROW_MISSING)
        elif concrete_label not in leaves:
            issues.append(PolymorphicCtypeIssue.POINTER_NOT_LEAF)
    elif current_label not in actual:
        issues.append(PolymorphicCtypeIssue.LEAF_ROW_MISSING)
    elif current_label not in leaves:
        issues.append(PolymorphicCtypeIssue.POINTER_NOT_LEAF)

    proposed_label: str | None = None
    reason: str | None = None

    if issues:
        if PolymorphicCtypeIssue.BROKEN_INHERITANCE_CHAIN in issues:
            reason = "Actual inheritance tables do not identify one valid concrete leaf."
        elif len(leaves) > 1:
            reason = "Multiple incompatible concrete leaf rows exist for the same primary key."
        elif unique_leaf:
            if snapshot.proxy_options.get(unique_leaf):
                issues.append(PolymorphicCtypeIssue.PROXY_TYPE_UNRESOLVABLE)
                reason = (
                    f"Concrete leaf {unique_leaf} has proxy types; "
                    "table rows cannot identify identity."
                )
            else:
                proposed_label = unique_leaf
        else:
            reason = "No concrete leaf row exists, so the real subclass cannot be proven."

    evidence = (
        f"current_ctype_id={current_ctype_id}; current_model={current_label}; "
        f"reachable={list(actual)}; leaves={list(leaves)}; "
        f"broken_pointer_rows={list(broken)}"
    )
    if current_ctype_id is not None and ctype_info and ctype_info[0] is None:
        evidence = f"content_type_id={current_ctype_id} is absent; {evidence}"
    elif current_ctype_id is not None and ctype_info and ctype_info[1] is None:
        evidence = f"content_type_id={current_ctype_id} has no installed model; {evidence}"

    if not issues:
        return None

    return PolymorphicCtypeFinding(
        root_label=root_label,
        pk=pk,
        issues=tuple(issue.value for issue in issues),
        current_ctype_id=current_ctype_id,
        current_model_label=current_label,
        actual_tables=actual,
        actual_leaves=leaves,
        broken_tables=broken,
        proxy_options=proxy_options,
        proposed_model_label=proposed_label,
        evidence=evidence,
        reason_not_auto_repaired=reason,
        batch=batch,
    )


def audit_polymorphic_ctype(
    *models_: type[models.Model],
    using: str = DEFAULT_DB_ALIAS,
    batch_size: int = 500,
) -> PolymorphicCtypeReport:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")

    roots = _normalize_roots(models_)
    if not roots:
        raise ValueError("At least one polymorphic model must be provided")

    snapshots = [_snapshot_for(root) for root in roots]
    model_evidence = tuple(
        PolymorphicCtypeModelEvidence(
            root_label=snapshot.root._meta.label,
            concrete_models=snapshot.concrete_labels,
            proxy_models=snapshot.proxy_labels,
            table_schema=snapshot.table_schema,
        )
        for snapshot in snapshots
    )
    all_findings: list[PolymorphicCtypeFinding] = []
    all_batches: list[PolymorphicCtypeBatch] = []

    for snapshot in snapshots:
        base_qs = _plain_manager(snapshot.root, using).order_by("pk")
        after_pk = None
        batch_number = 0

        while True:
            batch_number += 1
            with transaction.atomic(using=using):
                qs = base_qs
                if after_pk is not None:
                    qs = qs.filter(pk__gt=after_pk)
                rows = list(qs[:batch_size])
                if not rows:
                    break

                pks = tuple(row.pk for row in rows)
                ctype_ids = tuple(getattr(row, "polymorphic_ctype_id") for row in rows)
                ctype_rows = _ctype_rows(tuple(set(ctype_ids)), using)
                physical, rows_by_pk = _table_rows(snapshot, pks, using)
                findings: list[PolymorphicCtypeFinding] = []

                for row in rows:
                    ctype_id = getattr(row, "polymorphic_ctype_id")
                    actual, leaves, broken = _evidence_for(
                        snapshot, row.pk, physical, rows_by_pk
                    )
                    finding = _make_finding(
                        snapshot=snapshot,
                        pk=row.pk,
                        current_ctype_id=ctype_id,
                        ctype_info=ctype_rows.get(ctype_id) if ctype_id is not None else None,
                        actual=actual,
                        leaves=leaves,
                        broken=broken,
                        batch=batch_number,
                    )
                    if finding is not None:
                        findings.append(finding)

                all_findings.extend(findings)
                all_batches.append(
                    PolymorphicCtypeBatch(
                        root_label=snapshot.root._meta.label,
                        number=batch_number,
                        first_pk=pks[0],
                        last_pk=pks[-1],
                        scanned=len(rows),
                        findings=len(findings),
                        auto_repairable=sum(
                            finding.proposed_model_label is not None for finding in findings
                        ),
                    )
                )
                after_pk = pks[-1]

    return PolymorphicCtypeReport(
        using=using,
        batch_size=batch_size,
        roots=tuple(snapshot.root._meta.label for snapshot in snapshots),
        model_evidence=model_evidence,
        findings=tuple(all_findings),
        batches=tuple(all_batches),
    )


def _current_snapshot(evidence: PolymorphicCtypeModelEvidence) -> _Snapshot | None:
    try:
        root = apps.get_model(evidence.root_label)
    except LookupError:
        return None
    snapshot = _snapshot_for(root)
    if (
        snapshot.concrete_labels != evidence.concrete_models
        or snapshot.proxy_labels != evidence.proxy_models
        or snapshot.table_schema != evidence.table_schema
    ):
        return None
    return snapshot


def _chain_exists(
    snapshot: _Snapshot,
    model: type[models.Model],
    expected_pk: Any,
    using: str,
) -> Exists:
    if model is snapshot.root:
        return Exists(_plain_manager(model, using).filter(pk=expected_pk))

    route = route_to_ancestor(model, snapshot.root)
    first_link = route[0]
    parent_exists = _chain_exists(
        snapshot, first_link.model, OuterRef(first_link.link.attname), using
    )
    return Exists(
        _plain_manager(model, using)
        .filter(
            **{
                snapshot.root_pointer_fields[model._meta.label]: expected_pk,
            }
        )
        .filter(parent_exists)
    )


def _precondition_filter(
    snapshot: _Snapshot,
    finding: PolymorphicCtypeFinding,
    target_label: str,
    using: str,
) -> models.QuerySet:
    qs = _plain_manager(snapshot.root, using).filter(pk=finding.pk)
    if finding.current_ctype_id is None:
        qs = qs.filter(polymorphic_ctype_id__isnull=True)
    else:
        qs = qs.filter(polymorphic_ctype_id=finding.current_ctype_id)

    target_route = {
        *(
            model._meta.label
            for model in snapshot.routes[target_label]
            if model in snapshot.concrete
        ),
    }
    for label in snapshot.concrete_labels:
        if label == snapshot.root._meta.label:
            continue
        model = apps.get_model(label)
        root_pointer = snapshot.root_pointer_fields[label]
        physical_exists = Exists(
            _plain_manager(model, using).filter(**{root_pointer: OuterRef("pk")})
        )
        chained_exists = _chain_exists(snapshot, model, OuterRef("pk"), using)
        if label in target_route:
            qs = qs.filter(physical_exists, chained_exists)
        else:
            qs = qs.exclude(physical_exists)
    return qs


def apply_polymorphic_ctype_report(
    report: PolymorphicCtypeReport,
    *,
    using: str | None = None,
    dry_run: bool = False,
) -> PolymorphicCtypeApplyResult:
    using = using or report.using
    snapshots = {
        evidence.root_label: _current_snapshot(evidence) for evidence in report.model_evidence
    }
    findings = sorted(
        report.findings,
        key=lambda finding: (finding.root_label, finding.batch, finding.pk),
    )
    grouped: dict[tuple[str, int], list[PolymorphicCtypeFinding]] = {}
    for finding in findings:
        grouped.setdefault((finding.root_label, finding.batch), []).append(finding)

    items: list[PolymorphicCtypeApplyItem] = []
    for (root_label, batch_number), batch_findings in grouped.items():
        snapshot = snapshots[root_label]
        with transaction.atomic(using=using):
            for finding in batch_findings:
                if finding.proposed_model_label is None:
                    items.append(
                        PolymorphicCtypeApplyItem(
                            root_label=finding.root_label,
                            pk=finding.pk,
                            status=PolymorphicCtypeApplyStatus.NOT_REPAIRABLE.value,
                            issues=finding.issues,
                            proposed_model_label=None,
                            detail=finding.reason_not_auto_repaired
                            or "No uniquely proven concrete model.",
                        )
                    )
                    continue

                if snapshot is None:
                    status = PolymorphicCtypeApplyStatus.STALE
                    detail = "Model schema changed after the audit."
                else:
                    target_model = apps.get_model(finding.proposed_model_label)
                    filtered = _precondition_filter(
                        snapshot, finding, finding.proposed_model_label, using
                    )
                    guarded = filtered
                    if connections[using].features.has_select_for_update:
                        guarded = guarded.select_for_update()
                    if not guarded.values_list("pk", flat=True).filter(pk=finding.pk).exists():
                        status = PolymorphicCtypeApplyStatus.STALE
                        detail = "Original pointer or inheritance evidence changed after audit."
                    elif dry_run:
                        status = PolymorphicCtypeApplyStatus.DRY_RUN
                        detail = "Preconditions verified; no update was made."
                    else:
                        updated = _plain_manager(snapshot.root, using).filter(
                            pk=finding.pk
                        ).update(polymorphic_ctype=_get_polymorphic_ctype(target_model, using))
                        if updated != 1:
                            raise RuntimeError("Expected exactly one polymorphic ctype update")
                        status = PolymorphicCtypeApplyStatus.APPLIED
                        detail = f"Updated polymorphic_ctype to {target_model._meta.label}."

                items.append(
                    PolymorphicCtypeApplyItem(
                        root_label=finding.root_label,
                        pk=finding.pk,
                        status=status.value,
                        issues=finding.issues,
                        proposed_model_label=finding.proposed_model_label,
                        detail=detail,
                    )
                )

    return PolymorphicCtypeApplyResult(
        using=using,
        dry_run=dry_run,
        roots=report.roots,
        items=tuple(items),
        batches=len(grouped),
    )
