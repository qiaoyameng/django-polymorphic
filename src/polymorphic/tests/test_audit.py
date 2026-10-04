from io import StringIO

import pytest
from django.contrib.contenttypes.models import ContentType
from django.core.management import call_command
from django.db import connection

from polymorphic.audit import (
    PolymorphicCtypeApplyStatus,
    PolymorphicCtypeIssue,
    apply_polymorphic_ctype_report,
    audit_polymorphic_ctype,
)
from polymorphic.tests.models import (
    Base,
    CustomPkBase,
    CustomPkInherit,
    Duck,
    Model2A,
    Model2B,
    Model2C,
    Model2D,
    ModelX,
    ModelY,
    RedheadDuck,
)

pytestmark = pytest.mark.django_db


def ctype(model):
    return ContentType.objects.get_for_model(model, for_concrete_model=False)


def raw(sql, params=None):
    with connection.cursor() as cursor:
        cursor.execute(sql, params or [])


def test_audit_identifies_repair_proposals_without_touching_healthy_rows():
    root = Model2A.objects.create(field1="root")
    leaf = Model2D.objects.create(field1="d", field2="d", field3="d", field4="d")
    Model2A.objects.filter(pk__in=[root.pk, leaf.pk]).update(polymorphic_ctype_id=None)

    report = audit_polymorphic_ctype(Model2A, batch_size=1)

    assert report.scanned == 2
    assert report.batches[0].scanned == 1
    assert report.auto_repairable == 2
    root_finding = next(finding for finding in report.findings if finding.pk == root.pk)
    leaf_finding = next(finding for finding in report.findings if finding.pk == leaf.pk)
    assert root_finding.issues == (PolymorphicCtypeIssue.NULL_POINTER.value,)
    assert root_finding.proposed_model_label == Model2A._meta.label
    assert leaf_finding.issues == (PolymorphicCtypeIssue.NULL_POINTER.value,)
    assert leaf_finding.proposed_model_label == Model2D._meta.label
    assert Model2A.objects.non_polymorphic().get(pk=root.pk).polymorphic_ctype_id is None

    result = apply_polymorphic_ctype_report(report)

    assert result.applied == 2
    assert result.batches == 2
    assert Model2A.objects.non_polymorphic().get(pk=root.pk).polymorphic_ctype_id == ctype(
        Model2A
    ).pk
    assert Model2A.objects.non_polymorphic().get(pk=leaf.pk).polymorphic_ctype_id == ctype(
        Model2D
    ).pk


def test_audit_reports_pointer_not_subclass_and_stale_content_type():
    wrong = Model2B.objects.create(field1="wrong", field2="wrong")
    stale = Model2B.objects.create(field1="stale", field2="stale")
    Model2A.objects.filter(pk=wrong.pk).update(polymorphic_ctype=ctype(Duck))
    stale_ctype = ContentType.objects.create(app_label="legacy", model="removedmodel")
    Model2A.objects.filter(pk=stale.pk).update(polymorphic_ctype=stale_ctype)

    report = audit_polymorphic_ctype(Model2B, batch_size=10)

    wrong_finding = next(finding for finding in report.findings if finding.pk == wrong.pk)
    stale_finding = next(finding for finding in report.findings if finding.pk == stale.pk)
    assert wrong_finding.issues == (
        PolymorphicCtypeIssue.POINTER_NOT_SUBCLASS.value,
    )
    assert wrong_finding.proposed_model_label == Model2B._meta.label
    assert PolymorphicCtypeIssue.STALE_CONTENT_TYPE.value in stale_finding.issues
    assert stale_finding.proposed_model_label == Model2B._meta.label


def test_audit_detects_missing_declared_leaf_and_repairs_to_existing_proven_leaf():
    instance = Model2D.objects.create(field1="d", field2="d", field3="d", field4="d")
    raw(f"DELETE FROM {Model2D._meta.db_table} WHERE model2c_ptr_id = %s", [instance.pk])

    report = audit_polymorphic_ctype(Model2A)
    finding = next(item for item in report.findings if item.pk == instance.pk)

    assert PolymorphicCtypeIssue.LEAF_ROW_MISSING.value in finding.issues
    assert finding.actual_leaves == (Model2C._meta.label,)
    assert finding.proposed_model_label == Model2C._meta.label

    result = apply_polymorphic_ctype_report(report)

    assert result.applied == 1
    assert Model2A.objects.non_polymorphic().get(
        pk=instance.pk
    ).polymorphic_ctype_id == ctype(Model2C).pk


def test_audit_detects_multiple_incompatible_leaves_and_refuses_auto_repair():
    instance = ModelX.objects.create(field_b="base", field_x="x")
    raw(
        f"INSERT INTO {ModelY._meta.db_table} (base_ptr_id, field_y) VALUES (%s, %s)",
        [instance.pk, "y"],
    )

    report = audit_polymorphic_ctype(Base)
    finding = next(item for item in report.findings if item.pk == instance.pk)

    assert PolymorphicCtypeIssue.INCOMPATIBLE_LEAVES.value in finding.issues
    assert finding.actual_leaves == (ModelX._meta.label, ModelY._meta.label)
    assert finding.proposed_model_label is None
    assert "Multiple incompatible" in finding.reason_not_auto_repaired

    result = apply_polymorphic_ctype_report(report, dry_run=True)

    assert result.not_repairable == 1
    assert result.applied == 0
    assert Base.objects.get(pk=instance.pk).polymorphic_ctype_id == ctype(ModelX).pk


def test_audit_leaves_proxy_identity_to_manual_decision():
    duck = Duck.objects.create(name="duck")
    Duck.objects.filter(pk=duck.pk).update(polymorphic_ctype_id=None)

    report = audit_polymorphic_ctype(Duck)
    finding = next(item for item in report.findings if item.pk == duck.pk)

    assert PolymorphicCtypeIssue.PROXY_TYPE_UNRESOLVABLE.value in finding.issues
    assert finding.proposed_model_label is None
    assert "proxy types" in finding.reason_not_auto_repaired

    result = apply_polymorphic_ctype_report(report)

    assert result.not_repairable == 1
    assert Duck.objects.non_polymorphic().get(pk=duck.pk).polymorphic_ctype_id is None


def test_audit_repairs_pointer_to_ancestor_using_full_leaf_evidence():
    instance = Model2D.objects.create(field1="d", field2="d", field3="d", field4="d")
    Model2A.objects.filter(pk=instance.pk).update(polymorphic_ctype=ctype(Model2B))

    report = audit_polymorphic_ctype(Model2A)
    finding = next(item for item in report.findings if item.pk == instance.pk)

    assert finding.issues == (PolymorphicCtypeIssue.POINTER_NOT_LEAF.value,)
    assert finding.actual_leaves == (Model2D._meta.label,)
    assert finding.proposed_model_label == Model2D._meta.label

    result = apply_polymorphic_ctype_report(report)

    assert result.applied == 1
    assert Model2A.objects.non_polymorphic().get(
        pk=instance.pk
    ).polymorphic_ctype_id == ctype(Model2D).pk


def test_audit_reports_broken_inheritance_chain_without_auto_repair():
    instance = Model2D.objects.create(field1="d", field2="d", field3="d", field4="d")
    raw(f"DELETE FROM {Model2C._meta.db_table} WHERE model2b_ptr_id = %s", [instance.pk])
    try:
        report = audit_polymorphic_ctype(Model2A)
        finding = next(item for item in report.findings if item.pk == instance.pk)

        assert PolymorphicCtypeIssue.BROKEN_INHERITANCE_CHAIN.value in finding.issues
        assert Model2D._meta.label in finding.broken_tables
        assert Model2D._meta.label not in finding.actual_tables
        assert finding.proposed_model_label is None

        result = apply_polymorphic_ctype_report(report, dry_run=True)

        assert result.not_repairable == 1
    finally:
        raw(f"DELETE FROM {Model2D._meta.db_table} WHERE model2c_ptr_id = %s", [instance.pk])


def test_audit_accepts_valid_proxy_pointer():
    duck = Duck.objects.create(name="red")
    Duck.objects.filter(pk=duck.pk).update(polymorphic_ctype=ctype(RedheadDuck))

    report = audit_polymorphic_ctype(Duck)

    assert report.findings == ()


def test_audit_supports_custom_primary_key_inheritance():
    child = CustomPkInherit.objects.create(b="c", i="i")
    base = CustomPkBase.objects.create(b="a")
    CustomPkBase.objects.filter(pk__in=[base.pk, child.pk]).update(
        polymorphic_ctype_id=None
    )

    report = audit_polymorphic_ctype(CustomPkBase, batch_size=1)
    result = apply_polymorphic_ctype_report(report)

    base_finding = next(item for item in report.findings if item.pk == base.pk)
    child_finding = next(item for item in report.findings if item.pk == child.pk)
    assert base_finding.proposed_model_label == CustomPkBase._meta.label
    assert child_finding.proposed_model_label == CustomPkInherit._meta.label
    assert result.applied == 2
    assert CustomPkBase.objects.non_polymorphic().get(
        pk=child.pk
    ).polymorphic_ctype_id == ctype(CustomPkInherit).pk


def test_apply_rejects_items_when_evidence_changes_after_audit():
    base = Model2A.objects.create(field1="base")
    Model2A.objects.filter(pk=base.pk).update(polymorphic_ctype_id=None)
    child = Model2B.objects.create(field1="child", field2="child")
    Model2A.objects.filter(pk=child.pk).update(polymorphic_ctype_id=None)

    report = audit_polymorphic_ctype(Model2A, batch_size=10)
    Model2A.objects.filter(pk=child.pk).update(polymorphic_ctype=ctype(Model2B))
    raw(
        f"INSERT INTO {Model2B._meta.db_table} (model2a_ptr_id, field2) VALUES (%s, %s)",
        [base.pk, "new"],
    )

    result = apply_polymorphic_ctype_report(report)

    statuses = {item.pk: item.status for item in result.items}
    assert statuses[child.pk] == PolymorphicCtypeApplyStatus.STALE.value
    assert statuses[base.pk] == PolymorphicCtypeApplyStatus.STALE.value
    assert Model2A.objects.non_polymorphic().get(
        pk=child.pk
    ).polymorphic_ctype_id == ctype(Model2B).pk
    assert Model2A.objects.non_polymorphic().get(pk=base.pk).polymorphic_ctype_id is None


def test_management_command_reports_and_applies_findings():
    root = Model2A.objects.create(field1="root")
    Model2A.objects.filter(pk=root.pk).update(polymorphic_ctype_id=None)

    audit_output = StringIO()
    call_command(
        "polymorphic_audit_ctype",
        "tests.Model2A",
        "--batch-size",
        "1",
        stdout=audit_output,
    )
    assert "found 1 issue(s), 1 auto-repairable" in audit_output.getvalue()
    assert Model2A.objects.non_polymorphic().get(pk=root.pk).polymorphic_ctype_id is None

    apply_output = StringIO()
    call_command(
        "polymorphic_audit_ctype",
        "tests.Model2A",
        "--apply",
        "--batch-size",
        "1",
        stdout=apply_output,
    )
    assert "Applied 1; stale rejected 0" in apply_output.getvalue()
    assert Model2A.objects.non_polymorphic().get(pk=root.pk).polymorphic_ctype_id == ctype(
        Model2A
    ).pk
