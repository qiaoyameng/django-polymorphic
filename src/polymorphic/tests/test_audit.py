import pytest
from django.contrib.contenttypes.models import ContentType
from django.db import connection
from django.test import TestCase, TransactionTestCase

from polymorphic.audit import AuditCategory, audit_polymorphic_ctype
from polymorphic.tests.models import (
    Model2A,
    Model2B,
    Model2C,
    Model2D,
    NonProxyChild,
    ProxiedBase,
    ProxyBase,
    ProxyChild,
    ProxyModelA,
    ProxyModelB,
)


def ctype_id(model):
    return ContentType.objects.get_for_model(model, for_concrete_model=False).pk


def set_pointer(base, pk, ct_id):
    base.objects.non_polymorphic().filter(pk=pk).update(polymorphic_ctype_id=ct_id)


def get_pointer(base, pk):
    return base.objects.non_polymorphic().values_list("polymorphic_ctype_id", flat=True).get(
        pk=pk
    )


def delete_table_row(model, pk):
    with connection.cursor() as cursor:
        cursor.execute(
            f"DELETE FROM {model._meta.db_table} WHERE {model._meta.pk.column} = %s", [pk]
        )


def insert_child_row(model, pk, **fields):
    columns = [model._meta.pk.column]
    values = [pk]
    for name, value in fields.items():
        columns.append(model._meta.get_field(name).column)
        values.append(value)
    placeholders = ", ".join(["%s"] * len(values))
    with connection.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO {model._meta.db_table} ({', '.join(columns)}) "
            f"VALUES ({placeholders})",
            values,
        )


class AuditTests(TestCase):
    def test_consistent_rows_have_no_findings(self):
        Model2A.objects.create(field1="a")
        Model2B.objects.create(field1="a", field2="b")
        Model2D.objects.create(field1="a", field2="b", field3="c", field4="d")

        report = audit_polymorphic_ctype(Model2A)

        assert report.scanned == 3
        assert report.ok == 3
        assert report.items == []
        assert report.summary()["findings"] == 0
        assert report.fix_plan() == []
        result = report.apply()
        assert result.summary() == {
            "fixed": 0,
            "rejected": 0,
            "manual": 0,
            "batches": 0,
            "fixed_by_model": {},
        }

    def test_audit_runs_from_any_tree_model(self):
        report = audit_polymorphic_ctype(Model2D)
        assert report.model is Model2A

    def test_non_polymorphic_model_rejected(self):
        with pytest.raises(TypeError):
            audit_polymorphic_ctype(ContentType)

    def test_pointer_missing_is_auto_fixed(self):
        obj = Model2B.objects.create(field1="a", field2="b")
        set_pointer(Model2A, obj.pk, None)

        report = audit_polymorphic_ctype(Model2A)

        (item,) = report.items
        assert item.category is AuditCategory.POINTER_MISSING
        assert item.stored == "NULL"
        assert item.fix_model is Model2B
        assert item.inferred_model is Model2B
        assert item.existing == (Model2A._meta.label, Model2B._meta.label)
        plan = report.fix_plan()
        assert len(plan) == 1
        assert plan[0]["to"] == Model2B._meta.label
        assert "uniquely determines" in plan[0]["reason"]

        result = report.apply()
        assert result.fixed == 1
        assert result.rejected == 0
        assert result.fixed_by_model == {Model2B._meta.label: 1}
        assert get_pointer(Model2A, obj.pk) == ctype_id(Model2B)
        assert audit_polymorphic_ctype(Model2A).ok == 1

    def test_pointer_to_foreign_model(self):
        obj = Model2A.objects.create(field1="a")
        set_pointer(Model2A, obj.pk, ctype_id(ProxiedBase))

        report = audit_polymorphic_ctype(Model2A)

        (item,) = report.items
        assert item.category is AuditCategory.POINTER_NOT_SUBCLASS
        assert item.stored == ProxiedBase._meta.label
        assert item.fix_model is Model2A

        result = report.apply()
        assert result.fixed == 1
        assert get_pointer(Model2A, obj.pk) == ctype_id(Model2A)

    def test_leaf_row_missing_falls_back_to_deepest_existing(self):
        obj = Model2D.objects.create(field1="a", field2="b", field3="c", field4="d")
        # Simulate a leaf deleted outside of Django's collector: the parent
        # rows survive and still point at the gone leaf.
        delete_table_row(Model2D, obj.pk)
        delete_table_row(Model2C, obj.pk)

        report = audit_polymorphic_ctype(Model2A)

        (item,) = report.items
        assert item.category is AuditCategory.LEAF_ROW_MISSING
        assert item.inferred_model is Model2B
        assert item.fix_model is Model2B
        assert Model2D._meta.db_table in item.reason

        result = report.apply()
        assert result.fixed == 1
        assert get_pointer(Model2A, obj.pk) == ctype_id(Model2B)
        # No business rows were deleted or created by the repair.
        assert Model2A.objects.non_polymorphic().filter(pk=obj.pk).exists()
        assert Model2B.objects.non_polymorphic().filter(pk=obj.pk).exists()
        assert not Model2C.objects.non_polymorphic().filter(pk=obj.pk).exists()

    def test_pointer_not_leaf(self):
        obj = Model2D.objects.create(field1="a", field2="b", field3="c", field4="d")
        set_pointer(Model2A, obj.pk, ctype_id(Model2B))

        report = audit_polymorphic_ctype(Model2A)

        (item,) = report.items
        assert item.category is AuditCategory.POINTER_NOT_LEAF
        assert item.inferred_model is Model2D
        assert item.fix_model is Model2D

        result = report.apply()
        assert result.fixed == 1
        assert get_pointer(Model2A, obj.pk) == ctype_id(Model2D)

    def test_multiple_incompatible_leaves_need_manual_decision(self):
        obj = ProxyModelA.objects.create(name="a", field1="x")
        # A second, incompatible leaf row appears for the same primary key.
        insert_child_row(ProxyModelB, obj.pk, field2="y")

        report = audit_polymorphic_ctype(ProxiedBase)

        (item,) = report.items
        assert item.category is AuditCategory.MULTIPLE_INCOMPATIBLE_LEAVES
        assert item.fix_model is None
        assert item.inferred_model is None
        assert ProxyModelA._meta.label in item.reason
        assert ProxyModelB._meta.label in item.reason

        result = report.apply()
        assert result.fixed == 0
        assert result.manual == 1
        assert result.batches == 0
        assert get_pointer(ProxiedBase, obj.pk) == ctype_id(ProxyModelA)

    def test_proxy_identity_cannot_be_inferred(self):
        obj = ProxyBase.objects.create(some_data="a")
        set_pointer(ProxyBase, obj.pk, None)

        report = audit_polymorphic_ctype(ProxyBase)

        (item,) = report.items
        assert item.category is AuditCategory.PROXY_UNRESOLVABLE
        assert item.inferred_model is ProxyBase
        assert item.fix_model is None
        assert ProxyChild._meta.label in item.reason

        result = report.apply()
        assert result.fixed == 0
        assert result.manual == 1
        assert get_pointer(ProxyBase, obj.pk) is None

    def test_concrete_child_of_proxied_base_is_fixable(self):
        obj = NonProxyChild.objects.create(some_data="a", name="n")
        set_pointer(ProxyBase, obj.pk, None)

        report = audit_polymorphic_ctype(ProxyBase)

        (item,) = report.items
        assert item.category is AuditCategory.POINTER_MISSING
        assert item.fix_model is NonProxyChild

        result = report.apply()
        assert result.fixed == 1
        assert get_pointer(ProxyBase, obj.pk) == ctype_id(NonProxyChild)

    def test_proxy_pointer_is_accepted(self):
        ProxyChild.objects.create(some_data="a")
        report = audit_polymorphic_ctype(ProxyBase)
        assert report.ok == 1
        assert report.items == []

    def test_apply_rejects_pointer_changed_after_audit(self):
        obj = Model2B.objects.create(field1="a", field2="b")
        set_pointer(Model2A, obj.pk, None)
        report = audit_polymorphic_ctype(Model2A)

        # The data changes between audit and apply.
        set_pointer(Model2A, obj.pk, ctype_id(Model2A))

        result = report.apply()
        assert result.fixed == 0
        assert result.rejected == 1
        assert result.rejected_items[0].pk == obj.pk
        assert get_pointer(Model2A, obj.pk) == ctype_id(Model2A)

    def test_apply_rejects_existence_changed_after_audit(self):
        obj = Model2B.objects.create(field1="a", field2="b")
        set_pointer(Model2A, obj.pk, None)
        report = audit_polymorphic_ctype(Model2A)

        # A new leaf row appears between audit and apply.
        insert_child_row(Model2C, obj.pk, field3="c")

        result = report.apply()
        assert result.fixed == 0
        assert result.rejected == 1
        assert get_pointer(Model2A, obj.pk) is None

    def test_apply_rejects_row_deleted_after_audit(self):
        obj = Model2B.objects.create(field1="a", field2="b")
        set_pointer(Model2A, obj.pk, None)
        report = audit_polymorphic_ctype(Model2A)

        obj.delete()

        result = report.apply()
        assert result.fixed == 0
        assert result.rejected == 1

    def test_apply_batches_and_statistics(self):
        objs = [Model2B.objects.create(field1="a", field2="b") for _ in range(3)]
        for obj in objs:
            set_pointer(Model2A, obj.pk, None)
        # One manual item mixed in.
        proxy_obj = ProxyBase.objects.create(some_data="a")
        set_pointer(ProxyBase, proxy_obj.pk, None)

        report = audit_polymorphic_ctype(Model2A)
        result = report.apply(batch_size=1)

        assert result.fixed == 3
        assert result.batches == 3
        assert result.rejected == 0
        summary = report.summary()
        assert summary["scanned"] == 3
        assert summary["auto_fixable"] == 3
        assert summary["by_category"]["pointer_missing"] == 3

        proxy_report = audit_polymorphic_ctype(ProxyBase)
        proxy_summary = proxy_report.summary()
        assert proxy_summary["manual"] == 1
        assert proxy_summary["by_category"]["proxy_unresolvable"] == 1

    def test_audit_scans_in_primary_key_batches(self):
        for _ in range(5):
            Model2A.objects.create(field1="a")
        report = audit_polymorphic_ctype(Model2A, batch_size=2)
        assert report.scanned == 5
        assert report.ok == 5

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            audit_polymorphic_ctype(Model2A, batch_size=0)


class DeletedContentTypeAuditTests(TransactionTestCase):
    """
    Simulating a stale (deleted) content type requires disabling foreign key
    checks, which is only possible outside of a transaction.
    """

    def test_pointer_to_deleted_content_type(self):
        obj = Model2A.objects.create(field1="a")
        ghost = ContentType.objects.create(app_label="ghost", model="ghostmodel")
        set_pointer(Model2A, obj.pk, ghost.pk)
        with connection.constraint_checks_disabled():
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM django_content_type WHERE id = %s", [ghost.pk])

        report = audit_polymorphic_ctype(Model2A)

        (item,) = report.items
        assert item.category is AuditCategory.POINTER_NOT_SUBCLASS
        assert "deleted content type" in item.stored
        assert item.fix_model is Model2A

        result = report.apply()
        assert result.fixed == 1
        assert get_pointer(Model2A, obj.pk) == ctype_id(Model2A)
