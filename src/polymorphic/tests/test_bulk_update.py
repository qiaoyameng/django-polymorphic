from __future__ import annotations

import pytest
from django.contrib.contenttypes.models import ContentType
from django.db import IntegrityError
from django.db.models import F

from polymorphic.bulk_update import (
    BulkUpdateValidationError,
    ObjectUpdate,
    StrictBulkUpdateError,
    UpdateStatus,
    polymorphic_bulk_update,
)
from polymorphic.tests.models import (
    Base,
    CustomPkInherit,
    Model2A,
    Model2B,
    Model2C,
    Model2D,
    ModelX,
    ModelY,
    NatKeyChild,
    NatKeyParent,
    One2OneRelatingModel,
    One2OneRelatingModelDerived,
    ProxiedBase,
    ProxyChild,
    ProxyModelA,
    ProxyBase,
)

pytestmark = pytest.mark.django_db()


def _refresh(*objs):
    for obj in objs:
        obj.__class__.objects.non_polymorphic().filter(pk=obj.pk)
        obj.refresh_from_db()


# --------------------------------------------------------------------------- #
# Basic / mixed ancestor and leaf fields
# --------------------------------------------------------------------------- #


def test_mixed_list_ancestor_and_leaf_fields():
    a = Model2A.objects.create(field1="a")
    b = Model2B.objects.create(field1="b", field2="b2")
    c = Model2C.objects.create(field1="c", field2="c2", field3="c3")
    d = Model2D.objects.create(field1="d", field2="d2", field3="d3", field4="d4")

    report = Model2A.objects.polymorphic_bulk_update(
        [
            ObjectUpdate(a, {"field1": "A"}),
            (b, {"Model2A___field1": "B", "field2": "B2"}),
            ObjectUpdate(c, {"field1": "C", "Model2C___field3": "C3"}),
            (d, {"Model2A___field1": "D", "Model2B___field2": "D2",
                  "Model2C___field3": "D3", "field4": "D4"}),
        ]
    )

    assert len(report) == 4
    assert report.all_succeeded
    assert [r.status for r in report] == [UpdateStatus.UPDATED] * 4

    _refresh(a, b, c, d)
    assert a.field1 == "A"
    assert (b.field1, b.field2) == ("B", "B2")
    assert (c.field1, c.field2, c.field3) == ("C", "c2", "C3")
    assert (d.field1, d.field2, d.field3, d.field4) == ("D", "D2", "D3", "D4")


def test_only_supplied_fields_are_written():
    c = Model2C.objects.create(field1="c1", field2="c2", field3="c3")
    report = polymorphic_bulk_update([(c, {"field3": "new"})])
    assert report.all_succeeded
    c.refresh_from_db()
    assert (c.field1, c.field2, c.field3) == ("c1", "c2", "new")


def test_tuple_list_input_reads_values_from_instance():
    c = Model2C.objects.create(field1="c1", field2="c2", field3="c3")
    c.field3 = "mutated"
    report = polymorphic_bulk_update([(c, ["field3"])])
    assert report.all_succeeded
    c.refresh_from_db()
    assert c.field3 == "mutated"


def test_returns_per_object_results_in_input_order():
    a = Model2A.objects.create(field1="a")
    b = Model2B.objects.create(field1="b", field2="b2")
    deleted = Model2C.objects.create(field1="x", field2="y", field3="z")
    deleted_pk = deleted.pk
    deleted.delete()
    ghost = Model2C(field1="x", field2="y", field3="z")
    ghost.pk = deleted_pk

    report = polymorphic_bulk_update(
        [
            ObjectUpdate(a, {"field1": "A"}),
            ObjectUpdate(ghost, {"field3": "Z"}),
            (b, {"field2": "B2"}),
        ]
    )
    assert [r.status for r in report] == [
        UpdateStatus.UPDATED,
        UpdateStatus.NOT_FOUND,
        UpdateStatus.UPDATED,
    ]
    assert report.failed_results[0].missing_tables == [Model2A._meta.db_table]
    assert not report.all_succeeded
    assert len(report.updated_objects) == 2


# --------------------------------------------------------------------------- #
# Field ownership validation
# --------------------------------------------------------------------------- #


def test_sibling_subclass_field_is_rejected():
    x = ModelX.objects.create(field_b="b", field_x="x")
    report = polymorphic_bulk_update([ObjectUpdate(x, {"ModelY___field_y": "?"})])
    result = report[0]
    assert result.status is UpdateStatus.ERROR
    assert isinstance(result.error, BulkUpdateValidationError)
    assert "not an ancestor" in str(result.error)
    x.refresh_from_db()
    assert x.field_x == "x"


def test_unknown_field_is_rejected():
    a = Model2A.objects.create(field1="a")
    report = polymorphic_bulk_update([(a, {"nope": "x"})])
    assert report[0].status is UpdateStatus.ERROR
    assert "no field" in str(report[0].error)


def test_unknown_model_name_is_rejected():
    a = Model2A.objects.create(field1="a")
    report = polymorphic_bulk_update([(a, {"Nope___field1": "x"})])
    assert report[0].status is UpdateStatus.ERROR


def test_polymorphic_ctype_and_pk_are_protected():
    ctype = ContentType.objects.get_for_model(Model2A)
    a = Model2A.objects.create(field1="a")
    report = polymorphic_bulk_update(
        [
            (a, {"polymorphic_ctype": ctype}),
            (a, {"polymorphic_ctype_id": ctype.pk}),
            (a, {"id": 999}),
            (a, {"model2a_ptr_id": 5}),
        ]
    )
    assert all(r.status is UpdateStatus.ERROR for r in report)


def test_parent_link_field_is_protected():
    c = Model2C.objects.create(field1="c", field2="c2", field3="c3")
    report = polymorphic_bulk_update([(c, {"model2b_ptr_id": c.pk})])
    assert report[0].status is UpdateStatus.ERROR
    message = str(report[0].error).lower()
    assert "parent link" in message or "primary key" in message


def test_relation_path_is_rejected():
    a = Model2A.objects.create(field1="a")
    report = polymorphic_bulk_update([(a, {"field1__upper": "A"})])
    assert report[0].status is UpdateStatus.ERROR


def test_queryset_constrains_polymorphic_tree():
    x = ModelX.objects.create(field_b="b", field_x="x")
    # ModelX belongs to the Base tree, not the Model2A tree.
    report = Model2A.objects.polymorphic_bulk_update([(x, {"field_x": "?"})])
    assert report[0].status is UpdateStatus.ERROR
    assert "polymorphic tree" in str(report[0].error)


# --------------------------------------------------------------------------- #
# Proxy models
# --------------------------------------------------------------------------- #


def test_proxy_model_maps_to_concrete_table():
    ProxyBase.objects.create(
        some_data="data",
        polymorphic_ctype=ContentType.objects.get_for_model(ProxyChild, for_concrete_model=False),
    )
    proxy = ProxyChild.objects.get()
    assert type(proxy) is ProxyChild
    report = polymorphic_bulk_update([(proxy, {"some_data": "new"})])
    assert report.all_succeeded
    assert ProxyBase.objects.get().some_data == "new"


def test_proxy_named_field_resolves_against_concrete_table():
    a = ProxyModelA.objects.create(name="n", field1="f1")
    report = polymorphic_bulk_update(
        [ObjectUpdate(a, {"ProxiedBase___name": "N", "field1": "F1"})]
    )
    assert report.all_succeeded
    a.refresh_from_db()
    assert a.name == "N"
    assert a.field1 == "F1"


# --------------------------------------------------------------------------- #
# F expressions
# --------------------------------------------------------------------------- #


def test_f_expression_same_table():
    parent = NatKeyParent.objects.create(slug="p", content="old")
    child = NatKeyChild.objects.create(slug="c", content="hello", val=1)
    report = polymorphic_bulk_update(
        [
            ObjectUpdate(child, {"val": F("val") + 4}),
            ObjectUpdate(parent, {"content": F("slug")}),
        ]
    )
    assert report.all_succeeded
    child.refresh_from_db()
    parent.refresh_from_db()
    assert child.val == 5
    assert parent.content == "p"


def test_f_expression_cross_table_is_rejected():
    c = Model2C.objects.create(field1="c", field2="c2", field3="c3")
    # field2 lives on Model2B, field3 lives on Model2C -> cross table F.
    report = polymorphic_bulk_update([(c, {"field3": F("field2")})])
    assert report[0].status is UpdateStatus.ERROR
    assert "cannot cross tables" in str(report[0].error)

    # Ancestor column referenced by a child-named target is also forbidden.
    report = polymorphic_bulk_update([(c, {"Model2C___field3": F("field2")})])
    assert report[0].status is UpdateStatus.ERROR


def test_f_expression_relation_path_rejected():
    c = Model2C.objects.create(field1="c", field2="c2", field3="c3")
    report = polymorphic_bulk_update([(c, {"field3": F("field2__upper")})])
    assert report[0].status is UpdateStatus.ERROR


# --------------------------------------------------------------------------- #
# Stale input / deleted rows / constraint failures
# --------------------------------------------------------------------------- #


def test_stale_content_type_detected():
    c = Model2C.objects.create(field1="c", field2="c2", field3="c3")
    # Tamper with the stored content type so it now claims to be a Model2B.
    Model2A.objects.non_polymorphic().filter(pk=c.pk).update(
        polymorphic_ctype=ContentType.objects.get_for_model(Model2B)
    )
    tampered = Model2C.objects.non_polymorphic().get(pk=c.pk)
    report = polymorphic_bulk_update([(tampered, {"field3": "new"})])
    assert report[0].status is UpdateStatus.STALE
    tampered_base = Model2A.objects.non_polymorphic().get(pk=c.pk)
    assert tampered_base.field1 == "c"


def test_missing_leaf_row_detected():
    c = Model2C.objects.create(field1="c", field2="c2", field3="c3")
    # Delete just the leaf row, orphaning the ancestor rows.
    leaf_pk = c.pk
    # Delete only the leaf table row, orphaning its ancestors (raw SQL avoids
    # Django's cascade collector).
    from django.db import connection

    with connection.cursor() as cursor:
        cursor.execute(
            f"DELETE FROM {Model2C._meta.db_table} WHERE {Model2C._meta.pk.column} = %s",
            [leaf_pk],
        )
    stale = Model2C(field1="c", field2="c2", field3="c3")
    stale.pk = leaf_pk
    report = polymorphic_bulk_update([(stale, {"field3": "new"})])
    assert report[0].status is UpdateStatus.NOT_FOUND
    assert Model2C._meta.db_table in report[0].missing_tables


def test_constraint_failure_does_not_report_success_non_strict():
    one = NatKeyChild.objects.create(slug="one", val=1)
    two = NatKeyChild.objects.create(slug="two", val=2)
    # 'one' already exists -> unique violation on 'two'; 'one' updates cleanly.
    report = polymorphic_bulk_update(
        [
            ObjectUpdate(one, {"val": F("val") + 10}),
            ObjectUpdate(two, {"slug": "one"}),
        ]
    )
    statuses = {r.object.obj.pk: r.status for r in report}
    assert statuses[one.pk] is UpdateStatus.UPDATED
    assert statuses[two.pk] is UpdateStatus.ERROR
    assert isinstance(report.failed_results[0].error, IntegrityError)
    one.refresh_from_db()
    two.refresh_from_db()
    assert one.val == 11
    assert two.slug == "two"


def test_strict_mode_raises_validation_errors():
    a = Model2A.objects.create(field1="a")
    with pytest.raises(StrictBulkUpdateError):
        polymorphic_bulk_update([(a, {"nope": "x"})], strict=True)
    a.refresh_from_db()
    assert a.field1 == "a"


def test_strict_mode_rolls_back_on_constraint_error():
    one = NatKeyChild.objects.create(slug="one", val=1)
    two = NatKeyChild.objects.create(slug="two", val=2)
    with pytest.raises(StrictBulkUpdateError):
        polymorphic_bulk_update(
            [
                ObjectUpdate(one, {"val": F("val") + 10}),
                ObjectUpdate(two, {"slug": "one"}),
            ],
            strict=True,
        )
    one.refresh_from_db()
    two.refresh_from_db()
    # Whole batch rolled back.
    assert one.val == 1
    assert two.slug == "two"


def test_strict_mode_rolls_back_on_not_found():
    a = Model2A.objects.create(field1="a")
    ghost = Model2B(field1="g", field2="g2", id=999999)
    with pytest.raises(StrictBulkUpdateError):
        polymorphic_bulk_update(
            [ObjectUpdate(a, {"field1": "A"}), (ghost, {"field2": "x"})],
            strict=True,
        )
    a.refresh_from_db()
    assert a.field1 == "a"


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #


def test_duplicate_object_is_rejected_non_strict():
    a = Model2A.objects.create(field1="a")
    same = Model2A.objects.non_polymorphic().get(pk=a.pk)
    report = polymorphic_bulk_update(
        [
            ObjectUpdate(a, {"field1": "first"}),
            ObjectUpdate(same, {"field1": "second"}),
        ]
    )
    assert report[0].status is UpdateStatus.UPDATED
    assert report[1].status is UpdateStatus.DUPLICATE
    a.refresh_from_db()
    assert a.field1 == "first"


def test_duplicate_object_strict_fails():
    a = Model2A.objects.create(field1="a")
    with pytest.raises(StrictBulkUpdateError):
        polymorphic_bulk_update(
            [ObjectUpdate(a, {"field1": "x"}), ObjectUpdate(a, {"field1": "y"})],
            strict=True,
        )


# --------------------------------------------------------------------------- #
# Batching & misc
# --------------------------------------------------------------------------- #


def test_batch_size_chunks_statements():
    objs = [Model2A.objects.create(field1=str(i)) for i in range(5)]
    report = polymorphic_bulk_update(
        [ObjectUpdate(obj, {"field1": f"v{i}"}) for i, obj in enumerate(objs)],
        batch_size=2,
    )
    assert report.all_succeeded
    values = list(
        Model2A.objects.order_by("pk").values_list("field1", flat=True)
    )
    assert values == [f"v{i}" for i in range(5)]


def test_invalid_batch_size():
    a = Model2A.objects.create(field1="a")
    with pytest.raises(ValueError):
        polymorphic_bulk_update([(a, {"field1": "x"})], batch_size=0)


def test_unsaved_object_rejected():
    a = Model2A(field1="a")
    report = polymorphic_bulk_update([(a, {"field1": "x"})])
    assert report[0].status is UpdateStatus.ERROR
    assert "primary key" in str(report[0].error)


def test_in_memory_instance_attributes_reflect_plain_values():
    c = Model2C.objects.create(field1="c", field2="c2", field3="c3")
    polymorphic_bulk_update([(c, {"field3": "mem"})])
    assert c.field3 == "mem"  # no refresh needed for plain values


def test_empty_input_returns_empty_report():
    report = polymorphic_bulk_update([])
    assert report.results == []
    assert report.all_succeeded


def test_foreign_key_plain_and_instance_values():
    target = Model2A.objects.create(field1="target")
    rel = One2OneRelatingModel.objects.create(one2one=target, field1="r1")

    new_target = Model2A.objects.create(field1="new-target")
    report = polymorphic_bulk_update(
        [ObjectUpdate(rel, {"one2one_id": new_target.pk})]
    )
    assert report.all_succeeded
    rel.refresh_from_db()
    assert rel.one2one_id == new_target.pk

    # Passing a model instance as value also works.
    report = polymorphic_bulk_update([ObjectUpdate(rel, {"one2one": target})])
    assert report.all_succeeded
    rel.refresh_from_db()
    assert rel.one2one_id == target.pk


def test_unsaved_related_instance_is_rejected():
    rel = One2OneRelatingModel.objects.create(
        one2one=Model2A.objects.create(field1="t"), field1="r"
    )
    unsaved = Model2A(field1="unsaved")
    report = polymorphic_bulk_update([ObjectUpdate(rel, {"one2one": unsaved})])
    assert report[0].status is UpdateStatus.ERROR


def test_invalid_item_not_partially_written_across_tables():
    # The derived object writes an ancestor field and an invalid (unsaved) leaf
    # relation value; nothing should be written on either table.
    target = Model2A.objects.create(field1="t")
    rel = One2OneRelatingModelDerived.objects.create(
        one2one=target, field1="old", field2="leaf"
    )
    unsaved = Model2A(field1="unsaved")
    report = polymorphic_bulk_update(
        [
            ObjectUpdate(rel, {
                "field1": "new-ancestor",
                "field2": "new-leaf",
                "one2one": unsaved,
            })
        ]
    )
    assert report[0].status is UpdateStatus.ERROR
    rel.refresh_from_db()
    assert rel.field1 == "old"
    assert rel.field2 == "leaf"
    assert rel.one2one_id == target.pk


def test_custom_primary_key_model():
    obj = CustomPkInherit.objects.create(b="x", i="y")
    report = polymorphic_bulk_update([(obj, {"b": "B", "i": "I"})])
    assert report.all_succeeded
    obj.refresh_from_db()
    assert (obj.b, obj.i) == ("B", "I")

    bad = polymorphic_bulk_update([(obj, {"custom_id": obj.custom_id + 999})])
    assert bad[0].status is UpdateStatus.ERROR
