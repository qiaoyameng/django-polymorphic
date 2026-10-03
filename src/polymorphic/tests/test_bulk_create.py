"""
Tests for PolymorphicQuerySet.bulk_create_polymorphic and the manager equivalent.
"""

from unittest import mock

import pytest
from django.contrib.contenttypes.models import ContentType
from django.db import (
    DEFAULT_DB_ALIAS,
    IntegrityError,
    NotSupportedError,
    connection,
    connections,
)
from django.db.models.signals import post_save, pre_save
from django.test import TestCase, TransactionTestCase
from django.test.utils import CaptureQueriesContext

from polymorphic.tests.models import (
    BlogBase,
    Bottom,
    Duck,
    Enhance_Inherit,
    Middle,
    Model2A,
    Model2B,
    Model2C,
    Model2D,
    MROBase1,
    MRODerived,
    NonProxyChild,
    One2OneRelatingModel,
    One2OneRelatingModelDerived,
    PlainA,
    ProxiedBase,
    ProxyBase,
    ProxyChild,
    ProxyModelA,
    ProxyModelB,
    RedheadDuck,
    RubberDuck,
    Top,
    UUIDArtProject,
    UUIDArtProjectD,
    UUIDProject,
    UUIDResearchProject,
)


def warm_content_types(*models, using="default"):
    for model in models:
        ContentType.objects.db_manager(using).get_for_model(
            model, for_concrete_model=False
        )


def count_queries(using=DEFAULT_DB_ALIAS):
    """Count the queries of a block, grouped by SQL verb."""
    return CaptureQueriesContext(connections[using])


def inserts(ctx):
    return [q["sql"] for q in ctx.captured_queries if q["sql"].startswith("INSERT")]


class BulkCreatePolymorphicTests(TestCase):
    def test_mixed_multi_table_inheritance(self):
        objs = Model2A.objects.bulk_create_polymorphic(
            [
                Model2C(field1="C1", field2="C2", field3="C3"),
                Model2A(field1="A1"),
                Model2D(field1="D1", field2="D2", field3="D3", field4="D4"),
                Model2B(field1="B1", field2="B2"),
            ]
        )
        # input order is preserved and primary keys are set
        assert [type(obj) for obj in objs] == [Model2C, Model2A, Model2D, Model2B]
        assert all(obj.pk is not None for obj in objs)
        assert len({obj.pk for obj in objs}) == 4

        # one row per table along each inheritance chain
        assert Model2A.objects.non_polymorphic().count() == 4
        assert Model2B.objects.non_polymorphic().count() == 3
        assert Model2C.objects.non_polymorphic().count() == 2
        assert Model2D.objects.non_polymorphic().count() == 1

        # the real types and field values round-trip through the database
        fetched = list(Model2A.objects.order_by("pk"))
        assert [type(obj) for obj in fetched] == [Model2C, Model2A, Model2D, Model2B]
        assert [obj.field1 for obj in fetched] == ["C1", "A1", "D1", "B1"]
        assert fetched[2].field4 == "D4"

    def test_shared_ancestors_are_not_inserted_twice(self):
        objs = Model2A.objects.bulk_create_polymorphic(
            [
                Model2D(field1="D1", field2="D2", field3="D3", field4="D4"),
                Model2C(field1="C1", field2="C2", field3="C3"),
                Model2B(field1="B1", field2="B2"),
            ]
        )
        assert Model2A.objects.non_polymorphic().count() == 3
        assert Model2B.objects.non_polymorphic().count() == 3
        assert Model2C.objects.non_polymorphic().count() == 2
        assert Model2D.objects.non_polymorphic().count() == 1
        # every parent link points at the shared root primary key
        for obj in objs:
            real = Model2A.objects.get(pk=obj.pk)
            assert real.pk == obj.pk

    def test_deep_inheritance_chain(self):
        (obj,) = Top.objects.bulk_create_polymorphic(
            [Bottom(name="n", description="d", author="a")]
        )
        assert obj.pk is not None
        assert Top.objects.non_polymorphic().count() == 1
        assert Middle.objects.non_polymorphic().count() == 1
        assert Bottom.objects.non_polymorphic().count() == 1
        fetched = Top.objects.get()
        assert type(fetched) is Bottom
        assert fetched.author == "a"

    def test_proxy_models(self):
        objs = Duck.objects.bulk_create_polymorphic(
            [
                RedheadDuck(name="red"),
                Duck(name="plain"),
                RubberDuck(name="rubber"),
            ]
        )
        assert [type(obj) for obj in objs] == [RedheadDuck, Duck, RubberDuck]
        assert Duck.objects.non_polymorphic().count() == 3
        # the proxy content types are preserved
        assert [type(obj) for obj in Duck.objects.order_by("pk")] == [
            RedheadDuck,
            Duck,
            RubberDuck,
        ]

    def test_proxy_between_concrete_models(self):
        ProxyBase.objects.bulk_create_polymorphic(
            [
                ProxyChild(some_data="proxy"),
                NonProxyChild(some_data="np", name="x"),
            ]
        )
        assert ProxyBase.objects.non_polymorphic().count() == 2
        assert NonProxyChild.objects.non_polymorphic().count() == 1
        assert [type(obj) for obj in ProxyBase.objects.order_by("pk")] == [
            ProxyChild,
            NonProxyChild,
        ]

        ProxiedBase.objects.bulk_create_polymorphic(
            [
                ProxyModelA(name="a", field1="f1"),
                ProxyModelB(name="b", field2="f2"),
            ]
        )
        assert [type(obj) for obj in ProxiedBase.objects.order_by("pk")] == [
            ProxyModelA,
            ProxyModelB,
        ]

    def test_explicit_and_auto_primary_keys_mixed(self):
        objs = Model2A.objects.bulk_create_polymorphic(
            [
                Model2B(id=100, field1="B1", field2="B2"),
                Model2A(field1="A1"),
                Model2C(model2b_ptr_id=200, field1="C1", field2="C2", field3="C3"),
            ]
        )
        assert objs[0].pk == 100
        assert objs[2].pk == 200
        assert objs[1].pk not in (100, 200)
        assert Model2A.objects.non_polymorphic().count() == 3
        assert Model2B.objects.non_polymorphic().count() == 2
        assert Model2C.objects.non_polymorphic().count() == 1
        assert Model2A.objects.get(pk=100).field2 == "B2"
        assert Model2A.objects.get(pk=200).field3 == "C3"

    def test_uuid_primary_key_default_is_generated(self):
        objs = UUIDProject.objects.bulk_create_polymorphic(
            [
                UUIDArtProjectD(topic="t1", artist="a1"),
                UUIDResearchProject(topic="t2", supervisor="s1"),
            ]
        )
        assert all(obj.pk is not None for obj in objs)
        # six tables deep for UUIDArtProjectD, two for UUIDResearchProject
        assert UUIDProject.objects.non_polymorphic().count() == 2
        assert UUIDArtProject.objects.non_polymorphic().count() == 1
        assert UUIDArtProjectD.objects.non_polymorphic().count() == 1
        assert [type(obj) for obj in UUIDProject.objects.all()] == [
            UUIDArtProjectD,
            UUIDResearchProject,
        ]

    def test_child_model_manager(self):
        objs = Model2B.objects.bulk_create_polymorphic(
            [
                Model2C(field1="C1", field2="C2", field3="C3"),
                Model2B(field1="B1", field2="B2"),
            ]
        )
        assert [type(obj) for obj in objs] == [Model2C, Model2B]
        assert Model2A.objects.non_polymorphic().count() == 2
        assert Model2B.objects.non_polymorphic().count() == 2
        assert Model2C.objects.non_polymorphic().count() == 1

    def test_proxy_model_manager(self):
        (obj,) = RedheadDuck.objects.bulk_create_polymorphic([RedheadDuck(name="red")])
        assert type(obj) is RedheadDuck
        assert type(Duck.objects.get()) is RedheadDuck

    def test_empty_batch(self):
        with self.assertNumQueries(0):
            assert Model2A.objects.bulk_create_polymorphic([]) == []

    def test_auto_primary_keys_require_returning_support(self):
        with (
            mock.patch.object(
                type(connection.features),
                "can_return_rows_from_bulk_insert",
                new_callable=mock.PropertyMock,
                return_value=False,
            ),
            pytest.raises(NotSupportedError, match="can return them"),
        ):
            Model2A.objects.bulk_create_polymorphic([Model2A(field1="A1")])
        assert Model2A.objects.count() == 0
        # explicit primary keys still work without returning support
        with mock.patch.object(
            type(connection.features),
            "can_return_rows_from_bulk_insert",
            new_callable=mock.PropertyMock,
            return_value=False,
        ):
            (obj,) = Model2A.objects.bulk_create_polymorphic(
                [Model2B(id=10, field1="B1", field2="B2")]
            )
        assert obj.pk == 10
        assert Model2B.objects.non_polymorphic().count() == 1

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="Batch size"):
            Model2A.objects.bulk_create_polymorphic([Model2A(field1="A1")], batch_size=0)

    def test_wrong_type_rejected(self):
        with pytest.raises(TypeError, match="requires instances of tests.Model2A"):
            Model2A.objects.bulk_create_polymorphic([BlogBase(name="b")])
        with pytest.raises(TypeError):
            Model2A.objects.bulk_create_polymorphic([PlainA()])
        with pytest.raises(TypeError):
            Model2A.objects.bulk_create_polymorphic([object()])
        assert Model2A.objects.count() == 0

    def test_duplicate_primary_key_rejected_before_writing(self):
        with pytest.raises(ValueError, match="duplicate primary key"):
            Model2A.objects.bulk_create_polymorphic(
                [
                    Model2B(id=1, field1="B1", field2="B2"),
                    Model2C(id=1, field1="C1", field2="C2", field3="C3"),
                ]
            )
        assert Model2A.objects.count() == 0
        assert Model2B.objects.non_polymorphic().count() == 0

    def test_same_object_twice_rejected_before_writing(self):
        obj = Model2B(field1="B1", field2="B2")
        with pytest.raises(ValueError, match="same tests.Model2B object"):
            Model2A.objects.bulk_create_polymorphic([obj, obj])
        assert Model2A.objects.count() == 0

    def test_unsaved_related_object_rejected_before_writing(self):
        with pytest.raises(ValueError, match="unsaved related object 'one2one'"):
            One2OneRelatingModel.objects.bulk_create_polymorphic(
                [
                    One2OneRelatingModel(
                        one2one=Model2A.objects.create(field1="saved"), field1="F1"
                    ),
                    One2OneRelatingModelDerived(
                        one2one=Model2A(field1="unsaved"), field1="F1", field2="F2"
                    ),
                ]
            )
        # nothing was written, not even the valid object's parent rows
        assert Model2A.objects.count() == 1
        assert One2OneRelatingModel.objects.count() == 0

    def test_unsupported_multiple_concrete_parents_rejected(self):
        with pytest.raises(ValueError, match="multiple concrete parents"):
            MROBase1.objects.bulk_create_polymorphic([MRODerived()])
        with pytest.raises(ValueError, match="multiple concrete parents"):
            Enhance_Inherit.objects.bulk_create_polymorphic(
                [Enhance_Inherit()]
            )
        assert MRODerived.objects.count() == 0

    def test_batch_rolls_back_completely(self):
        # field2 is NOT NULL: the child table insert fails after the parent
        # rows have been written, and the whole batch must be rolled back.
        with pytest.raises(IntegrityError):
            Model2A.objects.bulk_create_polymorphic(
                [
                    Model2B(field1="B1", field2="B2"),
                    Model2B(field1="B3", field2=None),
                ]
            )
        assert Model2A.objects.count() == 0
        assert Model2B.objects.non_polymorphic().count() == 0

    def test_no_save_method_or_signals(self):
        pre_save_receiver = mock.Mock()
        post_save_receiver = mock.Mock()
        pre_save.connect(pre_save_receiver, weak=False)
        post_save.connect(post_save_receiver, weak=False)
        try:
            with mock.patch.object(
                Model2B, "save", autospec=True, side_effect=AssertionError("save called")
            ):
                Model2A.objects.bulk_create_polymorphic(
                    [Model2B(field1="B1", field2="B2")]
                )
        finally:
            pre_save.disconnect(pre_save_receiver)
            post_save.disconnect(post_save_receiver)
        pre_save_receiver.assert_not_called()
        post_save_receiver.assert_not_called()
        assert Model2B.objects.non_polymorphic().count() == 1

    def test_sqlite_returns_primary_keys(self):
        # the whole feature relies on the database returning generated keys
        assert connection.features.can_return_rows_from_bulk_insert
        (obj,) = Model2A.objects.bulk_create_polymorphic(
            [Model2D(field1="D1", field2="D2", field3="D3", field4="D4")]
        )
        assert obj.pk is not None
        assert obj.model2c_ptr_id == obj.pk
        assert Model2D.objects.non_polymorphic().get().pk == obj.pk


class BulkCreatePolymorphicMultipleDatabaseTests(TestCase):
    databases = ["default", "secondary"]

    def test_bulk_create_on_secondary_database(self):
        objs = Model2A.objects.db_manager("secondary").bulk_create_polymorphic(
            [
                Model2C(field1="C1", field2="C2", field3="C3"),
                Model2A(field1="A1"),
            ]
        )
        assert all(obj.pk is not None for obj in objs)
        assert Model2A.objects.db_manager("secondary").non_polymorphic().count() == 2
        assert Model2C.objects.db_manager("secondary").non_polymorphic().count() == 1
        assert Model2A.objects.count() == 0
        fetched = Model2A.objects.db_manager("secondary").order_by("pk")
        assert [type(obj) for obj in fetched] == [Model2C, Model2A]
        # the content type was fetched from the secondary alias
        secondary_ctype = ContentType.objects.db_manager("secondary").get_for_model(
            Model2C, for_concrete_model=False
        )
        assert objs[0].polymorphic_ctype_id == secondary_ctype.pk


class BulkCreatePolymorphicQueryCountTests(TransactionTestCase):
    """
    Assert the number of queries per insert, not just the results. Uses a
    TransactionTestCase so the batch transaction is the outermost one and no
    savepoint queries are emitted (compare test_performance.py).
    """

    databases = ["default", "secondary"]

    def test_query_count_matches_table_count(self):
        warm_content_types(Model2A, Model2B, Model2C, Model2D)
        # three tables: Model2A, Model2B, Model2C
        with count_queries() as ctx:
            Model2A.objects.bulk_create_polymorphic(
                [
                    Model2C(field1="C1", field2="C2", field3="C3"),
                    Model2B(field1="B1", field2="B2"),
                ]
            )
        assert len(inserts(ctx)) == 3
        # four tables for a single Model2D
        with count_queries() as ctx:
            Model2A.objects.bulk_create_polymorphic(
                [Model2D(field1="D1", field2="D2", field3="D3", field4="D4")]
            )
        assert len(inserts(ctx)) == 4
        # a single table for any mixture of proxies
        warm_content_types(Duck, RedheadDuck, RubberDuck)
        with count_queries() as ctx:
            Duck.objects.bulk_create_polymorphic(
                [RedheadDuck(name="r"), Duck(name="d"), RubberDuck(name="b")]
            )
        assert len(inserts(ctx)) == 1

    def test_query_count_with_batch_size(self):
        warm_content_types(Model2A, Model2B)
        # 3 objects in 2 batches for the root table + 2 batches for the child
        with count_queries() as ctx:
            Model2A.objects.bulk_create_polymorphic(
                [Model2B(field1=f"B{i}", field2=f"B{i}") for i in range(3)],
                batch_size=2,
            )
        assert len(inserts(ctx)) == 4
        assert Model2B.objects.non_polymorphic().count() == 3

    def test_content_type_queries_per_database_alias(self):
        ContentType.objects.clear_cache()
        # the content types are fetched per database alias: warm the
        # secondary alias cache, so only the default alias needs a query
        warm_content_types(Model2A, Model2B, using="secondary")
        # one content type query on the default alias, then the table inserts
        with count_queries() as ctx:
            Model2A.objects.bulk_create_polymorphic([Model2B(field1="B1", field2="B2")])
        assert len(inserts(ctx)) == 2
        ctype_queries = [
            q["sql"] for q in ctx.captured_queries if "django_content_type" in q["sql"]
        ]
        assert len(ctype_queries) == 1
        # cached per alias: the secondary database needs no content type query
        with count_queries(using="secondary") as ctx:
            Model2A.objects.db_manager("secondary").bulk_create_polymorphic(
                [Model2B(field1="B1", field2="B2")]
            )
        assert len(inserts(ctx)) == 2
        assert not [
            q for q in ctx.captured_queries if "django_content_type" in q["sql"]
        ]
