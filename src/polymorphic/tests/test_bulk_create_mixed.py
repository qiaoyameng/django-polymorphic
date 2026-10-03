"""
Tests for :meth:`polymorphic.query.PolymorphicQuerySet.bulk_create_mixed`,
the mixed-type bulk creation API for polymorphic models.
"""

from __future__ import annotations

import pytest
from django.db import connections
from django.db.models.signals import post_save, pre_save
from django.db.utils import IntegrityError
from django.test.utils import CaptureQueriesContext

from polymorphic.models import PolymorphicBulkCreateError
from polymorphic.tests.models import (
    Bottom,
    BulkTimestampBase,
    BulkTimestampChild,
    BulkTimestampGrandChild,
    CustomPkBase,
    CustomPkInherit,
    Duck,
    InlineModelA,
    InlineParent,
    Middle,
    Model2A,
    Model2B,
    Model2C,
    Model2D,
    MRODerived,
    ProxiedBase,
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


def insert_queries(context: CaptureQueriesContext) -> list[str]:
    return [
        query["sql"]
        for query in context.captured_queries
        if query["sql"].lower().lstrip().startswith("insert")
    ]


pytestmark = pytest.mark.django_db(transaction=True)


class TestBulkCreateMixed:
    def test_proxy_types_share_one_table(self, django_assert_max_num_queries):
        # All three types are stored in the same Duck table: a single INSERT.
        objs = [Duck(name="d"), RedheadDuck(name="r"), RubberDuck(name="ru")]
        with CaptureQueriesContext(connections["default"]) as context:
            created = Duck.objects.bulk_create_mixed(objs)

        assert len(insert_queries(context)) == 1
        assert [type(obj) for obj in created] == [Duck, RedheadDuck, RubberDuck]
        assert [obj.name for obj in created] == ["d", "r", "ru"]
        assert all(obj.pk is not None and not obj._state.adding for obj in created)

        downcast = list(Duck.objects.order_by("pk"))
        assert [(type(obj), obj.name) for obj in downcast] == [
            (Duck, "d"),
            (RedheadDuck, "r"),
            (RubberDuck, "ru"),
        ]

    def test_multi_table_writes_parent_before_child_and_wires_links(self):
        objs = [
            Top(name="t"),
            Middle(name="m", description="m-desc"),
            Bottom(name="b", description="b-desc", author="b-author"),
        ]
        with CaptureQueriesContext(connections["default"]) as context:
            created = Top.objects.bulk_create_mixed(objs)

        # One INSERT per concrete table, ancestor first.
        queries = insert_queries(context)
        assert len(queries) == 3
        assert "tests_top" in queries[0]
        assert "tests_middle" in queries[1]
        assert "tests_bottom" in queries[2]
        # The child insert carries the auto generated parent-link pk.
        top_pk = created[1].pk
        bottom_pk = created[2].pk
        assert f"({top_pk}," in queries[1].replace(" ", "")
        # Bottom links directly to Middle (its immediate parent).
        assert f"({bottom_pk}," in queries[2].replace(" ", "")

        downcast = list(Top.objects.order_by("pk"))
        assert [type(obj) for obj in downcast] == [Top, Middle, Bottom]
        assert downcast[2].author == "b-author"

    def test_multi_level_and_shared_ancestor_not_duplicated(self):
        objs = [
            Model2A(field1="a"),
            Model2B(field1="b1", field2="b2"),
            Model2C(field1="c1", field2="c2", field3="c3"),
            Model2D(field1="d1", field2="d2", field3="d3", field4="d4"),
        ]
        with CaptureQueriesContext(connections["default"]) as context:
            created = Model2A.objects.bulk_create_mixed(objs)

        # A, B, C, D are four distinct tables; the deepest object traverses all
        # four but every ancestor table receives exactly one row per object.
        assert len(insert_queries(context)) == 4
        assert [type(obj) for obj in created] == [Model2A, Model2B, Model2C, Model2D]

        with connections["default"].cursor() as cursor:
            for table, count in (
                ("tests_model2a", 4),
                ("tests_model2b", 3),
                ("tests_model2c", 2),
                ("tests_model2d", 1),
            ):
                cursor.execute(f"SELECT count(*) FROM {table}")
                assert cursor.fetchone()[0] == count

        assert [type(obj) for obj in Model2A.objects.order_by("pk")] == [
            Model2A,
            Model2B,
            Model2C,
            Model2D,
        ]

    def test_deep_chain_produces_one_insert_per_distinct_table(self):
        objs = [
            UUIDProject(topic="p"),
            UUIDArtProject(topic="a", artist="ar"),
            UUIDResearchProject(topic="r", supervisor="s"),
            UUIDArtProjectD(topic="d", artist="ad"),
        ]
        with CaptureQueriesContext(connections["default"]) as context:
            created = UUIDProject.objects.bulk_create_mixed(objs)

        # UUIDArtProjectD is stored across D, C, B, A and Project: the union of
        # distinct tables is Project, ArtProject, ResearchProject, ArtProjectA,
        # ArtProjectB, ArtProjectC, ArtProjectD => 7 INSERTs.
        assert len(insert_queries(context)) == 7
        assert all(obj.pk is not None for obj in created)
        assert {type(obj) for obj in UUIDProject.objects.all()} == {
            UUIDProject,
            UUIDArtProject,
            UUIDResearchProject,
            UUIDArtProjectD,
        }

    def test_mixed_explicit_and_auto_primary_keys(self):
        objs = [
            Top(pk=100, name="explicit"),
            Middle(name="auto-middle", description="d"),
            Top(name="auto-top"),
            Middle(pk=200, name="explicit-middle", description="d2"),
        ]
        with CaptureQueriesContext(connections["default"]) as context:
            created = Top.objects.bulk_create_mixed(objs)

        # Each table emits up to two INSERTs (explicit-pk rows and auto-pk rows
        # are partitioned, mirroring Django's own bulk_create()).
        queries = insert_queries(context)
        top_inserts = [q for q in queries if "tests_top" in q]
        assert len(top_inserts) == 2
        assert created[0].pk == 100
        # Middle(pk=200) assigns the shared Top row pk via the parent link.
        assert created[3].top_ptr_id == 200
        assert created[3].pk == 200

        rows = dict(Top.objects.values_list("pk", "name"))
        assert rows[100] == "explicit"
        assert rows[200] == "explicit-middle"
        assert created[1].pk not in (100, 200)
        assert {type(obj) for obj in Top.objects.filter(pk__in=(100, 200))} == {Top, Middle}

    def test_custom_child_primary_key_is_returned(self):
        objs = [CustomPkBase(b="base"), CustomPkInherit(b="child", i="iv")]
        created = CustomPkBase.objects.bulk_create_mixed(objs)

        base, child = created
        assert base.pk is not None and child.pk is not None
        # CustomPkInherit.pk is its own AutoField (custom_id), independent of
        # the inherited (shared) parent primary key.
        assert child.custom_id is not None
        assert child.custompkbase_ptr_id is not None
        fetched = CustomPkInherit.objects.get(pk=child.pk)
        assert fetched.b == "child"
        assert fetched.i == "iv"
        assert fetched.custom_id == child.custom_id
        assert fetched.custompkbase_ptr_id == child.custompkbase_ptr_id
        # The parent (base) row exists with the linked shared id.
        assert CustomPkBase.objects.filter(pk=fetched.custompkbase_ptr_id).exists()

    def test_input_order_is_preserved(self):
        objs = [
            Bottom(name="1", description="x", author="y"),
            Top(name="2"),
            Middle(name="3", description="z"),
            Top(name="4"),
        ]
        created = Top.objects.bulk_create_mixed(objs)
        assert [type(obj).__name__ for obj in created] == [
            "Bottom",
            "Top",
            "Middle",
            "Top",
        ]
        assert [obj.name for obj in created] == ["1", "2", "3", "4"]

    def test_proxy_concrete_chain(self):
        objs = [
            ProxiedBase(name="base"),
            ProxyModelA(name="a", field1="f1"),
            ProxyModelB(name="b", field2="f2"),
        ]
        created = ProxiedBase.objects.bulk_create_mixed(objs)
        assert [type(obj) for obj in created] == [ProxiedBase, ProxyModelA, ProxyModelB]
        assert [type(obj) for obj in ProxiedBase.objects.order_by("pk")] == [
            ProxiedBase,
            ProxyModelA,
            ProxyModelB,
        ]

    def test_proxy_queryset_rejects_base_instances(self):
        with pytest.raises(PolymorphicBulkCreateError):
            RedheadDuck.objects.bulk_create_mixed([Duck(name="d"), RedheadDuck(name="r")])
        # Nothing was written.
        assert Duck.objects.count() == 0

    def test_empty_batch_returns_empty_list_without_queries(self):
        with CaptureQueriesContext(connections["default"]) as context:
            result = Top.objects.bulk_create_mixed([])
        assert result == []
        assert insert_queries(context) == []

    def test_rejects_wrong_type(self):
        with pytest.raises(PolymorphicBulkCreateError):
            Top.objects.bulk_create_mixed([Duck(name="d")])
        assert Top.objects.count() == 0

    def test_rejects_already_saved_instance(self):
        saved = Top.objects.create(name="saved")
        with pytest.raises(PolymorphicBulkCreateError):
            Top.objects.bulk_create_mixed([saved, Top(name="new")])
        assert Top.objects.count() == 1

    def test_rejects_duplicate_object_identity(self):
        obj = Top(name="x")
        with pytest.raises(PolymorphicBulkCreateError):
            Top.objects.bulk_create_mixed([obj, obj])
        assert Top.objects.count() == 0

    def test_rejects_duplicate_explicit_pk_same_table(self):
        with pytest.raises(PolymorphicBulkCreateError):
            Top.objects.bulk_create_mixed([Top(pk=5, name="a"), Top(pk=5, name="b")])
        assert Top.objects.count() == 0

    def test_rejects_duplicate_explicit_pk_across_levels(self):
        with pytest.raises(PolymorphicBulkCreateError):
            Top.objects.bulk_create_mixed(
                [Top(pk=6, name="a"), Middle(pk=6, name="b", description="x")]
            )
        assert Top.objects.count() == 0

    def test_rejects_unsaved_related_object(self):
        unsaved_parent = InlineParent(title="p")
        with pytest.raises(PolymorphicBulkCreateError):
            InlineModelA.objects.bulk_create_mixed(
                [InlineModelA(parent=unsaved_parent, field1="x")]
            )

    def test_accepts_saved_related_object(self):
        parent = InlineParent.objects.create(title="p")
        created = InlineModelA.objects.bulk_create_mixed(
            [InlineModelA(parent=parent, field1="x")]
        )
        assert created[0].parent_id == parent.pk

    def test_rejects_unsupported_multiple_concrete_parents(self):
        with pytest.raises(PolymorphicBulkCreateError):
            Model2A.objects.bulk_create_mixed([MRODerived(field1="x")])

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            Top.objects.bulk_create_mixed([Top(name="x")], batch_size=0)

    def test_batch_size_chunks_insert_statements(self):
        objs = [Top(name=f"t{i}") for i in range(5)]
        with CaptureQueriesContext(connections["default"]) as context:
            Top.objects.bulk_create_mixed(objs, batch_size=2)
        assert len(insert_queries(context)) == 3

    @pytest.mark.parametrize(
        "objs_factory, expected_tables",
        [
            # proxy only -> single shared table
            (lambda: [Duck(name="d"), RedheadDuck(name="r")], ["tests_duck"]),
            # base + two sibling children -> three tables
            (
                lambda: [
                    Model2A(field1="a"),
                    Model2B(field1="b", field2="b2"),
                    Model2C(field1="c", field2="c2", field3="c3"),
                ],
                ["tests_model2a", "tests_model2b", "tests_model2c"],
            ),
        ],
    )
    def test_query_count_matches_distinct_tables(self, objs_factory, expected_tables):
        objs = objs_factory()
        manager = (
            Duck.objects if isinstance(objs[0], Duck) else Model2A.objects
        )
        with CaptureQueriesContext(connections["default"]) as context:
            manager.bulk_create_mixed(objs)

        queries = insert_queries(context)
        assert len(queries) == len(expected_tables)
        for table in expected_tables:
            assert any(table in query for query in queries)

    def test_auto_now_and_auto_now_add_are_populated(self):
        from django.utils import timezone

        before = timezone.now()
        created = BulkTimestampBase.objects.bulk_create_mixed(
            [
                BulkTimestampBase(name="base"),
                BulkTimestampChild(name="child", detail="d"),
                BulkTimestampGrandChild(name="gc", detail="d2", extra="e"),
            ]
        )
        after = timezone.now()
        for obj in created:
            assert before <= obj.created_at <= after
            assert before <= obj.modified_at <= after
        # Values persisted to the (root) table.
        assert all(
            ts is not None
            for ts in BulkTimestampBase.objects.values_list("created_at", flat=True)
        )
        assert BulkTimestampGrandChild.objects.get().extra == "e"

    def test_save_signals_are_not_sent(self, settings):
        sent = []

        def receiver_pre(*args, **kwargs):
            sent.append("pre")

        def receiver_post(*args, **kwargs):
            sent.append("post")

        pre_save.connect(receiver_pre)
        post_save.connect(receiver_post)
        try:
            Top.objects.bulk_create_mixed([Top(name="a"), Middle(name="b", description="d")])
        finally:
            pre_save.disconnect(receiver_pre)
            post_save.disconnect(receiver_post)
        assert sent == []

    def test_transaction_rolls_back_parent_rows_on_failure(self):
        Top.objects.create(pk=50, name="existing")
        with pytest.raises(IntegrityError):
            Top.objects.bulk_create_mixed(
                [
                    Top(name="would-be-parent"),
                    Top(pk=50, name="collision"),
                ]
            )
        # The first object must not leave an orphaned parent row behind.
        assert set(Top.objects.values_list("pk", flat=True)) == {50}

    def test_rollback_when_child_table_fails_after_parents(self, monkeypatch):
        """
        Simulate a failure while inserting a child table after the parent rows
        have been written and assert no orphaned parent rows remain.
        """
        original_batched_insert = type(Top.objects.all())._batched_insert
        call_count = {"n": 0}

        def failing_batched_insert(self, objs, fields, batch_size, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:  # the Middle table insert
                raise IntegrityError("simulated child insert failure")
            return original_batched_insert(self, objs, fields, batch_size, **kwargs)

        monkeypatch.setattr(
            type(Top.objects.all()), "_batched_insert", failing_batched_insert
        )
        with pytest.raises(IntegrityError):
            Top.objects.bulk_create_mixed(
                [Top(name="t"), Middle(name="m", description="d")]
            )
        assert Top.objects.count() == 0
        assert Middle.objects.count() == 0


@pytest.mark.django_db(transaction=True, databases="__all__")
class TestBulkCreateMixedMultiDb:
    def test_content_type_resolved_per_database_alias(self):
        created = Top.objects.db_manager("secondary").bulk_create_mixed(
            [Top(name="s"), Middle(name="sm", description="d"), Bottom(
                name="sb", description="bd", author="ba"
            )]
        )
        assert [obj._state.db for obj in created] == ["secondary"] * 3
        assert [type(obj) for obj in Top.objects.db_manager("secondary").all()] == [
            Top,
            Middle,
            Bottom,
        ]
        # Nothing leaked to the default database.
        assert Top.objects.using("default").count() == 0
