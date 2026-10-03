"""
Tests for ``PolymorphicQuerySet.polymorphic_prefetch_related()``:
per-model prefetch plans that are applied only to the real subclasses that
actually carry the relation.
"""

import pytest

from django.db import connection
from django.db.models import Prefetch
from django.test import TransactionTestCase
from django.test.utils import CaptureQueriesContext

from polymorphic.tests.models import (
    PrefetchAlbum,
    PrefetchArticle,
    PrefetchAuthor,
    PrefetchFeaturedArticle,
    PrefetchMedia,
    PrefetchMediaTag,
    PrefetchTrack,
    PrefetchVideo,
)


def _warm_content_types():
    # Ensure the ContentType cache is populated so that query counting only
    # reflects the queries under test.
    list(PrefetchMedia.objects.all())


def _make_media():
    author_a = PrefetchAuthor.objects.create(name="Alice")
    author_b = PrefetchAuthor.objects.create(name="Bob")
    album = PrefetchAlbum.objects.create(name="Album")
    track_one = PrefetchTrack.objects.create(title="T1", album=album)
    track_two = PrefetchTrack.objects.create(title="T2", album=album)
    tag = PrefetchMediaTag.objects.create(name="hot")

    article = PrefetchArticle.objects.create(title="Article", lead_author=author_a)
    article.authors.set([author_a, author_b])
    article.media_tags.set([tag])

    featured = PrefetchFeaturedArticle.objects.create(title="Featured", lead_author=author_a)
    featured.authors.set([author_a])
    featured.media_tags.set([tag])

    video = PrefetchVideo.objects.create(title="Video")
    video.tracks.set([track_one, track_two])
    video.media_tags.set([tag])

    return article, featured, video


class PolymorphicPrefetchTests(TransactionTestCase):
    def setUp(self):
        _warm_content_types()
        self.article, self.featured, self.video = _make_media()
        PrefetchMedia.objects.create(title="Plain media")

    def test_prefetch_applied_only_to_applicable_subclass(self):
        qs = PrefetchMedia.objects.order_by("id").polymorphic_prefetch_related(
            {
                PrefetchArticle: "authors",
                PrefetchVideo: "tracks",
            }
        )
        with CaptureQueriesContext(connection) as exec_ctx:
            objs = list(qs)
        with CaptureQueriesContext(connection) as read_ctx:
            for obj in objs:
                if isinstance(obj, PrefetchArticle):
                    list(obj.authors.all())
                elif isinstance(obj, PrefetchVideo):
                    list(obj.tracks.all())

        # One query for each occurring (model, relation) combination:
        # authors for Article + FeaturedArticle, tracks for Video.
        sqls = [q["sql"] for q in exec_ctx.captured_queries]
        assert sum("prefetcharticle_authors" in sql for sql in sqls) == 2
        assert sum("prefetchvideo_tracks" in sql for sql in sqls) == 1
        # Reading the prefetched relations requires no extra query.
        assert read_ctx.captured_queries == []

    def test_prefetch_object_to_attr_and_custom_queryset(self):
        qs = PrefetchMedia.objects.order_by("id").polymorphic_prefetch_related(
            {
                PrefetchArticle: Prefetch(
                    "authors",
                    queryset=PrefetchAuthor.objects.order_by("-name"),
                    to_attr="authors_desc",
                ),
            }
        )
        objs = list(qs)
        article = objs[0]
        assert isinstance(article, PrefetchArticle)
        assert [author.name for author in article.authors_desc] == ["Bob", "Alice"]
        with CaptureQueriesContext(connection) as ctx:
            assert [author.name for author in article.authors_desc] == ["Bob", "Alice"]
        assert len(ctx.captured_queries) == 0

    def test_nested_path(self):
        qs = PrefetchMedia.objects.order_by("id").polymorphic_prefetch_related(
            {
                PrefetchVideo: "tracks__album",
            }
        )
        objs = list(qs)
        video = next(obj for obj in objs if isinstance(obj, PrefetchVideo))
        with CaptureQueriesContext(connection) as ctx:
            albums = [track.album.name for track in video.tracks.all()]
        assert albums == ["Album", "Album"]
        assert len(ctx.captured_queries) == 0

    def test_base_plan_inherited_by_subclasses(self):
        qs = PrefetchMedia.objects.order_by("id").polymorphic_prefetch_related(
            {
                # media_tags lives on the common base and is inherited by all
                # concrete subclasses.
                PrefetchMedia: "media_tags",
                PrefetchArticle: Prefetch("authors", to_attr="writers"),
                PrefetchVideo: "tracks",
            }
        )
        with CaptureQueriesContext(connection) as ctx:
            objs = list(qs)
        with CaptureQueriesContext(connection) as read_ctx:
            for obj in objs:
                list(obj.media_tags.all())
                if isinstance(obj, PrefetchArticle):
                    getattr(obj, "writers")
                elif isinstance(obj, PrefetchVideo):
                    list(obj.tracks.all())
        assert read_ctx.captured_queries == []

        sqls = [q["sql"] for q in ctx.captured_queries]
        # media_tags is prefetched once per concrete model group:
        # Article, FeaturedArticle, Video (plain media needs none of the
        # subclass-only plans but does inherit media_tags).
        assert sum("prefetchmedia_media_tags" in sql for sql in sqls) == 4
        assert sum("prefetcharticle_authors" in sql for sql in sqls) == 2
        assert sum("prefetchvideo_tracks" in sql for sql in sqls) == 1

    def test_derived_plan_overrides_inherited_plan(self):
        qs = PrefetchMedia.objects.order_by("id").polymorphic_prefetch_related(
            {
                PrefetchArticle: "media_tags",
                PrefetchFeaturedArticle: Prefetch("media_tags", to_attr="tag_list"),
            }
        )
        objs = list(qs)
        article = next(obj for obj in objs if type(obj) is PrefetchArticle)
        featured = next(obj for obj in objs if type(obj) is PrefetchFeaturedArticle)
        # Base lookup for the direct subclass.
        assert list(article.media_tags.all())
        # Overridden to_attr lookup for the more derived subclass.
        assert hasattr(featured, "tag_list")
        assert list(featured.tag_list)

    def test_conflicting_querysets_raise_before_prefetch_query(self):
        qs = PrefetchMedia.objects.order_by("id").polymorphic_prefetch_related(
            {
                PrefetchArticle: [
                    Prefetch("authors", queryset=PrefetchAuthor.objects.filter(name="Alice")),
                    Prefetch("authors", queryset=PrefetchAuthor.objects.filter(name="Bob")),
                ],
            }
        )
        with CaptureQueriesContext(connection) as ctx:
            with pytest.raises(ValueError):
                list(qs)
        # Validation runs before any prefetch query is executed.
        assert not any("_prefetch_related" in q["sql"] for q in ctx.captured_queries)

    def test_plain_and_prefetch_lookup_conflict(self):
        qs = PrefetchMedia.objects.order_by("id").polymorphic_prefetch_related(
            {
                PrefetchArticle: [
                    "authors",
                    Prefetch("authors", queryset=PrefetchAuthor.objects.all()),
                ],
            }
        )
        with pytest.raises(ValueError):
            list(qs)

    def test_invalid_path_raises_before_prefetch_query(self):
        # tracks only exists on Video; inheriting it from the base onto an
        # Article object must fail before queries are issued.
        qs = PrefetchMedia.objects.order_by("id").polymorphic_prefetch_related(
            {PrefetchMedia: "tracks"}
        )
        with CaptureQueriesContext(connection) as ctx:
            with pytest.raises(AttributeError):
                list(qs)
        assert not any("_prefetch_related" in q["sql"] for q in ctx.captured_queries)

    def test_non_relation_endpoint_raises(self):
        qs = PrefetchMedia.objects.order_by("id").polymorphic_prefetch_related(
            {PrefetchArticle: Prefetch("title")}
        )
        with pytest.raises(ValueError):
            list(qs)

    def test_to_attr_on_plain_field_raises(self):
        qs = PrefetchMedia.objects.order_by("id").polymorphic_prefetch_related(
            {PrefetchArticle: Prefetch("title", to_attr="headline")}
        )
        with pytest.raises(ValueError):
            list(qs)

    def test_invalid_lookup_type_raises(self):
        with pytest.raises(TypeError):
            PrefetchMedia.objects.polymorphic_prefetch_related({PrefetchArticle: 42})

    def test_non_model_key_raises(self):
        with pytest.raises(TypeError):
            PrefetchMedia.objects.polymorphic_prefetch_related({"nope": "authors"})

    def test_empty_result_emits_no_extra_queries(self):
        PrefetchMedia.objects.all().delete()
        with CaptureQueriesContext(connection) as ctx:
            objs = list(
                PrefetchMedia.objects.polymorphic_prefetch_related(
                    {PrefetchArticle: ["authors"], PrefetchVideo: ["tracks"]}
                )
            )
        assert objs == []
        assert len(ctx.captured_queries) == 1

    def test_absent_subclass_emits_no_query(self):
        # Delete video objects so only articles occur.
        PrefetchVideo.objects.all().delete()
        with CaptureQueriesContext(connection) as ctx:
            list(
                PrefetchMedia.objects.order_by("id").polymorphic_prefetch_related(
                    {
                        PrefetchArticle: "authors",
                        PrefetchVideo: "tracks",
                    }
                )
            )
        sqls = [q["sql"] for q in ctx.captured_queries]
        assert not any("prefetchvideo" in sql for sql in sqls)
        assert any("prefetcharticle_authors" in sql for sql in sqls)

    def test_iterator_chunking(self):
        # Use an isolated result set of two articles and two videos so the
        # chunk prefetch query count is deterministic.
        author = PrefetchAuthor.objects.create(name="Solo")
        article_ids = []
        for index in range(2):
            article = PrefetchArticle.objects.create(title=f"A{index}")
            article.authors.set([author])
            article_ids.append(article.pk)
        track = PrefetchTrack.objects.create(title="SoloT")
        video_ids = []
        for index in range(2):
            video = PrefetchVideo.objects.create(title=f"V{index}")
            video.tracks.set([track])
            video_ids.append(video.pk)

        ids = article_ids + video_ids
        qs = (
            PrefetchMedia.objects.filter(pk__in=ids)
            .order_by("id")
            .polymorphic_prefetch_related(
                {PrefetchArticle: "authors", PrefetchVideo: "tracks"}
            )
        )
        with CaptureQueriesContext(connection) as ctx:
            objs = list(qs.iterator(chunk_size=2))
        with CaptureQueriesContext(connection) as read_ctx:
            for obj in objs:
                if isinstance(obj, PrefetchArticle):
                    assert len(obj.authors.all()) == 1
                else:
                    assert len(obj.tracks.all()) == 1
        assert read_ctx.captured_queries == []

        sqls = [q["sql"] for q in ctx.captured_queries]
        # Prefetch queries are per occurring model/relation per chunk, never
        # per row.
        assert sum("prefetcharticle_authors" in sql for sql in sqls) >= 1
        assert sum("prefetchvideo_tracks" in sql for sql in sqls) >= 1
        assert sum("prefetcharticle_authors" in sql for sql in sqls) <= 2
        assert sum("prefetchvideo_tracks" in sql for sql in sqls) <= 2

        # Reading the prefetched relations afterwards costs nothing.
        with CaptureQueriesContext(connection) as read_ctx:
            for obj in objs:
                if isinstance(obj, PrefetchArticle):
                    list(obj.authors.all())
                else:
                    list(obj.tracks.all())
        assert read_ctx.captured_queries == []

    def test_only_and_defer_with_m2m(self):
        qs = PrefetchMedia.objects.only("title").order_by("id").polymorphic_prefetch_related(
            {
                PrefetchMedia: "media_tags",
                PrefetchVideo: "tracks",
            }
        )
        with CaptureQueriesContext(connection) as ctx:
            objs = list(qs)
        with CaptureQueriesContext(connection) as read_ctx:
            for obj in objs:
                list(obj.media_tags.all())
                if isinstance(obj, PrefetchVideo):
                    list(obj.tracks.all())
        assert read_ctx.captured_queries == []
        assert objs[0].title

        qs = PrefetchMedia.objects.defer("title").order_by("id").polymorphic_prefetch_related(
            {PrefetchMedia: "media_tags", PrefetchVideo: "tracks"}
        )
        with CaptureQueriesContext(connection) as ctx:
            objs = list(qs)
        with CaptureQueriesContext(connection) as read_ctx:
            for obj in objs:
                list(obj.media_tags.all())
                if isinstance(obj, PrefetchVideo):
                    list(obj.tracks.all())
        assert read_ctx.captured_queries == []

    def test_annotate_is_preserved(self):
        from django.db.models import Count

        qs = (
            PrefetchMedia.objects.annotate(tag_count=Count("media_tags"))
            .order_by("id")
            .polymorphic_prefetch_related(
                {
                    PrefetchMedia: "media_tags",
                    PrefetchArticle: "media_tags",
                    PrefetchVideo: "media_tags",
                }
            )
        )
        with CaptureQueriesContext(connection) as ctx:
            objs = list(qs)
        counts = {type(obj).__name__: obj.tag_count for obj in objs}
        assert counts["PrefetchArticle"] == 1
        assert counts["PrefetchVideo"] == 1
        with CaptureQueriesContext(connection) as read_ctx:
            for obj in objs:
                list(obj.media_tags.all())
        assert read_ctx.captured_queries == []

    def test_ordering_and_duplicate_objects_preserved(self):
        base = list(PrefetchMedia.objects.order_by("id"))
        duplicated = [base[0], base[1], base[0]]
        qs = PrefetchMedia.objects.polymorphic_prefetch_related(
            {PrefetchArticle: "authors", PrefetchVideo: "tracks"}
        )
        objs = qs.get_real_instances(duplicated)
        assert [type(obj).__name__ for obj in objs] == [
            "PrefetchArticle",
            "PrefetchFeaturedArticle",
            "PrefetchArticle",
        ]

    def test_coexists_with_vanilla_prefetch(self):
        qs = (
            PrefetchMedia.objects.order_by("id")
            .prefetch_related("media_tags")
            .polymorphic_prefetch_related(
                {PrefetchArticle: "authors", PrefetchVideo: "tracks"}
            )
        )
        with CaptureQueriesContext(connection) as exec_ctx:
            objs = list(qs)
        with CaptureQueriesContext(connection) as read_ctx:
            for obj in objs:
                list(obj.media_tags.all())
                if isinstance(obj, PrefetchArticle):
                    list(obj.authors.all())
                elif isinstance(obj, PrefetchVideo):
                    list(obj.tracks.all())
        assert read_ctx.captured_queries == []

    def test_non_polymorphic_skips_plans(self):
        qs = PrefetchMedia.objects.non_polymorphic().polymorphic_prefetch_related(
            {PrefetchArticle: "authors"}
        )
        objs = list(qs)
        assert all(type(obj) is PrefetchMedia for obj in objs)

    def test_chaining_merges_and_is_immutable(self):
        base = PrefetchMedia.objects.polymorphic_prefetch_related(
            {PrefetchArticle: "authors"}
        )
        chained = base.polymorphic_prefetch_related({PrefetchVideo: "tracks"})
        assert PrefetchArticle in base._polymorphic_prefetch_plans
        assert PrefetchVideo not in base._polymorphic_prefetch_plans
        assert PrefetchArticle in chained._polymorphic_prefetch_plans
        assert PrefetchVideo in chained._polymorphic_prefetch_plans


@pytest.mark.django_db(databases=["default", "secondary"])
def test_prefetch_uses_query_database():
    from django.db import connections

    _warm_content_types()
    # Warm the ContentType cache on the secondary database as well.
    list(PrefetchMedia.objects.using("secondary"))
    author = PrefetchAuthor.objects.using("secondary").create(name="Alice")
    article = PrefetchArticle.objects.using("secondary").create(title="Article")
    article.authors.db_manager("secondary").set([author])

    qs = PrefetchMedia.objects.db_manager("secondary").order_by("id").polymorphic_prefetch_related(
        {PrefetchArticle: "authors"}
    )
    with CaptureQueriesContext(connections["default"]) as default_ctx, CaptureQueriesContext(
        connections["secondary"]
    ) as secondary_ctx:
        objs = list(qs)
        names = [name for obj in objs for name in obj.authors.values_list("name", flat=True)]

    assert names == ["Alice"]
    assert default_ctx.captured_queries == []
    assert any("prefetcharticle_authors" in q["sql"] for q in secondary_ctx.captured_queries)
