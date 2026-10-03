"""
Tests for PolymorphicQuerySet.polymorphic_prefetch().
"""

from django.db import connection
from django.db.models import Count, Prefetch
from django.test import TestCase
from django.test.utils import CaptureQueriesContext

from polymorphic.tests.models import (
    MediaArticle,
    MediaAuthor,
    MediaBase,
    MediaFeaturedArticle,
    MediaLabel,
    MediaTag,
    MediaTrack,
    MediaVideo,
)


class PolymorphicPrefetchTests(TestCase):
    def create_media(self):
        self.author1 = MediaAuthor.objects.create(name="author1")
        self.author2 = MediaAuthor.objects.create(name="author2")
        self.label = MediaLabel.objects.create(name="label1")
        self.tag1 = MediaTag.objects.create(name="tag1")
        self.tag2 = MediaTag.objects.create(name="tag2")

        self.article1 = MediaArticle.objects.create(title="article1")
        self.article1.authors.add(self.author1)
        self.article1.tags.add(self.tag1)
        self.article2 = MediaArticle.objects.create(title="article2")
        self.article2.authors.add(self.author1, self.author2)

        self.video1 = MediaVideo.objects.create(title="video1")
        self.track1 = MediaTrack.objects.create(
            name="track1", video=self.video1, label=self.label
        )
        self.track2 = MediaTrack.objects.create(
            name="track2", video=self.video1, label=self.label
        )
        self.video2 = MediaVideo.objects.create(title="video2")
        self.video2.tags.add(self.tag2)

    def test_basic_per_model_prefetch(self):
        self.create_media()
        qs = MediaBase.objects.order_by("pk").polymorphic_prefetch(
            {MediaArticle: ["authors"], MediaVideo: ["tracks"]}
        )
        # warm up the content type cache
        list(qs)

        # 1 base query, 1 real instance query per occurring model,
        # 1 prefetch query per applicable relation
        with self.assertNumQueries(5):
            objects = list(qs.all())

        assert [type(o) for o in objects] == [
            MediaArticle,
            MediaArticle,
            MediaVideo,
            MediaVideo,
        ]
        with self.assertNumQueries(0):
            assert [a.name for a in objects[0].authors.all()] == ["author1"]
            assert sorted(a.name for a in objects[1].authors.all()) == [
                "author1",
                "author2",
            ]
            assert sorted(t.name for t in objects[2].tracks.all()) == ["track1", "track2"]
            assert list(objects[3].tracks.all()) == []

    def test_query_count_does_not_depend_on_row_count(self):
        self.create_media()
        # add more rows of the same models
        for i in range(10):
            article = MediaArticle.objects.create(title=f"article-x{i}")
            article.authors.add(self.author1)
            video = MediaVideo.objects.create(title=f"video-x{i}")
            MediaTrack.objects.create(name=f"track-x{i}", video=video)

        qs = MediaBase.objects.polymorphic_prefetch(
            {MediaArticle: ["authors"], MediaVideo: ["tracks"]}
        )
        list(qs)  # warm up the content type cache
        with self.assertNumQueries(5):
            objects = list(qs.all())
        assert len(objects) == 24

    def test_prefetch_object_with_queryset_and_to_attr(self):
        self.create_media()
        qs = MediaBase.objects.order_by("pk").polymorphic_prefetch(
            {
                MediaVideo: [
                    Prefetch(
                        "tracks",
                        queryset=MediaTrack.objects.order_by("-name"),
                        to_attr="sorted_tracks",
                    )
                ]
            }
        )
        objects = list(qs)
        video = next(o for o in objects if isinstance(o, MediaVideo))
        with self.assertNumQueries(0):
            assert [t.name for t in video.sorted_tracks] == ["track2", "track1"]

    def test_nested_lookup_path(self):
        self.create_media()
        qs = MediaBase.objects.polymorphic_prefetch({MediaVideo: ["tracks__label"]})
        objects = list(qs)
        video = next(o for o in objects if isinstance(o, MediaVideo))
        with self.assertNumQueries(0):
            assert {t.label.name for t in video.tracks.all()} == {"label1"}

    def test_nested_prefetch_queryset(self):
        self.create_media()
        qs = MediaBase.objects.polymorphic_prefetch(
            {
                MediaVideo: [
                    Prefetch(
                        "tracks",
                        queryset=MediaTrack.objects.prefetch_related("label"),
                    )
                ]
            }
        )
        objects = list(qs)
        video = next(o for o in objects if isinstance(o, MediaVideo))
        with self.assertNumQueries(0):
            assert {t.label.name for t in video.tracks.all()} == {"label1"}

    def test_base_model_plan_applies_to_all_subclasses(self):
        self.create_media()
        qs = MediaBase.objects.order_by("pk").polymorphic_prefetch({MediaBase: ["tags"]})
        objects = list(qs)
        with self.assertNumQueries(0):
            assert [t.name for t in objects[0].tags.all()] == ["tag1"]
            assert list(objects[1].tags.all()) == []
            assert list(objects[2].tags.all()) == []
            assert [t.name for t in objects[3].tags.all()] == ["tag2"]

    def test_plan_is_inherited_by_subclasses(self):
        self.create_media()
        featured = MediaFeaturedArticle.objects.create(title="featured", featured=True)
        featured.authors.add(self.author2)

        qs = MediaBase.objects.order_by("pk").polymorphic_prefetch(
            {MediaArticle: ["authors"]}
        )
        objects = list(qs)
        featured_obj = objects[-1]
        assert type(featured_obj) is MediaFeaturedArticle
        with self.assertNumQueries(0):
            assert [a.name for a in featured_obj.authors.all()] == ["author2"]

    def test_subclass_plan_specializes_base_plan(self):
        self.create_media()
        self.article1.tags.add(self.tag2)
        qs = MediaBase.objects.order_by("pk").polymorphic_prefetch(
            {
                MediaBase: ["tags"],
                MediaArticle: [
                    Prefetch(
                        "tags",
                        queryset=MediaTag.objects.order_by("-name"),
                        to_attr="sorted_tags",
                    )
                ],
            }
        )
        objects = list(qs)
        article, video = objects[0], objects[3]
        with self.assertNumQueries(0):
            # the custom Prefetch of the subclass plan takes precedence
            assert [t.name for t in article.sorted_tags] == ["tag2", "tag1"]
            # the plain lookup of the base plan still applies to other models
            assert [t.name for t in video.tags.all()] == ["tag2"]

    def test_plans_merge_across_ancestors(self):
        self.create_media()
        qs = MediaBase.objects.order_by("pk").polymorphic_prefetch(
            {MediaBase: ["tags"], MediaArticle: ["authors"], MediaVideo: ["tracks"]}
        )
        objects = list(qs)
        with self.assertNumQueries(0):
            assert [a.name for a in objects[0].authors.all()] == ["author1"]
            assert [t.name for t in objects[0].tags.all()] == ["tag1"]
            assert sorted(t.name for t in objects[2].tracks.all()) == ["track1", "track2"]
            assert [t.name for t in objects[3].tags.all()] == ["tag2"]

    def test_chained_calls_merge_plans(self):
        self.create_media()
        qs = (
            MediaBase.objects.order_by("pk")
            .polymorphic_prefetch({MediaArticle: ["authors"]})
            .polymorphic_prefetch({MediaVideo: ["tracks"], MediaArticle: ["tags"]})
        )
        objects = list(qs)
        with self.assertNumQueries(0):
            assert [a.name for a in objects[0].authors.all()] == ["author1"]
            assert [t.name for t in objects[0].tags.all()] == ["tag1"]
            assert sorted(t.name for t in objects[2].tracks.all()) == ["track1", "track2"]

    def test_manager_proxy(self):
        self.create_media()
        objects = list(
            MediaBase.objects.polymorphic_prefetch({MediaArticle: ["authors"]}).order_by("pk")
        )
        with self.assertNumQueries(0):
            assert [a.name for a in objects[0].authors.all()] == ["author1"]

    def test_duplicate_plain_lookups_are_deduplicated(self):
        self.create_media()
        qs = MediaBase.objects.polymorphic_prefetch(
            {MediaArticle: ["authors", "authors", Prefetch("authors")]}
        )
        list(qs)  # warm up the content type cache
        with CaptureQueriesContext(connection) as captured:
            list(qs.all())
        author_queries = [q for q in captured if "mediaauthor" in q["sql"].lower()]
        assert len(author_queries) == 1

    def test_identical_prefetch_objects_are_deduplicated(self):
        self.create_media()
        prefetch = Prefetch("authors", queryset=MediaAuthor.objects.order_by("name"))
        qs = MediaBase.objects.polymorphic_prefetch({MediaArticle: [prefetch, prefetch]})
        list(qs)  # warm up the content type cache
        with CaptureQueriesContext(connection) as captured:
            list(qs.all())
        author_queries = [q for q in captured if "mediaauthor" in q["sql"].lower()]
        assert len(author_queries) == 1

    def test_conflicting_querysets_are_rejected_before_execution(self):
        self.create_media()
        with self.assertRaises(ValueError):
            MediaBase.objects.polymorphic_prefetch(
                {
                    MediaArticle: [
                        Prefetch("authors", queryset=MediaAuthor.objects.all()),
                        Prefetch("authors", queryset=MediaAuthor.objects.none()),
                    ]
                }
            )

    def test_conflicting_inherited_querysets_are_rejected_before_execution(self):
        self.create_media()
        with self.assertRaises(ValueError):
            MediaBase.objects.polymorphic_prefetch(
                {
                    MediaArticle: [Prefetch("authors", queryset=MediaAuthor.objects.all())],
                    MediaFeaturedArticle: [
                        Prefetch("authors", queryset=MediaAuthor.objects.none())
                    ],
                }
            )

    def test_conflicting_to_attr_is_rejected_before_execution(self):
        self.create_media()
        with self.assertRaises(ValueError):
            MediaBase.objects.polymorphic_prefetch(
                {
                    MediaArticle: [
                        Prefetch("authors", queryset=MediaAuthor.objects.all(), to_attr="a1"),
                        Prefetch("authors", queryset=MediaAuthor.objects.all(), to_attr="a1"),
                    ]
                }
            )

    def test_same_path_with_different_to_attr_both_apply(self):
        self.create_media()
        qs = MediaBase.objects.order_by("pk").polymorphic_prefetch(
            {
                MediaArticle: [
                    Prefetch("authors", to_attr="authors_a"),
                    Prefetch(
                        "authors",
                        queryset=MediaAuthor.objects.order_by("-name"),
                        to_attr="authors_b",
                    ),
                ]
            }
        )
        objects = list(qs)
        article = objects[1]
        with self.assertNumQueries(0):
            assert sorted(a.name for a in article.authors_a) == ["author1", "author2"]
            assert [a.name for a in article.authors_b] == ["author2", "author1"]

    def test_invalid_plan_arguments(self):
        with self.assertRaises(TypeError):
            MediaBase.objects.polymorphic_prefetch(["authors"])
        with self.assertRaises(ValueError):
            MediaBase.objects.polymorphic_prefetch({MediaAuthor: ["articles"]})
        with self.assertRaises(ValueError):
            MediaBase.objects.polymorphic_prefetch({"authors": ["authors"]})
        with self.assertRaises(TypeError):
            MediaBase.objects.polymorphic_prefetch({MediaArticle: [123]})
        with self.assertRaises(ValueError):
            MediaBase.objects.polymorphic_prefetch({MediaArticle: [""]})

    def test_empty_result_causes_no_prefetch_queries(self):
        self.create_media()
        qs = MediaBase.objects.filter(title="does-not-exist").polymorphic_prefetch(
            {MediaArticle: ["authors"], MediaVideo: ["tracks"]}
        )
        list(qs)  # warm up the content type cache
        with self.assertNumQueries(1):
            assert list(qs.all()) == []

    def test_absent_subclass_causes_no_queries(self):
        self.create_media()
        MediaVideo.objects.all().delete()
        qs = MediaBase.objects.order_by("pk").polymorphic_prefetch(
            {MediaArticle: ["authors"], MediaVideo: ["tracks"]}
        )
        list(qs)  # warm up the content type cache
        # 1 base query, 1 real instance query, 1 authors prefetch; nothing for videos
        with self.assertNumQueries(3):
            objects = list(qs.all())
        assert len(objects) == 2

    def test_iterator_chunks(self):
        # interleave articles and videos so that every chunk contains both models
        author = MediaAuthor.objects.create(name="author1")
        article1 = MediaArticle.objects.create(title="article1")
        article1.authors.add(author)
        video1 = MediaVideo.objects.create(title="video1")
        MediaTrack.objects.create(name="track1", video=video1)
        article2 = MediaArticle.objects.create(title="article2")
        article2.authors.add(author)
        video2 = MediaVideo.objects.create(title="video2")
        MediaTrack.objects.create(name="track2", video=video2)

        qs = MediaBase.objects.order_by("pk").polymorphic_prefetch(
            {MediaArticle: ["authors"], MediaVideo: ["tracks"]}
        )
        list(qs.iterator(chunk_size=2))  # warm up the content type cache

        # 1 base query, then per chunk of 2 objects:
        # 2 real instance queries and 2 prefetch queries
        with self.assertNumQueries(9):
            objects = list(qs.all().iterator(chunk_size=2))

        assert [o.pk for o in objects] == [
            article1.pk,
            video1.pk,
            article2.pk,
            video2.pk,
        ]
        with self.assertNumQueries(0):
            assert [a.name for a in objects[0].authors.all()] == ["author1"]
            assert [t.name for t in objects[1].tracks.all()] == ["track1"]
            assert [a.name for a in objects[2].authors.all()] == ["author1"]
            assert [t.name for t in objects[3].tracks.all()] == ["track2"]

    def test_combines_with_only(self):
        self.create_media()
        qs = MediaBase.objects.order_by("pk").only("title").polymorphic_prefetch(
            {MediaArticle: ["authors"], MediaVideo: ["tracks"]}
        )
        objects = list(qs)
        assert objects[0].title == "article1"
        with self.assertNumQueries(0):
            assert [a.name for a in objects[0].authors.all()] == ["author1"]
            assert sorted(t.name for t in objects[2].tracks.all()) == ["track1", "track2"]

    def test_combines_with_defer(self):
        self.create_media()
        qs = MediaBase.objects.order_by("pk").defer("title").polymorphic_prefetch(
            {MediaArticle: ["authors"]}
        )
        objects = list(qs)
        with self.assertNumQueries(0):
            assert [a.name for a in objects[0].authors.all()] == ["author1"]

    def test_combines_with_annotate(self):
        self.create_media()
        qs = (
            MediaBase.objects.order_by("pk")
            .annotate(num_tags=Count("tags"))
            .polymorphic_prefetch({MediaArticle: ["authors"], MediaBase: ["tags"]})
        )
        objects = list(qs)
        assert [o.num_tags for o in objects] == [1, 0, 0, 1]
        with self.assertNumQueries(0):
            assert [a.name for a in objects[0].authors.all()] == ["author1"]
            assert [t.name for t in objects[3].tags.all()] == ["tag2"]

    def test_ordering_is_preserved_and_objects_are_not_duplicated(self):
        self.create_media()
        qs = MediaBase.objects.order_by("-pk").polymorphic_prefetch(
            {MediaArticle: ["authors"], MediaVideo: ["tracks"], MediaBase: ["tags"]}
        )
        objects = list(qs)
        pks = [o.pk for o in objects]
        assert pks == sorted(pks, reverse=True)
        assert len(set(pks)) == 4

    def test_non_polymorphic_evaluation_ignores_plan(self):
        self.create_media()
        qs = MediaBase.objects.polymorphic_prefetch({MediaArticle: ["authors"]})
        objects = list(qs.non_polymorphic())
        assert all(type(o) is MediaBase for o in objects)


class PolymorphicPrefetchMultiDbTests(TestCase):
    databases = ["default", "secondary"]

    def test_prefetch_uses_the_queryset_database(self):
        author = MediaAuthor.objects.db_manager("secondary").create(name="author1")
        article = MediaArticle.objects.db_manager("secondary").create(title="article1")
        article.authors.add(author)

        qs = MediaBase.objects.db_manager("secondary").polymorphic_prefetch(
            {MediaArticle: ["authors"]}
        )
        list(qs)  # warm up the content type cache
        with self.assertNumQueries(3, using="secondary"):
            objects = list(qs.all())
        with self.assertNumQueries(0, using="secondary"):
            assert [a.name for a in objects[0].authors.all()] == ["author1"]
        assert not MediaArticle.objects.using("default").exists()
