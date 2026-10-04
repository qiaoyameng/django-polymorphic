"""
Tests for PolymorphicModel.retype().
"""

import pytest
from django.contrib.contenttypes.models import ContentType
from django.test import TestCase

from polymorphic.retype import (
    RetypeBlocked,
    RetypeError,
    RetypeNotAllowed,
    RetypePlan,
    RetypeValidationError,
)
from polymorphic.tests.models import (
    DocumentBase,
    DocumentImage,
    DocumentNote,
    DocumentText,
    DocumentTextProxy,
    DocumentTextV2,
    Model2A,
    NullableDocumentRef,
    ProtectedDocumentRef,
)

ALLOWED = {DocumentBase, DocumentText, DocumentTextV2, DocumentImage, DocumentTextProxy}


def ctype_of(model):
    return ContentType.objects.get_for_model(model, for_concrete_model=False)


class SiblingRetypeTests(TestCase):
    """Retyping between two sibling types (DocumentText <-> DocumentImage)."""

    def test_retype_to_sibling(self):
        doc = DocumentText.objects.create(title="draft", body="some text")
        pk = doc.pk

        new = doc.retype(
            DocumentImage, allowed=ALLOWED, fields={"image_url": "http://x/img.png"}
        )

        # base primary key and common fields are preserved
        assert new.pk == pk
        assert isinstance(new, DocumentImage)
        assert new.title == "draft"
        assert new.image_url == "http://x/img.png"

        # old child row is gone, new child row exists, base row untouched
        assert not DocumentText.objects.non_polymorphic().filter(pk=pk).exists()
        assert DocumentImage.objects.non_polymorphic().filter(pk=pk).exists()
        assert DocumentBase.objects.non_polymorphic().filter(pk=pk).exists()

        # content type was switched
        assert (
            DocumentBase.objects.non_polymorphic().get(pk=pk).polymorphic_ctype_id
            == ctype_of(DocumentImage).pk
        )
        # polymorphic fetch downcasts to the new type
        assert isinstance(DocumentBase.objects.get(pk=pk), DocumentImage)

    def test_retype_back_and_forth(self):
        doc = DocumentText.objects.create(title="draft", body="text")
        pk = doc.pk
        img = doc.retype(DocumentImage, allowed=ALLOWED, fields={"image_url": "u"})
        back = img.retype(DocumentText, allowed=ALLOWED, fields={"body": "new text"})
        assert isinstance(back, DocumentText)
        assert back.pk == pk
        assert back.body == "new text"
        assert not DocumentImage.objects.non_polymorphic().filter(pk=pk).exists()

    def test_base_row_never_deleted(self):
        """Relations to the kept base table survive a retype."""
        doc = DocumentText.objects.create(title="draft", body="text")
        note = DocumentNote.objects.create(document=doc, text="keep me")
        doc.retype(DocumentImage, allowed=ALLOWED, fields={"image_url": "u"})
        assert DocumentNote.objects.filter(pk=note.pk).exists()
        assert DocumentBase.objects.non_polymorphic().filter(pk=doc.pk).exists()

    def test_ancestor_field_update(self):
        doc = DocumentText.objects.create(title="draft", body="text")
        new = doc.retype(
            DocumentImage,
            allowed=ALLOWED,
            fields={"image_url": "u", "title": "renamed"},
        )
        assert new.title == "renamed"
        assert DocumentBase.objects.non_polymorphic().get(pk=doc.pk).title == "renamed"


class AncestorDowngradeTests(TestCase):
    """Retyping from a deeper type down to an ancestor type."""

    def test_downgrade_one_level(self):
        doc = DocumentTextV2.objects.create(title="t", body="b", extra="e")
        pk = doc.pk
        new = doc.retype(DocumentText, allowed=ALLOWED)
        assert isinstance(new, DocumentText)
        assert not isinstance(new, DocumentTextV2)
        assert new.pk == pk
        assert new.body == "b"
        assert not DocumentTextV2.objects.non_polymorphic().filter(pk=pk).exists()
        assert DocumentText.objects.non_polymorphic().filter(pk=pk).exists()

    def test_downgrade_to_root(self):
        doc = DocumentTextV2.objects.create(title="t", body="b", extra="e")
        pk = doc.pk
        new = doc.retype(DocumentBase, allowed=ALLOWED)
        assert type(new) is DocumentBase
        assert new.pk == pk
        assert new.title == "t"
        assert not DocumentTextV2.objects.non_polymorphic().filter(pk=pk).exists()
        assert not DocumentText.objects.non_polymorphic().filter(pk=pk).exists()
        assert DocumentBase.objects.non_polymorphic().filter(pk=pk).exists()
        assert (
            DocumentBase.objects.non_polymorphic().get(pk=pk).polymorphic_ctype_id
            == ctype_of(DocumentBase).pk
        )

    def test_upgrade_deeper_chain(self):
        doc = DocumentText.objects.create(title="t", body="b")
        new = doc.retype(DocumentTextV2, allowed=ALLOWED, fields={"extra": "e"})
        assert isinstance(new, DocumentTextV2)
        assert new.extra == "e"
        assert DocumentText.objects.non_polymorphic().filter(pk=doc.pk).exists()


class ProtectedRelationTests(TestCase):
    """Relations pointing at removed child tables are reported and can block."""

    def test_protected_fk_blocks_retype(self):
        doc = DocumentText.objects.create(title="t", body="b")
        ProtectedDocumentRef.objects.create(text_doc=doc, label="ref")
        original_ctype_id = doc.polymorphic_ctype_id

        with pytest.raises(RetypeBlocked) as excinfo:
            doc.retype(DocumentImage, allowed=ALLOWED, fields={"image_url": "u"})

        assert len(excinfo.value.issues) == 1
        issue = excinfo.value.issues[0]
        assert issue.related_model is ProtectedDocumentRef
        assert issue.field_name == "text_doc"
        assert issue.blocking

        # nothing changed: old child row, content type and in-memory type intact
        assert DocumentText.objects.non_polymorphic().filter(pk=doc.pk).exists()
        assert not DocumentImage.objects.non_polymorphic().filter(pk=doc.pk).exists()
        assert (
            DocumentBase.objects.non_polymorphic().get(pk=doc.pk).polymorphic_ctype_id
            == ctype_of(DocumentText).pk
        )
        assert doc.polymorphic_ctype_id == original_ctype_id

    def test_protected_fk_reported_in_dry_run(self):
        doc = DocumentText.objects.create(title="t", body="b")
        ProtectedDocumentRef.objects.create(text_doc=doc, label="ref")
        plan = doc.retype(
            DocumentImage, allowed=ALLOWED, fields={"image_url": "u"}, dry_run=True
        )
        assert len(plan.blockers) == 1
        assert plan.blockers[0].related_model is ProtectedDocumentRef
        # dry run changed nothing
        assert DocumentText.objects.non_polymorphic().filter(pk=doc.pk).exists()

    def test_unreferenced_protected_fk_does_not_block(self):
        doc = DocumentText.objects.create(title="t", body="b")
        other = DocumentText.objects.create(title="t2", body="b2")
        ProtectedDocumentRef.objects.create(text_doc=other, label="ref")
        new = doc.retype(DocumentImage, allowed=ALLOWED, fields={"image_url": "u"})
        assert isinstance(new, DocumentImage)

    def test_set_null_relation_is_warning_only(self):
        doc = DocumentText.objects.create(title="t", body="b")
        ref = NullableDocumentRef.objects.create(text_doc=doc, label="ref")
        plan = doc.retype(
            DocumentImage, allowed=ALLOWED, fields={"image_url": "u"}, dry_run=True
        )
        assert not plan.blockers
        assert len(plan.warnings) == 1
        assert plan.warnings[0].related_model is NullableDocumentRef
        assert len(plan.fixups) == 1
        new = plan.execute()
        assert isinstance(new, DocumentImage)
        # the declared SET_NULL behavior was applied, no dangling reference
        ref.refresh_from_db()
        assert ref.text_doc_id is None


class DryRunTests(TestCase):
    def test_dry_run_returns_plan_without_changes(self):
        doc = DocumentTextV2.objects.create(title="t", body="b", extra="e")
        plan = doc.retype(
            DocumentImage, allowed=ALLOWED, fields={"image_url": "u"}, dry_run=True
        )
        assert isinstance(plan, RetypePlan)
        assert plan.source_model is DocumentTextV2
        assert plan.target_model is DocumentImage
        assert plan.common_ancestor is DocumentBase
        assert plan.root_model is DocumentBase
        assert plan.create_models == [DocumentImage]
        assert plan.delete_models == [DocumentTextV2, DocumentText]
        assert plan.changes_tables
        assert not plan.is_proxy_switch
        assert plan.new_row_values == {DocumentImage: {"image_url": "u"}}
        # database untouched
        assert DocumentTextV2.objects.non_polymorphic().filter(pk=doc.pk).exists()
        assert not DocumentImage.objects.non_polymorphic().filter(pk=doc.pk).exists()

    def test_plan_execute(self):
        doc = DocumentText.objects.create(title="t", body="b")
        plan = doc.retype(
            DocumentImage, allowed=ALLOWED, fields={"image_url": "u"}, dry_run=True
        )
        new = plan.execute()
        assert isinstance(new, DocumentImage)
        assert new.pk == doc.pk


class ProxySwitchTests(TestCase):
    """Proxy switches change only the content type, never the tables."""

    def test_retype_to_proxy(self):
        doc = DocumentText.objects.create(title="t", body="b")
        plan = doc.retype(DocumentTextProxy, allowed=ALLOWED, dry_run=True)
        assert plan.is_proxy_switch
        assert not plan.changes_tables
        assert plan.create_models == []
        assert plan.delete_models == []

        new = plan.execute()
        assert isinstance(new, DocumentTextProxy)
        assert (
            DocumentBase.objects.non_polymorphic().get(pk=doc.pk).polymorphic_ctype_id
            == ctype_of(DocumentTextProxy).pk
        )
        # table rows unchanged
        assert DocumentText.objects.non_polymorphic().filter(pk=doc.pk).exists()

    def test_retype_from_proxy_to_concrete(self):
        doc = DocumentText.objects.create(title="t", body="b")
        proxied = doc.retype(DocumentTextProxy, allowed=ALLOWED)
        assert isinstance(proxied, DocumentTextProxy)
        back = proxied.retype(DocumentText, allowed=ALLOWED)
        assert type(back) is DocumentText
        assert (
            DocumentBase.objects.non_polymorphic().get(pk=doc.pk).polymorphic_ctype_id
            == ctype_of(DocumentText).pk
        )

    def test_proxy_to_sibling_is_real_switch(self):
        doc = DocumentText.objects.create(title="t", body="b")
        proxied = doc.retype(DocumentTextProxy, allowed=ALLOWED)
        plan = proxied.retype(
            DocumentImage, allowed=ALLOWED, fields={"image_url": "u"}, dry_run=True
        )
        assert not plan.is_proxy_switch
        assert plan.create_models == [DocumentImage]
        assert plan.delete_models == [DocumentText]


class CacheInvalidationTests(TestCase):
    def test_old_instance_cache_invalidated(self):
        doc = DocumentText.objects.create(title="t", body="b")
        # populate the field cache of the old instance
        doc.polymorphic_ctype
        assert doc._state.fields_cache

        new = doc.retype(DocumentImage, allowed=ALLOWED, fields={"image_url": "u"})

        # cached related state of the old python instance is cleared and its
        # content type reflects the new concrete type
        assert not doc._state.fields_cache
        assert doc.polymorphic_ctype_id == ctype_of(DocumentImage).pk
        assert isinstance(new, DocumentImage)

    def test_fresh_fetch_has_new_type(self):
        doc = DocumentText.objects.create(title="t", body="b")
        doc.retype(DocumentImage, allowed=ALLOWED, fields={"image_url": "u"})
        fresh = DocumentBase.objects.get(pk=doc.pk)
        assert isinstance(fresh, DocumentImage)
        assert fresh.get_real_instance_class() is DocumentImage


class ValidationTests(TestCase):
    def test_target_must_be_allowed(self):
        doc = DocumentText.objects.create(title="t", body="b")
        with pytest.raises(RetypeNotAllowed):
            doc.retype(DocumentImage, allowed={DocumentText})

    def test_allowed_is_required(self):
        doc = DocumentText.objects.create(title="t", body="b")
        with pytest.raises(RetypeError):
            doc.retype(DocumentImage, allowed=None)

    def test_missing_required_field(self):
        doc = DocumentText.objects.create(title="t", body="b")
        with pytest.raises(RetypeValidationError) as excinfo:
            doc.retype(DocumentImage, allowed=ALLOWED)
        assert "image_url" in str(excinfo.value)

    def test_unknown_field(self):
        doc = DocumentText.objects.create(title="t", body="b")
        with pytest.raises(RetypeValidationError):
            doc.retype(DocumentImage, allowed=ALLOWED, fields={"nope": 1})

    def test_unsaved_instance(self):
        with pytest.raises(RetypeError):
            DocumentText(title="t", body="b").retype(DocumentImage, allowed=ALLOWED)

    def test_cross_hierarchy_target_rejected(self):
        doc = DocumentText.objects.create(title="t", body="b")
        with pytest.raises(RetypeError):
            doc.retype(Model2A, allowed={Model2A})

    def test_non_model_target_rejected(self):
        doc = DocumentText.objects.create(title="t", body="b")
        with pytest.raises(RetypeError):
            doc.retype("DocumentImage", allowed=ALLOWED)
