import pytest
from django.contrib.contenttypes.models import ContentType
from django.test import TestCase

from polymorphic.retype import (
    ACTION_BLOCK,
    ACTION_CASCADE,
    RetypeError,
    RetypeNotAllowedError,
    RetypeProtectedError,
    RetypeValidationError,
)
from polymorphic.tests.models import (
    Base,
    Model2A,
    Model2B,
    Model2C,
    Model2D,
    ModelX,
    ModelXAttachment,
    ModelXGuardian,
    ModelY,
    NonProxyChild,
    ProxyBase,
    ProxyChild,
)


def ctype_of(model):
    return ContentType.objects.get_for_model(model, for_concrete_model=False)


class RetypeSiblingTests(TestCase):
    def test_retype_sibling(self):
        obj = ModelX.objects.create(field_b="base", field_x="x")
        pk = obj.pk
        plan = obj.retype(ModelY, field_values={"field_y": "y"})
        assert plan.executed
        assert plan.common_ancestor is Base
        assert plan.removed_models == [ModelX]
        assert plan.created_models == [ModelY]

        # base row and shared fields are preserved
        assert Base.objects.non_polymorphic().count() == 1
        base_row = Base.objects.non_polymorphic().get(pk=pk)
        assert base_row.field_b == "base"
        assert base_row.polymorphic_ctype_id == ctype_of(ModelY).pk

        # old child row removed, new child row created
        assert ModelX.objects.non_polymorphic().count() == 0
        new_row = ModelY.objects.non_polymorphic().get(pk=pk)
        assert new_row.field_y == "y"

        # polymorphic fetch returns the new type
        real = Base.objects.get(pk=pk)
        assert type(real) is ModelY
        assert type(plan.get_instance()) is ModelY

    def test_retype_sibling_back_and_forth(self):
        obj = ModelX.objects.create(field_b="base", field_x="x")
        obj.retype(ModelY, field_values={"field_y": "y"})
        y_obj = Base.objects.get(pk=obj.pk)
        y_obj.retype(ModelX, field_values={"field_x": "x2"})
        real = Base.objects.get(pk=obj.pk)
        assert type(real) is ModelX
        assert real.field_x == "x2"
        assert real.field_b == "base"


class RetypeChainTests(TestCase):
    def test_retype_downgrade_to_ancestor(self):
        obj = Model2C.objects.create(field1="1", field2="2", field3="3")
        pk = obj.pk
        plan = obj.retype(Model2A)
        assert plan.executed
        assert plan.common_ancestor is Model2A
        assert plan.removed_models == [Model2C, Model2B]
        assert plan.created_models == []

        # only the base row remains, the base class itself is never deleted
        assert Model2A.objects.non_polymorphic().count() == 1
        assert Model2B.objects.non_polymorphic().count() == 0
        assert Model2C.objects.non_polymorphic().count() == 0
        row = Model2A.objects.non_polymorphic().get(pk=pk)
        assert row.field1 == "1"
        assert row.polymorphic_ctype_id == ctype_of(Model2A).pk
        assert type(Model2A.objects.get(pk=pk)) is Model2A

    def test_retype_upgrade_to_descendant(self):
        obj = Model2A.objects.create(field1="1")
        pk = obj.pk
        plan = obj.retype(Model2C, field_values={"field2": "2"})
        assert plan.executed
        assert plan.removed_models == []
        assert plan.created_models == [Model2B, Model2C]

        real = Model2A.objects.get(pk=pk)
        assert type(real) is Model2C
        assert real.field1 == "1"
        assert real.field2 == "2"
        assert real.field3 == ""  # field default applied

    def test_retype_between_branches_keeps_shared_ancestor_rows(self):
        obj = Model2C.objects.create(field1="1", field2="2", field3="3")
        pk = obj.pk
        # Model2C -> Model2D shares the whole chain except the leaf
        plan = obj.retype(Model2D, field_values={"field4": "4"})
        assert plan.common_ancestor is Model2C
        assert plan.removed_models == []
        assert plan.created_models == [Model2D]
        real = Model2A.objects.get(pk=pk)
        assert type(real) is Model2D
        assert real.field3 == "3"
        assert real.field4 == "4"


class RetypeValidationTests(TestCase):
    def test_missing_required_field_restores_type(self):
        obj = ModelX.objects.create(field_b="b", field_x="x")
        old_ctype = obj.polymorphic_ctype_id
        with pytest.raises(RetypeValidationError) as excinfo:
            obj.retype(ModelY)
        assert "tests.ModelY.field_y" in str(excinfo.value)
        # original concrete type is restored, nothing was written
        assert obj.polymorphic_ctype_id == old_ctype
        assert ModelX.objects.non_polymorphic().count() == 1
        assert ModelY.objects.non_polymorphic().count() == 0
        assert type(Base.objects.get(pk=obj.pk)) is ModelX

    def test_unknown_field_rejected(self):
        obj = ModelX.objects.create(field_b="b", field_x="x")
        with pytest.raises(RetypeValidationError) as excinfo:
            obj.retype(ModelY, field_values={"field_y": "y", "bogus": 1})
        assert "bogus" in str(excinfo.value)
        assert type(Base.objects.get(pk=obj.pk)) is ModelX

    def test_target_must_be_in_allowed_set(self):
        obj = ModelX.objects.create(field_b="b", field_x="x")
        with pytest.raises(RetypeNotAllowedError):
            obj.retype(ModelY, allowed_types=[ModelX], field_values={"field_y": "y"})
        assert type(Base.objects.get(pk=obj.pk)) is ModelX

        plan = obj.retype(
            ModelY, allowed_types=[ModelX, ModelY], field_values={"field_y": "y"}
        )
        assert plan.executed

    def test_class_level_allowed_set(self):
        obj = ModelX.objects.create(field_b="b", field_x="x")
        try:
            Base.polymorphic_retype_allowed_types = [ModelX]
            with pytest.raises(RetypeNotAllowedError):
                obj.retype(ModelY, field_values={"field_y": "y"})
        finally:
            del Base.polymorphic_retype_allowed_types

    def test_retype_same_type_raises(self):
        obj = ModelX.objects.create(field_b="b", field_x="x")
        with pytest.raises(RetypeError):
            obj.retype(ModelX)

    def test_retype_to_other_tree_raises(self):
        obj = ModelX.objects.create(field_b="b", field_x="x")
        with pytest.raises(RetypeError):
            obj.retype(Model2A)

    def test_retype_unsaved_instance_raises(self):
        with pytest.raises(RetypeError):
            ModelX(field_b="b", field_x="x").retype(ModelY)

    def test_retype_to_non_polymorphic_raises(self):
        obj = ModelX.objects.create(field_b="b", field_x="x")
        with pytest.raises(RetypeError):
            obj.retype(ModelXGuardian)


class RetypeProtectedTests(TestCase):
    def test_protected_relation_blocks_retype(self):
        obj = ModelX.objects.create(field_b="b", field_x="x")
        ModelXGuardian.objects.create(x=obj)
        old_ctype = obj.polymorphic_ctype_id
        with pytest.raises(RetypeProtectedError) as excinfo:
            obj.retype(ModelY, field_values={"field_y": "y"})
        assert excinfo.value.effects[0].model is ModelXGuardian
        assert excinfo.value.effects[0].action == ACTION_BLOCK
        # nothing changed: original concrete type preserved
        assert obj.polymorphic_ctype_id == old_ctype
        assert ModelX.objects.non_polymorphic().count() == 1
        assert ModelXGuardian.objects.count() == 1
        assert type(Base.objects.get(pk=obj.pk)) is ModelX

    def test_dry_run_reports_blocking_relation(self):
        obj = ModelX.objects.create(field_b="b", field_x="x")
        ModelXGuardian.objects.create(x=obj)
        plan = obj.retype(ModelY, field_values={"field_y": "y"}, dry_run=True)
        assert plan.blocked
        assert not plan.executed
        effect = plan.blocking_effects[0]
        assert effect.on_delete == "PROTECT"
        assert effect.count == 1
        assert "PROTECT" in plan.describe()

    def test_cascade_relation_is_removed_and_reported(self):
        obj = ModelX.objects.create(field_b="b", field_x="x")
        ModelXAttachment.objects.create(x=obj)
        plan = obj.retype(ModelY, field_values={"field_y": "y"})
        assert plan.executed
        assert ModelXAttachment.objects.count() == 0
        cascades = [e for e in plan.relation_effects if e.action == ACTION_CASCADE]
        assert len(cascades) == 1
        assert cascades[0].model is ModelXAttachment


class RetypeDryRunTests(TestCase):
    def test_dry_run_makes_no_changes(self):
        obj = ModelX.objects.create(field_b="b", field_x="x")
        plan = obj.retype(ModelY, field_values={"field_y": "y"}, dry_run=True)
        assert not plan.executed
        description = plan.describe()
        assert "tests.ModelX" in description
        assert "tests.ModelY" in description
        assert "dry-run" in description
        assert ModelX.objects.non_polymorphic().count() == 1
        assert ModelY.objects.non_polymorphic().count() == 0
        assert type(Base.objects.get(pk=obj.pk)) is ModelX
        with pytest.raises(RetypeError):
            plan.get_instance()


class RetypeProxyTests(TestCase):
    def test_retype_to_proxy_only_switches_ctype(self):
        obj = ProxyBase.objects.create(some_data="d")
        plan = obj.retype(ProxyChild)
        assert plan.proxy_only
        assert plan.removed_models == []
        assert plan.created_models == []
        assert ProxyBase.objects.non_polymorphic().count() == 1
        real = ProxyBase.objects.get(pk=obj.pk)
        assert type(real) is ProxyChild
        assert real.some_data == "d"

    def test_retype_from_proxy_back_to_base(self):
        obj = ProxyChild.objects.create(some_data="d")
        plan = obj.retype(ProxyBase)
        assert plan.proxy_only
        assert type(ProxyBase.objects.get(pk=obj.pk)) is ProxyBase

    def test_retype_base_to_real_child_is_not_proxy_only(self):
        obj = ProxyBase.objects.create(some_data="d")
        plan = obj.retype(NonProxyChild, field_values={"name": "n"})
        assert not plan.proxy_only
        assert plan.created_models == [NonProxyChild]
        real = ProxyBase.objects.get(pk=obj.pk)
        assert type(real) is NonProxyChild
        assert real.name == "n"


class RetypeInstanceStateTests(TestCase):
    def test_stale_instance_cache_is_invalidated(self):
        obj = ModelX.objects.create(field_b="b", field_x="x")
        # populate the relation cache on the python instance
        assert obj.polymorphic_ctype.model_class() is ModelX
        assert obj._state.fields_cache
        obj.retype(ModelY, field_values={"field_y": "y"})
        # cached state explicitly invalidated, ctype updated in memory
        assert not obj._state.fields_cache
        assert obj.polymorphic_ctype_id == ctype_of(ModelY).pk
        # a fresh fetch through the stale instance resolves the new type
        assert obj.polymorphic_ctype.model_class() is ModelY
