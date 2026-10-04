"""
Tests for the explicit polymorphic projection
``PolymorphicQuerySet.polymorphic_values``.
"""

from __future__ import annotations

import pytest
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import FieldError
from django.db import connection
from django.db.models import F, Value
from django.db.models.functions import Upper
from django.test.utils import CaptureQueriesContext

from polymorphic.polyvalues import NOT_PRESENT, PolymorphicValuesRow
from polymorphic.tests.models import (
    PolyValuesAlpha,
    PolyValuesAlphaProxy,
    PolyValuesBeta,
    PolyValuesGamma,
    PolyValuesItem,
)


def _ctype(model):
    return ContentType.objects.get_for_model(model, for_concrete_model=False).pk


@pytest.fixture
def items(transactional_db):
    alpha = PolyValuesAlpha.objects.create(
        name="alpha", shared_number=None, alpha_text=None, payload="a-payload"
    )
    base = PolyValuesItem.objects.create(name="base", shared_number=10)
    beta = PolyValuesBeta.objects.create(
        name="beta", shared_number=20, beta_text="", payload=42
    )
    beta2 = PolyValuesBeta.objects.create(
        name="beta2", shared_number=20, beta_text="b2", payload=77
    )
    gamma = PolyValuesGamma.objects.create(
        name="gamma", shared_number=20, gamma_text="g", payload=1.5
    )
    proxy = PolyValuesAlphaProxy.objects.create(
        name="proxy-alpha", shared_number=5, alpha_text="p", payload="proxy-payload"
    )
    return {
        "base": base,
        "alpha": alpha,
        "beta": beta,
        "beta2": beta2,
        "gamma": gamma,
        "proxy": proxy,
    }


def project(**kwargs):
    return PolyValuesItem.objects.polymorphic_values(
        "id",
        "name",
        "PolyValuesAlpha___alpha_text",
        beta_text="PolyValuesBeta___beta_text",
        gamma_text="PolyValuesGamma___gamma_text",
        **kwargs,
    ).order_by("id")


def test_basic_projection_presence_and_null_distinction(items):
    rows = list(project())
    assert len(rows) == 6
    assert all(isinstance(row, PolymorphicValuesRow) for row in rows)
    by_name = {row["name"]: row for row in rows}

    base_row = by_name["base"]
    assert base_row["PolyValuesAlpha___alpha_text"] is NOT_PRESENT
    assert base_row["beta_text"] is NOT_PRESENT
    assert base_row["gamma_text"] is NOT_PRESENT
    assert base_row.is_present("beta_text") is False
    assert "beta_text" not in base_row.present_fields()
    assert bool(base_row["beta_text"]) is False

    alpha_row = by_name["alpha"]
    # exists but NULL => None, distinguishable from NOT_PRESENT
    assert alpha_row["PolyValuesAlpha___alpha_text"] is None
    assert alpha_row["PolyValuesAlpha___alpha_text"] is not NOT_PRESENT
    assert alpha_row.is_present("PolyValuesAlpha___alpha_text") is True
    assert alpha_row["beta_text"] is NOT_PRESENT

    beta_row = by_name["beta"]
    assert beta_row["beta_text"] == ""  # empty string preserved, not NULL
    assert beta_row.is_present("beta_text") is True
    assert beta_row["PolyValuesAlpha___alpha_text"] is NOT_PRESENT

    gamma_row = by_name["gamma"]
    assert gamma_row["gamma_text"] == "g"

    for row in rows:
        for key in (
            "beta_text__present",
            "gamma_text__present",
            "PolyValuesAlpha___alpha_text__present",
        ):
            assert key in row


def test_stable_model_identifier_and_proxy_identity(items):
    rows = {
        row["name"]: row
        for row in PolyValuesItem.objects.polymorphic_values(
            "name", "PolyValuesAlpha___payload"
        )
    }

    assert rows["base"]["polymorphic_ctype_id"] == _ctype(PolyValuesItem)
    assert rows["alpha"]["polymorphic_ctype_id"] == _ctype(PolyValuesAlpha)
    assert rows["beta"]["polymorphic_ctype_id"] == _ctype(PolyValuesBeta)
    assert rows["gamma"]["polymorphic_ctype_id"] == _ctype(PolyValuesGamma)
    assert rows["proxy-alpha"]["polymorphic_ctype_id"] == _ctype(PolyValuesAlphaProxy)
    # concrete column comes from the alpha table
    assert rows["proxy-alpha"]["PolyValuesAlpha___payload"] == "proxy-payload"


def test_single_query_with_database_pagination(items):
    qs = project()
    with CaptureQueriesContext(connection) as ctx:
        page = list(qs[1:4])
    assert len(page) == 3
    # database-side slicing follows the same id ordering as a full fetch
    full_names = [row["name"] for row in list(qs.order_by("id"))]
    assert [row["name"] for row in page] == full_names[1:4]
    assert len(ctx.captured_queries) == 1
    sql = ctx.captured_queries[0]["sql"]
    assert "LIMIT" in sql and "OFFSET" in sql

    with CaptureQueriesContext(connection) as ctx:
        assert qs.count() == 6
    assert len(ctx.captured_queries) == 1


def test_filter_order_and_duplicate_common_values(items):
    qs = (
        PolyValuesItem.objects.polymorphic_values(
            "name",
            "shared_number",
            beta_payload="PolyValuesBeta___payload",
        )
        .filter(PolyValuesBeta___payload__gte=10)
        .order_by("-PolyValuesBeta___payload", "name")
    )
    rows = list(qs)
    assert [row["name"] for row in rows] == ["beta2", "beta"]
    assert rows[0]["beta_payload"] == 77
    assert all(row["polymorphic_ctype_id"] == _ctype(PolyValuesBeta) for row in rows)

    qs2 = (
        PolyValuesItem.objects.polymorphic_values("name", "shared_number")
        .filter(shared_number=20)
        .order_by("name")
    )
    assert [row["name"] for row in qs2] == ["beta", "beta2", "gamma"]


def test_aliases_and_basic_expressions(items):
    qs = PolyValuesItem.objects.polymorphic_values(
        "name",
        upper_name=Upper("name"),
        const=Value("x"),
        alpha_alias="PolyValuesAlpha___alpha_text",
    ).order_by("name")
    by_name = {row["name"]: row for row in qs}
    assert by_name["alpha"]["upper_name"] == "ALPHA"
    assert by_name["alpha"]["const"] == "x"
    assert by_name["alpha"]["alpha_alias"] is None
    assert by_name["beta"]["alpha_alias"] is NOT_PRESENT

    qs2 = PolyValuesItem.objects.polymorphic_values(
        "name",
        p=F("PolyValuesBeta___payload"),
    ).order_by("id")
    by_name2 = {row["name"]: row for row in qs2}
    assert by_name2["beta"]["p"] == 42
    assert by_name2["alpha"]["p"] is NOT_PRESENT


def test_coalesced_compatible_sibling_union(items):
    # alpha_text and beta_text are both CharField -> one coalesced alias
    qs = (
        PolyValuesItem.objects.polymorphic_values(
            "name",
            ("text", "PolyValuesAlpha___alpha_text"),
            ("text", "PolyValuesBeta___beta_text"),
        )
        .order_by("name")
    )
    with CaptureQueriesContext(connection) as ctx:
        rows = list(qs)
    assert len(ctx.captured_queries) == 1
    assert "COALESCE" in ctx.captured_queries[0]["sql"]
    assert "CASE" in ctx.captured_queries[0]["sql"]

    by_name = {row["name"]: row for row in rows}
    assert by_name["alpha"]["text"] is None           # present, NULL
    assert by_name["beta"]["text"] == ""              # present, empty
    assert by_name["beta2"]["text"] == "b2"
    assert by_name["gamma"]["text"] is NOT_PRESENT    # neither alpha nor beta
    assert by_name["base"]["text"] is NOT_PRESENT
    # the alias-level presence bit is true for either contributor
    assert by_name["alpha"]["text__present"] is True
    assert by_name["beta"]["text__present"] is True
    assert by_name["gamma"]["text__present"] is False


def test_invalid_paths_raise_before_query(items):
    with pytest.raises(FieldError):
        list(PolyValuesItem.objects.polymorphic_values("does_not_exist"))

    with pytest.raises(FieldError):
        list(PolyValuesItem.objects.polymorphic_values("Nope___field1"))

    with pytest.raises(FieldError):
        list(PolyValuesItem.objects.polymorphic_values("PolyValuesAlpha___nope"))

    with pytest.raises(FieldError):
        list(PolyValuesItem.objects.polymorphic_values("PolyValuesAlpha___"))

    # raises at construction time (before evaluation)
    with pytest.raises(FieldError):
        PolyValuesItem.objects.polymorphic_values("PolyValuesAlpha___nope").order_by("id")


def test_incompatible_sibling_types_raise(items):
    with pytest.raises(FieldError, match="incompatible"):
        PolyValuesItem.objects.polymorphic_values(
            "name",
            ("payload", "PolyValuesAlpha___payload"),  # CharField
            ("payload", "PolyValuesBeta___payload"),   # IntegerField
        )


def test_existing_values_behavior_unchanged(items):
    names = list(
        PolyValuesItem.objects.values("name")
        .order_by("name")
        .values_list("name", flat=True)
    )
    assert names == ["alpha", "base", "beta", "beta2", "gamma", "proxy-alpha"]

    row = PolyValuesItem.objects.values("name", "shared_number").get(name="alpha")
    assert set(row.keys()) == {"name", "shared_number"}
    assert isinstance(row, dict) and not isinstance(row, PolymorphicValuesRow)


def test_inherited_field_conflict_raises(items):
    # 'name' lives on the base table and is requested both directly and via a
    # subclass -> ambiguous output name
    with pytest.raises(FieldError, match="ambiguous"):
        PolyValuesItem.objects.polymorphic_values(
            "name",
            ("name", "PolyValuesAlpha___name"),
        )


def test_proxy_of_base_presence(transactional_db):
    from polymorphic.tests.models import Duck, RedheadDuck, RubberDuck

    Duck.objects.create(name="donald")
    RedheadDuck.objects.create(name="red")
    RubberDuck.objects.create(name="rubber")

    rows = {
        row["name"]: row
        for row in Duck.objects.polymorphic_values(
            "name",
            red="RedheadDuck___name",
        )
    }
    assert rows["red"]["red"] == "red"
    assert rows["red"]["red__present"] is True
    assert rows["donald"]["red"] is NOT_PRESENT
    assert rows["donald"]["red__present"] is False
    assert rows["rubber"]["red"] is NOT_PRESENT
    assert rows["rubber"]["red__present"] is False

    # stable ctype identity for proxies of the base table
    assert rows["red"]["polymorphic_ctype_id"] == _ctype(RedheadDuck)
    assert rows["rubber"]["polymorphic_ctype_id"] == _ctype(RubberDuck)


def test_proxy_of_concrete_child_presence(transactional_db):
    from polymorphic.tests.models import (
        ProxiedBase,
        ProxyModelA,
        ProxyModelB,
    )

    ProxiedBase.objects.create(name="base")
    ProxyModelA.objects.create(name="a", field1="f1")
    ProxyModelB.objects.create(name="b", field2="f2")

    rows = {
        row["name"]: row
        for row in ProxiedBase.objects.polymorphic_values(
            "name",
            fa="ProxyModelA___field1",
            fb="ProxyModelB___field2",
        )
    }
    assert rows["a"]["fa"] == "f1"
    assert rows["a"]["fb"] is NOT_PRESENT
    assert rows["b"]["fb"] == "f2"
    assert rows["base"]["fa"] is NOT_PRESENT
    assert rows["a"]["polymorphic_ctype_id"] == _ctype(ProxyModelA)


def test_bare_and_inherited_field_sharing_output_name_raises(items):
    # The positional bare "name" and a positional subclass path whose output
    # alias is also "name" collapse onto one key.
    with pytest.raises(FieldError):
        PolyValuesItem.objects.polymorphic_values(
            "name",
            ("name", "PolyValuesAlpha___name"),
        )
    # ...but an explicitly aliased inherited field alongside the bare common
    # field is perfectly valid.
    rows = list(
        PolyValuesItem.objects.polymorphic_values(
            "name",
            alpha_name="PolyValuesAlpha___name",
        )
    )
    alpha_row = next(r for r in rows if r["name"] == "alpha")
    assert alpha_row["alpha_name"] == "alpha"
