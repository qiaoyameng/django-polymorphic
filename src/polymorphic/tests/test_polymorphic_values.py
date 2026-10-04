"""
Tests for :meth:`PolymorphicQuerySet.polymorphic_values` - a lightweight
polymorphic ``values()`` projection that never constructs subclass instances.
"""

from __future__ import annotations

from django.core.exceptions import FieldError
from django.db.models import F
from django.db.models.functions import Upper
from django.test import TransactionTestCase

from polymorphic.tests.models import (
    Model2A,
    ValuesProjectAlpha,
    ValuesProjectAlphaProxy,
    ValuesProjectBase,
    ValuesProjectBeta,
    ValuesProjectGamma,
)


def create_projection_rows() -> None:
    # A plain base row.
    ValuesProjectBase.objects.create(name="plain")
    # A subclass column that exists (the row is an Alpha) but is SQL NULL.
    ValuesProjectAlpha.objects.create(name="alpha-null", alpha_code=None, shared_note=None)
    ValuesProjectAlpha.objects.create(name="alpha", alpha_code="A1", shared_note="alpha-note")
    ValuesProjectBeta.objects.create(name="beta", beta_amount=7, shared_note="beta-note")
    # A grandchild: beta_amount is inherited and NULL here.
    ValuesProjectGamma.objects.create(
        name="gamma", beta_amount=None, gamma_flag=True, shared_note="gamma-note"
    )
    # A proxy row: no own table, identity must still point at the proxy.
    ValuesProjectAlphaProxy.objects.create(name="proxy", alpha_code="P1", shared_note=None)


class PolymorphicValuesTests(TransactionTestCase):
    def setUp(self) -> None:
        create_projection_rows()

    def test_projection_shape_and_presence(self) -> None:
        rows = list(
            ValuesProjectBase.objects.polymorphic_values(
                "name",
                "ValuesProjectAlpha___alpha_code",
                beta_amount=(ValuesProjectBeta, "beta_amount"),
                gamma_flag="ValuesProjectGamma___gamma_flag",
            ).order_by("pk")
        )

        by_name = {row["name"]: row for row in rows}

        # Every row carries the stable identity columns.
        for row in rows:
            self.assertIn("id", row)
            self.assertIsInstance(row["polymorphic_ctype_id"], int)
            self.assertIn(".", row["polymorphic_label"])

        # A plain base row: no subclass field applies.
        plain = by_name["plain"]
        self.assertEqual(plain["polymorphic_label"], "tests.valuesprojectbase")
        self.assertIsNone(plain["alpha_code"])
        self.assertFalse(plain["alpha_code__present"])
        self.assertFalse(plain["beta_amount__present"])
        self.assertFalse(plain["gamma_flag__present"])

        # Existing but NULL column vs. a field that does not apply to the row.
        alpha_null = by_name["alpha-null"]
        self.assertIsNone(alpha_null["alpha_code"])
        self.assertTrue(alpha_null["alpha_code__present"])
        self.assertFalse(alpha_null["beta_amount__present"])

        # Concrete values are read from the proper tables.
        self.assertEqual(by_name["alpha"]["alpha_code"], "A1")
        self.assertEqual(by_name["beta"]["beta_amount"], 7)
        self.assertTrue(by_name["beta"]["beta_amount__present"])
        self.assertEqual(by_name["gamma"]["gamma_flag"], True)
        # Gamma is also a Beta, so the inherited Beta column reports present.
        self.assertTrue(by_name["gamma"]["beta_amount__present"])
        self.assertIsNone(by_name["gamma"]["beta_amount"])

        # Proxy identity is preserved, concrete field comes from the Alpha table.
        proxy = by_name["proxy"]
        self.assertEqual(proxy["polymorphic_label"], "tests.valuesprojectalphaproxy")
        self.assertEqual(proxy["alpha_code"], "P1")
        self.assertTrue(proxy["alpha_code__present"])

    def test_proxy_ctype_differs_from_concrete(self) -> None:
        rows = list(ValuesProjectBase.objects.polymorphic_values("name").order_by("pk"))
        labels = {row["name"]: row["polymorphic_ctype_id"] for row in rows}
        self.assertNotEqual(labels["alpha"], labels["proxy"])
        self.assertEqual(labels["alpha"], labels["alpha-null"])

    def test_single_query_no_instance_construction(self) -> None:
        # One query for the whole projection: no per-subclass downcast queries.
        with self.assertNumQueries(1):
            rows = list(
                ValuesProjectBase.objects.polymorphic_values(
                    "name",
                    "ValuesProjectAlpha___alpha_code",
                    "ValuesProjectBeta___beta_amount",
                    "ValuesProjectGamma___gamma_flag",
                )
            )
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertNotIn("field1", row)

    def test_duplicate_common_values_are_kept(self) -> None:
        ValuesProjectAlpha.objects.create(name="dup", alpha_code="X")
        ValuesProjectBeta.objects.create(name="dup", beta_amount=1)
        rows = list(
            ValuesProjectBase.objects.polymorphic_values("name")
            .filter(name="dup")
            .order_by("pk")
        )
        # values() does not collapse rows: both duplicate names survive.
        self.assertEqual([row["name"] for row in rows], ["dup", "dup"])
        self.assertNotEqual(rows[0]["id"], rows[1]["id"])

    def test_filtering_before_and_after_projection(self) -> None:
        before = list(
            ValuesProjectBase.objects.filter(
                ValuesProjectGamma___gamma_flag=True
            )
            .polymorphic_values("name", gamma_flag="ValuesProjectGamma___gamma_flag")
            .order_by("pk")
        )
        self.assertEqual([row["name"] for row in before], ["gamma"])

        after = list(
            ValuesProjectBase.objects.polymorphic_values("name")
            .filter(ValuesProjectAlpha___alpha_code__isnull=False)
            .order_by("pk")
        )
        self.assertEqual([row["name"] for row in after], ["alpha", "proxy"])

    def test_ordering_and_database_pagination(self) -> None:
        queryset = (
            ValuesProjectBase.objects.polymorphic_values(
                "name", beta_amount="ValuesProjectBeta___beta_amount"
            )
            .order_by("-beta_amount", "pk")
        )
        sql = str(queryset.query)
        self.assertIn("ORDER BY", sql.upper())

        paged = queryset[2:4]
        paged_sql = str(paged.query).upper()
        self.assertIn("LIMIT", paged_sql)
        self.assertIn("OFFSET", paged_sql)

        names = [row["name"] for row in paged]
        self.assertEqual(len(names), 2)
        # The full ordering: only the Beta row has a non-null amount.
        self.assertEqual(
            [row["name"] for row in queryset][0],
            "beta",
        )

        with self.assertNumQueries(1):
            list(
                ValuesProjectBase.objects.polymorphic_values("name").order_by("pk")[:3]
            )

    def test_aliases_and_base_expressions(self) -> None:
        rows = list(
            ValuesProjectBase.objects.polymorphic_values(
                "ValuesProjectAlpha___alpha_code",
                upper=Upper(F("name")),
                code=Upper(F("ValuesProjectAlpha___alpha_code")),
            ).order_by("pk")
        )
        by_upper = {row["upper"]: row for row in rows}
        self.assertEqual(by_upper["ALPHA"]["code"], "A1")
        self.assertIsNone(by_upper["BETA"]["code"])
        # Existing but NULL code on alpha-null stays None with presence True.
        self.assertIsNone(by_upper["ALPHA-NULL"]["code"])
        self.assertTrue(by_upper["ALPHA-NULL"]["alpha_code__present"])
        self.assertFalse(by_upper["BETA"]["alpha_code__present"])

    def test_multi_spec_alias_coalesces_compatible_siblings(self) -> None:
        rows = list(
            ValuesProjectBase.objects.polymorphic_values(
                "name",
                note=[
                    "ValuesProjectAlpha___shared_note",
                    "ValuesProjectBeta___shared_note",
                ],
            ).order_by("pk")
        )
        by_name = {row["name"]: row for row in rows}
        self.assertEqual(by_name["alpha"]["note"], "alpha-note")
        self.assertTrue(by_name["alpha"]["note__present"])
        self.assertEqual(by_name["beta"]["note"], "beta-note")
        # Gamma inherits Beta's column: present through the Beta table.
        self.assertEqual(by_name["gamma"]["note"], "gamma-note")
        self.assertTrue(by_name["gamma"]["note__present"])
        self.assertIsNone(by_name["plain"]["note"])
        self.assertFalse(by_name["plain"]["note__present"])
        # Existing-but-NULL: Alpha proxy/alpha-null rows.
        self.assertIsNone(by_name["alpha-null"]["note"])
        self.assertTrue(by_name["alpha-null"]["note__present"])

    def test_tuple_and_applabel_specs(self) -> None:
        rows = list(
            ValuesProjectBase.objects.polymorphic_values(
                (ValuesProjectBeta, "beta_amount"),
                "tests__ValuesProjectAlpha___alpha_code",
            ).order_by("pk")
        )
        # The primary key is a valid common projection column.
        pk_rows = list(
            ValuesProjectBase.objects.polymorphic_values("pk").order_by("pk")
        )
        self.assertEqual([row["pk"] for row in pk_rows], [row["id"] for row in rows])

        # A multi-spec alias accepts (Model, field) pairs as well.
        tuple_rows = list(
            ValuesProjectBase.objects.polymorphic_values(
                "name",
                note=[
                    (ValuesProjectAlpha, "shared_note"),
                    (ValuesProjectBeta, "shared_note"),
                ],
            ).order_by("pk")
        )
        notes = {(row["name"], row["note"], row["note__present"]) for row in tuple_rows}
        self.assertIn(("alpha", "alpha-note", True), notes)
        self.assertIn(("beta", "beta-note", True), notes)
        self.assertIn(("plain", None, False), notes)

        by_name = {row["id"]: row for row in rows}
        beta_row = next(row for row in rows if row["polymorphic_label"] == "tests.valuesprojectbeta")
        self.assertEqual(beta_row["beta_amount"], 7)
        self.assertFalse(beta_row["alpha_code__present"])
        alpha_row = next(
            row for row in rows if row["polymorphic_label"] == "tests.valuesprojectalpha"
            and row["alpha_code"] == "A1"
        )
        self.assertEqual(alpha_row["alpha_code"], "A1")
        self.assertTrue(alpha_row["alpha_code__present"])

    def test_inheritance_field_conflict_is_an_alias_conflict(self) -> None:
        # Same implicit alias selected for two sibling tables with columns of the
        # same name but incompatible types must fail eagerly.
        with self.assertRaises(FieldError):
            ValuesProjectBase.objects.polymorphic_values(
                x=[
                    "ValuesProjectAlpha___alpha_code",
                    "ValuesProjectBeta___beta_amount",
                ]
            )

    def test_illegal_paths_fail_eagerly(self) -> None:
        cases = [
            lambda: ValuesProjectBase.objects.polymorphic_values("does_not_exist"),
            lambda: ValuesProjectBase.objects.polymorphic_values("NoSuchModel___x"),
            lambda: ValuesProjectBase.objects.polymorphic_values(
                "ValuesProjectAlpha___no_such_field"
            ),
            lambda: ValuesProjectBase.objects.polymorphic_values(
                "ValuesProjectAlphaProxy___alpha_code"
            ),
            lambda: ValuesProjectBase.objects.polymorphic_values(
                "ValuesProjectGamma___alpha_code"
            ),
            lambda: ValuesProjectBase.objects.polymorphic_values(
                "ValuesProjectBeta___valuesprojectbase_ptr"
            ),
            lambda: ValuesProjectBase.objects.polymorphic_values(
                "ValuesProjectBeta___beta_amount__gte"
            ),
            lambda: ValuesProjectBase.objects.polymorphic_values(
                "ValuesProjectBase___name"
            ),
            lambda: ValuesProjectBase.objects.polymorphic_values("name", "name"),
            lambda: ValuesProjectBase.objects.polymorphic_values("name__present"),
            lambda: ValuesProjectBase.objects.polymorphic_values(F("name")),
            lambda: ValuesProjectBase.objects.polymorphic_values(weird=object()),
            lambda: ValuesProjectBase.objects.polymorphic_values(
                twice=[
                    "ValuesProjectAlpha___alpha_code",
                    "ValuesProjectAlpha___shared_note",
                ]
            ),
            lambda: ValuesProjectBase.objects.polymorphic_values(
                id=F("name")
            ),
        ]
        for build in cases:
            with self.subTest(build=build):
                with self.assertRaises(FieldError):
                    build()

    def test_non_descendant_tuple_rejected(self) -> None:
        with self.assertRaises(FieldError):
            ValuesProjectBase.objects.polymorphic_values((Model2A, "field1"))

    def test_plain_values_still_non_polymorphic(self) -> None:
        # The existing values() behaviour is untouched and stays non polymorphic.
        rows = list(
            ValuesProjectBase.objects.values("id", "name").order_by("pk")
        )
        self.assertEqual(len(rows), 6)
        self.assertNotIn("alpha_code", rows[0])

        with self.assertNumQueries(1):
            list(Model2A.objects.values("pk"))
