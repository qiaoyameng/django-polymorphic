from __future__ import annotations

from django.apps import apps
from django.core.management.base import BaseCommand, CommandError

from polymorphic.audit import (
    apply_polymorphic_ctype_report,
    audit_polymorphic_ctype,
)


class Command(BaseCommand):
    help = "Audit polymorphic_ctype values against actual inheritance table rows."

    def add_arguments(self, parser) -> None:
        parser.add_argument("models", nargs="+", help="Polymorphic model labels, e.g. app.Model")
        parser.add_argument("--database", default="default", help="Database alias to audit")
        parser.add_argument("--batch-size", type=int, default=500)
        parser.add_argument(
            "--apply",
            action="store_true",
            help="Apply only findings whose preconditions still match after the audit.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Verify repair preconditions without changing polymorphic_ctype values.",
        )

    def handle(self, *args, **options) -> None:
        if options["dry_run"] and options["apply"]:
            raise CommandError("--apply and --dry-run cannot be used together")

        try:
            roots = [apps.get_model(label) for label in options["models"]]
        except LookupError as exc:
            raise CommandError(str(exc)) from exc

        report = audit_polymorphic_ctype(
            *roots,
            using=options["database"],
            batch_size=options["batch_size"],
        )

        self.stdout.write(
            f"Audited {report.scanned} row(s) across {len(report.batches)} batch(es); "
            f"found {len(report.findings)} issue(s), {report.auto_repairable} auto-repairable."
        )
        for issue, count in report.issue_counts.items():
            self.stdout.write(f"  {issue}: {count}")

        for finding in report.findings:
            self.stdout.write(
                f"{finding.root_label}#{finding.pk} issues={','.join(finding.issues)} "
                f"evidence={finding.evidence}"
            )
            if finding.proposed_model_label:
                self.stdout.write(f"  proposed={finding.proposed_model_label}")
            else:
                self.stdout.write(f"  manual={finding.reason_not_auto_repaired}")

        if not options["apply"] and not options["dry_run"]:
            return

        result = apply_polymorphic_ctype_report(
            report,
            using=options["database"],
            dry_run=options["dry_run"],
        )
        self.stdout.write(
            f"Applied {result.applied}; stale rejected {result.stale}; "
            f"not repairable {result.not_repairable}; dry-run verified "
            f"{result.dry_run_verified}; batches {result.batches}."
        )
