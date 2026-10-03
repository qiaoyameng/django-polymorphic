from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("contenttypes", "0002_remove_content_type_name"),
        ("tests", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="BulkTimestampBase",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "polymorphic_ctype",
                    models.ForeignKey(
                        editable=False,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="polymorphic_%(app_label)s.%(class)s_set+",
                        to="contenttypes.contenttype",
                    ),
                ),
                ("name", models.CharField(max_length=50)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("modified_at", models.DateTimeField(auto_now=True)),
            ],
            options={"abstract": False},
        ),
        migrations.CreateModel(
            name="BulkTimestampChild",
            fields=[
                (
                    "bulktimestampbase_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="tests.bulktimestampbase",
                    ),
                ),
                ("detail", models.CharField(max_length=50, default="")),
            ],
            options={"abstract": False},
            bases=("tests.bulktimestampbase",),
        ),
        migrations.CreateModel(
            name="BulkTimestampGrandChild",
            fields=[
                (
                    "bulktimestampchild_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="tests.bulktimestampchild",
                    ),
                ),
                ("extra", models.CharField(max_length=50, default="")),
            ],
            options={"abstract": False},
            bases=("tests.bulktimestampchild",),
        ),
    ]
