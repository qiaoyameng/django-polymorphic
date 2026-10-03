Managers & Querysets
====================

Using a Custom Manager
----------------------

A nice feature of Django is the possibility to define one's own custom object managers.
This is fully supported with :pypi:`django-polymorphic`. For creating a custom polymorphic
manager class, just derive your manager from :class:`~polymorphic.managers.PolymorphicManager`
instead of :class:`~django.db.models.Manager`. As with vanilla Django, in your model class, you
should explicitly add the default manager first, and then your custom manager:

.. code-block:: python

    from polymorphic.models import PolymorphicModel
    from polymorphic.managers import PolymorphicManager

    class TimeOrderedManager(PolymorphicManager):
        def get_queryset(self):
            qs = super(TimeOrderedManager,self).get_queryset()
            return qs.order_by('-start_date')

        def most_recent(self):
            qs = self.get_queryset()  # get my ordered queryset
            return qs[:10]            # limit => get ten most recent entries

    class Project(PolymorphicModel):
        objects = PolymorphicManager()          # add the default polymorphic manager first
        objects_ordered = TimeOrderedManager()  # then add your own manager
        start_date = DateTimeField()            # project start is this date/time

The first manager defined (:attr:`~django.db.models.Model.objects` in the example) is used by Django
as automatic manager for several purposes, including accessing related objects. It must not filter
objects and it's safest to use the plain :class:`~polymorphic.managers.PolymorphicManager` here.

Manager Inheritance
-------------------

Polymorphic models inherit/propagate all managers from their base models, as long as these are
polymorphic. This means that all managers defined in polymorphic base models continue to work as
expected in models inheriting from this base model:

.. code-block:: python

    from polymorphic.models import PolymorphicModel
    from polymorphic.managers import PolymorphicManager

    class TimeOrderedManager(PolymorphicManager):
        def get_queryset(self):
            qs = super(TimeOrderedManager,self).get_queryset()
            return qs.order_by('-start_date')

        def most_recent(self):
            qs = self.get_queryset()  # get my ordered queryset
            return qs[:10]            # limit => get ten most recent entries

    class Project(PolymorphicModel):
        objects = PolymorphicManager()          # add the default polymorphic manager first
        objects_ordered = TimeOrderedManager()  # then add your own manager
        start_date = DateTimeField()            # project start is this date/time

    class ArtProject(Project):  # inherit from Project, inheriting its fields and managers
        artist = models.CharField(max_length=30)

ArtProject inherited the managers ``objects`` and ``objects_ordered`` from Project.

``ArtProject.objects_ordered.all()`` will return all art projects ordered regarding their start time
and ``ArtProject.objects_ordered.most_recent()`` will return the ten most recent art projects.

Using a Custom Queryset Class
-----------------------------

The :class:`~polymorphic.managers.PolymorphicManager` class accepts one initialization argument,
which is the queryset class the manager should use. Just as with vanilla Django, you may define your
own custom queryset classes. Just use :class:`~polymorphic.managers.PolymorphicQuerySet` instead of
Django's :class:`~django.db.models.query.QuerySet` as the base class:

.. code-block:: python

        from polymorphic.models import PolymorphicModel
        from polymorphic.managers import PolymorphicManager
        from polymorphic.query import PolymorphicQuerySet

        class MyQuerySet(PolymorphicQuerySet):
            def my_queryset_method(self):
                ...

        class MyModel(PolymorphicModel):
            my_objects = PolymorphicManager.from_queryset(MyQuerySet)()
            ...

If you do not wish to extend from a custom :class:`~polymorphic.managers.PolymorphicManager` you
may also prefer the :meth:`~polymorphic.managers.PolymorphicQuerySet.as_manager`
shortcut:

.. code-block:: python

    from polymorphic.models import PolymorphicModel
    from polymorphic.query import PolymorphicQuerySet

    class MyQuerySet(PolymorphicQuerySet):
        def my_queryset_method(self):
            ...

    class MyModel(PolymorphicModel):
        my_objects = MyQuerySet.as_manager()
        ...

For further discussion see `this topic on the Q&A page
<https://github.com/django-commons/django-polymorphic/discussions/696#discussioncomment-15223661>`_.

Bulk creating objects of different types
----------------------------------------

Django's :meth:`~django.db.models.query.QuerySet.bulk_create` cannot insert
multi-table inherited models because it only writes a single table and cannot
hand the generated primary keys to the child tables. Polymorphic models that
only use *abstract* inheritance or *proxy* models share one table, so the
regular :meth:`~polymorphic.query.PolymorphicQuerySet.bulk_create` works for
them.

When a batch contains different concrete subclasses of a common polymorphic
base that are stored across multiple tables (multi-table inheritance, possibly
several levels deep, mixed with proxy types), use
:meth:`~polymorphic.query.PolymorphicQuerySet.bulk_create_mixed`:

.. code-block:: python

    from polymorphic.showfields import ShowFieldType
    from polymorphic.models import PolymorphicModel

    class Animal(ShowFieldType, PolymorphicModel):
        name = models.CharField(max_length=50)

    class Mammal(Animal):
        fur_color = models.CharField(max_length=50)

    class Bird(Animal):
        feather_color = models.CharField(max_length=50)

    animals = Animal.objects.bulk_create_mixed(
        [
            Animal(name="snake"),
            Mammal(name="cat", fur_color="tabby"),
            Bird(name="sparrow", feather_color="brown"),
        ]
    )

The API:

* writes parent tables before child tables and wires the automatically
  generated primary keys into the correct parent-link fields;
* supports batches that mix explicitly assigned primary keys with
  automatically generated primary keys;
* writes a shared ancestor table at most once per object, even across several
  inheritance levels;
* resolves :class:`~django.contrib.contenttypes.models.ContentType` objects
  through the database alias the queryset writes to;
* returns the created objects in the same order as the input, with the real
  polymorphic type retained.

The whole batch is validated *before* any row is written and inserted in a
single transaction, so a failure rolls back every row and never leaves an
object that only has a parent row. The following raise
:class:`~polymorphic.models.PolymorphicBulkCreateError` (a ``ValueError``
subclass) up front:

* objects that are not instances of the queryset's model;
* already saved objects, the same object passed twice, or two objects that
  would create the same table row with the same explicit primary key;
* a foreign key pointing at an unsaved related object;
* inheritance structures that are not a single linear multi-table chain (for
  example a model with multiple concrete parent tables).

The database backend must be able to return primary keys from bulk inserts
when the batch contains objects without an explicit primary key.

As with Django's own :meth:`~django.db.models.query.QuerySet.bulk_create`,
this differs from calling :meth:`~django.db.models.Model.save` on each object:

* ``save()`` is **not** called and ``pre_save`` / ``post_save`` signals are
  **not** sent;
* model validation and custom ``save()`` logic do not run;
* field-level ``pre_save`` hooks such as ``auto_now`` and ``auto_now_add``
  **are** executed;
* many-to-many relations are not saved.
