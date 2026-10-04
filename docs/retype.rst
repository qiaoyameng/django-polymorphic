Retyping
========

.. versionadded:: 4.12.0

Sometimes the type of a draft or otherwise mutable object needs to change after it has
been created - for example when a generic ``Document`` draft is fleshed out into an
``Article`` or a ``Report``. :meth:`~polymorphic.models.PolymorphicModel.retype` performs
this conversion explicitly and safely:

.. code-block:: python

    plan = draft.retype(Article, field_values={"author": request.user})
    article = plan.get_instance()

The retype operation:

* keeps the base row, the primary key and all shared fields,
* finds the common concrete ancestor of the current and the target type,
* creates the rows of the target inheritance chain below that ancestor -
  subclass specific fields must be supplied by the caller through
  ``field_values``,
* removes the old child rows that are no longer needed, level by level with
  ``keep_parents=True``, so a normal delete can never cascade into the kept
  base rows,
* switches the ``polymorphic_ctype`` in the same transaction as the table
  changes.

Allowed types
-------------

The target type must come from an allowed set. By default every concrete (and proxy)
model of the same polymorphic tree is allowed. Restrict the set per call with
``allowed_types`` or per model with the ``polymorphic_retype_allowed_types`` class
attribute:

.. code-block:: python

    draft.retype(Article, allowed_types=[Article, Report], field_values={...})

Targets outside the allowed set raise
:class:`~polymorphic.retype.RetypeNotAllowedError`. Targets from a different
polymorphic tree, abstract models and non-polymorphic models raise
:class:`~polymorphic.retype.RetypeError`.

Field values
------------

Fields that are specific to the target subclass chain and have no default must be
supplied through ``field_values``. Missing or unknown fields raise
:class:`~polymorphic.retype.RetypeValidationError` before anything is written, so the
original concrete type is always preserved on validation failure.

Relations
---------

Before anything is executed, all relations pointing at the child tables that would be
removed are inspected:

* ``PROTECT``, ``RESTRICT`` and ``DO_NOTHING`` relations with existing objects cannot
  be preserved and **block** the retype. They are reported through
  :class:`~polymorphic.retype.RetypeProtectedError` (and on the dry-run plan) before
  any row is touched.
* ``CASCADE`` relations are deleted together with the removed child rows and are
  listed on the plan.
* ``SET_NULL``/``SET_DEFAULT``/``SET(...)`` relations are updated by the normal
  deletion collector and are listed on the plan.

Dry-run plans
-------------

Pass ``dry_run=True`` to get a :class:`~polymorphic.retype.RetypePlan` without
executing it. The plan describes exactly which rows would be deleted and created,
which content type switch would happen and which relation effects were detected:

.. code-block:: python

    plan = draft.retype(Article, field_values={"author": user}, dry_run=True)
    print(plan.describe())
    if not plan.blocked:
        execute_retype(plan)  # from polymorphic.retype import execute_retype

Proxy models
------------

Switching between a concrete model and one of its proxies (or between two proxies of
the same concrete model) is handled separately from a real multi-table switch: only
the content type changes, no table is touched. Such plans have
:attr:`~polymorphic.retype.RetypePlan.proxy_only` set to ``True``.

Stale instances
---------------

The Python instance on which ``retype()`` is called still has its old class after the
conversion - Python cannot change the class of an existing object. Its cached state
(related object caches, prefetched objects) is explicitly invalidated and its
``polymorphic_ctype_id`` is updated, but any other in-memory attribute may be stale.
Always work with the freshly fetched instance afterwards:

.. code-block:: python

    plan = draft.retype(Article, field_values={"author": user})
    article = plan.get_instance()
