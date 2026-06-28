Extensions
==========

Random Network Distillation
---------------------------

.. automodule:: rsl_rl.extensions.rnd
   :members:
   :undoc-members:


Symmetry
--------

.. automodule:: rsl_rl.extensions.symmetry
   :members:
   :undoc-members:


Successor Features (Deprecated)
-------------------------------

``SuccessorFeatures`` is deprecated. For off-policy forward-backward training,
use :class:`rsl_rl.algorithms.ForwardBackward` with
:class:`rsl_rl.runners.OffPolicyRunner`. This is an explicit migration rather
than a drop-in rename: the unified learner owns replay, reward channels,
optimizer state, and checkpoint compatibility.

.. automodule:: rsl_rl.extensions.successor
   :members: SuccessorFeatures
   :undoc-members:
   :show-inheritance:

