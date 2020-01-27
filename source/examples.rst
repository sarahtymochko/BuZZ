.. _Examples:

Examples
========

.. toctree::
   :caption: Examples:

As described in :ref:`Background`, this package computes zigzag persistence (with the help of Dionysis 2)
for a collection of point clouds :math:`X_0,\ldots,X_n` where we have the following zigzag diagram

.. math::
  R(X_0,r) \hookrightarrow R(X_0 \cup X_1,r) \hookleftarrow R(X_1,r) \hookrightarrow \cdots \hookleftarrow R(X_{n-1}, r)\hookrightarrow R(X_{n-1} \cup X_n,r) \hookleftarrow R(X_n,r).

Additionally, we can have a zigzag diagram where the radius for each rips filtration is different for each point cloud, :math:`R(X_i,r_i)`,
then for the unions we choose the radius to be the max, :math:`R(X_i\cup X_{i+1}, max(r_i,r_{i+1}))`.
See below for an example of each of these cases.

Fixed radius
------------

.. code-block:: python

  import numpy as np
  import dionysus as dio
  from ZZPers.ZZPers.PD import PD
  from ZZPers.ZZPers.PtClouds import PtClouds

  ZZ = PtClouds(PCs, verbose=True)
  ZZ.run_Zigzag(r=2, k=2)

  ZZ.zz_dgms[0].drawDgm()
  ZZ.zz_dgms[1].drawDgm()
  plt.legend()

Changing radius
---------------
