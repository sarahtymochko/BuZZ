.. _Background:

Background
==========

.. toctree::
   :caption: Background:


Topological data analysis (TDA) is a relatively new field that has shown significant promise in theory and in applications.
One of the most popular tools in TDA is persistent homology.
This tool allows you to track the homology of a filtration, nested set of simplicial complexes,

.. math::
  X_0 \hookrightarrow X_1 \hookrightarrow \cdots \hookrightarrow X_n.

Applying homology, we get a diagram of vector spaces and linear maps,

.. math::
  H(X_0) \to H(X_1) \to \cdots \to H(X_n).

This setting is very useful for a variety of applications.
For example, the Vietoris-Rips complex on point cloud data gives the filtration and the inclusions induce the linear maps on homology.
However, if we do not have inclusions in our filtration, standard persistent homology cannot be applied.

Zigzag persistence is a modification of persistent homology except inclusions are not required to go strictly in one direction.
Instead we may have

.. math::
  X_0 \leftrightarrow X_1 \leftrightarrow \cdots \leftrightarrow X_n

where the arrows can be inclusions in either direction.
This is called a **zigzag diagram**.
This setting could be useful when looking at a collection of point clouds where we don't have an inclusion from one point cloud into the next.

In this software package, we are going to focus on a specific type of zigzag persistence.
We will consider an *ordered* (here the order will matter) set of point clouds :math:`X_0,X_1,\ldots,X_n` where we do not have an inclusion from one point cloud to the next.
To use zigzag persistence, we set up inclusions as follows:

.. math::
  X_0 \hookrightarrow X_0 \cup X_1 \hookleftarrow X_1 \hookrightarrow \cdots \hookleftarrow X_{n-1} \hookrightarrow X_{n-1} \cup X_n \hookleftarrow X_n.

However, this only provides a set of inclusions of point clouds, so we will compute the Rips complex with a radius :math:`r`, resulting in the following inclusions:

.. math::
  R(X_0,r) \hookrightarrow R(X_0 \cup X_1,r) \hookleftarrow R(X_1,r) \hookrightarrow \cdots \hookleftarrow R(X_{n-1}, r)\hookrightarrow R(X_{n-1} \cup X_n,r) \hookleftarrow R(X_n,r).

Further, the radius could be different for each :math:`i`, giving us :math:`R(X_i,r_i)`, then for the unions, we have :math:`R(X_i \cup X_{i+1}, max(r_i,r_{i+1}))`, ensuring we still have inclusions.

This package can compute zigzag persistence for both a fixed radius, or a different radius for each original point cloud.
For an example of both, see :ref:`Examples`.
