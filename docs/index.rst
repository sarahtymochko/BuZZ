.. ZZPers documentation master file, created by
   sphinx-quickstart on Thu Jan 16 08:36:21 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BuZZ documentation!
==================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   ./background
   ./examples
   ./API

Description
===========

This package provides python code to compute zigzag from a collection of point clouds.
Zigzag persistence is calculated using code from Dionysus2_.
This work is published _here.


See :ref:`Background` for a brief introduction and the specific zigzag diagram used in this code.
See :ref:`Examples` for an examples of how to use this code.


If you are using this method, please use the following citation/bibtex:

    Sarah Tymochko, Elizabeth Munch, and Firas A. Khasawneh (2020). Using Zigzag Persistent Homology to Detect Hopf Bifurcations in Dynamical Systems. arXiv:2009.08972.

::

    @Article{Tymochko2020a,
      author = {Sarah Tymochko and Elizabeth Munch and Firas A. Khasawneh},
      title = {Using Zigzag Persistent Homology to Detect Hopf Bifurcations in Dynamical Systems},
      journal = {arXiv:2009.08972},
      year = {2020},
    }


.. _here: https://arxiv.org/abs/2009.08972
.. _Dionysus2: https://mrzv.org/software/dionysus2/

Installation
============

Hopefully it will use pip :)

Requirements
============

Must have Dionysis 2 installed. Can install with

.. code-block:: bash

	pip install dionysis



Contact
========

Sarah Tymochko: `tymochko@egr.msu.edu <mailto:tymochko@egr.msu.edu>`_.



Index and Search
==================

* :ref:`genindex`
* :ref:`search`
