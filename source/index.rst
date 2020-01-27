.. ZZPers documentation master file, created by
   sphinx-quickstart on Thu Jan 16 08:36:21 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ZZPers's documentation!
==================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   ./background
   ./examples
   ./API

Description
===========

This package provides python code to compute zigzag and regular persistence from a collection of point clouds.
Regular persistence is calculated using ripser_ from Scikit-TDA_, while zigzag persistence is calculated using code from Dionysus2_.


See :ref:`Background` for a brief introduction and the specific zigzag diagram used in this code.
See :ref:`Examples` for an examples of how to use this code.

.. _Ripser: http://ripser.scikit-tda.org/reference/stubs/ripser.ripser.html#ripser.ripser
.. _Scikit-TDA: https://scikit-tda.org/
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
