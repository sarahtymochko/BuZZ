API
=======

.. toctree::
    :caption: API:


Persistence Diagram
________________________________
.. autoclass:: ZZPers.PersDgm.PD
   :members: __init__, __str__, drawDgm, drawDgm_BL, removeInfiniteClasses, maxPers, toBirthLifetime



Persistence on Point Clouds
____________________________________
.. autoclass:: ZZPers.Zigzag.PtClouds
   :members: __init__, __str__, run_Zigzag_from_PC, get_All_Cplx, all_Cplx_to_Filtration, run_Ripser


Helper Functions
____________________________________

.. automodule:: ZZPers.Zigzag
	:members: fix_dio_vert_nums, from_DistMat_to_Cpx, to_PD_Class