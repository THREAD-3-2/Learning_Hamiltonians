========================================
 Documentation of `Learning_Hamiltonians`
========================================

`Learning Hamiltonians <https://github.com/THREAD-3-2/Learning_Hamiltonians>`_ is a Python code for 
learning the Hamiltonian of a constrained mechanical system. 
It is part of the source code for the paper `(Celledoni, Leone, Murari and Owren, 2022) <https://doi.org/10.1016/j.cam.2022.114608>`_ and uses `PyTorch <https://pytorch.org/>`_ as 
deep learning library, in particular the modules related to `automatic differentiation <https://pytorch.org/docs/stable/autograd.html>`_, 
`optimization algorithms <https://pytorch.org/docs/stable/optim.html>`_ and `neural networks <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_.
An introduction to the problem and a description of the system under consideration are provided in :ref:`Example <intro>`. The :ref:`learning procedure <learn>` is presented in the homonymous section.   
The code can be run from the `main file <https://github.com/THREAD-3-2/Learning_Hamiltonians/blob/main/Learning_Hamiltonians/main.py>`_.


Contents
========

.. toctree::
   installation
   intro
   learn
   :maxdepth: 2


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
