.. _intro:

=========
 Example
=========


Problem description
===================

There is an increasing interest in modelling and computation of physical systems with neural networks, 
given the the outstanding results they achieved in learning patterns from data. We consider Hamiltonian mechanical systems, 
whose dynamics is fully determined by one scalar function, the Hamiltonian, which represent the energy of the system. 
This explains why multiple approaches have been proposed to approximate this energy function, 
and we focus here on the task of learning the Hamiltonians of contrained mechanical systems with neural networks, 
given sample data information of their solutions.  

Let us consider Hamiltonian functions of the form

.. math::
    :label: ham

    \begin{align}
        H(q,p) = \frac{1}{2}p^TM^{-1}(q)p + V(q),
    \end{align}

where :math:`M(q)` is the mass matrix of the system, possibly depending on the configuration :math:`q\in\mathbb{R}^n`, and :math:`V(q)` is 
the potential energy of the system.  The solution trajectories are often constrained to evolve on a submanifold of a linear vector space. 
In particular, we focus on systems that are holonomically constrained on some configuration 
manifold :math:`\mathcal{Q}=\{q\in\mathbb{R}^n:\,g(q)=0\}` embedded in :math:`\mathbb{R}^n` and we model them by means of some projection operator. 
As a result, the vector field is written in such a way that it directly respects the constraints, without the addition of algebraic equations. 
We assume that the components :math:`g_i:\mathbb{R}^n\rightarrow \mathbb{R}`, :math:`i=1,...,m`, are functionally independent on the zero level set, 
so that the Hamiltonian is defined on the :math:`(2n-2m)` dimensional cotangent bundle :math:`\mathcal{M}=T^*\mathcal{Q}`. Working with elements 
of the tangent space at :math:`q`, :math:`T_q\mathcal{Q}`, as vectors in :math:`\mathbb{R}^n`, we introduce a linear operator that defines the 
orthogonal projection of an arbitrary vector :math:`v\in\mathbb{R}^n` onto :math:`T_q \mathcal{Q}`, i.e.

.. math::
    :label: proj

    \begin{align}
        \forall q\in \mathcal{Q},\text{ we set }P(q):\mathbb{R}^n\rightarrow T_q\mathcal{Q},\;\;v\mapsto P(q)v.
    \end{align}

:math:`P(q)^T` can be seen as a map sending vectors of :math:`\mathbb{R}^n` into covectors in :math:`T_q^*\mathcal{Q}`. If :math:`g(q)` is differentiable, 
assuming :math:`G(q)` is the Jacobian matrix of :math:`g(q)`, we have :math:`T_q \mathcal{Q} = \mathrm{Ker}\,G(q)`, and 
so :math:`P(q) = I_n - G(q)\left(G(q)^TG(q)\right)^{-1}G(q)^T`, where :math:`I_n\in\mathbb{R}^{n\times n}` is the identity matrix. 
This projection map allows us to define Hamilton's equations as follows

.. math::
    :label: chameq

    \begin{align}
        \begin{cases}
        \dot{q} = P(q)\partial_pH(q,p)\\
        \dot{p} = -P(q)^T\partial_qH(q,p) + W(q,p)\partial_pH(q,p),
        \end{cases}
    \end{align}

where

.. math::
    :label: wmatr

    \begin{align}
        \begin{split}
        W(q,p)&=P(q)^T\Lambda(q,p)^T P(q) + \Lambda(q,p)P(q) -P(q)^T\Lambda(q,p)^T,\\ &\text{with}\quad \Lambda(q,p) = \frac{\partial P(q)^Tp}{\partial q}.
        \end{split}
    \end{align}


Systems defined on products of spheres
=======================================

Let the phase space of the system be :math:`\mathcal{M}=(T^*S^2)^k`, where :math:`S^2` is the unit sphere in :math:`\mathbb{R}^3`, :math:`k\in \mathbb{N}^+`. 
We coordinatize :math:`\mathcal{M}` with :math:`(q,p)=(q_1,\dots,q_k,p_1,\dots,p_k)\in \mathbb{R}^{6k}`. In this case, when :math:`p\in \mathbb{R}^{3k}` is 
intended as the vector of linear momenta, the matrix :math:`M(q)` in equation :eq:`ham` is a block matrix, with

.. math::
    :label: mmatrp

    \begin{align}
        i,j = 1,...,k,\quad \mathbb{R}^{3\times 3}\ni M(q)_{ij} = \begin{cases} m_{ii}I_3,\quad i=j\\
        m_{ij}(I_3-q_iq_i^T),\quad \text{otherwise,}
        \end{cases}
    \end{align}

The matrix having constant entries :math:`m_{ij}` is symmetric and positive definite. 

For example, in the case of a spherical pendulum we have :math:`k=1`, hence the Hamiltonian dynamics is defined on its cotangent bundle :math:`T^*S^2`. 
With this specific choice of the geometry, the formulation presented in equation :eq:`chameq` simplifies considerably. Indeed :math:`P(q) = I_3-qq^T` which 
implies :math:`W(q,p) = pq^T-qp^T`. Replacing these expressions in :eq:`chameq` and using the triple product rule we end up with the following set of ODEs

.. math::
    :label: hameqpend

    \begin{align}
        \begin{cases}
        \dot{q} &= (I-qq^T)\partial_pH(q,p)\\
        \dot{p} &= -(I-qq^T)\partial_qH(q,p) + \partial_pH(q,p)\times (p\times q).
        \end{cases}
    \end{align}

We remarks briefly that :math:`T^*S^2` is a homogeneous manifold, since it is acted upon transitively by the Lie group SE(3) through the group action 

.. math::
    :label: act

    \begin{align}
        \Psi : SE(3)\times T^*S^2\rightarrow T^*S^2,\;\;((R,r),(q,p^T))\mapsto (Rq,(Rp+r\times Rq)^T),
    \end{align}

where the transpose comes from the usual interpretation of covectors as row vectors. As a consequence, we can use also Lie group integrators to solve numerically 
the system :eq:`hameqpend`. In the `code <https://github.com/THREAD-3-2/Learning_Hamiltonians/blob/main/code/learn_hamiltonian.ipynb>`_, both Lie group 
integrators (Lie-Euler and commutator-free of order 4) and classical Runge-Kutta schemes (Euler and Runge-Kutta of order four)
are implemented for the time integration of :eq:`hameqpend` during the training procedure. We represent a generic element of the special Euclidean group :math:`G=SE(3)` as 
an ordered pair :math:`(R,r)`, where :math:`R\in SO(3)` is a rotation matrix and :math:`r\in\mathbb{R}^3` is a vector. The vector 
field :math:`X(q,p)` can be expressed as :math:`\psi_*(F[H](q,p))(q,p)` with

.. math::
    :label: infgen

    \begin{align}
        \psi_*((\xi,\eta))(q,p) = (\xi\times q,\xi\times p + \eta\times q), \quad (\xi,\eta)\in \mathfrak{g= se}(3)
    \end{align}

and

.. math::
    :label: mapfham

    \begin{align}
        F[H](q,p) = (\xi,\eta)=\left(q\times \frac{\partial H(q,p)}{\partial p},\frac{\partial H(q,p)}{\partial q}\times q + \frac{\partial H(q,p)}{\partial p}\times p \right).
    \end{align}

A similar reasoning can be extended to a chain of :math:`k` connected pendula, and hence to a system on :math:`(T^*S^2)^k`. The main idea is to replicate 
both the equations and the expression :math:`F[H]` for all the :math:`k` copies of :math:`T^*S^2`. A more detailed explanation can be found 
in `(Celledoni, Çokaj, Leone, Murari and Owren, 2021) <https://doi.org/10.1080/00207160.2021.1966772>`_.