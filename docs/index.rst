Welcome to QuChem's documentation!
==================================

.. image:: /Media/WelcomeGif.gif


.. QuChem documentation master file, created by
   sphinx-quickstart on Thu Nov 28 23:07:38 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


`quchem <https://github.com/AlexisRalli/VQE-code>`_ is a python library for quantum chemistry VQE simulations.


The ``quchem`` module:

* Uses `Psi4 <http://www.psicode.org/>`_ to obtain the molecular Hamiltonian of a system of interest
* The `openfermion <https://github.com/quantumlib/OpenFermion>`_ library converts this to a qubit Hamiltonian
   * Currently only the Jordan-Wigner transform is used
* Google's `cirq <https://github.com/quantumlib/Cirq>`_ library is used to build quantum circuits
   * Hartree-Fock (HF) + Unitary Coupled Cluster Single Double (UCCSD) excitations are currently used as the Ansatz
   * Each quantum  circuit is simulated using a wrapper for `cirq <https://github.com/quantumlib/Cirq>`_'s simulator.

.. image:: https://readthedocs.org/projects/quchem/badge/?version=latest
  :target: http://quchem.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

To Install
^^^^^^^^^^
**quchem** is a pure python package. The code is hosted on  `github <https://github.com/AlexisRalli/VQE-code/>`_
and if the dependencies are satisfied, a development version can be installed directly by cloning the github repo:

.. code-block:: bash

    git clone https://github.com/AlexisRalli/VQE-code.git

and running:

.. code-block:: bash

    python setup.py develop


Guide
^^^^^^

.. toctree::
  :numbered:
  :maxdepth: 1
  :caption: Contents:

  Installation
  LICENSE


.. toctree::
  :numbered:
  :maxdepth: 1
  :caption: AutoDocs:

  Ansatz_Generator_Functions
  Graph
  Hamiltonian_Generator_Functions
  Unitary_partitioning




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
