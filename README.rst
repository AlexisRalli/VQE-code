#######
QuChem
#######

.. image:: docs//Media/WelcomeGif.gif


.. image:: https://readthedocs.org/projects/quchem/badge/?version=latest
  :target: http://quchem.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/AlexisRalli/VQE-code/badge.svg?branch=master
  :target: https://coveralls.io/github/AlexisRalli/VQE-code?branch=master

.. image:: https://img.shields.io/lgtm/grade/python/g/AlexisRalli/VQE-code.svg
  :target: https://lgtm.com/projects/g/AlexisRalli/VQE-code/
  :alt: Code Quality

-----------------------------------------------------------------------------------------------

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

To Install
^^^^^^^^^^

First clone github repo then run:

.. code-block:: bash

    python setup.py develop

Documentation
^^^^^^^^^^^^^
The **full documentation** can be found at: `<http://quchem.readthedocs.io/en/latest/>`_.
