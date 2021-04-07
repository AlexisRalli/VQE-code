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

* Uses `PySCF <https://sunqm.github.io/pyscf/>`_ to obtain the molecular Hamiltonian of a system of interest
* Can use `Psi4 <http://www.psicode.org/>`_ to obtain the molecular Hamiltonian of a system of interest too
* The `openfermion <https://github.com/quantumlib/OpenFermion>`_ library converts this to a qubit Hamiltonian
   * Currently only the Jordan-Wigner and bravyi-kitaev transforms are used
* Google's `cirq <https://github.com/quantumlib/Cirq>`_ library is used to build quantum circuits
   * Hartree-Fock (HF) + Unitary Coupled Cluster Single Double (UCCSD) excitations are currently used as the Ansatz
   * Each quantum  circuit is simulated using a wrapper for `cirq <https://github.com/quantumlib/Cirq>`_'s simulator.

To Install
^^^^^^^^^^

**quchem** is a pure python package. The code is hosted on  `github <https://github.com/AlexisRalli/VQE-code/>`_
and if the dependencies are satisfied, a development version can be installed directly by cloning the github repo
and running:

.. code-block:: bash

    git clone https://github.com/AlexisRalli/VQE-code.git

    pip install -r requirements.txt
    python setup.py develop

Special Dependencies
^^^^^^^^^^^^^^^^^^^^
.. note::
    :class: quchemnote

    To install Psi4 please follow `Psi4_install <https://admiring-tesla-08529a.netlify.com/installs/v132/>`_
    for anaconda distribution do:

    .. code-block:: bash

        conda config --add channels http://conda.anaconda.org/psi4
        #check
        cat ~/.condarc
        >> channels:
          - http://conda.anaconda.org/psi4
          - defaults

        #to install
        conda install psi4

.. note::
    :class: quchem_Tensor_Note

    TensorFlow version 1 required for one optimizer (only requires installing if used)!
    To install for anaconda distribution do:

    .. code-block:: bash

        conda install -c conda-forge tensorflow=1.15


Documentation
^^^^^^^^^^^^^
The **full documentation** can be found at: `<http://quchem.readthedocs.io/en/latest/>`_.
