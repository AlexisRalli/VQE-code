############
Installation
############


**quchem** is a pure python package. The code is hosted on  `github <https://github.com/AlexisRalli/VQE-code/>`_
and if the dependencies are satisfied, a development version can be installed directly by cloning the github repo
and running:

.. code-block:: bash

    git clone https://github.com/AlexisRalli/VQE-code.git
    python setup.py develop



Required Dependencies
=====================

The core packages ``quchem`` requires are:

* python 3.5+
* `numpy <http://www.numpy.org/>`_
* `scipy <https://www.scipy.org/>`_
* `tqdm <https://github.com/tqdm/tqdm>`_
* `networkx <https://networkx.github.io/>`_
* `Psi4 <http://www.psicode.org/>`_
* `openfermion <https://github.com/quantumlib/OpenFermion>`_
* `openfermionpsi4 <https://github.com/quantumlib/OpenFermion-Psi4>`_
* `cirq <https://github.com/quantumlib/Cirq>`_
* `tensorflow <https://github.com/tensorflow/tensorflow>`_

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

