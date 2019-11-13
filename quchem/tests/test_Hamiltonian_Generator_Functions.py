if __name__ == '__main__':
    from VQE_methods.Hamiltonian_Generator_Functions import Hamiltonian
else:
    from .VQE_methods.Hamiltonian_Generator_Functions import Hamiltonian
import pytest



def test_geometry():
    """
    Standard use test
    """

    X = Hamiltonian('H2')

    pass

    #assert Blah