import numpy as np
import cirq
import random



class UCCSD_Trotter():
    """

    The UCCSD_Trotter object calculates trotterisation of UCCSD anatzse

    Args:
        Second_Quant_CC_Ops_list (list): List of Fermionic Operators (openfermion.ops._fermion_operator.FermionOperator)
        Theta_param_list (list): List of Angles corresponding to UCCSD excitation amplitudes


    Attributes:
        #TODO

    """
    def __init__(self, Second_Quant_CC_Ops_list, Theta_param_list):
        self.Second_Quant_CC_Ops_list = Second_Quant_CC_Ops_list # FermionOperator
        self.Theta_param_list = Theta_param_list

    def SingleTrotterStep(self):

        """
        Performs single trotter step approximation of UCCSD anstaz.
            U = exp [ t02 (a†2a0−a†0a2) + t13(a†3a1−a†1a3) +t0123 (a†3a†2a1a0−a†0a†1a2a3) ]
            becomes
            U=exp [t02(a†2a0−a†0a2)] × exp [t13(a†3a1−a†1a3)] × exp [t0123(a†3a†2a1a0−a†0a†1a2a3)]

        using the JORDAN WIGNER TRANSFORM

        Takes list of UCCSD fermionic excitation operators:

                    [
                       -(a†0 a2) + (a†2 a0),
                       -(a†1 a3) + (a†3 a1),
                       -(a†0 a†1 a2 a3) + a†3 a†2 a1 a0)
                    ]
        and returns JW transform of each term and appends it to a list yielding a list of QubitOperators
        performing UCCSD.

        [
            -0.5j [X0 Z1 Y2] + 0.5j [Y0 Z1 X2],
            -0.5j [X1 Z2 Y3] + 0.5j [Y1 Z2 X3],
            0.125j [X0 X1 X2 Y3] + 0.125j [X0 X1 Y2 X3] + -0.125j [X0 Y1 X2 X3] + 0.125j [X0 Y1 Y2 Y3] +
            -0.125j [Y0 X1 X2 X3] + 0.125j [Y0 X1 Y2 Y3] + -0.125j [Y0 Y1 X2 Y3] + -0.125j [Y0 Y1 Y2 X3]
        ]

        returns:
            Second_Quant_CC_JW_OP_list (list): List of QubitOperators (openfermion.ops._qubit_operator.QubitOperator)
                                               under JW transform. Each performs a UCCSD excitation.

        """

        Second_Quant_CC_JW_OP_list = []
        from openfermion.transforms import jordan_wigner
        for OP in self.Second_Quant_CC_Ops_list: # each OP = (T_i − T_i^†) i indicates if single, double, triple ... etc
            JW_OP = jordan_wigner(OP)
            Second_Quant_CC_JW_OP_list.append(JW_OP)
        return Second_Quant_CC_JW_OP_list

    def DoubleTrotterStep(self):
        #TODO
        # note for Theta_param_list, with 2nd order Trot will need to double angle params!
        pass

def Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list):
    """
     From a list of QubitOperators (openfermion.ops._qubit_operator.QubitOperator) generate corresponding
     list of PauliStrings with cofactor!

    Args:
        Second_Quant_CC_JW_OP_list (list): list of QubitOperators (openfermion.ops._qubit_operator.QubitOperator)

    Returns:
        PauliWord_str_Second_Quant_CC_JW_OP_list (list): List of tuples (PauliWord_str, cofactor)

    e.g.
    Second_Quant_CC_JW_OP_list =
    [
        -0.5j[X0 Z1 Y2] + 0.5j[Y0 Z1 X2],
        -0.5j[X1 Z2 Y3] +0.5j[Y1 Z2 X3],
        0.125j[X0 X1 X2 Y3] +0.125j[X0 X1 Y2 X3] + -0.125j[X0 Y1 X2 X3] + 0.125j[X0 Y1 Y2 Y3] +-0.125j[Y0 X1 X2 X3]
        +0.125j[Y0 X1 Y2 Y3] +-0.125j[Y0 Y1 X2 Y3] + -0.125j[Y0 Y1 Y2 X3]
    ]
    becomes

     [
        [('X0 Z1 Y2 I3', -0.5j), ('Y0 Z1 X2 I3', 0.5j)],
        [('I0 X1 Z2 Y3', -0.5j), ('I0 Y1 Z2 X3', 0.5j)],
        [('Y0 Y1 X2 Y3', -0.125j),('X0 X1 X2 Y3', 0.125j),('X0 Y1 Y2 Y3', 0.125j),('Y0 X1 Y2 Y3', 0.125j),
         ('X0 Y1 X2 X3', -0.125j),('Y0 X1 X2 X3', -0.125j),('Y0 Y1 Y2 X3', -0.125j),('X0 X1 Y2 X3', 0.125j)]
     ]
    """


    PauliWord_str_Second_Quant_CC_JW_OP_list = []
    max_No_terms = max([len(list(QubitOP.terms.keys())[0]) for QubitOP in Second_Quant_CC_JW_OP_list])
    all_indices = np.arange(0, max_No_terms, 1)

    for QubitOP in Second_Quant_CC_JW_OP_list:
        T_Tdagg_Op_list = []

        for tupleOfTuples, factor in QubitOP.terms.items():
            qubit_OP_list = [tupl[1] + str(tupl[0]) for tupl in tupleOfTuples]

            if len(qubit_OP_list) < max_No_terms:
                # fill missing terms with Identity
                indices_present = [int(qubitNo_and_OP[1::]) for qubitNo_and_OP in qubit_OP_list]
                missing_indices = [index for index in all_indices if index not in indices_present]

                for index in missing_indices:
                    qubit_OP_list.append('I{}'.format(index))

                qubit_OP_list = sorted(qubit_OP_list, key=lambda x: int(x[1::])) # sort by qubitNo!
            # T_Tdagg_Op_list.append((qubit_OP_list, factor))

            seperator = ' '
            PauliWord = seperator.join(qubit_OP_list)
            T_Tdagg_Op_list.append((PauliWord, factor))
        PauliWord_str_Second_Quant_CC_JW_OP_list.append(T_Tdagg_Op_list[::-1]) # reverse order (as need to do righthand side first!)
    return PauliWord_str_Second_Quant_CC_JW_OP_list

def HF_state_generator(n_electrons, n_qubits):
    """
     Generate ground state HF state (singlet) in occupation number basis (canonical orbitals)

    Args:
        n_electrons (int): Number of electrons
        n_qubits (int): Number of qubits

    Returns:
        np.array: HF singlet ground state in occupation number basis

    .. code-block:: python
       :emphasize-lines: 2

       from quchem.Ansatz_Generator_Functions import *
       state = HF_state_generator(2, 4)
       print(state)
       >> [0. 0. 1. 1.]
    """
    occupied = np.ones(n_electrons)
    unoccupied = np.zeros(n_qubits-n_electrons)
    return np.array([*unoccupied,*occupied])

from quchem.quantum_circuit_functions import *
class Ansatz_Circuit():
    """

    The Ansatz_Circuit object allows Hartree Fock UCCSD Ansatz Circuit to be generated

    Args:
        PauliWord_str_Second_Quant_CC_JW_OP_list (list): List of Fermionic Operators (openfermion.ops._fermion_operator.FermionOperator)
        n_electrons (int): Number of electrons
        n_qubits (int): Number of qubits

    Attributes:
        HF_QCirc ():

    """
    def __init__(self, PauliWord_str_Second_Quant_CC_JW_OP_list, n_electrons, n_qubits):
        self.PauliWord_str_Second_Quant_CC_JW_OP_list = PauliWord_str_Second_Quant_CC_JW_OP_list
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons

        self.HF_QCirc = None

    def Get_HF_Quantum_Circuit(self):
        HF_state = HF_state_generator(self.n_electrons, self.n_qubits)
        HF_state_prep = State_Prep(HF_state)
        HF_state_prep_circuit = cirq.Circuit.from_ops(cirq.decompose_once(
            (HF_state_prep(*cirq.LineQubit.range(HF_state_prep.num_qubits())))))
        self.HF_QCirc = list(HF_state_prep_circuit.all_operations())

    def Get_UCCSD_Quantum_Circuit(self, Theta_param_list):

        Q_Circuit_generator_list =[]

        for i in range(len(self.PauliWord_str_Second_Quant_CC_JW_OP_list)):
            ExcitationOp = self.PauliWord_str_Second_Quant_CC_JW_OP_list[i]
            Theta = Theta_param_list[i]
            for Paulistring_and_Cofactor in ExcitationOp:
                Q_circuit_gen = full_exponentiated_PauliWord_circuit(Paulistring_and_Cofactor, Theta)
                Q_circuit = cirq.Circuit.from_ops(cirq.decompose_once(
                    (Q_circuit_gen(*cirq.LineQubit.range(Q_circuit_gen.num_qubits())))))
                Q_Circuit_generator_list.append(Q_circuit.all_operations())
        return Q_Circuit_generator_list

    def Get_Full_HF_UCCSD_QC(self, Theta_param_list):

        if self.HF_QCirc is None:
            self.Get_HF_Quantum_Circuit()

        UCCSD_QC_List = self.Get_UCCSD_Quantum_Circuit(Theta_param_list)

        full_circuit = cirq.Circuit.from_ops(
            [
                self.HF_QCirc,
                *UCCSD_QC_List,
            ]
        )
        return full_circuit

if __name__ == '__main__':
    ####### REQUIRED TO USE ANSATZSE CLASS ######
    from quchem.Hamiltonian_Generator_Functions import *
    ### Variable Parameters
    Molecule = 'H2'
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.74))]
    num_shots = 10000
    ####

    ### Get Hamiltonian
    Hamilt = Hamiltonian(Molecule,
                         run_scf=1, run_mp2=1, run_cisd=1, run_ccsd=1, run_fci=1,
                         basis='sto-3g',
                         multiplicity=1,
                         geometry=geometry)  # normally None!

    Hamilt.Get_Molecular_Hamiltonian()
    SQ_CC_ops, THETA_params = Hamilt.Get_ia_and_ijab_terms(Coupled_cluser_param=False)
    print(SQ_CC_ops)

    HF_transformations = Hamiltonian_Transforms(Hamilt.MolecularHamiltonian, SQ_CC_ops, Hamilt.molecule.n_qubits)
    # fermionic_hamiltonian = HF_transformations.Get_Fermionic_Hamiltonian()
    # print(fermionic_hamiltonian)
    Qubit_Hamiltonian = HF_transformations.Get_Qubit_Hamiltonian_JW() # qubit Hamiltonian version of Molecular Hamiltonian
    print(Qubit_Hamiltonian)
    ####### FINISHED   #### REQUIRED TO USE ANSATZSE CLASS ######

    UCCSD = UCCSD_Trotter(SQ_CC_ops, THETA_params)

    Second_Quant_CC_JW_OP_list = UCCSD.SingleTrotterStep()

    PauliWord_list = Convert_QubitOperator_To_Pauliword_Str_list(Second_Quant_CC_JW_OP_list)
    HF_UCCSD_ansatz = Ansatz_Circuit(PauliWord_list, Hamilt.molecule.n_electrons, Hamilt.molecule.n_qubits)

    THETA_params = [np.pi, 3*np.pi, 2*np.pi]
    # THETA_params = [random.uniform(0, 2 * np.pi) for _ in range(len(THETA_params))]
    ansatz_Q_cicuit = HF_UCCSD_ansatz.Get_Full_HF_UCCSD_QC(THETA_params)
    #print(ansatz_Q_cicuit)
