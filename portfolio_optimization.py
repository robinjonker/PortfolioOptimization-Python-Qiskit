# Portfolio optimization
#
# README:   The code within this python file relates to all the functions required to do the computations for the
#           portfolio optimization. Here within you will find all the relative importing of libraries required in
#           each method. There are multiple functions created that gets called within each method. Towards the end
#           of the file you will find the computation methods, namely: classical_result, built_in_vqe,
#           custom_ansatz_vqe, and full_vqe.

from qiskit import Aer, QuantumCircuit,  QuantumRegister
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit.circuit import Parameter
import yfinance as yf
import numpy as np
import pennylane as qml
import time
import source_code
theta = Parameter('θ')


def get_stockData(stockList, start_date, end_date):
    stock_data = yf.download(stockList, start=start_date, end=end_date)['Adj Close']  # Retrieves adjusted closing price
    return stock_data


def get_mean_returns(stock_data):
    returns = stock_data.pct_change()
    ReturnMeans = returns.mean()
    mean_returns = []
    for i in range(len(ReturnMeans)):
        mean_returns.append(ReturnMeans[i])
    mean_returns = np.array(mean_returns)
    mean_returns = mean_returns.T
    return returns, mean_returns


def get_covariance_of_stocks(returns):
    covariance_matrix = returns.cov()
    covariance_matrix = covariance_matrix.to_numpy()
    return covariance_matrix


def get_quadratic_program(mean_returns, covariance_matrix, risk_factor, budget):
    portfolio = PortfolioOptimization(expected_returns=mean_returns, covariances=covariance_matrix,
                                      risk_factor=risk_factor, budget=budget)
    return portfolio.to_quadratic_program()


def ansatz(stock_list):
    qubits = len(stock_list)
    reps = 3
    ansatz = ansatz_entanglement(qubits, reps)
    ansatz.draw(output='mpl')
    return ansatz


def ansatz_entanglement(qubit, rep):
    qr = QuantumRegister(qubit, 'q')
    circ = QuantumCircuit(qr)
    y_rotations(circ, theta)
    for i in range(rep):
        circ.barrier()
        cx_entanglement(circ)
        y_rotations(circ, theta)
    return circ


def cx_entanglement(circ):
    for qubit in range(circ.num_qubits-1):
        for gate in range(circ.num_qubits-qubit-1):
            circ.cx(qubit, qubit+1+gate)


def y_rotations(circ, theta):
    for qubit in range(circ.num_qubits):
        circ.ry(theta, qubit)


# Function for the classical method
def classical_result(quadratic_program):
    start = time.time()
    exact_mes = NumPyMinimumEigensolver()
    eigen_solver = MinimumEigenOptimizer(exact_mes)
    result = eigen_solver.solve(quadratic_program)
    end = time.time()
    execution_time = end - start
    return result, execution_time


# Function for the built-in VQE method
def built_in_vqe(quadratic_program, stock_list):
    start = time.time()
    algorithm_globals.random_seed = 123
    backend = Aer.get_backend("statevector_simulator")
    ry = TwoLocal(len(stock_list), "ry", "cz", reps=3, entanglement="full")
    quantum_instance = QuantumInstance(backend=backend)
    cobyla = COBYLA()
    cobyla.set_options(maxiter= 500)
    vqe_mes = VQE(ry, optimizer=COBYLA(), quantum_instance=quantum_instance)
    vqe = MinimumEigenOptimizer(vqe_mes)
    result = vqe.solve(quadratic_program)
    end = time.time()
    execution_time = end - start
    return result, execution_time


# Function for the custom ansatz VQE method
def custom_ansatz_vqe(quadratic_program, stock_list):
    start = time.time()
    algorithm_globals.random_seed = 123
    backend = Aer.get_backend("statevector_simulator")
    ry = ansatz(stock_list)
    quantum_instance = QuantumInstance(backend=backend)
    cobyla = COBYLA()
    cobyla.set_options(maxiter=500)
    vqe_mes = VQE(ry, optimizer=COBYLA(), quantum_instance=quantum_instance)
    vqe = MinimumEigenOptimizer(vqe_mes)
    result = vqe.solve(quadratic_program)
    end = time.time()
    execution_time = end - start
    return result, execution_time


# Function for the VQE method
def full_vqe(quadratic_program, stock_list):
    wires = len(stock_list)

    def to_hamiltonian(pauli):
        # converting Ising Pauli Sum Operator to Hamiltonian
        x = pauli.primitive.to_list()
        a = []
        b = []
        for i in range(len(x)):
            b.append(qml.grouping.string_to_pauli_word(x[i][0]))
            a.append(x[i][1].real)
        return qml.Hamiltonian(a, b)

    def dict_of_hamiltonian():
        H, offset = QuadraticProgramToQubo().convert(quadratic_program).to_ising() # converting quadratic program to Ising Pauli Sum Operator
        Hamiltonian = to_hamiltonian(H)
        H = Hamiltonian.coeffs
        dict = {}
        for i in range(wires * 2):
            dict.update({str(i): H[i]})
        return dict, Hamiltonian     #returns a dictionary of hamiltonian coeffs

    def ansatz_ry(angle, x):    # applies rotation y gate of specified rotation parameter to qml node
        for i in range(wires):
            qml.RY(angle[i + x], wires=i)

    def ansatz_cz():        # applies cz gates for entanglement to qml node
        for i in range(wires - 1):
            for j in range(wires - i - 1):
                qml.CZ(wires=[i, i + j + 1])

    def ansatz_circuit(angle):      # creating temp ansatz circuit layout on QNode
        x = 0
        ansatz_ry(angle, x)
        for i in range(reps):
            x = x + wires
            ansatz_cz()
            ansatz_ry(angle, x)
        return ansatz_circuit

    def exp():      # creates range of identity and pauilZ operator lists to be used in creation of QNode circuit
        v = []
        for i in range(wires):
            c = []
            for j in range(wires):
                c.append(qml.Identity(j))
            c1 = c
            c1[wires - 1 - i] = qml.PauliZ(wires - 1 - i)
            v.append(c1)
        for i in range(wires):
            c = []
            for j in range(wires):
                c.append(qml.PauliZ(j))
            c1 = c
            c1[i] = qml.Identity(i)
            v.append(c1)
        return v

    dev1 = qml.device('qiskit.aer', wires=wires)
    @qml.qnode(dev1)
    def circuitx(params, expv, num):        # creates the circuit with applied operators onto a QNode
        ansatz_circuit(params)
        x = expv[num][2]
        for i in range(1, 0, -1):
            x = x @ expv[num][i]
        return qml.expval(x)

    def vqe_loop(parameters):       # creates the variational step values based on the input parameters of the circuits
        value = 0
        for i in range(wires * 2):
            value = value + dict[str(i)] * circuitx(parameters, expv, i)
        return value

    @qml.qnode(dev1)
    def final_circ(params):         # creates the final circuit with the resultant parameters for the y rotation gates
        ansatz_circuit(params)      # this occurs after the specified amount of recursions in finding the parameters
        a = []
        for i in range(wires):
            a.append(i)
        return qml.probs(wires=a)

    def y_rotations_vqe(circ, theta, x):       # applies y rotation with final parameters to quantum circuit
        for qubit in range(circ.num_qubits):
            circ.ry(theta[qubit + x], qubit)

    def draw_vqe_circuit(params):       # drawing of the final circuit
        circ = QuantumCircuit(wires)
        x = 0
        y_rotations_vqe(circ, params, x)
        for i in range(reps):
            x = x + wires
            cx_entanglement(circ)
            y_rotations_vqe(circ, params, x)
        circ.draw(output='mpl')

    start = time.time()
    dict, Hamiltonian = dict_of_hamiltonian()
    expv = exp()
    opt = qml.GradientDescentOptimizer(stepsize=0.5)
    value = []
    reps = 3
    params = np.random.rand(wires * (reps + 1) * 2)     # sets random initial parameters to be entered into the circuit
    steps = 100   # amount of recursion of the parameter values
    for i in range(steps):
        params = opt.step(vqe_loop, params)
        value.append(vqe_loop(params))
    end = time.time()
    execution_time = end - start
    draw_vqe_circuit(params)
    return final_circ(params), execution_time
