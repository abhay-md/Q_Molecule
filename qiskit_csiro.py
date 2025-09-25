import os
from qiskit import QuantumCircuit, transpile
import sys
from qiskit.algorithms import VQD
from qsubgisom.qsubgisom import ansatz, observable, s4_ansatz
from qsubgisom import sample_exact_thetas, perm_to_2line
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP, GradientDescent
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import Aer
from qiskit.quantum_info import Operator
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
from openbabel import openbabel, pybel


def pdb_to_adj_and_pos(pdb_file):
    # Load molecule
    mol = next(pybel.readfile("pdb", pdb_file))
    n_atoms = len(mol.atoms)
    # Build 0/1 adjacency matrix
    adj_matrix = np.zeros((n_atoms, n_atoms), dtype=int)
    for bond in openbabel.OBMolBondIter(mol.OBMol):
        #order = bond.GetBondOrder()
        i = bond.GetBeginAtomIdx() - 1 #order
        j = bond.GetEndAtomIdx() - 1 #order
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # symmetric
    # Atom symbols
    atom_symbols = [atom.type for atom in mol.atoms]
    # 2D positions (x, y) for plotting
    pos = {i: atom.coords[:2] for i, atom in enumerate(mol.atoms)}
    return adj_matrix, pos, atom_symbols


def pdb_to_adj_and_pos_exclude_hydrogen(pdb_file, exclude_h=True):
    """
    Returns adjacency matrix, positions, and atom symbols from a PDB file.
    exclude_h=True omits hydrogens from adjacency and positions.
    """
    mol = next(pybel.readfile("pdb", pdb_file))
    atom_symbols = [a.type for a in mol.atoms]
    atoms = [a for a in mol.atoms if not (exclude_h and a.type == "H")]
    idx_map = {a.idx: i for i, a in enumerate(atoms)}
    adj = np.zeros((len(atoms), len(atoms)), dtype=int)
    for b in openbabel.OBMolBondIter(mol.OBMol):
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idx_map and j in idx_map:
            adj[idx_map[i], idx_map[j]] = adj[idx_map[j], idx_map[i]] = 1
    pos = {i: a.coords[:2] for i, a in enumerate(atoms)}
    return adj, pos, atom_symbols


def pad_to_pow2(mat):
    n = mat.shape[0]
    next_pow2 = 1 << (n - 1).bit_length()  # smallest power of 2 ≥ n
    padded = np.zeros((next_pow2, next_pow2), dtype=mat.dtype)
    padded[:n, :n] = mat
    return padded


def checkboard(n):
    """
    Produce a matrix with the checkboard pattern.
    In the case n=4, the resulting 4x4 matrix corresponds to
    the adjacency matrix of a graph having 4 vertices interconnected
    to form a square.
    """
    c = np.arange(n//2)*2
    c = np.tile(c, len(c)), np.repeat(c + 1, len(c))
    m = np.zeros((n, n), dtype=int)
    m[c] = 1
    m[(c[1], c[0])] = 1
    return m


def rnd_perm_mat(n, *, seed=None):
    """
    Produce a nxn random permutation matrix.
    """
    rng = np.random.default_rng(seed=seed)
    m = np.eye(n, dtype=int)
    rng.shuffle(m)
    return m


def get_qc_for_n_qubit_GHZ_state(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n-1):
        qc.cx(i, i+1)
    return qc


def cost_f(adj1, adj2, p):
    m = p @ adj1 @ p.T
    m = m[:len(adj2), :len(adj2)]
    return np.linalg.norm(m - adj2)


def plot_solution_edges(g1, *, pos, perm, ax):
    vx_sel = perm_to_2line(perm, inverse=True)
    vx_sel = vx_sel[1, :len(adj2)]
    nx.draw_networkx_edges(g1.subgraph(vx_sel), pos,
                           width=5.0, alpha=0.5, ax=ax)


def molecule_substructure_match(adj1=None, adj2=None, max_trials=5, ansatz=ansatz):
    seed = 10283764
    algorithm_globals.random_seed = seed
    rng = np.random.default_rng(seed=seed)
    qc, params = ansatz(adj1, adj2)
    qi = QuantumInstance(Aer.get_backend('statevector_simulator'),
                         seed_transpiler=seed, seed_simulator=seed, shots=1024)
    optim = SLSQP(maxiter=1000)
    def trial():
        initial_point = (rng.uniform(size=len(qc.parameters)) - 0.5)  * np.pi #change to -pi to pi exploration
        vqe = VQE(ansatz=qc, optimizer=optim, quantum_instance=qi, initial_point=initial_point)
        obj = vqe.compute_minimum_eigenvalue(operator=observable(qc.num_qubits))
        return obj.optimal_value, obj
    results = [trial() for _ in tqdm(range(max_trials))]
    results = sorted(results, key=lambda obj: obj[0])# sort by energy
    for i in range(max_trials):
        result = results[i][1]
        print("Optimal value:", result.optimal_value)
        qc1 = s4_ansatz('circular', qreg=(qc.num_qubits - 1) // 2, params=params)[0]
        sampled_params_dicts = sample_exact_thetas(result.optimal_parameters, n=32, seed=seed)
        min_cost = np.inf
        for v in sampled_params_dicts:
            p1 = np.abs(Operator(qc1.bind_parameters(v)).data)
            p1 = np.round(p1)
            cost = cost_f(adj1, adj2, p1)
            if cost < min_cost:
                p2 = p1
                if cost < 1.:
                    break
        #plt.matshow(p2, vmin=0, vmax=1)
        fig, ax = plt.subplots()
        g1_pos = graphviz_layout(g1)
        nx.draw(g1, g1_pos, with_labels=True, ax=ax)
        plot_solution_edges(g1, pos=g1_pos, perm=p2, ax=ax)
        #fig, axs = plt.subplots(1, 3, figsize=(12, 5))
        #fig.patch.set_facecolor('white')
        # axs[0].matshow(adj1)
        # axs[0].set_title('Adj $A$')
        # axs[1].matshow(adj2)
        # axs[1].set_title('Adj $B$')
        # axs[2].matshow(p2 @ adj1 @ p2.T)
        # axs[2].set_title('Permuted adj $PAP^T$')
        # rect = patches.Rectangle((-0.5, -0.5), len(adj2), len(adj2), linewidth=2,
        #                          edgecolor='red', facecolor='white', alpha=0.5)
        # axs[2].add_patch(rect)
        #plt.show()
        fig.savefig('/Users/uqasha17/Abhay/code_csiro/' + 'rough_benzene' + f'{i}'+ '.png', dpi=300)
        print(qc.num_qubits)


def molecule_substructure_match_vqd(adj1=None, adj2=None, max_trials=2, ansatz=ansatz):
    seed = 10283764
    algorithm_globals.random_seed = seed
    rng = np.random.default_rng(seed=seed)
    qc, params = ansatz(adj1, adj2)
    qi = QuantumInstance(Aer.get_backend('statevector_simulator'),
                         seed_transpiler=seed, seed_simulator=seed, shots=1024)
    optim = SLSQP(maxiter=1000)
    def trial():
        initial_point = (rng.uniform(size=len(qc.parameters)) - 0.5) * np.pi
        vqd = VQD(
            ansatz=qc,
            optimizer=optim,
            quantum_instance=qi,
            k=3,  # number of minima you want per trial
            initial_point=initial_point
        )
        obj = vqd.compute_eigenvalues(operator=observable(qc.num_qubits))
        return obj.eigenvalues, obj
    results = [trial() for _ in tqdm(range(max_trials))]
    all_minima = []
    for vals, obj in results:
        for val in vals:
            all_minima.append((val, obj))
    all_minima = sorted(all_minima, key=lambda x: x[0])
    result = all_minima[4][1]
    print("Optimal value:", result.optimal_value)
    qc1 = s4_ansatz('circular', qreg=(qc.num_qubits - 1) // 2, params=params)[0]
    sampled_params_dicts = sample_exact_thetas(result.optimal_parameters, n=32, seed=seed)
    min_cost = np.inf
    for v in sampled_params_dicts:
        p1 = np.abs(Operator(qc1.bind_parameters(v)).data)
        p1 = np.round(p1)
        cost = cost_f(adj1, adj2, p1)
        if cost < min_cost:
            p2 = p1
            if cost < 1.:
                break
    plt.matshow(p2, vmin=0, vmax=1)
    fig, ax = plt.subplots()
    g1_pos = graphviz_layout(g1)
    nx.draw(g1, g1_pos, with_labels=True, ax=ax)
    plot_solution_edges(g1, pos=g1_pos, perm=p2, ax=ax)
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    axs[0].matshow(adj1)
    axs[0].set_title('Adj $A$')
    axs[1].matshow(adj2)
    axs[1].set_title('Adj $B$')
    axs[2].matshow(p2 @ adj1 @ p2.T)
    axs[2].set_title('Permuted adj $PAP^T$')
    rect = patches.Rectangle((-0.5, -0.5), len(adj2), len(adj2), linewidth=2,
                             edgecolor='red', facecolor='white', alpha=0.5)
    axs[2].add_patch(rect)
    plt.savefig('/Users/uqasha17/Abhay/code_csiro/' + 'vqe_8_trial_pi_to_pi' + '.png', dpi=300)
    #plt.show()
    print(qc.num_qubits)


if __name__ == "__main__":
    # Example usage
    # g1 = nx.generators.erdos_renyi_graph(8, 0.35, seed=14) graph generator with probability of 0.35 or 35 % chance of each edge between nodes.
    # adj1 = np.array(nx.adjacency_matrix(g1).todense()) #creates sparse matrix (conection) .todense() converts the matrix forms
    # adj2 = checkboard(4)
    # g2 = nx.from_numpy_array(adj2)
    # Folder where the current script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Load adjacency matrices from PDB files
    adj1, pos, atom_symbols = pdb_to_adj_and_pos_exclude_hydrogen(f"{base_dir}" + "/pdb/benzene.pdb")
    adj2, pos_1, atom_symbols_1 = pdb_to_adj_and_pos(f"{base_dir}"+"/pdb/acetylene.pdb")
    adj1 = pad_to_pow2(adj1)
    adj2 = pad_to_pow2(adj2)
    g1 = nx.from_numpy_array(adj1)
    g2 = nx.from_numpy_array(adj2)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    nx.draw(g1, with_labels=True, ax=axs[0][0])
    nx.draw(g2, with_labels=True, ax=axs[0][1])
    axs[0][0].set_title('Source graph ($\\mathcal{A}$)')
    axs[0][1].set_title('Subgraph ($\\mathcal{B}$)')
    axs[1][0].set_title('Adjacency matrix ($A$)')
    axs[1][0].matshow(adj1)
    axs[1][1].set_title('Adjacency matrix ($B$)')
    axs[1][1].matshow(adj2)
    #plt.savefig('/Users/uqasha17/Abhay/code_csiro/graphs_1.png', dpi=300)
    plt.show()
    molecule_substructure_match(adj1, adj2, max_trials=2)
    #molecule_substructure_match_match(adj1, adj2, max_trials=2)
    sys.exit("Stopped here after visualization.")




