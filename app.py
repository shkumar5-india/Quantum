from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

def compute_energy(molecule, bond_length):
    if molecule == "H2":
        atom_string = f"H 0 0 0; H 0 0 {bond_length}"
    else:
        atom_string = f"Li 0 0 0; H 0 0 {bond_length}"

    driver = PySCFDriver(atom=atom_string, basis="sto3g", unit=DistanceUnit.ANGSTROM)
    problem = driver.run()

    es_problem = ElectronicStructureProblem(problem.hamiltonian)
    second_q_op = es_problem.second_q_ops()[0]

    mapper = JordanWignerMapper()
    qubit_op = mapper.map(second_q_op)

    num_particles = es_problem.num_particles
    num_spatial_orbitals = es_problem.num_spatial_orbitals

    hf_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    ansatz = UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=hf_state)

    optimizer = SLSQP(maxiter=80)
    estimator = Estimator()

    vqe = VQE(estimator, ansatz, optimizer=optimizer)
    result = vqe.compute_minimum_eigenvalue(qubit_op)

    return result.eigenvalue.real

@app.route("/run_vqe", methods=["POST"])
def run_vqe():
    data = request.json
    molecule = data["molecule"]

    bond_lengths = np.linspace(0.5, 2.5, 6)
    energies = [compute_energy(molecule, bl) for bl in bond_lengths]

    # Plot
    plt.figure()
    plt.plot(bond_lengths, energies, marker='o')
    plt.xlabel("Bond Length (Å)")
    plt.ylabel("Ground State Energy (Hartree)")
    plt.title(f"Potential Energy Curve of {molecule}")
    plt.grid()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({
        "molecule": molecule,
        "bond_lengths": bond_lengths.tolist(),
        "energies": energies,
        "plot": plot_url
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
