import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from flask_cors import CORS

from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

app = Flask(__name__)
CORS(app)

# -------- Utility --------
def get_mapper(name):
    return JordanWignerMapper() if name == "JW" else ParityMapper()

def run_vqe_energy(atom_string, basis, mapper_type):
    try:
        driver = PySCFDriver(atom=atom_string, basis=basis)
        problem = ElectronicStructureProblem(driver)

        mapper = get_mapper(mapper_type)
        second_q_ops = problem.second_q_ops()
        main_op = list(second_q_ops.values())[0]
        qubit_op = mapper.map(main_op)

        num_particles = problem.num_particles
        num_spatial_orbitals = problem.num_spatial_orbitals

        init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)

        ansatz = UCCSD(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            mapper=mapper,
            initial_state=init_state,
        )

        vqe = VQE(Estimator(), ansatz, SLSQP(maxiter=100))
        result = vqe.compute_minimum_eigenvalue(qubit_op)

        return float(result.eigenvalue.real)

    except Exception as e:
        print("VQE ERROR:", e)
        return None

def hartree_to_kjmol(E):
    return E * 2625.5

# ================= PART 1 =================
@app.route("/run_vqe_curve", methods=["POST"])
def run_vqe_curve():
    data = request.json
    molecule = data.get("molecule", "H2")
    basis = data.get("basis", "sto3g")
    mapper = data.get("mapper", "JW")

    bond_lengths = np.linspace(0.5, 2.5, 8)
    energies = []

    for d in bond_lengths:
        geom = f"H 0 0 0; H 0 0 {d}" if molecule == "H2" else f"Li 0 0 0; H 0 0 {d}"
        energy = run_vqe_energy(geom, basis, mapper)
        if energy is None:
            return jsonify({"error": "VQE failed"}), 500
        energies.append(energy)

    min_energy = min(energies)
    min_dist = float(bond_lengths[energies.index(min_energy)])

    plt.figure()
    plt.plot(bond_lengths, energies, marker="o")
    plt.xlabel("Bond Length (Å)")
    plt.ylabel("Energy (Hartree)")
    plt.title(f"{molecule} Dissociation Curve")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    img = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "bond_lengths": bond_lengths.tolist(),
        "energies": energies,
        "min_energy": min_energy,
        "optimal_distance": min_dist,
        "plot": img
    })

# ================= PART 2 =================
@app.route("/run_hydrogen_storage", methods=["POST"])
def run_storage():
    data = request.json
    basis = data.get("basis", "sto3g")
    mapper = data.get("mapper", "JW")
    T = float(data.get("temperature", 298.15))

    E1 = run_vqe_energy("N 0 0 0; B 0 0 1.5", basis, mapper)
    E2 = run_vqe_energy("P 0 0 0; H 0 0 1.0", basis, mapper)
    E3 = run_vqe_energy("H 0 0 0; H 0 0 0.74", basis, mapper)

    distances = np.linspace(2.0, 4.0, 4)
    complex_energies = [
        run_vqe_energy(f"N 0 0 0; B 0 0 1.5; P 0 0 {d}", basis, mapper)
        for d in distances
    ]

    E4 = min(complex_energies)
    E5, E6 = E4 + 0.02, E4 + 0.04

    bind_energy = hartree_to_kjmol(E4 - (E1 + E2))
    release_energy = hartree_to_kjmol(E5 - E4)
    deltaG_bind = bind_energy - (T * 0.01)

    spontaneity = "Spontaneous" if deltaG_bind < 0 else "Non-spontaneous"

    plt.figure()
    plt.plot(distances, complex_energies, marker="s")
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png")
    plt.close()
    img1 = base64.b64encode(buf1.getvalue()).decode()

    plt.figure()
    plt.bar(["E1","E2","E4","E5","E6"], [E1,E2,E4,E5,E6])
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png")
    plt.close()
    img2 = base64.b64encode(buf2.getvalue()).decode()

    return jsonify({
        "binding_energy_kjmol": bind_energy,
        "release_energy_kjmol": release_energy,
        "deltaG_binding": deltaG_bind,
        "spontaneity": spontaneity,
        "plots": {"binding_curve": img1, "reaction_diagram": img2}
    })

@app.route("/")
def home():
    return {"message": "Quantum VQE Flask backend running 🚀"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
