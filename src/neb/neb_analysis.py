import numpy as np
from pathlib import Path
import pandas as pd
from .util import *
from .neb_analysis import *


def bond_distances(images, i, j):
    """Compute the distance between atoms i and j for every image."""
    images = np.asarray(images)
    return np.linalg.norm(images[:, i] - images[:, j], axis=1)



def reaction_coordinate(images, center=0, atom_a=4, atom_b=5):
    """Compute q = d(center, atom_a) - d(center, atom_b)."""
    da = bond_distances(images, center, atom_a)
    db = bond_distances(images, center, atom_b)
    return da - db



def min_pair_distances(images):
    """Compute the minimum interatomic distance in each image."""
    images = np.asarray(images)
    distances = []

    for frame in images:
        d = np.linalg.norm(
            frame[:, None, :] - frame[None, :, :],
            axis=-1
        )
        np.fill_diagonal(d, np.inf)
        distances.append(np.min(d))

    return np.asarray(distances)



def final_analysis(data_dir, 
                   atoms, 
                   calculator, 
                   *, 
                   force_tol=0.08,
                   stage="ci", 
                   atom_center=0, 
                   atom_a=4, 
                   atom_b=5):
    """Analyze a force-converged CI-NEB checkpoint and export path data.

    Parameters
    ----------
    data_dir : str or Path
        Root results directory.
    atoms : sequence of str
        Ordered chemical symbols corresponding to geometry rows.
    calculator : callable
        Function taking (n_atoms, 3) coordinates and returning
        (energy_eV, gradient_eV_per_A).
    force_tol : float
        CI convergence threshold in eV/Angstrom.
    stage : str
        Subdirectory containing checkpoint.npz.
    atom_center, atom_a, atom_b : int
        Indices for distance and reaction coordinate measurements.

    Returns
    -------
    images, energies, summary, ts_idx, history : tuple
    """
    data_dir = Path(data_dir)
    atoms = tuple(atoms)
    folder = data_dir / stage
    checkpoint = folder / "checkpoint.npz"

    with np.load(checkpoint, allow_pickle=False) as saved:
        images = np.asarray(saved["images"], dtype=float).copy()
        history = np.asarray(saved["history"], dtype=float).copy()

    energies = np.array([float(calculator(im)[0]) for im in images])

    da = bond_distances(images, atom_center, atom_a)
    db = bond_distances(images, atom_center, atom_b)
    q = da - db

    steps = np.diff(images, axis=0).reshape(len(images) - 1, -1)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(steps, axis=1))))

    ts_idx = 1 + int(np.argmax(energies[1:-1]))
    relative = energies - energies[0]
    summary = pd.DataFrame({
        "image": np.arange(len(images)),
        "arc_A": arc,
        "q_A": q,
        "C_ClA_A": da,
        "C_ClB_A": db,
        "relative_energy_eV": relative,
    })

    folder.mkdir(parents=True, exist_ok=True)
    ts_dir = data_dir / "ts"
    ts_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(folder / "final_band.csv", index=False)
    np.savez(folder / "final_band.npz", images=images,
             energies_eV=energies, ts_idx=ts_idx)
    write_xyz(ts_dir / "ts_candidate.xyz", atoms, images[ts_idx],
              title=f"Highest-energy CI image {ts_idx}")

    with (folder / "final_path.xyz").open("w", encoding="utf-8") as stream:
        for i, im in enumerate(images):
            stream.write(coords_to_xyz(
                atoms, im, title=f"image={i} E_rel_eV={relative[i]:.9f}"
            ))

    return images, energies, summary, ts_idx, history