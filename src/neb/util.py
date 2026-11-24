import numpy as np
from pathlib import Path


def coords_to_xyz(atoms, coords, title=""):
    """Return an XYZ-formatted string for (n_atoms, 3) Cartesian coordinates.

    Parameters
    ----------
    atoms : sequence of str
        Atom labels.
    coords : array-like, shape (n_atoms, 3)
        Cartesian coordinates of the atoms.
    title : str, optional
        Title line for the XYZ file.

    Returns
    -------
    str
        XYZ-formatted string.
    """
    atoms = tuple(atoms)
    coords = np.asarray(coords)
    lines = [str(len(atoms)), str(title)]
    lines.extend(
        f"{atom:2s} {x: .10f} {y: .10f} {z: .10f}"
        for atom, (x, y, z) in zip(atoms, coords)
    )
    return "\n".join(lines) + "\n"


def write_xyz(path, atoms, coords, title=""):
    """Write coordinates to an XYZ file and return its path.
    
    Parameters
    ----------
    path : str or Path
        Path to the XYZ file.
    atoms : sequence of str
        Atom labels.
    coords : array-like, shape (n_atoms, 3)
        Cartesian coordinates of the atoms.
    title : str, optional
        Title line for the XYZ file.

    Returns
    -------
    Path
        Path to the written XYZ file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(coords_to_xyz(atoms, coords, title), encoding="utf-8")
    return path


def read_xyz(path, atoms=None):
    """Read XYZ coordinates from a file and optionally validate atom labels.

    Parameters
    ----------
    path : str or Path
        Path to the XYZ file.
    atoms : sequence of str, optional
        Ordered atom labels to validate against the file.

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
        Cartesian coordinates of the atoms.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    n = int(lines[0].strip())
    labels, coords = [], []
    for line in lines[2:2 + n]:
        fields = line.split()
        labels.append(fields[0])
        coords.append([float(x) for x in fields[1:4]])
    return np.asarray(coords, dtype=float)

