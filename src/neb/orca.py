from pathlib import Path
import shutil
import subprocess
import numpy as np
from .util import *


def check_orca(executable="orca"):
    """Return the ORCA executable path."""
    path = Path(executable).expanduser()
    
    if path.is_absolute() or path.parent != Path("."):
        return str(path.resolve())
    
    found = shutil.which(executable)
    return found


def orca_input(coords, atoms, task="Opt", *,
               method="B3LYP", basis="def2-SVP",
               charge=-1, mult=1, nprocs=4):
    """Generate an ORCA input string."""
    coords = np.asarray(coords, dtype=float)
    atoms = tuple(atoms)

    lines = [
        f"! {method} {basis} TightSCF {task}",
        f"%pal nprocs {nprocs} end",
        "",
        f"* xyz {charge} {mult}",
    ]

    for atom, (x, y, z) in zip(atoms, coords):
        lines.append(
            f"{atom:2s} {x: .12f} {y: .12f} {z: .12f}"
        )

    return "\n".join(lines + ["*", ""])


def run_orca(job_dir, input_text, *,
             executable="orca", stem="calc"):
    """Execute ORCA and return the output file path."""
    exe = check_orca(executable)

    job_dir = Path(job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)

    inp = job_dir / f"{stem}.inp"
    out = job_dir / f"{stem}.out"

    inp.write_text(input_text, encoding="utf-8")

    with out.open("w") as stream:
        proc = subprocess.run(
            [exe, inp.name],
            cwd=job_dir,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )

    contents = out.read_text(errors="replace")

    if (proc.returncode != 0 or
            "ORCA TERMINATED NORMALLY" not in contents):
        raise RuntimeError(
            f"ORCA failed: {out}\n"
            f"Exit code: {proc.returncode}"
        )

    return out


def optimize_endpoint(name, guess, atoms, data_dir, *,
                      executable="orca", method="B3LYP",
                      basis="def2-SVP", charge=-1,
                      mult=1, nprocs=4):
    """Optimize an endpoint, reusing matching converged results."""
    job = Path(data_dir) / "endpoints" / name
    geom_file = job / "calc.xyz"
    out_file = job / "calc.out"
    inp_file = job / "calc.inp"

    input_text = orca_input(
        guess, atoms, "Opt",
        method=method,
        basis=basis,
        charge=charge,
        mult=mult,
        nprocs=nprocs,
    )

    # Reuse only results from the same input
    if all(p.exists() for p in
           (geom_file, out_file, inp_file)):
        contents = out_file.read_text(errors="replace")

        if (inp_file.read_text() == input_text and
                "THE OPTIMIZATION HAS CONVERGED" in contents and
                "ORCA TERMINATED NORMALLY" in contents):
            return read_xyz(geom_file, atoms)

    # Prevent reuse of stale geometry
    if geom_file.exists():
        geom_file.unlink()

    out = run_orca(
        job, input_text,
        executable=executable,
    )

    contents = out.read_text(errors="replace")

    if "THE OPTIMIZATION HAS CONVERGED" not in contents:
        raise RuntimeError(
            f"Optimization did not converge: {out}"
        )

    if not geom_file.exists():
        raise FileNotFoundError(
            f"Optimized geometry not found: {geom_file}"
        )

    return read_xyz(geom_file, atoms)




import hashlib
import json
import re

EH_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
GRAD_FACTOR = EH_TO_EV / BOHR_TO_ANG

FLOAT_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?$"
)


def section_numbers(lines, heading, count):
    """Extract numeric values following a section heading."""
    locations = [
        i for i, line in enumerate(lines)
        if heading.lower() in line.lower()
    ]

    if not locations:
        raise ValueError(f"Cannot locate .engrad heading: {heading}")

    values = []

    for line in lines[locations[-1] + 1:]:
        s = line.strip()

        if s.startswith("#") or not s:
            continue

        if FLOAT_RE.fullmatch(s):
            values.append(
                float(s.replace("D", "E").replace("d", "e"))
            )

            if len(values) == count:
                return np.asarray(values, dtype=float)

        elif values:
            raise ValueError(
                f"Unexpected .engrad content after {heading}: {s[:60]}"
            )

    raise ValueError(
        f"Incomplete .engrad section {heading}: "
        f"got {len(values)}, expected {count}"
    )


def parse_engrad(filename, n_atoms):
    """Parse ORCA .engrad into energy (eV) and gradient (eV/Å)."""
    lines = Path(filename).read_text(
        errors="replace"
    ).splitlines()

    n = int(section_numbers(lines, "Number of atoms", 1)[0])

    if n != n_atoms:
        raise ValueError(
            f"Expected {n_atoms} atoms, .engrad reports {n}"
        )

    energy = section_numbers(
        lines, "current total energy in Eh", 1
    )[0]

    gradient = section_numbers(
        lines, "current gradient in Eh/bohr", 3 * n
    ).reshape(n, 3)

    if not np.isfinite(energy) or not np.isfinite(gradient).all():
        raise ValueError("Non-finite ORCA energy or gradient")

    return float(energy * EH_TO_EV), gradient * GRAD_FACTOR


def energy_gradient(coords, atoms, data_dir, *,
                    executable="orca", method="B3LYP",
                    basis="def2-SVP", charge=-1,
                    mult=1, nprocs=4):
    """Calculate energy and gradient with ORCA, using disk caching.

    Returns
    -------
    energy : float
        Energy in eV.
    gradient : ndarray, shape (n_atoms, 3)
        Gradient in eV/Å.
    """
    atoms = tuple(atoms)
    coords = np.asarray(coords, dtype=np.float64)

    settings = dict(
        method=method,
        basis=basis,
        charge=charge,
        mult=mult,
        nprocs=nprocs
    )

    input_text = orca_input(
        coords, atoms, task="EnGrad", **settings
    )

    # Identify calculation by geometry and input settings.
    fingerprint = {
        "input": input_text,
        "atoms": atoms,
        "method": method,
        "basis": basis,
        "charge": charge,
        "multiplicity": mult,
    }

    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode()
    ).hexdigest()[:20]

    job = Path(data_dir) / "orca_cache" / digest
    cache = job / "result.npz"

    if cache.exists():
        with np.load(cache) as result:
            energy = float(result["energy"])
            gradient = result["gradient"].copy()

        if (
            np.isfinite(energy)
            and gradient.shape == coords.shape
            and np.isfinite(gradient).all()
        ):
            return energy, gradient

        raise ValueError(f"Invalid cached ORCA result: {cache}")

    # Execute ORCA.
    out = run_orca(
        job,
        input_text,
        executable=executable,
        stem="calc"
    )

    engrad_file = job / "calc.engrad"

    if not engrad_file.exists():
        raise FileNotFoundError(
            f"Missing .engrad after ORCA calculation: {out}"
        )

    energy, gradient = parse_engrad(
        engrad_file,
        n_atoms=len(atoms)
    )

    np.savez(
        cache,
        energy=energy,
        gradient=gradient
    )

    return energy, gradient
