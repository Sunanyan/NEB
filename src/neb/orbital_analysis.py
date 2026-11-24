from pathlib import Path
import re
import numpy as np
from html import escape
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import py3Dmol
from .util import *
from .orca import *


EH_TO_EV = 27.211386245988


def parse_orbitals(output):
    """Read the last ORBITAL ENERGIES table (restricted closed shell)."""
    lines = Path(output).read_text(errors="replace").splitlines()
    headers = [i for i, line in enumerate(lines) if line.strip() == "ORBITAL ENERGIES"]
    records = []
    started = False
    for line in lines[headers[-1] + 1:]:
        pieces = line.split()
        if len(pieces) >= 4 and pieces[0].isdigit():
            try:
                idx = int(pieces[0])
                occ = float(pieces[1])
                eh = float(pieces[2].replace('D', 'E'))
                ev = float(pieces[3].replace('D', 'E'))
            except ValueError:
                if started:
                    break
                continue
            records.append((idx, occ, eh, ev))
            started = True
        elif started:
            break
    if not records:
        raise ValueError(f"Could not parse orbital table: {output}")
    # Only support restricted calculations, not separate alpha/beta tables.
    occupied = [r for r in records if r[1] > 1.5]
    virtual = [r for r in records if r[1] < 0.5]
    if any(0.5 <= r[1] <= 1.5 for r in records) or not occupied or not virtual:
        raise ValueError("Expected restricted closed-shell orbital occupations (2/0).")
    homo = max(occupied, key=lambda r:r[0])
    lumo = min((r for r in virtual if r[0] > homo[0]), key=lambda r:r[0])
    if lumo[0] != homo[0] + 1:
        raise ValueError("Nonconsecutive HOMO/LUMO indices; inspect ORCA output")
    return {"homo_idx":homo[0], "lumo_idx":lumo[0],
            "homo_eV":homo[3], "lumo_eV":lumo[3]}


def _input_with_cubes(coords, atoms, mo_homo, mo_lumo, *, method, basis, charge, mult, nprocs, grid):
    if mult != 1:
        raise ValueError("This example expects closed-shell multiplicity=1")
    lines = [
        f"! {method} {basis} TightSCF SP",
        f"%pal nprocs {nprocs} end",
        "%plots",
        f"  dim1 {grid}", f"  dim2 {grid}", f"  dim3 {grid}",
        "  Format Gaussian_Cube",
        f'  MO("homo.cube",{mo_homo},0);',
        f'  MO("lumo.cube",{mo_lumo},0);',
        "end", "", f"* xyz {charge} {mult}",
    ]
    for atom, (x, y, z) in zip(atoms, coords):
        lines.append(f"{atom:2s} {x: .12f} {y: .12f} {z: .12f}")
    return "\n".join(lines + ["*", ""])


def compute_orbitals(images, atoms, data_dir, *, executable, method,
                     basis, charge, mult, nprocs=1, grid=50,
                     homo_idx=21, lumo_idx=22, overwrite=False):
    """Compute one ORCA SP and HOMO/LUMO cube pair per CI image.

    Note: HOMO/LUMO indices are initial guesses, verified against ORCA output.
    If a different index is discovered, rerun with those indices.
    """
    images = np.asarray(images, dtype=float)
    atoms = tuple(atoms)
    if images.ndim != 3 or images.shape[1:] != (len(atoms), 3):
        raise ValueError("images must have shape (n_images, n_atoms, 3)")
    root = Path(data_dir) / "orbitals_ci"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    from tqdm.auto import tqdm
    for i, coords in enumerate(tqdm(images, desc="ORCA MO cubes", unit="image")):
        folder = root / f"image_{i:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / "calc.out"
        cubes = [folder / "homo.cube", folder / "lumo.cube"]
        config = _input_with_cubes(coords, atoms, homo_idx, lumo_idx,
            method=method, basis=basis, charge=charge, mult=mult,
            nprocs=nprocs, grid=grid)
        inp = folder / "calc.inp"
        valid = (not overwrite and out.exists() and
                 all(p.exists() for p in cubes) and inp.exists() and
                 inp.read_text() == config and
                 "ORCA TERMINATED NORMALLY" in out.read_text(errors="replace"))
        if not valid:
            for p in cubes:
                p.unlink(missing_ok=True)
            run_orca(folder, config, executable=executable)
        if not all(p.exists() for p in cubes):
            raise FileNotFoundError(f"ORCA did not produce cube files in {folder}; inspect calc.out")
        orbital = parse_orbitals(out)
        if (orbital["homo_idx"], orbital["lumo_idx"]) != (homo_idx, lumo_idx):
            raise ValueError(f"Image {i}: ORCA reports HOMO/LUMO "
                f"{orbital['homo_idx']}/{orbital['lumo_idx']}, but generated cubes "
                f"for {homo_idx}/{lumo_idx}. Re-run using the reported indices.")
        rows.append([orbital["homo_eV"], orbital["lumo_eV"]])
    energies = np.array(rows)
    np.savez(root / "orbital_energies.npz", energies_eV=energies)
    return energies


def plot_orbital_energies(energies, *, save_path=None):
    energies = np.asarray(energies, dtype=float)
    x = np.linspace(0, 1, len(energies))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(x, energies[:,0], 'o-', label='HOMO')
    axs[0].plot(x, energies[:,1], 's-', label='LUMO')
    axs[0].set(xlabel='Normalized image index', ylabel='Orbital energy (eV)', title='Frontier orbital energies')
    axs[0].legend()
    axs[1].plot(x, energies[:,1] - energies[:,0], 'o-')
    axs[1].set(xlabel='Normalized image index', ylabel='Gap (eV)', title='HOMO–LUMO gap')
    for ax in axs:
        ax.grid(alpha=.2)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    return fig, axs


def show_orbitals(images, atoms, data_dir, image_index=4, *,
                  isoval=0.035, width=1000, height=360,
                  save_html=None):

    images = np.asarray(images, dtype=float)
    folder = Path(data_dir) / "orbitals_ci" / f"image_{image_index:02d}"

    view = py3Dmol.view(
        width=width,
        height=height,
        viewergrid=(1, 2),
        linked=True,
        js="https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"
    )

    view.setBackgroundColor("white")
    view.setProjection("orthographic")

    xyz = coords_to_xyz(atoms, images[image_index])

    for col, name in enumerate(("homo", "lumo")):
        panel = (0, col)
        cube_path = folder / f"{name}.cube"

        cube_data = cube_path.read_text()

        view.addModel(xyz, "xyz", viewer=panel)
        view.setStyle(
            {},
            {
                "stick": {"radius": 0.10},
                "sphere": {"scale": 0.23}
            },
            viewer=panel
        )

        for sign, color in ((1, "blue"), (-1, "red")):
            view.addVolumetricData(
                cube_data,
                "cube",
                {
                    "isoval": sign * isoval,
                    "color": color,
                    "opacity": 0.65,
                    "smoothness": 3
                },
                viewer=panel
            )

        view.zoomTo(viewer=panel)

    if save_html is not None:
        save_html = Path(save_html)
        save_html.parent.mkdir(parents=True, exist_ok=True)
        view.write_html(str(save_html))

    view.show()
    return view



def show_all_orbitals(
    images, atoms, data_dir,
    images_per_row=3,
    panel_width=240,
    panel_height=260,
    isoval=0.035,
    opacity=0.65,
    linked=True,
    labels=False
):
    """Display HOMO and LUMO for every image in one linked 3D viewer."""

    images = np.asarray(images, dtype=float)
    atoms = tuple(atoms)
    n_images = len(images)

    if images.shape[1:] != (len(atoms), 3):
        raise ValueError("Invalid molecular coordinate shape")

    n_groups = min(images_per_row, n_images)
    ncols = 2 * n_groups
    nrows = int(np.ceil(n_images / n_groups))

    width = panel_width * ncols
    height = panel_height * nrows

    view = py3Dmol.view(
        width=width,
        height=height,
        viewergrid=(nrows, ncols),
        linked=linked
    )

    view.setProjection("orthographic")
    view.setBackgroundColor("white")

    headings = [""] * (nrows * ncols)

    for i, coords in enumerate(images):
        row = i // n_groups
        group_col = i % n_groups

        xyz = coords_to_xyz(atoms, coords)

        folder = (
            Path(data_dir)
            / "orbitals_ci"
            / f"image_{i:02d}"
        )

        for j, orbital in enumerate(("homo", "lumo")):
            col = 2 * group_col + j
            panel = (row, col)

            headings[row * ncols + col] = (
                f"Image {i} · {orbital.upper()}"
            )

            cube_file = folder / f"{orbital}.cube"

            if not cube_file.exists():
                raise FileNotFoundError(
                    f"Missing orbital cube: {cube_file}"
                )

            cube = cube_file.read_text()

            # Molecular geometry
            view.addModel(xyz, "xyz", viewer=panel)

            view.setStyle(
                {},
                {
                    "stick": {"radius": 0.10},
                    "sphere": {"scale": 0.23}
                },
                viewer=panel
            )

            # Positive and negative orbital phases
            for sign, color in (
                (1, "blue"),
                (-1, "red")
            ):
                view.addVolumetricData(
                    cube,
                    "cube",
                    {
                        "isoval": sign * isoval,
                        "color": color,
                        "opacity": opacity,
                        "smoothness": 3
                    },
                    viewer=panel
                )

            if labels:
                for k, (atom, position) in enumerate(
                    zip(atoms, coords)
                ):
                    view.addLabel(
                        f"{atom}{k}",
                        {
                            "position": {
                                "x": float(position[0]),
                                "y": float(position[1]),
                                "z": float(position[2])
                            },
                            "fontSize": 10,
                            "fontColor": "black",
                            "showBackground": False,
                            "alignment": "center"
                        },
                        viewer=panel
                    )

            view.zoomTo(viewer=panel)

    # Fixed titles above each 3D panel
    titles_html = "".join(
        f'<div style="text-align:center;'
        f'font-size:14px;font-weight:600;'
        f'padding-top:10px;color:black;">'
        f'{escape(title)}</div>'
        for title in headings
    )

    html = f"""
    <div style="position:relative;
                width:{width}px;
                height:{height}px;">

        {view.write_html()}

        <div style="
            position:absolute;
            inset:0;
            display:grid;
            grid-template-columns:repeat({ncols},1fr);
            grid-template-rows:repeat({nrows},1fr);
            pointer-events:none;
            z-index:10;
        ">
            {titles_html}
        </div>
    </div>
    """

    display(HTML(html))
    return view
