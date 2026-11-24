from __future__ import annotations
from html import escape
import numpy as np
import py3Dmol
from IPython.display import HTML, display
from .util import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import math


def show_images(atoms, images, titles=None, columns=4,
                panel_width=300, panel_height=300,
                linked=True, labels=False):
    """Display all molecular images in one linked 3D viewer."""

    atoms = tuple(atoms)
    images = np.asarray(images, dtype=float)

    if images.ndim == 2:
        images = images[None, :, :]

    n_images = len(images)

    titles = (
        [f"Image {i}" for i in range(n_images)]
        if titles is None else list(titles)
    )

    ncols = min(columns, n_images)
    nrows = (n_images + ncols - 1) // ncols

    width = panel_width * ncols
    height = panel_height * nrows

    # One viewer for the entire path
    view = py3Dmol.view(
        width=width,
        height=height,
        viewergrid=(nrows, ncols),
        linked=linked
    )

    view.setProjection("orthographic")
    view.setBackgroundColor("white")

    for i, coords in enumerate(images):
        row, col = divmod(i, ncols)
        panel = (row, col)

        view.addModel(
            coords_to_xyz(atoms, coords),
            "xyz",
            viewer=panel
        )

        view.setStyle(
            {},
            {
                "stick": {"radius": 0.13},
                "sphere": {"scale": 0.30}
            },
            viewer=panel
        )

        # Chlorine colors and equal radii
        for serial, color in ((4, "lightgreen"), (5, "cyan")):
            if serial < len(atoms) and atoms[serial] == "Cl":
                view.setStyle(
                    {"serial": serial},
                    {
                        "sphere": {
                            "radius": 0.65,
                            "color": color
                        },
                        "stick": {"radius": 0.13}
                    },
                    viewer=panel
                )

        # Atom labels at atom centers
        if labels:
            for j, (atom, xyz) in enumerate(zip(atoms, coords)):
                view.addLabel(
                    f"{atom}{j}",
                    {
                        "position": {
                            "x": float(xyz[0]),
                            "y": float(xyz[1]),
                            "z": float(xyz[2])
                        },
                        "fontSize": 11,
                        "fontColor": "black",
                        "fontFace": "Arial",
                        "showBackground": False,
                        "alignment": "center",
                        "screenOffset": {"x": 0, "y": 0}
                    },
                    viewer=panel
                )

        view.zoomTo(viewer=panel)

    # Fixed titles, overlaid on the entire viewer grid
    title_cells = "".join(
        f'<div style="text-align:center;padding:10px 4px;'
        f'font-size:16px;font-weight:600;color:black;">'
        f'{escape(str(title))}</div>'
        for title in titles
    )

    viewer_html = view.write_html()

    html = f"""
    <div style="
        position:relative;
        width:{width}px;
        height:{height}px;
        max-width:100%;
    ">
        {viewer_html}

        <div style="
            position:absolute;
            inset:0;
            display:grid;
            grid-template-columns:repeat({ncols}, 1fr);
            grid-template-rows:repeat({nrows}, 1fr);
            pointer-events:none;
            z-index:10;
        ">
            {title_cells}
        </div>
    </div>
    """

    display(HTML(html))
    return view


def show_molecules(atoms, reactant, product,
                   width=1000, height=350,
                   linked=True, labels=True):
    """Display reactant and product with synchronized rotation."""

    return show_images(
        atoms,
        [reactant, product],
        titles=["Reactant", "Product"],
        columns=2,
        panel_width=width // 2,
        panel_height=height,
        linked=linked,
        labels=labels
    )






def load_history(data_dir, stage):
    """Read [iteration, Fmax, barrier, peak image index] from checkpoint."""
    checkpoint = Path(data_dir) / stage / "checkpoint.npz"
    with np.load(checkpoint, allow_pickle=False) as data:
        history = np.asarray(data["history"], dtype=float).copy()
    return history


def animate_convergence(data_dir, stages=("neb", "ci"), *,
                        tolerance=None, stride=2, fps=12,
                        save_path=None, show=True):
    """Animate recorded NEB/CI Fmax and barrier *versus iteration*.

    `stages` may include one or both of ('neb', 'ci'). Their iterations are
    concatenated on a single horizontal *progress* axis; stage boundaries
    are marked. Uses real saved history, not interpolated energy profiles.

    Returns (animation, saved_path_or_None). Requires Pillow for saving GIF.
    """
    stages = tuple(stages)

    records = []
    boundaries = []
    labels = []
    running = 0
    for stage in stages:
        h = load_history(data_dir, stage)
        # Keep a monotonic global iteration count across stages, independent
        # of the stored iteration number (which resets to zero at CI).
        x = np.arange(running, running + len(h), dtype=float)
        records.append(np.column_stack((x, h[:, 1], h[:, 2], h[:, 3])))
        labels.append((stage.upper(), running, running + len(h) - 1))
        if running:
            boundaries.append(running - 0.5)
        running += len(h)

    rows = np.concatenate(records)
    xs, fs, es, peaks = rows.T

    fig, (axf, axe) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    line_f, = axf.plot([], [], lw=2, marker="o", markersize=2.2)
    line_e, = axe.plot([], [], lw=2, marker="o", markersize=2.2)
    dot_f, = axf.plot([], [], "o", markersize=7)
    dot_e, = axe.plot([], [], "o", markersize=7)

    def pad_ylim(ax, vals, zero=False):
        lo, hi = float(np.min(vals)), float(np.max(vals))
        span = max(hi - lo, abs(hi) * 0.02, 0.03)
        ax.set_ylim(min(0.0, lo - span * .08) if zero else lo - span * .12,
                    hi + span * .12)

    for ax in (axf, axe):
        ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
        ax.set_xlabel("Recorded iteration (NEB then CI)")
        ax.grid(alpha=.23)
        for boundary in boundaries:
            ax.axvline(boundary, color="0.5", lw=1, ls=":")
    axf.set(ylabel="Fmax (eV/Å)")
    axe.set(ylabel="Highest image − reactant (eV)")

    axf.set_title("NEB force convergence", y=1.08, pad=10)
    axe.set_title("Barrier estimate", y=1.08, pad=10)
    pad_ylim(axf, fs, zero=True)
    pad_ylim(axe, es)

    if tolerance is not None:
        if isinstance(tolerance, dict):
            # Separate stage-specific tolerance line segments.
            for stage, start, end in labels:
                value = tolerance.get(stage.lower())
                if value is not None:
                    axf.hlines(value, start, end, color="tab:red", ls="--", lw=1.1)
        else:
            axf.axhline(float(tolerance), color="tab:red", ls="--", lw=1.1)

    # Mark the region for each stage without inventing iterations.
    for stage, start, end in labels:
        axf.text((start + end)/2, 1.02, stage,
                 transform=axf.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=9)

    status = fig.suptitle("", fontsize=12)
    indices = list(range(0, len(xs), stride))
    if indices[-1] != len(xs) - 1:
        indices.append(len(xs) - 1)

    def update(frame):
        j = indices[frame]
        line_f.set_data(xs[:j + 1], fs[:j + 1])
        line_e.set_data(xs[:j + 1], es[:j + 1])
        dot_f.set_data([xs[j]], [fs[j]])
        dot_e.set_data([xs[j]], [es[j]])
        current_stage = next(stage for stage, a, b in labels if a <= xs[j] <= b)
        status.set_text(f"{current_stage} | step {j + 1}/{len(xs)} | "
                        f"Fmax = {fs[j]:.4f} eV/Å | "
                        f"barrier = {es[j]:.4f} eV | peak = {int(peaks[j])}")
        return line_f, line_e, dot_f, dot_e, status

    ani = FuncAnimation(fig, update, frames=len(indices), interval=1000 / fps,
                        blit=False, repeat=False)
    fig.tight_layout(rect=(0, 0, 1, .91))

    saved = None
    if save_path is not None:
        saved = Path(save_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        if saved.suffix.lower() != ".gif":
            raise ValueError("save_path must end with .gif")
        ani.save(str(saved), writer=PillowWriter(fps=fps), dpi=100)
    if show:
        try:
            from IPython.display import display, HTML
            display(HTML(ani.to_jshtml()))
        except ImportError:
            plt.show()
    plt.close(fig)
    return ani, saved


def show_final_images(data_dir, atoms, *, stage="ci", columns=4,
                      panel_width=300, panel_height=280, labels=True,
                      linked=True, titles=None):
    """One py3Dmol viewer with synchronized rotation across *all* rows.

    Reads final geometry from stage/checkpoint.npz; returns py3Dmol viewer.
    Requires py3Dmol and your existing util.coords_to_xyz.
    """
    import py3Dmol
    from IPython.display import publish_display_data
    from util import coords_to_xyz

    ckpt = Path(data_dir) / stage / "checkpoint.npz"
    with np.load(ckpt, allow_pickle=False) as data:
        images = np.asarray(data["images"], dtype=float).copy()

    atoms = tuple(atoms)
    if (images.ndim != 3 or images.shape[1:] != (len(atoms), 3)
            or len(images) == 0 or not np.isfinite(images).all()):
        raise ValueError("Invalid molecular images in checkpoint")
    if columns < 1 or panel_width <= 0 or panel_height <= 0:
        raise ValueError("Invalid panel layout")
    n = len(images)
    cols = min(columns, n)
    rows = math.ceil(n / cols)
    width, height = panel_width * cols, panel_height * rows

    if titles is None:
        titles = [f"Image {i}" for i in range(n)]
    if len(titles) != n:
        raise ValueError("titles length must match image count")

    view = py3Dmol.view(width=width, height=height,
                        viewergrid=(rows, cols), linked=linked)
    view.setBackgroundColor("white")
    view.setProjection("orthographic")
    for i, coords in enumerate(images):
        panel = divmod(i, cols)
        view.addModel(coords_to_xyz(atoms, coords), "xyz", viewer=panel)
        view.setStyle({}, {"sphere": {"scale": 0.30},
                           "stick": {"radius": 0.13}}, viewer=panel)
        for atom_index, color in ((4, "green"), (5, "cyan")):
            if atom_index < len(atoms) and atoms[atom_index] == "Cl":
                view.setStyle({"serial": atom_index},
                              {"sphere": {"radius": 0.65, "color": color},
                               "stick": {"radius": 0.13}}, viewer=panel)
        if labels:
            for atom_index, (atom, xyz) in enumerate(zip(atoms, coords)):
                view.addLabel(f"{atom}{atom_index}", {
                    "position": dict(zip(("x", "y", "z"), map(float, xyz))),
                    "fontSize": 11, "fontColor": "black",
                    "showBackground": False, "alignment": "center",
                    "screenOffset": {"x": 0, "y": 0},
                }, viewer=panel)
        view.zoomTo(viewer=panel)

    # Fixed overlay aligns with all panels in a single multirow grid.
    title_cells = "".join(
        '<div style="text-align:center;padding:8px 4px;'
        'font:600 15px Arial,sans-serif;color:#171717;">'
        f'{escape(str(t))}</div>' for t in titles
    )
    html = (
        f'<div style="position:relative;width:{width}px;height:{height}px;">'
        + view._make_html()
        + '<div style="position:absolute;inset:0;display:grid;'
          f'grid-template-columns:repeat({cols},minmax(0,1fr));'
          f'grid-template-rows:repeat({rows},minmax(0,1fr));'
          'pointer-events:none;z-index:10;">'
        + title_cells + '</div></div>'
    )
    publish_display_data({"application/3dmoljs_load.v0": html, "text/html": html})
    return view


def plot_final_energy_profile(data_dir, *, stage="ci", save_path=None):
    """Plot FINAL energy versus normalized cumulative path length.

    Requires stage/final_band.npz generated by the postprocessing module.
    This is one final profile, not a historical profile animation.
    """
    path = Path(data_dir) / stage / "final_band.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run final_analysis() to save final energies first."
        )
    with np.load(path, allow_pickle=False) as data:
        images = np.asarray(data["images"], dtype=float)
        energies = np.asarray(data["energies_eV"], dtype=float)
    if images.ndim != 3 or energies.shape != (len(images),):
        raise ValueError("Invalid saved final band")
    distances = np.linalg.norm(np.diff(images, axis=0).reshape(len(images)-1, -1), axis=1)
    arc = np.r_[0., np.cumsum(distances)]
    t = arc / arc[-1] if arc[-1] > 0 else np.linspace(0, 1, len(images))
    relative = energies - energies[0]
    peak = int(np.argmax(energies[1:-1])) + 1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, relative, "o-", lw=2)
    ax.scatter([t[peak]], [relative[peak]], marker="*", s=190,
               color="tab:red", zorder=4, label=f"Peak image {peak}")
    ax.set(xlabel="Normalized cumulative path length",
           ylabel="Relative electronic energy (eV)",
           title=f"{stage.upper()} final energy profile")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.show()
    return fig, ax
