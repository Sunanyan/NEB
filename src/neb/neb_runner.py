from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from .neb_core import *


def checkpoint_write(stage, 
                     images, 
                     history, 
                     endpoint_e, 
                     data_dir, 
                     *,
                     config=None, 
                     converged=False):
    """Atomically save a full NEB state after each iteration."""
    folder = Path(data_dir) / stage
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "checkpoint.npz"
    temp = folder / "checkpoint.tmp.npz"
    payload = dict(
        images=np.asarray(images),
        history=np.asarray(history).reshape(-1, 4),
        endpoint_e=np.asarray(endpoint_e),
        converged=np.array(bool(converged)),
    )
    if config is not None:
        payload["config_json"] = np.array(json.dumps(config, sort_keys=True))
    np.savez(temp, **payload)
    temp.replace(path)
    return path


def _draw_history(history, tolerance, stage, handle=None, save_to=None):
    """Update a Jupyter display in place; optionally persist a PNG."""
    import matplotlib.pyplot as plt

    h = np.asarray(history, dtype=float).reshape(-1, 4)
    if len(h) == 0:
        return handle

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4))
    iteration, fmax, barrier = h[:, 0], h[:, 1], h[:, 2]
    axs[0].plot(iteration, fmax, "o-", markersize=3, linewidth=1.4)
    axs[0].axhline(tolerance, color="tab:red", linestyle="--", linewidth=1,
                   label=f"tol = {tolerance:g}")
    axs[0].set(xlabel="Iteration", ylabel="Fmax (eV/Å)", title="NEB force convergence")
    axs[0].legend(fontsize=8)
    axs[1].plot(iteration, barrier, "o-", markersize=3, linewidth=1.4)
    axs[1].set(xlabel="Iteration", ylabel="Barrier estimate (eV)",
               title="Highest image - reactant energy")
    for ax in axs:
        ax.grid(alpha=0.25)
    fig.suptitle(f"{stage.upper()} · iteration {int(iteration[-1])}", fontsize=12)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=130, bbox_inches="tight")
    if handle is not False:
        try:
            from IPython.display import display
            if handle is None:
                handle = display(fig, display_id=True)
            else:
                handle.update(fig)
        except Exception:
            handle = False
    plt.close(fig)
    return handle


def run_band(stage, starting_images, calculator, data_dir, *,
             max_iter=120, tol=0.15, climb=False,
             step=0.002, k=2.0, max_move=0.025,
             resume=True, live_plot=True, plot_every=5,
             report_every=5, checkpoint_config=None):
    """Optimize a band with live iteration/image progress and convergence plots.

    Returns (images, history, converged), where history columns are
    [iteration, Fmax in eV/Å, barrier estimate in eV, highest-energy image index].

    Notes
    -----
    The image progress bar updates *after* a calculator call completes. While
    ORCA is running, its last 'Computing image ...' status identifies the job.
    Endpoint gradient arrays are zeros, since endpoints are fixed.
    The iteration history describes geometries *before* each nonconverged step.
    """
    stage = str(stage)
    folder = Path(data_dir) / stage
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = folder / "checkpoint.npz"
    input_images = np.asarray(starting_images, dtype=np.float64)
    config = dict(stage=stage, shape=list(input_images.shape), max_iter=int(max_iter),
                  tol=float(tol), climb=bool(climb), step=float(step),
                  k=float(k), max_move=float(max_move))
    if checkpoint_config:
        config["calculator_settings"] = checkpoint_config

    if resume and checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            images = saved["images"].copy()
            history = saved["history"].reshape(-1, 4).tolist()
            endpoint_e = saved["endpoint_e"].copy()
            was_converged = bool(saved["converged"]) if "converged" in saved else False
            saved_config = str(saved["config_json"]) if "config_json" in saved else None
        tqdm.write(f"[{stage.upper()}] Resuming after {len(history)} recorded iterations")
        if was_converged:
            tqdm.write(f"[{stage.upper()}] Checkpoint already marked converged")
            return images, np.asarray(history, dtype=float).reshape(-1, 4), True
    else:
        images = input_images.copy()
        history = []
        tqdm.write(f"[{stage.upper()}] Calculating fixed endpoint energies...")
        endpoint_e = np.array([float(calculator(images[0])[0]),
                               float(calculator(images[-1])[0])], dtype=float)

    t0 = time.perf_counter()
    plot_handle = None
    if live_plot and history:
        plot_handle = _draw_history(history, tol, stage, save_to=folder / "progress.png")

    with tqdm(total=max_iter, initial=min(len(history), max_iter),
              desc=f"{stage.upper()} iterations", unit="iter", position=0,
              dynamic_ncols=True, leave=True) as iterations_bar:
        for iteration in range(len(history), max_iter):
            energies = np.empty(len(images), dtype=float)
            gradients = np.zeros_like(images)
            energies[0], energies[-1] = endpoint_e
            n_interior = len(images) - 2
            with tqdm(total=n_interior, desc=f"ORCA · iter {iteration}",
                      unit="image", position=1, leave=False,
                      dynamic_ncols=True) as image_bar:
                for i in range(1, len(images) - 1):
                    image_bar.set_description(f"ORCA · iter {iteration} · image {i}/{n_interior}")
                    energy, gradient = calculator(images[i])
                    gradient = np.asarray(gradient, dtype=float)
                    energies[i] = float(energy)
                    gradients[i] = gradient
                    image_bar.update(1)

            updated, fmax = update(images, gradients, energies,
                                   step=step, k=k, climb=climb,
                                   max_move=max_move, dtype=np.float64)
            updated = np.asarray(updated, dtype=np.float64)
            fmax = float(fmax)
            if updated.shape != images.shape or not np.isfinite(updated).all() or not np.isfinite(fmax):
                raise ValueError("NEB update returned invalid images or Fmax")
            # Always keep endpoints fixed, even if update() does not enforce this.
            updated[[0, -1]] = images[[0, -1]]

            highest_image = 1 + int(np.argmax(energies[1:-1]))
            barrier = float(np.max(energies[1:-1]) - energies[0])
            history.append([iteration, fmax, barrier, highest_image])
            converged = fmax < tol
            # If converged do NOT take an unnecessary additional step.
            if not converged:
                images = updated
            checkpoint_write(stage, images, history, endpoint_e, data_dir,
                             config=config, converged=converged)
            iterations_bar.update(1)
            iterations_bar.set_postfix_str(
                f"Fmax={fmax:.4f} | barrier={barrier:.4f}eV | peak={highest_image}")
            if live_plot and (iteration == len(history) - 1 and
                              ((iteration + 1) % plot_every == 0 or
                               iteration == 0 or converged or iteration + 1 == max_iter)):
                plot_handle = _draw_history(history, tol, stage, handle=plot_handle,
                                            save_to=folder / "progress.png")
            if (iteration % report_every == 0) or converged:
                elapsed = (time.perf_counter() - t0) / 60
                tqdm.write(f"[{stage.upper()}] iter={iteration:3d} "
                           f"Fmax={fmax:.5f} eV/Å | barrier={barrier:.4f} eV "
                           f"| peak={highest_image} | elapsed={elapsed:.1f} min")
            if converged:
                tqdm.write(f"[{stage.upper()}] CONVERGED (Fmax < {tol:g} eV/Å)")
                return images, np.asarray(history, dtype=float), True

    tqdm.write(f"[{stage.upper()}] Reached max_iter={max_iter} without convergence")
    return images, np.asarray(history, dtype=float).reshape(-1, 4), False
