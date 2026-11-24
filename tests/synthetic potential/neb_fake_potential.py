"""Run two-stage NEB / CI-NEB on six 2-D model potentials from the screenshot.

Example:
    python neb_demo.py --out results

Requires numpy, matplotlib, pillow. Never uses ORCA or ASE.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from neb_core import linear, update

# ---------------------- TEST POTENTIALS ----------------------

def potential(n, xy):
    """Return vectorized (energy, gradient) for screenshot E1..E6."""
    xy = np.asarray(xy, float)
    x, y = xy[..., 0], xy[..., 1]
    if n == 1:
        E = (x*x - 1)**2 + y*y
        gx, gy = 4*x*(x*x-1), 2*y
    elif n == 2:
        q = y - .8 * (1 - x*x)
        E = (x*x-1)**2 + 8*q*q
        gx, gy = 4*x*(x*x-1) + 25.6*x*q, 16*q
    elif n == 3:
        q = y - np.sin(np.pi*x/2)
        E = (x*x-4)**2 + 8*q*q
        gx, gy = 4*x*(x*x-4) - 8*np.pi*q*np.cos(np.pi*x/2), 16*q
    elif n == 4:
        ex = np.exp(-4*x*x-8*y*y)
        a = (x*x-1)**2
        E = a*(1 + 2*ex) + .3*y*y
        gx = 4*x*(x*x-1)*(1+2*ex) -16*x*a*ex
        gy = -32*y*a*ex + .6*y
    elif n == 5:
        E = (x*x-1)**2 + (y*y-1)**2
        gx, gy = 4*x*(x*x-1), 4*y*(y*y-1)
    elif n == 6:
        p1 = -(x-1)**2 -10*y*y
        p2 = -x*x-10*(y-.5)**2
        u, v = x+.5, y-1.5
        p3 = -6.5*u*u + 11*u*v - 6.5*v*v
        w, z = x+1, y-1
        p4 = .7*w*w+.6*w*z+.7*z*z
        a = -200*np.exp(p1)
        b = -100*np.exp(p2)
        c = -170*np.exp(p3)
        d = 15*np.exp(p4)
        E = a+b+c+d
        gx = a*(-2*(x-1)) + b*(-2*x) + c*(-13*u+11*v) + d*(1.4*w+.6*z)
        gy = a*(-20*y) + b*(-20*(y-.5)) + c*(11*u-13*v) + d*(.6*w+1.4*z)
    else:
        raise ValueError('E must be one of 1,2,3,4,5,6')
    return E, np.stack((gx, gy), axis=-1)


def initial(case, n_images=13):
    if case in (1,2,4):
        start, end = [-1,0], [1,0]
    elif case == 3:
        start, end = [-2,0], [2,0]
    elif case == 5:
        start, end = [-1,-1], [1,-1]
    else:
        # Two nearby minima of the Mueller-Brown potential.
        start, end = [-.558224,1.441726], [-.050011,.466694]
    images = linear(start, end, n_images)
    t = np.linspace(0, 1, n_images)
    if case == 1:
        images[:, 1] += .25*np.sin(np.pi*t)
    elif case == 2:
        images[:, 1] += .10*np.sin(np.pi*t)
    elif case == 3:
        images[:, 1] += .15 * np.sin(np.pi*images[:, 0]/2)
    elif case == 4:
        # Positive/negative symmetric paths: choose the upper saddle.
        images[:, 1] += .75*np.sin(np.pi*t)
    elif case == 5:
        images[:, 1] += .2*np.sin(np.pi*t)
    elif case == 6:
        images[:, 0] -= .08 * np.sin(np.pi*t)
    return images


CONFIG = {
    1: dict(step=.035, k=2, max_move=.05, normal_tol=3e-4, ci_tol=3e-4, limit=9000),
    2: dict(step=.013, k=3, max_move=.05, normal_tol=6e-4, ci_tol=6e-4, limit=20000),
    3: dict(step=.005, k=3, max_move=.04, normal_tol=2e-3, ci_tol=2e-3, limit=25000),
    4: dict(step=.009, k=2, max_move=.035, normal_tol=6e-4, ci_tol=6e-4, limit=20000),
    5: dict(step=.03, k=2, max_move=.05, normal_tol=3e-4, ci_tol=3e-4, limit=9000),
    6: dict(step=.00035, k=8, max_move=.025, normal_tol=.012, ci_tol=.012, limit=33000),
}


def solve(case, n_images=13):
    cfg = CONFIG[case]
    images = initial(case, n_images)
    history = [images.copy()]
    iteration_at = [0]
    climb_started = None
    fmax = float('inf')
    mode = False
    step = cfg['step']
    limit = cfg['limit']
    sample_every = 5  # Record enough snapshots; downsample after convergence.
    for it in range(limit):
        energy, grad = potential(case, images)
        new, fmax = update(images, grad, energy,
                           step=step, k=cfg['k'], climb=mode,
                           max_move=cfg['max_move'])
        tol = cfg['ci_tol'] if mode else cfg['normal_tol']
        if not mode and fmax < tol:
            mode = True
            climb_started = it
            history.append(images.copy())
            iteration_at.append(it)
            continue
        if mode and fmax < tol:
            break
        images = new
        if (it+1) % sample_every == 0:
            history.append(images.copy())
            iteration_at.append(it+1)
    history.append(images.copy())
    iteration_at.append(it+1)
    # Roughly 70 animation frames, plus the exact NEB -> CI switch frame.
    keep = np.linspace(0, len(history)-1, min(len(history), 70)).astype(int)
    if climb_started is not None:
        switch_frame = np.argmin(abs(np.array(iteration_at)-climb_started))
        keep = np.r_[keep, switch_frame]
    keep = np.unique(keep)
    history = [history[j] for j in keep]
    iteration_at = [iteration_at[j] for j in keep]
    energy, grad = potential(case, images)
    ci = int(np.argmax(energy[1:-1])+1)
    return dict(case=case, images=images, history=history, its=iteration_at,
                climb_started=climb_started, converged=(mode and fmax < cfg['ci_tol']),
                iterations=it+1, fmax=fmax, ci=ci, energy=energy)


def reaction_coordinate(images):
    delta = np.diff(images.reshape(len(images),-1), axis=0)
    s = np.r_[0.,np.cumsum(np.linalg.norm(delta,axis=1))]
    return s / s[-1] if s[-1]>1e-14 else np.linspace(0,1,len(images))


def limits(case):
    return {
        1: (-1.45,1.45,-.75,.90),
        2: (-1.45,1.45,-.55,1.20),
        3: (-2.40,2.40,-1.45,1.45),
        4: (-1.45,1.45,-1.10,1.10),
        5: (-1.40,1.40,-1.60,-.40),
        6: (-1.10,.55,-.05,1.85),
    }[case]


def draw_background(ax, case):
    xmin,xmax,ymin,ymax=limits(case)
    xx,yy=np.meshgrid(np.linspace(xmin,xmax,260),np.linspace(ymin,ymax,250))
    ZZ=potential(case,np.stack([xx,yy],axis=-1))[0]
    low,high=np.percentile(ZZ,[2,90])
    if case==6:
        low,high=np.percentile(ZZ,[3,85])
    lev=np.linspace(low,high,24)
    ax.contourf(xx,yy,ZZ,levels=lev,extend='both',cmap='viridis',alpha=.85)
    ax.contour(xx,yy,ZZ,levels=lev[::3],colors='white',alpha=.28,linewidths=.55)
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='x',ylabel='y')
    ax.set_aspect('equal', adjustable='box')


def save_case(result, out):
    """Save a contour+E(t) GIF using one Matplotlib background and fast Pillow overlays."""
    case = result['case']
    images, hist = result['images'], result['history']
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    fig, (ax, ae) = plt.subplots(1, 2, figsize=(10, 4.1), layout='constrained')
    draw_background(ax, case)
    init = hist[0]
    ax.plot(init[:, 0], init[:, 1], '--', color='white', linewidth=1.5,
            alpha=.8, label='Initial path')
    ax.plot([], [], '-o', color='#ff6e35', label='Current images')
    ax.plot([], [], '*', color='red', markersize=10, label='Climbing image')
    ax.legend(loc='upper right', fontsize=7, facecolor='white', framealpha=.9)

    values = [potential(case, r)[0] for r in hist]
    min_E = min(v.min() for v in values)
    max_E = max(v.max() for v in values)
    rng = max(max_E - min_E, .25)
    ae.set(xlim=(-.03, 1.03), ylim=(min_E-.1*rng, max_E+.16*rng),
           xlabel='t (normalized path length)', ylabel='Energy E(t)')
    ae.grid(alpha=.25)
    fig.suptitle(f'E{case} | NEB -> CI-NEB', fontsize=12)
    fig.canvas.draw()  # fixes layout and transforms for the entire animation

    bg = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert('RGB')
    height = bg.height

    def pixel(axes, xy):
        px, py = axes.transData.transform(xy)
        return (int(round(px)), int(round(height - py)))

    def draw_circle(draw, p, r, fill):
        draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r), fill=fill)

    def draw_star(draw, p, r, fill):
        angles = np.arange(10) * (np.pi/5) - np.pi/2
        rad = np.where(np.arange(10) % 2 == 0, r, r*.42)
        pts = [(int(p[0]+rad[j]*np.cos(angles[j])),
                int(p[1]+rad[j]*np.sin(angles[j]))) for j in range(10)]
        draw.polygon(pts, fill=fill)

    frames = []
    for j, r in enumerate(hist):
        canvas = bg.copy()
        d = ImageDraw.Draw(canvas)
        pos = [pixel(ax, q) for q in r]
        reaction_t = reaction_coordinate(r)
        en = values[j]
        profile = [pixel(ae, p) for p in zip(reaction_t, en)]
        d.line(pos, fill=(255, 110, 53), width=4, joint='curve')
        d.line(profile, fill=(18, 121, 188), width=4, joint='curve')
        for p in pos:
            draw_circle(d, p, 4, (255, 110, 53))
        for p in profile:
            draw_circle(d, p, 3, (18, 121, 188))

        is_ci = (result['climb_started'] is not None and
                 result['its'][j] >= result['climb_started'])
        if is_ci:
            ci = int(np.argmax(en[1:-1])) + 1
            draw_star(d, pos[ci], 11, (229, 22, 25))
            draw_star(d, profile[ci], 11, (229, 22, 25))
        stage = 'CI-NEB' if is_ci else 'NEB'
        d.text((17, 8), f'Iteration {result["its"][j]} | {stage}',
               fill=(22, 22, 22))
        frames.append(canvas)
    frames[0].save(out/f'E{case}_animation.gif',save_all=True,
                   append_images=frames[1:],duration=95,loop=0,optimize=True)

    # Also generate a conventional, high-resolution Matplotlib final energy plot.
    en = result['energy']
    t = reaction_coordinate(images)
    ax.plot(images[:,0],images[:,1], '-o', color='#ff6e35',lw=2.2,ms=5)
    ae.plot(t,en,'-o',color='#1279bc',lw=1.8,ms=4)
    if result['climb_started'] is not None:
        ci = result['ci']
        ax.plot(images[ci,0],images[ci,1],'r*',markersize=15)
        ae.plot(t[ci],en[ci],'r*',markersize=15)
    fig.savefig(out/f'E{case}_final.png', dpi=150)
    plt.close(fig)
    np.save(out/f'E{case}_images.npy', images)
    np.savetxt(out/f'E{case}_energy.csv', np.column_stack((t,en)),
               delimiter=',',header='t_normalized_path,E',comments='')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',type=int,default=[1,2,3,4,5,6])
    parser.add_argument('--out',default='neb_results')
    parser.add_argument('--images',type=int,default=12)
    args=parser.parse_args()
    for case in args.cases:
        result=solve(case,args.images)
        print(f'E{case}: steps={result["iterations"]}, CI started={result["climb_started"]}, '
              f'converged={result["converged"]}, Fmax={result["fmax"]:.3g}, '
              f'Epeak={result["energy"][result["ci"]]:.5f}',flush=True)
        save_case(result,args.out)

if __name__=='__main__':
    main()
