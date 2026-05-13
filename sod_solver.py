"""
SOD shock-tube finite-volume solver for Homework 9.

Features in this updated version:
1. First-order finite volume and second-order TVD finite volume with minmod limiter.
2. Third-order TVD Runge-Kutta time integration.
3. Automatic comparison over multiple grid numbers.
4. CSV outputs and publication-ready figures generated with relative paths.

Run from this folder or any folder:
    python code/sod_solver.py

Optional examples:
    python code/sod_solver.py --grids 50 100 200 400 800 --cfl 0.5
    python code/sod_solver.py --outdir homework9_updated
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

GAMMA = 1.4
FLOOR = 1.0e-12


def setup_matplotlib() -> None:
    """Use a CJK font when it is available; otherwise fall back silently."""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for fp in candidates:
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fp).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"


def primitive_to_conserved(q: np.ndarray) -> np.ndarray:
    rho = q[..., 0]
    u = q[..., 1]
    p = q[..., 2]
    U = np.empty_like(q, dtype=float)
    U[..., 0] = rho
    U[..., 1] = rho * u
    U[..., 2] = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return U


def conserved_to_primitive(U: np.ndarray) -> np.ndarray:
    rho = np.maximum(U[..., 0], FLOOR)
    u = U[..., 1] / rho
    E = U[..., 2]
    p = (GAMMA - 1.0) * (E - 0.5 * rho * u * u)
    p = np.maximum(p, FLOOR)
    q = np.empty_like(U, dtype=float)
    q[..., 0] = rho
    q[..., 1] = u
    q[..., 2] = p
    return q


def euler_flux(U: np.ndarray) -> np.ndarray:
    q = conserved_to_primitive(U)
    rho, u, p = q[..., 0], q[..., 1], q[..., 2]
    F = np.empty_like(U, dtype=float)
    F[..., 0] = rho * u
    F[..., 1] = rho * u * u + p
    F[..., 2] = u * (U[..., 2] + p)
    return F


def hll_flux(UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
    qL = conserved_to_primitive(UL)
    qR = conserved_to_primitive(UR)
    rhoL, uL, pL = qL[..., 0], qL[..., 1], qL[..., 2]
    rhoR, uR, pR = qR[..., 0], qR[..., 1], qR[..., 2]

    aL = np.sqrt(GAMMA * pL / rhoL)
    aR = np.sqrt(GAMMA * pR / rhoR)
    sL = np.minimum(uL - aL, uR - aR)
    sR = np.maximum(uL + aL, uR + aR)
    FL = euler_flux(UL)
    FR = euler_flux(UR)

    out = np.empty_like(UL, dtype=float)
    maskL = sL >= 0.0
    maskR = sR <= 0.0
    maskM = ~(maskL | maskR)

    out[maskL] = FL[maskL]
    out[maskR] = FR[maskR]
    if np.any(maskM):
        smL = sL[maskM][..., None]
        smR = sR[maskM][..., None]
        den = np.maximum(smR - smL, FLOOR)
        out[maskM] = (smR * FL[maskM] - smL * FR[maskM] + smL * smR * (UR[maskM] - UL[maskM])) / den
    return out


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same_sign = (a * b) > 0.0
    return np.where(same_sign, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def rhs(U: np.ndarray, dx: float, order: int) -> np.ndarray:
    ng = 2
    Ue = np.empty((U.shape[0] + 2 * ng, 3), dtype=float)
    Ue[ng:-ng] = U
    Ue[:ng] = U[0]
    Ue[-ng:] = U[-1]

    if order == 1:
        UL = Ue[1:-2]
        UR = Ue[2:-1]
    elif order == 2:
        qe = conserved_to_primitive(Ue)
        dq_left = qe[1:-1] - qe[:-2]
        dq_right = qe[2:] - qe[1:-1]
        slope = minmod(dq_left, dq_right)
        q_left_cell = qe[1:-1] - 0.5 * slope
        q_right_cell = qe[1:-1] + 0.5 * slope
        UL = primitive_to_conserved(q_right_cell[:-1])
        UR = primitive_to_conserved(q_left_cell[1:])
    else:
        raise ValueError("order must be 1 or 2")

    Fh = hll_flux(UL, UR)
    return -(Fh[1:] - Fh[:-1]) / dx


def rk3_step(U: np.ndarray, dt: float, dx: float, order: int) -> np.ndarray:
    k1 = rhs(U, dx, order)
    U1 = U + dt * k1
    k2 = rhs(U1, dx, order)
    U2 = 0.75 * U + 0.25 * (U1 + dt * k2)
    k3 = rhs(U2, dx, order)
    return (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * k3)


def initial_condition(x: np.ndarray) -> np.ndarray:
    q = np.zeros((x.size, 3), dtype=float)
    left = x < 0.5
    q[left] = np.array([1.0, 0.0, 1.0])
    q[~left] = np.array([0.125, 0.0, 0.1])
    return q


def solve_sod(n: int = 200, t_end: float = 0.2, order: int = 2, cfl: float = 0.5) -> tuple[np.ndarray, np.ndarray, int]:
    xmin, xmax = 0.0, 1.0
    dx = (xmax - xmin) / n
    x = xmin + (np.arange(n) + 0.5) * dx
    U = primitive_to_conserved(initial_condition(x))

    t = 0.0
    nsteps = 0
    while t < t_end - 1.0e-14:
        qp = conserved_to_primitive(U)
        a = np.sqrt(GAMMA * qp[:, 2] / qp[:, 0])
        spectral_radius = np.max(np.abs(qp[:, 1]) + a)
        dt = cfl * dx / spectral_radius
        if t + dt > t_end:
            dt = t_end - t
        U = rk3_step(U, dt, dx, order)
        t += dt
        nsteps += 1
    return x, conserved_to_primitive(U), nsteps


def exact_sod(x: np.ndarray, t: float) -> np.ndarray:
    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    g = GAMMA
    aL = np.sqrt(g * pL / rhoL)
    aR = np.sqrt(g * pR / rhoR)

    def pressure_function(p: float, rho: float, p0: float, a: float) -> tuple[float, float]:
        if p > p0:
            A = 2.0 / ((g + 1.0) * rho)
            B = (g - 1.0) / (g + 1.0) * p0
            f = (p - p0) * np.sqrt(A / (p + B))
            fd = np.sqrt(A / (p + B)) * (1.0 - 0.5 * (p - p0) / (p + B))
        else:
            pr = p / p0
            f = 2.0 * a / (g - 1.0) * (pr ** ((g - 1.0) / (2.0 * g)) - 1.0)
            fd = (1.0 / (rho * a)) * pr ** (-(g + 1.0) / (2.0 * g))
        return f, fd

    p = 0.5 * (pL + pR)
    for _ in range(80):
        fL, fdL = pressure_function(p, rhoL, pL, aL)
        fR, fdR = pressure_function(p, rhoR, pR, aR)
        p_new = p - (fL + fR + uR - uL) / (fdL + fdR)
        p_new = max(float(p_new), FLOOR)
        if abs(p_new - p) < 1.0e-13:
            p = p_new
            break
        p = p_new

    fL, _ = pressure_function(p, rhoL, pL, aL)
    fR, _ = pressure_function(p, rhoR, pR, aR)
    u = 0.5 * (uL + uR + fR - fL)

    xi = (x - 0.5) / t
    rho = np.empty_like(x)
    uu = np.empty_like(x)
    pp = np.empty_like(x)

    rho_star_L = rhoL * (p / pL) ** (1.0 / g)
    rho_star_R = rhoR * ((p / pR + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * p / pR + 1.0))
    a_star_L = aL * (p / pL) ** ((g - 1.0) / (2.0 * g))
    s_head = uL - aL
    s_tail = u - a_star_L
    s_shock = uR + aR * np.sqrt((g + 1.0) / (2.0 * g) * p / pR + (g - 1.0) / (2.0 * g))

    for j, s in enumerate(xi):
        if s < s_head:
            rho[j], uu[j], pp[j] = rhoL, uL, pL
        elif s < s_tail:
            uu[j] = 2.0 / (g + 1.0) * (aL + 0.5 * (g - 1.0) * uL + s)
            a = 2.0 / (g + 1.0) * (aL + 0.5 * (g - 1.0) * (uL - s))
            rho[j] = rhoL * (a / aL) ** (2.0 / (g - 1.0))
            pp[j] = pL * (a / aL) ** (2.0 * g / (g - 1.0))
        elif s < u:
            rho[j], uu[j], pp[j] = rho_star_L, u, p
        elif s < s_shock:
            rho[j], uu[j], pp[j] = rho_star_R, u, p
        else:
            rho[j], uu[j], pp[j] = rhoR, uR, pR
    return np.vstack([rho, uu, pp]).T


def l1_error(q: np.ndarray, qe: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(q - qe), axis=0)


def run_grid_study(grids: list[int], t_end: float, cfl: float) -> tuple[list[dict], dict[int, dict]]:
    rows: list[dict] = []
    sols: dict[int, dict] = {}
    for n in grids:
        x, q1, steps1 = solve_sod(n=n, t_end=t_end, order=1, cfl=cfl)
        _, q2, steps2 = solve_sod(n=n, t_end=t_end, order=2, cfl=cfl)
        qe = exact_sod(x, t_end)
        err1 = l1_error(q1, qe)
        err2 = l1_error(q2, qe)
        rows.append(
            {
                "N": n,
                "dx": 1.0 / n,
                "steps_first": steps1,
                "steps_tvd": steps2,
                "rho_first_L1": err1[0],
                "u_first_L1": err1[1],
                "p_first_L1": err1[2],
                "rho_tvd_L1": err2[0],
                "u_tvd_L1": err2[1],
                "p_tvd_L1": err2[2],
            }
        )
        sols[n] = {"x": x, "q_first": q1, "q_tvd": q2, "q_exact": qe}
        print(f"N={n:4d}: rho L1 first={err1[0]:.6e}, TVD={err2[0]:.6e}; steps={steps2}")
    return rows, sols


def add_observed_orders(rows: list[dict]) -> None:
    # Error order between consecutive mesh refinements: p = log(e_h/e_h2)/log(h/h2).
    keys = [
        "rho_first_L1", "u_first_L1", "p_first_L1",
        "rho_tvd_L1", "u_tvd_L1", "p_tvd_L1",
    ]
    for key in keys:
        rows[0][key.replace("_L1", "_order")] = np.nan
        for i in range(1, len(rows)):
            e0 = rows[i - 1][key]
            e1 = rows[i][key]
            h0 = rows[i - 1]["dx"]
            h1 = rows[i]["dx"]
            rows[i][key.replace("_L1", "_order")] = np.log(e0 / e1) / np.log(h0 / h1)


def write_csv(rows: list[dict], path: Path) -> None:
    keys = list(rows[0].keys())
    arr = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    header = ",".join(keys)
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def make_solution_figures(sol: dict, n: int, t_end: float, outdir: Path) -> None:
    x = sol["x"]
    q1 = sol["q_first"]
    q2 = sol["q_tvd"]
    qe = sol["q_exact"]
    names = [r"密度 $\rho$", r"速度 $u$", r"压力 $p$"]
    filenames = ["density.png", "velocity.png", "pressure.png"]

    for k, name in enumerate(names):
        plt.figure(figsize=(7.2, 4.6), dpi=220)
        plt.plot(x, qe[:, k], "-", linewidth=1.2, label="精确解")
        plt.plot(x, q1[:, k], "--", linewidth=1.0, label="一阶有限体积")
        plt.plot(x, q2[:, k], "-", linewidth=1.0, label="二阶 TVD")
        plt.xlabel(r"$x$")
        plt.ylabel(name)
        plt.title(f"SOD 激波管 t={t_end:g}, N={n}")
        plt.grid(True, alpha=0.28)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outdir / filenames[k])
        plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(7.3, 8.8), dpi=220, sharex=True)
    for k, ax in enumerate(axes):
        ax.plot(x, qe[:, k], "-", linewidth=1.2, label="精确解")
        ax.plot(x, q1[:, k], "--", linewidth=1.0, label="一阶有限体积")
        ax.plot(x, q2[:, k], "-", linewidth=1.0, label="二阶 TVD")
        ax.set_ylabel(names[k])
        ax.grid(True, alpha=0.28)
        if k == 0:
            ax.legend(loc="best", frameon=False)
    axes[-1].set_xlabel(r"$x$")
    fig.suptitle("SOD 激波管数值结果比较")
    fig.tight_layout()
    fig.savefig(outdir / "sod_comparison.png")
    plt.close(fig)


def make_grid_overlay(sols: dict[int, dict], grids: list[int], outdir: Path) -> None:
    # Show convergence of density profile for multiple grid numbers.
    plt.figure(figsize=(7.2, 4.8), dpi=220)
    finest = max(grids)
    plt.plot(sols[finest]["x"], sols[finest]["q_exact"][:, 0], "-", linewidth=1.4, label="精确解")
    for n in grids:
        x = sols[n]["x"]
        rho = sols[n]["q_tvd"][:, 0]
        # Light decimation for readable plot at high resolution.
        stride = max(1, n // 220)
        plt.plot(x[::stride], rho[::stride], linewidth=0.95, label=f"TVD, N={n}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"密度 $\rho$")
    plt.title("二阶 TVD 格式在不同网格数下的密度剖面")
    plt.grid(True, alpha=0.28)
    plt.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "grid_density_overlay.png")
    plt.close()


def make_convergence_fig(rows: list[dict], outdir: Path) -> None:
    N = np.array([r["N"] for r in rows], dtype=float)
    keys = ["rho", "u", "p"]
    labels = [r"$\rho$", r"$u$", r"$p$"]
    plt.figure(figsize=(7.2, 4.8), dpi=220)
    for key, lab in zip(keys, labels):
        e1 = np.array([r[f"{key}_first_L1"] for r in rows])
        e2 = np.array([r[f"{key}_tvd_L1"] for r in rows])
        plt.loglog(N, e1, "--o", linewidth=1.0, markersize=3.2, label=f"一阶 {lab}")
        plt.loglog(N, e2, "-s", linewidth=1.0, markersize=3.2, label=f"TVD {lab}")
    plt.xlabel("网格数 N")
    plt.ylabel(r"平均绝对误差 $L^1$")
    plt.title("SOD 激波管网格加密误差比较")
    plt.grid(True, which="both", alpha=0.28)
    plt.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "grid_convergence.png")
    plt.close()


def save_n200_data(sol: dict, path: Path) -> None:
    x = sol["x"]
    q1 = sol["q_first"]
    q2 = sol["q_tvd"]
    qe = sol["q_exact"]
    data = np.column_stack([x, q1, q2, qe])
    header = "x,rho_first,u_first,p_first,rho_tvd,u_tvd,p_tvd,rho_exact,u_exact,p_exact"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", nargs="+", type=int, default=[50, 100, 200, 400, 800])
    parser.add_argument("--t-end", type=float, default=0.2)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    setup_matplotlib()

    root = Path(__file__).resolve().parents[1]
    outroot = Path(args.outdir).resolve() if args.outdir else root
    imgdir = outroot / "images"
    datadir = outroot / "data"
    imgdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    grids = sorted(set(args.grids))
    if 200 not in grids:
        grids.append(200)
        grids = sorted(grids)

    rows, sols = run_grid_study(grids, args.t_end, args.cfl)
    add_observed_orders(rows)

    write_csv(rows, datadir / "sod_grid_study.csv")
    save_n200_data(sols[200], datadir / "sod_results.csv")
    make_solution_figures(sols[200], 200, args.t_end, imgdir)
    make_grid_overlay(sols, grids, imgdir)
    make_convergence_fig(rows, imgdir)

    print("\nOutputs written to:")
    print(f"  {imgdir}")
    print(f"  {datadir}")


if __name__ == "__main__":
    main()
