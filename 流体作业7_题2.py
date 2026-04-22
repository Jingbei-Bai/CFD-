"""
Homework 7, Problem 2: 1st- and 2nd-order Steger–Warming flux-vector splitting (FVS)
for the 1-D Euler equations, time integration by 3rd-order SSP Runge–Kutta.
Same Sod shock-tube setup as 流体作业6.py (t_final = 0.3).

Spatial:
  1st order: piecewise constant states at each face.
  2nd order: MUSCL (minmod-limited) reconstruction on primitive variables
             (rho, u, p), then F_{i+1/2} = F^+(U_L) + F^-(U_R).

Flux split: F^± = A^± U with A = ∂F/∂U and A^± = Q diag(λ^±) Q^{-1} (Steger–Warming–type
spectral split; ensures F^+ + F^- = F).

All figure text is in English.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

import 流体作业6 as hw6

gamma = hw6.gamma
xL, xR = hw6.xL, hw6.xR
x0 = hw6.x0
rhoL, uL, pL = hw6.rhoL, hw6.uL, hw6.pL
rhoR, uR, pR = hw6.rhoR, hw6.uR, hw6.pR

primitive_to_conservative = hw6.primitive_to_conservative
conservative_to_primitive = hw6.conservative_to_primitive
flux = hw6.flux

OUTPUT_DIR = "fvs_hw7_q2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
T_FINAL = 0.3
CFL = 0.45
GRID_LIST = [100, 200, 400, 800, 1000]


def exact_at_t(x, t):
    rho = np.zeros_like(x, dtype=float)
    u = np.zeros_like(x, dtype=float)
    p = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        rho[i], u[i], p[i] = hw6.exact_sample(xi, t)
    return rho, u, p


def jacobian_1d(Uvec):
    """
    Jacobian A = ∂F/∂U for 1-D Euler in conservative variables U = [ρ, ρu, E]^T.
    For ideal gas, F is homogeneous of degree 1 in U, hence F = A @ U (Toro).
    """
    Uvec = np.asarray(Uvec, dtype=float).reshape(3)
    rho, rhou, E = Uvec[0], Uvec[1], Uvec[2]
    rho = float(np.maximum(rho, 1e-14))
    u = rhou / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u * u)
    p = float(np.maximum(p, 1e-14))
    H = (E + p) / rho
    gm1 = gamma - 1.0
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [gm1 * u * u / 2.0 - u * u, (3.0 - gamma) * u, gm1],
            [gm1 * u**3 / 2.0 - u * H, H - gm1 * u**2, gamma * u],
        ],
        dtype=float,
    )


def steger_warming_split(U):
    """
    Steger–Warming–type spectral split: F^+ = A^+ U, F^- = A^- U with
    A^± = Q diag(λ^±) Q^{-1}, λ_i eigenvalues of A = ∂F/∂U.

    Then F^+ + F^- = A U = F(U), so the split is consistent with the flux.
    U: shape (3,) or (3, N).
    """
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U.reshape(3, 1)
        squeeze = True
    else:
        squeeze = False
    Fp = np.zeros_like(U)
    Fm = np.zeros_like(U)
    for j in range(U.shape[1]):
        Uj = U[:, j]
        A = jacobian_1d(Uj)
        lam, Q = np.linalg.eig(A)
        Qinv = np.linalg.inv(Q)
        lam_p = np.maximum(np.real(lam), 0.0)
        lam_m = np.minimum(np.real(lam), 0.0)
        Fp[:, j] = np.real(Q @ np.diag(lam_p) @ Qinv @ Uj)
        Fm[:, j] = np.real(Q @ np.diag(lam_m) @ Qinv @ Uj)
    if squeeze:
        return Fp[:, 0], Fm[:, 0]
    return Fp, Fm


def fvs_interface_flux(UL, UR):
    """Numerical flux: F_hat = F^+(U_L) + F^-(U_R)."""
    FpL, _ = steger_warming_split(UL)
    _, FmR = steger_warming_split(UR)
    return FpL + FmR


def minmod_scalar(a, b):
    """Element-wise minmod for 1D arrays of the same shape."""
    s = 0.5 * (np.sign(a) + np.sign(b))
    return s * np.minimum(np.abs(a), np.abs(b))


def muscl_primitive_states(Ue):
    """
    Ue: (3, N+4) with 2 ghost cells per side. Physical cells at indices 2..N+1.
    Interface k (k = 0..N) lies between Ue[:, k+1] and Ue[:, k+2].

    MUSCL (minmod):
      q_L = q_{k+1} + 0.5*minmod(q_{k+1}-q_k, q_{k+2}-q_{k+1})
      q_R = q_{k+2} - 0.5*minmod(q_{k+3}-q_{k+2}, q_{k+2}-q_{k+1})
    on primitive (rho, u, p). Returns UL, UR with shape (3, N+1).
    """
    rho, u, p = conservative_to_primitive(Ue)
    M = Ue.shape[1]
    n = M - 4
    k = np.arange(n + 1, dtype=int)

    rho_L = rho[k + 1] + 0.5 * minmod_scalar(
        rho[k + 1] - rho[k], rho[k + 2] - rho[k + 1]
    )
    rho_R = rho[k + 2] - 0.5 * minmod_scalar(
        rho[k + 3] - rho[k + 2], rho[k + 2] - rho[k + 1]
    )
    u_L = u[k + 1] + 0.5 * minmod_scalar(
        u[k + 1] - u[k], u[k + 2] - u[k + 1]
    )
    u_R = u[k + 2] - 0.5 * minmod_scalar(
        u[k + 3] - u[k + 2], u[k + 2] - u[k + 1]
    )
    p_L = p[k + 1] + 0.5 * minmod_scalar(
        p[k + 1] - p[k], p[k + 2] - p[k + 1]
    )
    p_R = p[k + 2] - 0.5 * minmod_scalar(
        p[k + 3] - p[k + 2], p[k + 2] - p[k + 1]
    )

    rho_L = np.maximum(rho_L, 1e-12)
    rho_R = np.maximum(rho_R, 1e-12)
    p_L = np.maximum(p_L, 1e-12)
    p_R = np.maximum(p_R, 1e-12)

    UL = primitive_to_conservative(rho_L, u_L, p_L)
    UR = primitive_to_conservative(rho_R, u_R, p_R)
    return UL, UR


def build_extended_U(U, nghost=2):
    """Constant extrapolation at boundaries."""
    left = np.broadcast_to(U[:, :1], (3, nghost))
    right = np.broadcast_to(U[:, -1:], (3, nghost))
    return np.concatenate([left, U, right], axis=1)


def compute_interface_fluxes(U, order: int):
    """
    U: (3, N). order 1 or 2. Returns F_hat shape (3, N+1).
    """
    N = U.shape[1]
    if order == 1:
        Ue = build_extended_U(U, nghost=1)
        F_hat = np.zeros((3, N + 1))
        for k in range(N + 1):
            UL = Ue[:, k]
            UR = Ue[:, k + 1]
            F_hat[:, k] = fvs_interface_flux(UL, UR)
        return F_hat

    if order == 2:
        Ue = build_extended_U(U, nghost=2)
        UL, UR = muscl_primitive_states(Ue)
        F_hat = np.zeros((3, N + 1))
        for k in range(N + 1):
            F_hat[:, k] = fvs_interface_flux(UL[:, k], UR[:, k])
        return F_hat

    raise ValueError("order must be 1 or 2")


def rhs_fvs(U, dx, order: int):
    F_hat = compute_interface_fluxes(U, order)
    return -(F_hat[:, 1:] - F_hat[:, :-1]) / dx


def rk3_step(U, dx, dt, order: int):
    def L(U_):
        return rhs_fvs(U_, dx, order)

    L0 = L(U)
    U1 = U + dt * L0
    U2 = 0.75 * U + 0.25 * U1 + 0.25 * dt * L(U1)
    U3 = (1.0 / 3.0) * U + (2.0 / 3.0) * U2 + (2.0 / 3.0) * dt * L(U2)
    return U3


def fvs_solver(N, order: int, t_final=T_FINAL):
    dx = (xR - xL) / N
    x = xL + (np.arange(N) + 0.5) * dx

    rho = np.where(x < x0, rhoL, rhoR)
    u = np.where(x < x0, uL, uR)
    p = np.where(x < x0, pL, pR)
    U = primitive_to_conservative(rho, u, p)

    t = 0.0
    while t < t_final:
        rho, u, p = conservative_to_primitive(U)
        rho = np.maximum(rho, 1e-14)
        p = np.maximum(p, 1e-14)
        c = np.sqrt(gamma * p / rho)
        smax = float(np.max(np.abs(u) + c))
        dt = CFL * dx / smax
        if t + dt > t_final:
            dt = t_final - t
        if dt <= 0:
            break
        U = rk3_step(U, dx, dt, order)
        U[0] = np.maximum(U[0], 1e-12)
        t += dt

    rho, u, p = conservative_to_primitive(U)
    return x, rho, u, p


def l1_errors(x, rho, u, p, t):
    rho_ex, u_ex, p_ex = exact_at_t(x, t)
    dx = float(x[1] - x[0])
    e_rho = dx * np.sum(np.abs(rho - rho_ex))
    e_u = dx * np.sum(np.abs(u - u_ex))
    e_p = dx * np.sum(np.abs(p - p_ex))
    return e_rho, e_u, e_p


def verify_split_consistency():
    rng = np.random.default_rng(0)
    for _ in range(50):
        rho = 0.5 + rng.random()
        u = -0.5 + rng.random()
        p = 0.3 + rng.random()
        U = primitive_to_conservative(
            np.array([rho]), np.array([u]), np.array([p])
        )[:, 0]
        Fp, Fm = steger_warming_split(U)
        Ftot = flux(U)
        A = jacobian_1d(U)
        err_au = np.max(np.abs(A @ U - Ftot))
        assert err_au < 1e-10, err_au
        err = np.max(np.abs(Fp + Fm - Ftot))
        assert err < 1e-8, err


def run_grid_study():
    errs1_rho, errs2_rho = [], []
    errs1_u, errs2_u = [], []
    errs1_p, errs2_p = [], []
    results1, results2 = {}, {}

    for N in GRID_LIST:
        x, r1, u1, p1 = fvs_solver(N, 1, T_FINAL)
        x, r2, u2, p2 = fvs_solver(N, 2, T_FINAL)
        e1r, e1u, e1p = l1_errors(x, r1, u1, p1, T_FINAL)
        e2r, e2u, e2p = l1_errors(x, r2, u2, p2, T_FINAL)
        errs1_rho.append(e1r)
        errs2_rho.append(e2r)
        errs1_u.append(e1u)
        errs2_u.append(e2u)
        errs1_p.append(e1p)
        errs2_p.append(e2p)
        results1[N] = (x, r1, u1, p1)
        results2[N] = (x, r2, u2, p2)
        print(
            f"N={N:5d}  L1(rho) 1st={e1r:.6e}  2nd={e2r:.6e}  |  "
            f"L1(u) 1st={e1u:.6e}  2nd={e2u:.6e}  |  "
            f"L1(p) 1st={e1p:.6e}  2nd={e2p:.6e}"
        )

    x_ref = np.linspace(xL, xR, 4000)
    rref, uref, pref = exact_at_t(x_ref, T_FINAL)

    # Multi-N profiles (2nd order)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    keys = ("rho", "u", "p")
    labs = (r"$\rho$", r"$u$", r"$p$")
    refd = {"rho": rref, "u": uref, "p": pref}
    for ax, key, lab in zip(axes, keys, labs):
        for N in GRID_LIST:
            x, r, uu, pp = results2[N]
            ax.plot(x, {"rho": r, "u": uu, "p": pp}[key], lw=1.0, label=f"N={N}")
        ax.plot(x_ref, refd[key], "k--", lw=1.6, label="Exact")
        ax.set_ylabel(lab)
        ax.grid(True)
    axes[0].set_title(
        rf"FVS 2nd-order (Steger–Warming + MUSCL), $t={T_FINAL}$"
    )
    axes[-1].set_xlabel(r"$x$")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fvs2_profiles_multiN.png"), dpi=200)
    plt.close(fig)

    # Zoom 2nd order
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, key, lab in zip(axes, keys, labs):
        for N in GRID_LIST:
            x, r, uu, pp = results2[N]
            ax.plot(x, {"rho": r, "u": uu, "p": pp}[key], lw=1.0, label=f"N={N}")
        ax.plot(x_ref, refd[key], "k--", lw=1.6, label="Exact")
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylabel(lab)
        ax.grid(True)
    axes[0].set_title(r"Zoom near discontinuities ($x \in [-0.2,\,0.2]$), FVS 2nd")
    axes[-1].set_xlabel(r"$x$")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fvs2_zoom.png"), dpi=200)
    plt.close(fig)

    # 1st vs 2nd same N
    Nc = 400
    x1, r1, u1, p1 = results1[Nc]
    x2, r2, u2, p2 = results2[Nc]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, key, lab in zip(axes, keys, labs):
        ax.plot(x1, {"rho": r1, "u": u1, "p": p1}[key], lw=1.2, label="FVS 1st order")
        ax.plot(x2, {"rho": r2, "u": u2, "p": p2}[key], lw=1.2, label="FVS 2nd order")
        ax.plot(x_ref, refd[key], "k--", lw=1.5, label="Exact")
        ax.set_ylabel(lab)
        ax.grid(True)
    axes[0].set_title(rf"FVS comparison at $N={Nc}$, $t={T_FINAL}$")
    axes[-1].set_xlabel(r"$x$")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fvs1_vs_fvs2_N400.png"), dpi=200)
    plt.close(fig)

    # L1 error vs N
    Ns = np.asarray(GRID_LIST, dtype=float)
    plt.figure(figsize=(7, 5.5))
    plt.loglog(Ns, errs1_rho, "o-", lw=2, label=r"$L^1(\rho)$ FVS 1st")
    plt.loglog(Ns, errs2_rho, "s-", lw=2, label=r"$L^1(\rho)$ FVS 2nd")
    plt.loglog(Ns, errs1_rho[0] * (Ns[0] / Ns), "k--", lw=1.5, label=r"slope $N^{-1}$")
    plt.xlabel(r"grid size $N$")
    plt.ylabel(r"$L^1$ error in $\rho$")
    plt.title(r"$L^1(\rho)$ vs. $N$ (Steger–Warming FVS + RK3)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fvs_l1_rho_vs_N.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5.5))
    plt.loglog(Ns, errs1_u, "o-", lw=2, label=r"$L^1(u)$ FVS 1st")
    plt.loglog(Ns, errs2_u, "s-", lw=2, label=r"$L^1(u)$ FVS 2nd")
    plt.xlabel(r"grid size $N$")
    plt.ylabel(r"$L^1$ error in $u$")
    plt.title(r"$L^1(u)$ vs. $N$")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fvs_l1_u_vs_N.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5.5))
    plt.loglog(Ns, errs1_p, "o-", lw=2, label=r"$L^1(p)$ FVS 1st")
    plt.loglog(Ns, errs2_p, "s-", lw=2, label=r"$L^1(p)$ FVS 2nd")
    plt.loglog(Ns, errs1_p[0] * (Ns[0] / Ns), "k--", lw=1.5, label=r"slope $N^{-1}$")
    plt.xlabel(r"grid size $N$")
    plt.ylabel(r"$L^1$ error in $p$")
    plt.title(r"$L^1(p)$ vs. $N$")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fvs_l1_p_vs_N.png"), dpi=200)
    plt.close()


def main():
    verify_split_consistency()
    print("Steger–Warming split F^+ + F^- = F: OK")
    print("Homework 7 Problem 2: FVS (1st/2nd space) + RK3, t_final =", T_FINAL)
    print("--- L1 errors ---")
    run_grid_study()
    print("Output directory:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
