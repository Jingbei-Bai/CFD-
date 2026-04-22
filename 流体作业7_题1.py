"""
Homework 7, Problem 1: Jameson-type central scheme (central flux + 2nd/4th-order
artificial viscosity) for the same 1-D Sod shock tube as in 流体作业6.py.

All figure text (titles, axes, legends) is in English. Outputs: multi-N profiles,
zoom near discontinuities, L1 error vs. N, and k2/k4 sweeps as overlays of all
curves on one figure (rho, u, p) near discontinuities.
"""

import math
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

OUTPUT_DIR = "jameson_hw7_q1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

T_FINAL = 0.3
# 中心格式对时间步较敏感，略保守的 CFL 更稳
CFL = 0.35
GRID_LIST = [100, 200, 400, 800, 1000]
# K-sweep: wide window (like main zoom); contact window centres on exact contact
SWEEP_WIDE_LO, SWEEP_WIDE_HI = -0.2, 0.2
SWEEP_CONTACT_HALF = 0.14  # half-width for local discontinuity zoom [x_c ± half]


def exact_at_t(x, t):
    rho = np.zeros_like(x, dtype=float)
    u = np.zeros_like(x, dtype=float)
    p = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        rho[i], u[i], p[i] = hw6.exact_sample(xi, t)
    return rho, u, p


def pressure_switch(p):
    """Jameson 压强开关 ν∈[0,1]；光滑区接近 0，间断附近增大。边界用邻点延拓。"""
    nu = np.zeros_like(p)
    num = np.abs(p[2:] - 2.0 * p[1:-1] + p[:-2])
    den = np.abs(p[2:]) + 2.0 * np.abs(p[1:-1]) + np.abs(p[:-2]) + 1e-20
    nu[1:-1] = num / den
    nu[0] = nu[1]
    nu[-1] = nu[-2]
    return nu


def spectral_radius(rho, u, p):
    c = np.sqrt(gamma * p / rho)
    return np.abs(u) + c


def rhs_jameson(U_int, dx, k2, k4):
    """
    内部单元 U_int 形状 (3, N)；两侧各 2 层虚拟单元，零阶外推。

    半离散：dU/dt = -∂F/∂x（中心差分）+ 人工粘性。

    二阶（激波捕捉）：界面耗散系数 α₂ = k₂·Δx·|λ|·ν，
    D₂ = ((α₂)_{j+1/2}(U_{j+1}-U_j) - (α₂)_{j-1/2}(U_j-U_{j-1})) / Δx²。

    四阶（光滑区抑制振荡）：采用与 Jameson 经典形式等价的尺度
    ε₄ ∼ k₄·Δx³·|λ|·(1-ν)，离散为 D₄ = -k₄·(1-ν)·|λ|·δ⁴U / Δx
    （δ⁴ 为未除以 Δx 的五点四阶差分），避免出现 δ⁴U/Δx⁴ 导致的溢出。
    """
    assert U_int.shape[0] == 3
    N = U_int.shape[1]
    pad = 2
    Ue = np.zeros((3, N + 2 * pad))
    Ue[:, pad : pad + N] = U_int
    for k in range(pad):
        Ue[:, k] = Ue[:, pad]
        Ue[:, pad + N + k] = Ue[:, pad + N - 1]

    Ue[0] = np.maximum(Ue[0], 1e-14)
    rho, u, p = conservative_to_primitive(Ue)
    p = np.maximum(p, 1e-14)
    lam = spectral_radius(rho, u, p)
    nu = pressure_switch(p)

    Fc = flux(Ue)
    Rc = np.zeros_like(U_int)
    for j in range(pad, pad + N):
        Rc[:, j - pad] = (Fc[:, j + 1] - Fc[:, j - 1]) / (2.0 * dx)

    # 界面 j 位于 Ue[:,j] 与 Ue[:,j+1] 之间，j = 0 .. N+2
    lam_ip = 0.5 * (lam[:-1] + lam[1:])
    nu_ip = np.maximum(nu[:-1], nu[1:])
    dU_ip = Ue[:, 1:] - Ue[:, :-1]
    alpha2_ip = k2 * dx * lam_ip * nu_ip

    D2 = np.zeros_like(U_int)
    D4 = np.zeros_like(U_int)
    for j in range(pad, pad + N):
        D2[:, j - pad] = (
            alpha2_ip[j] * dU_ip[:, j] - alpha2_ip[j - 1] * dU_ip[:, j - 1]
        ) / (dx**2)

        jj = j
        d4 = (
            Ue[:, jj + 2]
            - 4.0 * Ue[:, jj + 1]
            + 6.0 * Ue[:, jj]
            - 4.0 * Ue[:, jj - 1]
            + Ue[:, jj - 2]
        )
        damp = max(0.0, 1.0 - nu[jj])
        D4[:, j - pad] = -k4 * damp * lam[jj] * d4 / dx

    return -Rc + D2 + D4


def rk3_step(U, dx, dt, k2, k4):
    def L(U_):
        return rhs_jameson(U_, dx, k2, k4)

    L0 = L(U)
    U1 = U + dt * L0
    U2 = 0.75 * U + 0.25 * U1 + 0.25 * dt * L(U1)
    U3 = (1.0 / 3.0) * U + (2.0 / 3.0) * U2 + (2.0 / 3.0) * dt * L(U2)
    return U3


def jameson_solver(N, k2, k4, t_final=T_FINAL):
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
        U = rk3_step(U, dx, dt, k2, k4)
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
    return e_rho, e_u, e_p, rho_ex, u_ex, p_ex


def run_grid_study(k2=0.25, k4=0.06):
    errs_rho, errs_u, errs_p = [], [], []
    results = {}
    for N in GRID_LIST:
        x, rho, u, p = jameson_solver(N, k2, k4, T_FINAL)
        er, eu, ep, rex, uex, pex = l1_errors(x, rho, u, p, T_FINAL)
        errs_rho.append(er)
        errs_u.append(eu)
        errs_p.append(ep)
        results[N] = dict(x=x, rho=rho, u=u, p=p, rho_ex=rex, u_ex=uex, p_ex=pex)
        print(f"N={N:5d}  L1(rho)={er:.6e}  L1(u)={eu:.6e}  L1(p)={ep:.6e}")

    x_ref = np.linspace(xL, xR, 4000)
    rref, uref, pref = exact_at_t(x_ref, T_FINAL)

    ref_profiles = {"rho": rref, "u": uref, "p": pref}
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, key, lab in zip(
        axes,
        ("rho", "u", "p"),
        (r"$\rho$", r"$u$", r"$p$"),
    ):
        for N in GRID_LIST:
            ax.plot(results[N]["x"], results[N][key], lw=1.0, label=f"N={N}")
        ax.plot(x_ref, ref_profiles[key], "k--", lw=1.6, label="Exact")
        ax.set_ylabel(lab)
        ax.grid(True)
    axes[0].set_title(
        rf"Jameson-type scheme (central + artificial viscosity), $t={T_FINAL}$, "
        rf"$k_2={k2}$, $k_4={k4}$"
    )
    axes[-1].set_xlabel(r"$x$")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "jameson_profiles_multiN.png"),
        dpi=200,
    )
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, key, lab in zip(axes, ("rho", "u", "p"), (r"$\rho$", r"$u$", r"$p$")):
        for N in GRID_LIST:
            ax.plot(results[N]["x"], results[N][key], lw=1.0, label=f"N={N}")
        ax.plot(
            x_ref,
            {"rho": rref, "u": uref, "p": pref}[key],
            "k--",
            lw=1.6,
            label="Exact",
        )
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylabel(lab)
        ax.grid(True)
    axes[0].set_title(r"Zoom near discontinuities ($x \in [-0.2,\,0.2]$)")
    axes[-1].set_xlabel(r"$x$")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "jameson_zoom_contact.png"),
        dpi=200,
    )
    plt.close(fig)

    Ns = np.asarray(GRID_LIST, dtype=float)
    ref = errs_rho[0] * (Ns[0] / Ns)
    plt.figure(figsize=(7, 5.5))
    plt.loglog(Ns, errs_rho, "o-", lw=2, label=r"$L^1(\rho)$")
    plt.loglog(Ns, errs_u, "s-", lw=2, label=r"$L^1(u)$")
    plt.loglog(Ns, errs_p, "^-", lw=2, label=r"$L^1(p)$")
    plt.loglog(Ns, ref, "k--", lw=1.5, label=r"reference slope $N^{-1}$")
    plt.xlabel(r"grid size $N$")
    plt.ylabel(r"$L^1$ error")
    plt.title(rf"$L^1$ error vs. $N$ ($k_2={k2}$, $k_4={k4}$)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "jameson_error_vs_N.png"), dpi=200)
    plt.close()


def _exact_wave_positions(t):
    """左波特征位置、接触间断 \(x_c\)、右波结构位置（与作业6 精确 Riemann 解一致）。"""
    p_star, u_star = hw6.star_pu(rhoL, uL, pL, rhoR, uR, pR)
    aL = math.sqrt(gamma * pL / rhoL)
    x_contact, x_right = hw6.exact_wave_locations(t)
    if p_star <= pL:
        x_left = x0 + (uL - aL) * t
    else:
        qL = math.sqrt(
            1.0 + (gamma + 1.0) / (2.0 * gamma) * (p_star / pL - 1.0)
        )
        x_left = x0 + (uL - aL * qL) * t
    return x_left, x_contact, x_right


def _add_exact_wave_vlines(ax, t, *, show_wave_legend: bool):
    x_left, x_c, x_r = _exact_wave_positions(t)
    lines = [
        (x_left, "0.45", "left wave (exact)"),
        (x_c, "C0", "contact (exact)"),
        (x_r, "C3", "right wave (exact)"),
    ]
    for xv, c, lab in lines:
        ax.axvline(
            xv,
            color=c,
            ls="--",
            lw=1.25,
            zorder=2,
            alpha=0.92,
            label=lab if show_wave_legend else "_nolegend_",
        )


def run_viscosity_sweep(N=400):
    """
    Each k-sweep: wide window + contact zoom; rho,u,p; dashed lines = exact wave
    positions from the Riemann solution.
    """
    k2_list = [0.05, 0.12, 0.2, 0.3, 0.45]
    k4_list = [0.0, 0.02, 0.05, 0.1, 0.15]
    assert len(k2_list) >= 4 and len(k4_list) >= 4
    k4_fixed_for_k2_sweep = 0.06
    k2_fixed_for_k4_sweep = 0.25

    x_ref = np.linspace(xL, xR, 4000)
    rref, uref, pref = exact_at_t(x_ref, T_FINAL)
    _, x_contact, _ = _exact_wave_positions(T_FINAL)
    xlim_contact = (
        x_contact - SWEEP_CONTACT_HALF,
        x_contact + SWEEP_CONTACT_HALF,
    )
    xlim_wide = (SWEEP_WIDE_LO, SWEEP_WIDE_HI)

    def plot_overlay_sweep(
        sweep_name,
        param_values,
        param_label,
        fixed_k2,
        fixed_k4,
        vary_k2,
        *,
        xlim,
        window_tag,
    ):
        """vary_k2=True: sweep k2 with fixed_k4; else sweep k4 with fixed_k2."""
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        n = len(param_values)
        colors = plt.cm.tab10(np.linspace(0, 0.95, n))
        keys = ("rho", "u", "p")
        ylabs = (r"$\rho$", r"$u$", r"$p$")
        refs = (rref, uref, pref)

        for ax, key, ylab, yref in zip(axes, keys, ylabs, refs):
            for i, val in enumerate(param_values):
                if vary_k2:
                    x, rho, u, p = jameson_solver(N, val, fixed_k4, T_FINAL)
                else:
                    x, rho, u, p = jameson_solver(N, fixed_k2, val, T_FINAL)
                y = {"rho": rho, "u": u, "p": p}[key]
                ax.plot(
                    x,
                    y,
                    lw=1.25,
                    color=colors[i],
                    label=rf"${param_label}={val}$",
                    zorder=1,
                )
            ax.plot(x_ref, yref, "k--", lw=1.6, label="Exact", zorder=1)
            _add_exact_wave_vlines(ax, T_FINAL, show_wave_legend=(ax is axes[-1]))
            ax.set_ylabel(ylab)
            ax.set_xlim(xlim[0], xlim[1])
            ax.grid(True, zorder=0)
            ax.legend(fontsize=6.5, ncol=2, loc="upper right")

        te = (
            rf"$k_4={fixed_k4}$ fixed, sweep $k_2$"
            if vary_k2
            else rf"$k_2={fixed_k2}$ fixed, sweep $k_4$"
        )
        axes[0].set_title(rf"Jameson: {te}, $N={N}$, $t={T_FINAL}$ | {window_tag}")
        axes[-1].set_xlabel(r"$x$")
        fig.tight_layout()
        fn = f"Jameson_sweep_{sweep_name}_rho_u_p_{window_tag}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, fn), dpi=200)
        plt.close(fig)

    for sn, vals, plab, vk2, fk2, fk4 in (
        ("k2", k2_list, "k_2", True, None, k4_fixed_for_k2_sweep),
        ("k4", k4_list, "k_4", False, k2_fixed_for_k4_sweep, None),
    ):
        plot_overlay_sweep(
            sn,
            vals,
            plab,
            fk2,
            fk4,
            vk2,
            xlim=xlim_wide,
            window_tag="wide",
        )
        plot_overlay_sweep(
            sn,
            vals,
            plab,
            fk2,
            fk4,
            vk2,
            xlim=xlim_contact,
            window_tag="contact",
        )


def main():
    print("Homework 7, Problem 1: Jameson-type scheme + artificial viscosity, t_final =", T_FINAL)
    print("Default coefficients: k2 = 0.25, k4 = 0.06 (see rhs_jameson for dx scaling).")
    print("--- L1 errors for several grid sizes ---")
    run_grid_study(k2=0.25, k4=0.06)
    print(
        "--- k sweeps: Jameson_sweep_k{2,4}_rho_u_p_{wide,contact}.png ---"
    )
    run_viscosity_sweep(N=400)
    print("Output directory:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
