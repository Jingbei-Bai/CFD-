import os
import numpy as np
import math
import matplotlib.pyplot as plt

gamma = 1.4
CFL = 0.8

# ----------------------------
# 初始条件（Sod shock tube）
# ----------------------------
rhoL, uL, pL = 1.0, 0.75, 1.0
rhoR, uR, pR = 0.125, 0.0, 0.1

xL, xR = -0.5, 0.5
x0 = 0.0
tf = 0.3
TIME_LIST = [0.25, 0.30, 0.35]
OUTPUT_DIR = "rusanov_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 基本函数
# ----------------------------
def primitive_to_conservative(rho, u, p):
    """从原始变量到守恒变量，返回形状 (3, N)"""
    rhou = rho * u
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return np.array([rho, rhou, E])


def conservative_to_primitive(U):
    """从守恒变量到原始变量，U 形状为 (3, N)"""
    rho, rhou, E = U
    u = rhou / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
    return rho, u, p


def flux(U):
    """欧拉方程通量，U 形状可为 (3,) 或 (3, N)"""
    rho, rhou, E = U
    u = rhou / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
    F = np.zeros_like(U)
    F[0] = rhou
    F[1] = rhou * u + p
    F[2] = u * (E + p)
    return F


def rusanov_flux(UL, UR):
    """Rusanov 数值通量，UL/UR 形状为 (3,)"""
    rhoL, rhouL, EL = UL
    uL_loc = rhouL / rhoL
    pL_loc = (gamma - 1.0) * (EL - 0.5 * rhoL * uL_loc**2)
    cL = np.sqrt(gamma * pL_loc / rhoL)

    rhoR, rhouR, ER = UR
    uR_loc = rhouR / rhoR
    pR_loc = (gamma - 1.0) * (ER - 0.5 * rhoR * uR_loc**2)
    cR = np.sqrt(gamma * pR_loc / rhoR)

    lambda_max = max(abs(uL_loc) + cL, abs(uR_loc) + cR)
    FL = flux(UL)
    FR = flux(UR)
    return 0.5 * (FL + FR) - 0.5 * lambda_max * (UR - UL)

# ----------------------------
# Rusanov 格式
# ----------------------------
def rusanov_solver(N):
    dx = (xR - xL) / N
    x = xL + (np.arange(N) + 0.5) * dx

    rho = np.where(x < x0, rhoL, rhoR)
    u = np.where(x < x0, uL, uR)
    p = np.where(x < x0, pL, pR)
    U = primitive_to_conservative(rho, u, p)

    t = 0.0
    while t < tf:
        rho, u, p = conservative_to_primitive(U)
        c = np.sqrt(gamma * p / rho)
        smax = np.max(np.abs(u) + c)
        dt = CFL * dx / smax
        if t + dt > tf:
            dt = tf - t

        # 零梯度边界条件（保持你当前网格与计算位置设置不变）
        U_left = U[:, [0]]
        U_right = U[:, [-1]]
        U_ext = np.concatenate([U_left, U, U_right], axis=1)

        # 逐界面计算数值通量
        F = np.zeros((3, N + 1))
        for i in range(N + 1):
            F[:, i] = rusanov_flux(U_ext[:, i], U_ext[:, i + 1])

        U = U - dt / dx * (F[:, 1:] - F[:, :-1])
        t += dt

    rho, u, p = conservative_to_primitive(U)
    return x, rho, u, p

# ----------------------------
# 精确 Riemann 解（Toro 标准算法）
# ----------------------------
def prefun(p, dk, pk, ak):
    if p <= pk:   # rarefaction
        pr = p / pk
        f = (2.0 * ak / (gamma - 1.0)) * (pr ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        fd = (1.0 / (dk * ak)) * pr ** (-(gamma + 1.0) / (2.0 * gamma))
    else:         # shock
        A = 2.0 / ((gamma + 1.0) * dk)
        B = (gamma - 1.0) / (gamma + 1.0) * pk
        sq = math.sqrt(A / (p + B))
        f = (p - pk) * sq
        fd = sq * (1.0 - 0.5 * (p - pk) / (p + B))
    return f, fd

def star_pu(rhoL, uL, pL, rhoR, uR, pR):
    aL = math.sqrt(gamma * pL / rhoL)
    aR = math.sqrt(gamma * pR / rhoR)
    p = max(1e-8, 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * (aL + aR))
    for _ in range(50):
        fL, fdL = prefun(p, rhoL, pL, aL)
        fR, fdR = prefun(p, rhoR, pR, aR)
        p_new = p - (fL + fR + uR - uL) / (fdL + fdR)
        if abs(p_new - p) / (0.5 * (p_new + p)) < 1e-8:
            p = max(1e-8, p_new)
            break
        p = max(1e-8, p_new)
    u = 0.5 * (uL + uR + fR - fL)
    return p, u

def exact_sample(x, t):
    if t == 0:
        if x < x0:
            return rhoL, uL, pL
        else:
            return rhoR, uR, pR

    p_star, u_star = star_pu(rhoL, uL, pL, rhoR, uR, pR)
    aL = math.sqrt(gamma * pL / rhoL)
    aR = math.sqrt(gamma * pR / rhoR)
    s = (x - x0) / t

    # 左波
    if p_star <= pL:
        # 左稀疏波
        shL = uL - aL
        a_starL = aL * (p_star / pL) ** ((gamma - 1.0) / (2.0 * gamma))
        stL = u_star - a_starL
    else:
        qL = math.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (p_star / pL - 1.0))
        sL = uL - aL * qL

    # 右波
    if p_star > pR:
        qR = math.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (p_star / pR - 1.0))
        sR = uR + aR * qR
    else:
        shR = uR + aR
        a_starR = aR * (p_star / pR) ** ((gamma - 1.0) / (2.0 * gamma))
        stR = u_star + a_starR

    if s <= u_star:
        # 左侧
        if p_star <= pL:
            if s <= shL:
                return rhoL, uL, pL
            elif s > stL:
                rho = rhoL * (p_star / pL) ** (1.0 / gamma)
                return rho, u_star, p_star
            else:
                u = 2.0 / (gamma + 1.0) * (aL + 0.5 * (gamma - 1.0) * uL + s)
                a = 2.0 / (gamma + 1.0) * (aL + 0.5 * (gamma - 1.0) * (uL - s))
                rho = rhoL * (a / aL) ** (2.0 / (gamma - 1.0))
                p = pL * (a / aL) ** (2.0 * gamma / (gamma - 1.0))
                return rho, u, p
        else:
            if s <= sL:
                return rhoL, uL, pL
            else:
                rho = rhoL * ((p_star / pL) + (gamma - 1.0) / (gamma + 1.0)) / \
                      (((gamma - 1.0) / (gamma + 1.0)) * (p_star / pL) + 1.0)
                return rho, u_star, p_star
    else:
        # 右侧
        if p_star > pR:
            if s >= sR:
                return rhoR, uR, pR
            else:
                rho = rhoR * ((p_star / pR) + (gamma - 1.0) / (gamma + 1.0)) / \
                      (((gamma - 1.0) / (gamma + 1.0)) * (p_star / pR) + 1.0)
                return rho, u_star, p_star
        else:
            if s >= shR:
                return rhoR, uR, pR
            elif s <= stR:
                rho = rhoR * (p_star / pR) ** (1.0 / gamma)
                return rho, u_star, p_star
            else:
                u = 2.0 / (gamma + 1.0) * (-aR + 0.5 * (gamma - 1.0) * uR + s)
                a = 2.0 / (gamma + 1.0) * (aR - 0.5 * (gamma - 1.0) * (uR - s))
                rho = rhoR * (a / aR) ** (2.0 / (gamma - 1.0))
                p = pR * (a / aR) ** (2.0 * gamma / (gamma - 1.0))
                return rho, u, p

def exact_solution(x):
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)
    for i, xi in enumerate(x):
        rho[i], u[i], p[i] = exact_sample(xi, tf)
    return rho, u, p


def exact_wave_locations(t):
    """返回接触间断和右行波位置，用于局部放大。"""
    p_star, u_star = star_pu(rhoL, uL, pL, rhoR, uR, pR)
    aR = math.sqrt(gamma * pR / rhoR)
    x_contact = x0 + u_star * t

    if p_star > pR:
        qR = math.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * (p_star / pR - 1.0))
        sR = uR + aR * qR
        x_right = x0 + sR * t
    else:
        # 右稀疏波时取波头作为右侧局部放大中心
        shR = uR + aR
        x_right = x0 + shR * t
    return x_contact, x_right


def time_tag(t):
    return f"t{int(round(100 * t)):03d}"

def run_for_time(t_final):
    global tf
    tf = t_final

    grid_list = [100, 200, 400, 800, 1000, 4000]
    results = {}
    err_rho_list, err_u_list, err_p_list = [], [], []

    print(f"\n===== t = {tf} =====")
    print("   N        L1(rho)      L1(u)        L1(p)")
    for N in grid_list:
        x, rho, u, p = rusanov_solver(N)
        rho_ex, u_ex, p_ex = exact_solution(x)
        dx = (xR - xL) / N
        err_rho = dx * np.sum(np.abs(rho - rho_ex))
        err_u = dx * np.sum(np.abs(u - u_ex))
        err_p = dx * np.sum(np.abs(p - p_ex))
        print(f"{N:6d}   {err_rho:12.6e} {err_u:12.6e} {err_p:12.6e}")

        results[N] = {
            "x": x,
            "rho": rho,
            "u": u,
            "p": p,
            "rho_ex": rho_ex,
            "u_ex": u_ex,
            "p_ex": p_ex,
        }
        err_rho_list.append(err_rho)
        err_u_list.append(err_u)
        err_p_list.append(err_p)

    x_ref = np.linspace(xL, xR, 5000)
    rho_ref, u_ref, p_ref = exact_solution(x_ref)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    labels = [r"$\rho$", r"$u$", r"$p$"]
    keys = ["rho", "u", "p"]
    for k, ax in enumerate(axes):
        for N in grid_list:
            ax.plot(results[N]["x"], results[N][keys[k]], lw=1.0, label=f"N={N}")
        ax.plot(x_ref, [rho_ref, u_ref, p_ref][k], "k--", lw=2.0, label="Exact")
        ax.set_ylabel(labels[k])
        ax.set_xlabel(r"$x$")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True)
    axes[0].set_title(f"Rusanov solution at t = {tf}")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"rusanov_profiles_{time_tag(tf)}.png"), dpi=200)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for k, ax in enumerate(axes):
        for N in grid_list:
            ax.plot(results[N]["x"], results[N][keys[k]], lw=1.1, label=f"N={N}")
        ax.plot(x_ref, [rho_ref, u_ref, p_ref][k], "k--", lw=2.0, label="Exact")
        ax.set_ylabel(labels[k])
        ax.set_xlabel(r"$x$")
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xlim(-0.2, 0.2)
        ax.grid(True)
    axes[0].set_title(f"Central region zoom at t = {tf} (x in [-0.2, 0.2])")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"rusanov_contact_zoom_density_{time_tag(tf)}.png"), dpi=200)

    x_contact, x_right = exact_wave_locations(tf)
    half_width = 0.08
    x_min = max(xL, x_right - half_width)
    x_max = min(xR, x_right + half_width)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for k, ax in enumerate(axes):
        for N in grid_list:
            ax.plot(results[N]["x"], results[N][keys[k]], lw=1.1, label=f"N={N}")
        ax.plot(x_ref, [rho_ref, u_ref, p_ref][k], "k--", lw=2.0, label="Exact")
        ax.set_ylabel(labels[k])
        ax.set_xlabel(r"$x$")
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xlim(0.2, 0.5)
        ax.grid(True)
    axes[0].set_title(f"Right local zoom at t = {tf} (x_right ≈ {x_right:.4f})")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"rusanov_right_zoom_{time_tag(tf)}.png"), dpi=200)

    N_arr = np.asarray(grid_list, dtype=float)
    ref_line = err_rho_list[0] * (N_arr[0] / N_arr)
    plt.figure(figsize=(8, 6))
    plt.loglog(N_arr, err_rho_list, "o-", lw=2, label=r"$L^1$ error of $\rho$")
    plt.loglog(N_arr, err_u_list, "s-", lw=2, label=r"$L^1$ error of $u$")
    plt.loglog(N_arr, err_p_list, "^-", lw=2, label=r"$L^1$ error of $p$")
    plt.loglog(N_arr, ref_line, "k--", lw=1.8, label=r"reference slope $N^{-1}$")
    plt.xlabel("N")
    plt.ylabel(r"$L^1$ error")
    plt.title(f"Error vs grid number at t = {tf}")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"rusanov_error_{time_tag(tf)}.png"), dpi=200)


if __name__ == "__main__":
    for t_val in TIME_LIST:
        run_for_time(t_val)

    plt.show()