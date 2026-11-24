"""Per-plane metrics in float64 against the ORIGINAL HR truth (plan 7).

All functions take physical-unit float64 arrays unless stated.  Normalized
quantities use the frozen group scale U_g and length ell_g.

    E_v   = sum (z - q*)^2 / sum (q* - mu)^2            primary speed error
    E_d   = (ell/U)^2 * mean (D z - D q*)^2             divergence error
    A_d   = (ell/U)   * sqrt mean (D z)^2               predicted divergence rms
    A_d*  = (ell/U)   * sqrt mean (D q*)^2              reference divergence rms
    a     = <d, d*> / <d*, d*>  divergence-map gain;  r_d = d - a d*
    chi2  = mean (D q*)^2 / sum_c sum_b mean (d_b q_c*)^2
    C_r z - y consistency = mean (block_avg(z) - y)^2 / U_g^2
"""
import numpy as np

from . import spec as S
from .data import block_average
from .operator_torch import divergence_numpy, gradient_numpy


def e_v(z64, q64, mu64):
    num = float(np.sum((z64 - q64) ** 2))
    den = float(np.sum((q64 - mu64[:, None, None]) ** 2))
    return num, den, (num / den if den > 0 else float("nan"))


def per_component_errors(z64, q64, mu64):
    out = {}
    for c in range(2):
        num = float(np.sum((z64[c] - q64[c]) ** 2))
        den = float(np.sum((q64[c] - mu64[c]) ** 2))
        out[c] = (num, den, num / den if den > 0 else float("nan"))
    return out


def divergence_metrics(z64, q64, grid, U_g, ell_g):
    """E_d, A_d, A_d* plus raw physical divergence maps and gain/residual."""
    Nx, Ns, Lx, Ls = grid["Nx"], grid["Ns"], grid["Lx"], grid["Ls"]
    dz = divergence_numpy(z64, Nx, Ns, Lx, Ls)
    dq = divergence_numpy(q64, Nx, Ns, Lx, Ls)
    E_d = (ell_g / U_g) ** 2 * float(np.mean((dz - dq) ** 2))
    A_d = (ell_g / U_g) * float(np.sqrt(np.mean(dz ** 2)))
    A_d_star = (ell_g / U_g) * float(np.sqrt(np.mean(dq ** 2)))
    ref_energy = float(np.sum(dq * dq))
    gain_a = float(np.sum(dz * dq) / ref_energy) if ref_energy > 0 else float("nan")
    if ref_energy > 0:
        rd = dz - gain_a * dq
        r_energy = float(np.sum(rd * rd))
    else:
        rd, r_energy = None, float("nan")
    return {
        "E_d": E_d, "A_d": A_d, "A_d_star": A_d_star,
        "gain_a": gain_a, "residual_energy": r_energy,
        "ref_energy": ref_energy,
        "identity_check": ((gain_a - 1.0) ** 2 * ref_energy + r_energy
                           - float(np.sum((dz - dq) ** 2)))
        if ref_energy > 0 else float("nan"),
    }


def chi_squared(q64, grid):
    """True divergence share of all in-plane velocity gradients (plan 7.3).

    chi2 = mean (D q*)^2 / sum_c sum_{b in {x,s}} mean (d_b q_c*)^2.
    Returns (chi2, numerator, denominator); the caller marks unresolved
    denominators (numerically indistinguishable from zero) explicitly.
    """
    Nx, Ns, Lx, Ls = grid["Nx"], grid["Ns"], grid["Lx"], grid["Ls"]
    dq = divergence_numpy(q64, Nx, Ns, Lx, Ls)
    num = float(np.mean(dq ** 2))
    grads = gradient_numpy(q64, Nx, Ns, Lx, Ls)
    den = float(sum(np.mean(g ** 2) for g in grads.values()))
    return (num / den if den > 0 else float("nan")), num, den


def observation_consistency(z64, y64, U_g):
    """mean (block_avg_r(z) - y)^2 / U_g^2 — is the LR input still fitted."""
    if z64.shape[-1] != y64.shape[-1]:
        r = z64.shape[-1] // y64.shape[-1]
        cz = block_average(z64, r)
    else:
        cz = z64
    return float(np.mean((cz - y64) ** 2)) / U_g ** 2


def plane_metrics(z64, q64, y64, mu64, grid, U_g, ell_g):
    """Full per-plane metric dict (plan 7.1-7.4)."""
    num, den, ev = e_v(z64, q64, mu64)
    d = {
        "e_v_num": num, "e_v_den": den, "E_v": ev,
        "sqrt_E_v": float(np.sqrt(ev)) if ev >= 0 else float("nan"),
    }
    d.update(divergence_metrics(z64, q64, grid, U_g, ell_g))
    chi2, chi2_num, chi2_den = chi_squared(q64, grid)
    d.update({"chi2": chi2, "chi2_num": chi2_num, "chi2_den": chi2_den})
    d["obs_consistency"] = observation_consistency(z64, y64, U_g)
    for c, (n, dd, r) in per_component_errors(z64, q64, mu64).items():
        d["E_v_c{}".format(c)] = r
    return d


def equal_time_mean(per_plane, dataset):
    """Iso: mean positions within a time first, then times. Channel: all times."""
    by_m = {}
    for (m, _c), v in per_plane.items():
        by_m.setdefault(m, []).append(v)
    if dataset.startswith("isotropic"):
        return float(np.mean([np.mean(vs) for vs in by_m.values()]))
    return float(np.mean([v for vs in by_m.values() for v in vs]))
