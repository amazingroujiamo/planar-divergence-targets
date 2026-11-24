"""Three loss groups on normalized fields (plan 4.2).

    L_A = mean_{c,i,j} (z' - q*')^2
    L_Z = L_A + lam * mean_{i,j} (D_nd z')^2
    L_T = L_A + lam * mean_{i,j} (D_nd z' - D_nd q*')^2

D_nd = ell_g * D_physical acting on U_g-normalized velocity; the reference
map D_nd q*' is precomputed by the SAME torch operator under no_grad (cached
in PlaneStore).  lam = 0 reduces all three to L_A; the runner enforces that
the baseline is a single shared configuration (target 'A').
"""


def loss_terms(z, q, dref, div_op, lam, target):
    """Return (L_total, L_v, div_raw, div_weighted, dz_map).

    z, q: (B, 2, Ns, Nx) float32 normalized.  dref: (B, Ns, Nx) or None
    (required for target 'T').  div_raw is the UNWEIGHTED divergence term.
    """
    L_v = (z - q).pow(2).mean()
    if target == "A" or lam == 0.0:
        return L_v, L_v, None, None, None
    dz = div_op(z)
    if target == "Z":
        div_raw = dz.pow(2).mean()
    elif target == "T":
        div_raw = (dz - dref).pow(2).mean()
    else:
        raise ValueError("unknown target {}".format(target))
    total = L_v + lam * div_raw
    return total, L_v, div_raw, lam * div_raw, dz
