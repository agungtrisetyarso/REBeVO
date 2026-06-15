"""
Corrected decoupling demonstration.

WHAT THIS SHOWS: the coherent advantage (Theorem 4.2) depends ONLY on the
eigenvalue gap delta and is INVARIANT under changes to the off-diagonal
coherence, whereas Wigner negativity depends on the coherence. Demonstrating
this requires TWO INDEPENDENT KNOBS:
    - delta : relative eigenvalue gap   -> drives the advantage
    - c     : coherence fraction        -> drives the negativity
A single delta-sweep (as in the original script) ties the two together and can
only ever show correlation, never decoupling.

CAVEAT (important): the state below is an ABSTRACT 2x2 block embedded in a
continuous-variable cat for visualization. It is NOT the MRTA optimizer state.
delta here is the gap of a hand-built 2-level block, not of the real
Hamiltonian's density matrix. This figure illustrates a general fact about
2x2-in-phase-space; it is not evidence about the MRTA problem itself.
"""
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib import cm

# ----------------------------------------------------------------------
def advantage(delta):
    """Theorem 4.2: per-pair coherent advantage = 1 - delta/arctanh(delta)."""
    delta = abs(delta)
    if delta < 1e-12:
        return 0.0
    return 1 - delta / np.arctanh(delta)

# ----------------------------------------------------------------------
def build_cv_cat(delta, c, N=40, alpha0=3.5):
    """
    Build an abstract cat state with INDEPENDENT gap and coherence.
      p_plus, p_minus = (1 +/- delta)/2     -> eigenvalue populations
      coher           = c * sqrt(p+ p-)     -> c in [0,1], 0=incoherent mixture,
                                                1=maximal (pure-state ceiling)
    Returns (rho_cv, min_eig) and asserts physicality.
    """
    p_plus  = (1 + delta) / 2
    p_minus = (1 - delta) / 2
    coher   = c * np.sqrt(p_plus * p_minus)        # <-- c is INDEPENDENT of delta

    psi_g = qt.coherent(N,  alpha0)
    psi_e = qt.coherent(N, -alpha0)

    rho = (p_plus  * psi_g * psi_g.dag()
         + p_minus * psi_e * psi_e.dag()
         + coher   * (psi_g * psi_e.dag() + psi_e * psi_g.dag()))

    rho = rho / rho.tr()                            # renormalize (overlap leakage)
    min_eig = np.min(rho.eigenenergies())
    return rho, min_eig

def negativity_volume(rho, xvec):
    """Wigner negativity volume  int |W| d^2 alpha - 1  (= 0 for classical states)."""
    W  = qt.wigner(rho, xvec, xvec)
    dx = xvec[1] - xvec[0]
    return np.sum(np.abs(W)) * dx**2 - 1.0

# ----------------------------------------------------------------------
def run_2d_sweep():
    deltas = np.linspace(0.02, 0.9, 16)
    cfracs = np.linspace(0.0, 0.95, 16)
    xvec   = np.linspace(-7, 7, 120)

    ADV = np.zeros((len(cfracs), len(deltas)))
    NEG = np.zeros((len(cfracs), len(deltas)))
    worst_min_eig = 0.0

    for i, c in enumerate(cfracs):
        for j, d in enumerate(deltas):
            rho, mineig = build_cv_cat(d, c)
            worst_min_eig = min(worst_min_eig, mineig)
            ADV[i, j] = advantage(d)            # depends on d only
            NEG[i, j] = negativity_volume(rho, xvec)

    # physicality guard
    assert worst_min_eig > -1e-6, f"Unphysical state encountered: min eig {worst_min_eig:.2e}"
    print(f"Positivity check passed: worst min-eigenvalue = {worst_min_eig:.2e}")

    # ---- diagnostics: variation along each axis ----
    # advantage variation along coherence axis (should be ~0)
    adv_var_along_c = np.mean(np.std(ADV, axis=0))
    adv_var_along_d = np.mean(np.std(ADV, axis=1))
    neg_var_along_c = np.mean(np.std(NEG, axis=0))
    neg_var_along_d = np.mean(np.std(NEG, axis=1))
    print("\nDecoupling diagnostics (mean std along each axis):")
    print(f"  Advantage  varies along coherence axis: {adv_var_along_c:.5f}  (expect ~0)")
    print(f"  Advantage  varies along gap axis:       {adv_var_along_d:.5f}  (expect large)")
    print(f"  Negativity varies along coherence axis: {neg_var_along_c:.5f}  (expect large)")
    print(f"  Negativity varies along gap axis:       {neg_var_along_d:.5f}")

    # ---- figure: two heatmaps with shared structure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    im0 = axes[0].imshow(ADV, origin='lower', aspect='auto',
                         extent=[deltas[0], deltas[-1], cfracs[0], cfracs[-1]],
                         cmap='viridis')
    axes[0].set_title('Coherent advantage (Thm 4.2)\nflat along coherence axis')
    axes[0].set_xlabel(r'eigenvalue gap  $\delta$')
    axes[0].set_ylabel(r'coherence fraction  $c$')
    fig.colorbar(im0, ax=axes[0], label='advantage')

    im1 = axes[1].imshow(NEG, origin='lower', aspect='auto',
                         extent=[deltas[0], deltas[-1], cfracs[0], cfracs[-1]],
                         cmap='magma')
    axes[1].set_title('Wigner negativity volume\nvaries along coherence axis')
    axes[1].set_xlabel(r'eigenvalue gap  $\delta$')
    axes[1].set_ylabel(r'coherence fraction  $c$')
    fig.colorbar(im1, ax=axes[1], label=r'$\int|W|\,d^2\alpha - 1$')

    fig.suptitle('Decoupling (abstract 2x2-in-CV illustration, NOT the MRTA state):\n'
                 r'advantage depends only on $\delta$; negativity depends on coherence $c$',
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('/home/claude/decoupling_2d.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ---- companion line plot: the two clean 1D cuts ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # cut 1: fix delta, sweep c -> advantage flat, negativity moves
    d_fix = deltas[10]
    adv_cut = [advantage(d_fix) for _ in cfracs]
    neg_cut = [negativity_volume(build_cv_cat(d_fix, c)[0], xvec) for c in cfracs]
    a1.plot(cfracs, adv_cut, 'o-', color='blue', label='advantage')
    a1.plot(cfracs, neg_cut, 's--', color='red', label='negativity')
    a1.set_title(rf'Fix gap $\delta={d_fix:.2f}$, sweep coherence $c$')
    a1.set_xlabel(r'coherence fraction $c$'); a1.set_ylabel('value'); a1.legend(); a1.grid(alpha=0.3)
    a1.annotate('advantage INVARIANT\nunder coherence', xy=(0.5, adv_cut[0]),
                xytext=(0.3, max(neg_cut)*0.5), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))

    # cut 2: fix c (physically), sweep delta -> advantage moves
    c_fix = cfracs[8]
    adv_cut2 = [advantage(d) for d in deltas]
    neg_cut2 = [negativity_volume(build_cv_cat(d, c_fix)[0], xvec) for d in deltas]
    a2.plot(deltas, adv_cut2, 'o-', color='blue', label='advantage')
    a2.plot(deltas, neg_cut2, 's--', color='red', label='negativity')
    a2.set_title(rf'Fix coherence $c={c_fix:.2f}$, sweep gap $\delta$')
    a2.set_xlabel(r'eigenvalue gap $\delta$'); a2.set_ylabel('value'); a2.legend(); a2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/decoupling_cuts.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nsaved decoupling_2d.png and decoupling_cuts.png")

if __name__ == "__main__":
    run_2d_sweep()
