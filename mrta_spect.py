import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

np.random.seed(2)

# ============================================================
#  Quantized Acemoglu-Restrepo task-based production as H(m)
# ------------------------------------------------------------
#  N tasks, one qubit each: |1> = task done by LABOUR, |0> = by CAPITAL (robot).
#  Task output:  M(i) = A_L * gamma_L(i) if labour ; A_K * gamma_K(i) if capital.
#  CES aggregate: Y = ( (1/N) sum_i M(i)^((s-1)/s) )^(s/(s-1)),  s = elasticity sigma.
#  Diagonal "energy":  H_diag = -Y(config)  (we MINIMIZE -Y, i.e. maximize output).
#  Off-diagonal: sigma-controlled transverse field  Gamma * sum_i sigma^x_i,
#                Gamma = Gamma0 / s  (higher substitutability s => more mixing).
#  "Robots" m  := number of capital-tasks available, imposed as a CONSTRAINT-style
#                penalty that at most m tasks may be capital (>= N-m must be labour).
#                Degeneracy is expected from permutation symmetry among identical
#                capital units (S_m) once m is large enough to form multiplets.
# ============================================================

# ---- comparative-advantage profiles (A-R: capital better at "routine" low-i tasks) ----
def profiles(N):
    i = np.arange(N)
    gamma_L = 1.0 + 0.6*np.sin(2*np.pi*i/N)          # labour comparative adv varies
    gamma_K = 1.0 + 0.6*np.cos(2*np.pi*i/N)          # capital comparative adv varies
    return gamma_L, gamma_K

def task_output(config, gL, gK, A_L=1.0, A_K=1.1):
    # config: array of 0/1, 1=labour
    M = np.where(config==1, A_L*gL, A_K*gK)
    return M

def ces_Y(config, gL, gK, s):
    M = task_output(config, gL, gK)
    N = len(config)
    if abs(s-1.0) < 1e-9:                            # Cobb-Douglas limit
        return np.exp(np.mean(np.log(M)))
    e = (s-1)/s
    inner = np.mean(M**e)
    return inner**(1.0/e)

def build_H(N, m, s, Gamma0=0.5, penalty=20.0):
    dim = 2**N
    configs = list(product([0,1], repeat=N))         # 0=capital,1=labour
    diag = np.zeros(dim)
    for idx, c in enumerate(configs):
        c = np.array(c)
        gL, gK = profiles(N)
        Y = ces_Y(c, gL, gK, s)
        n_capital = np.sum(c==0)
        # robots constraint: at most m capital units; penalize excess capital use
        excess = max(0, n_capital - m)
        diag[idx] = -Y + penalty*excess
    H = np.diag(diag).astype(complex)
    # transverse field: sigma^x on each qubit flips labour<->capital allocation
    Gamma = Gamma0 / s
    for idx, c in enumerate(configs):
        c = np.array(c)
        for q in range(N):
            c2 = c.copy(); c2[q] = 1-c2[q]
            jdx = sum(int(b)<<(N-1-k) for k,b in enumerate(c2))
            # map config tuple -> index consistently
        # rebuild flip via bit ops for correctness:
    # redo off-diagonal cleanly with bit indexing
    H = np.diag(diag).astype(complex)
    for idx in range(dim):
        for q in range(N):
            jdx = idx ^ (1<<q)
            H[idx, jdx] += Gamma
    return H

def ground_gap_and_degeneracy(H, beta, tol=1e-6):
    w, V = np.linalg.eigh(H)
    E0 = w[0]
    # ground-state degeneracy: count eigenvalues within tol of E0
    deg = int(np.sum(w < E0 + tol))
    energy_gap = w[deg] - E0 if deg < len(w) else 0.0   # gap to first excited *level*
    # Gibbs populations at inverse temp beta
    p = np.exp(-beta*(w - E0)); p /= p.sum()
    # population gap between ground manifold and first excited
    p_g = p[:deg].sum()/deg            # avg pop per ground state
    p_e = p[deg] if deg < len(w) else 0.0
    g = (p_g - p_e)/(p_g + p_e) if (p_g+p_e)>0 else 1.0
    return deg, energy_gap, g, w

# ---- universal advantage curve (from previous step): advantage ~ depends on pop gap g
def log_mean(a,b):
    return a if abs(a-b)<1e-14 else (a-b)/(np.log(a)-np.log(b))

def universal_advantage(g):
    # two-level model: p_g = (1+g)/2 * scale, p_e=(1-g)/2 ; relative AM-LM gap
    pg, pe = (1+g)/2, (1-g)/2
    AM = 0.5*(pg+pe); LM = log_mean(pg,pe)
    return (AM-LM)/AM

# ============================================================
#  Sweep robot count m, with N tasks fixed
# ============================================================
N = 6
s = 1.5            # elasticity of substitution (CES sigma)
beta = 4.0         # effective inverse temperature of the variational state
Gamma0 = 0.5

print(f"N={N} tasks, sigma={s}, beta={beta}, Gamma0={Gamma0}\n")
print(f"{'m':>3} {'gs_deg':>7} {'E_gap':>9} {'pop_gap g':>10} {'univ_adv':>9}")
ms, degs, gaps, advs = [], [], [], []
for m in range(1, N+1):
    H = build_H(N, m, s, Gamma0)
    deg, egap, g, w = ground_gap_and_degeneracy(H, beta)
    adv = universal_advantage(g)
    ms.append(m); degs.append(deg); gaps.append(g); advs.append(adv)
    print(f"{m:>3} {deg:>7} {egap:9.4f} {g:10.4f} {adv:9.4f}")

# ============================================================
#  Figure: degeneracy onset and advantage collapse vs robot count
# ============================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

ax[0].plot(ms, degs, 'o-', color='#d62728', lw=2)
ax[0].set_xlabel('robot count  m'); ax[0].set_ylabel('ground-state degeneracy')
ax[0].set_title('Degeneracy onset')
ax[0].axhline(1, color='gray', lw=0.5, ls=':')

ax[1].plot(ms, gaps, 's-', color='#1f77b4', lw=2)
ax[1].set_xlabel('robot count  m'); ax[1].set_ylabel('population gap  g')
ax[1].set_title('Population gap closes as m grows')

ax[2].plot(ms, advs, '^-', color='#2ca02c', lw=2, label='predicted from H(m)')
gg = np.linspace(0.001, max(gaps) if max(gaps)>0 else 1, 100)
ax[2].set_xlabel('robot count  m'); ax[2].set_ylabel('coherent advantage')
ax[2].set_title('Advantage vs robot count')
ax[2].legend(frameon=False)

plt.tight_layout()
plt.savefig('/home/claude/ar_mrta_spectrum.png', dpi=150, bbox_inches='tight')
print("\nsaved /home/claude/ar_mrta_spectrum.png")
