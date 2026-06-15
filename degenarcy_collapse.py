import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(1)

# ---------- means ----------
def log_mean(a, b):
    if abs(a - b) < 1e-14:
        return a
    return (a - b) / (np.log(a) - np.log(b))

# ---------- build a density matrix with a controlled ground-state gap ----------
# Spectrum: one "ground" level with population p_g, and a near-degenerate cluster
# of (m-1) excited levels. We tune the gap g = (p_g - p_excited)/(p_g + p_excited).
# g -> 0 is full degeneracy (ground state merges into the cluster).
def make_spectrum(d, g):
    # excited cluster all equal at p_e; ground at p_g with relative gap g
    # p_g = p_e (1+g)/(1-g); normalize sum = 1 over d levels (1 ground + (d-1) excited)
    # let p_e = x, p_g = x*(1+g)/(1-g); total = x[(1+g)/(1-g) + (d-1)] = 1
    ratio = (1+g)/(1-g)
    x = 1.0 / (ratio + (d-1))
    p = np.array([x*ratio] + [x]*(d-1))
    return p / p.sum()

def rand_unitary(d):
    A = np.random.randn(d, d) + 1j*np.random.randn(d, d)
    Q, R = np.linalg.qr(A)
    return Q @ np.diag(np.exp(1j*np.angle(np.diag(R))))

def rand_herm(d):
    A = np.random.randn(d, d) + 1j*np.random.randn(d, d)
    return (A + A.conj().T)/2

# ---------- the three curvature quadratic forms in a random direction G ----------
def curvatures(p, U, G):
    Gb = U.conj().T @ G @ U
    d = len(p)
    C_AM = C_LM = C_HM = 0.0
    for i in range(d):
        for j in range(d):
            w2 = abs(Gb[i, j])**2
            pi, pj = p[i], p[j]
            C_AM += 0.5*(pi+pj) * w2
            C_LM += log_mean(pi, pj) * w2
            C_HM += (2*pi*pj/(pi+pj)) * w2
    return C_AM, C_LM, C_HM

# ---------- sweep ----------
gaps = np.linspace(0.02, 0.98, 40)
n_avg = 200   # average over random G and random eigenbasis orientation
results = {}

for d in [3, 4, 5]:
    adv_LM = []   # advantage measured by GT slack at log-mean level: (C_AM - C_LM)/C_AM
    adv_HM = []   # lower bracket: (C_AM - C_HM)/C_AM
    for g in gaps:
        p = make_spectrum(d, g)
        accAM = accLM = accHM = 0.0
        for _ in range(n_avg):
            U = rand_unitary(d)
            G = rand_herm(d)
            cAM, cLM, cHM = curvatures(p, U, G)
            accAM += cAM; accLM += cLM; accHM += cHM
        adv_LM.append((accAM - accLM)/accAM)
        adv_HM.append((accAM - accHM)/accAM)
    results[d] = (np.array(adv_LM), np.array(adv_HM))

# ---------- print a small table at representative gaps ----------
print(f"{'d':>3} {'gap g':>7} {'adv_LM':>9} {'adv_HM':>9}")
for d in [3,4,5]:
    aLM, aHM = results[d]
    for gi in [0, len(gaps)//2, len(gaps)-1]:
        print(f"{d:>3} {gaps[gi]:7.3f} {aLM[gi]:9.4f} {aHM[gi]:9.4f}")

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

colors = {3:'#1f77b4', 4:'#d62728', 5:'#2ca02c'}
for d in [3,4,5]:
    aLM, aHM = results[d]
    ax[0].plot(gaps, aLM, color=colors[d], lw=2, label=f'd={d}')
    ax[1].plot(gaps, aLM, color=colors[d], lw=2, label=f'log-mean (KM), d={d}')
    ax[1].plot(gaps, aHM, color=colors[d], lw=1.3, ls='--', alpha=0.7)

for a in ax:
    a.set_xlabel(r'ground-state spectral gap  $g=(p_g-p_e)/(p_g+p_e)$')
    a.axvline(0, color='k', lw=0.5)
ax[0].set_ylabel(r'coherent advantage  $(C_{AM}-C_{LM})/C_{AM}$')
ax[0].set_title('Advantage collapses as ground state degenerates')
ax[0].legend(frameon=False)
ax[1].set_ylabel('relative advantage')
ax[1].set_title('Two-sided bracket: log-mean (solid) vs measured/HM (dashed)')
ax[1].legend(frameon=False, fontsize=8)

# annotate the degeneracy limit
ax[0].annotate('full degeneracy\n(advantage → 0)', xy=(0.02, results[5][0][0]),
               xytext=(0.25, 0.02), fontsize=9,
               arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('/home/claude/degeneracy_collapse.png', dpi=150, bbox_inches='tight')
print("\nsaved figure")

# ---------- quadratic-law check near degeneracy: advantage ~ g^2 ? ----------
print("\nNear-degeneracy scaling (expect advantage ~ g^2, slope ~2 on log-log):")
for d in [3,4,5]:
    aLM, _ = results[d]
    small = gaps < 0.2
    slope = np.polyfit(np.log(gaps[small]), np.log(aLM[small]), 1)[0]
    print(f"  d={d}: log-log slope of advantage vs gap = {slope:.3f}")
