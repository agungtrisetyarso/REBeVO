# VQE-MMW Experiment - Final Version
import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt

def mmw_update(rho, G, eta=0.12):
    log_rho = logm(rho + 1e-12*np.eye(len(rho), dtype=complex))
    rho_new = expm(log_rho - eta * G)
    rho_new /= np.trace(rho_new).real
    return rho_new

def spectral_advantage(rho):
    evals = np.real(np.linalg.eigvalsh(rho))
    evals = np.maximum(evals, 1e-12)
    evals /= np.sum(evals)
    d = len(evals)
    num = den = 0.0
    for i in range(d):
        for j in range(d):
            if abs(evals[i] - evals[j]) < 1e-10:
                lmean = evals[i]
            else:
                lmean = (evals[i] - evals[j]) / (np.log(evals[i]) - np.log(evals[j]))
            arith = 0.5 * (evals[i] + evals[j])
            num += lmean
            den += arith
    return max(0.0, 1.0 - num/den)

# Run experiment (already produced your plot)
# ... (use the previous code)

plt.figure(figsize=(10, 8))
plt.scatter(res['adv'], res['energy'], c=res['gap'], cmap='viridis', s=100, edgecolor='k', linewidth=0.8)
plt.colorbar(label='Final Spectral Gap')
plt.xlabel('Coherent Advantage Adv(p) (Theorem 5.1)')
plt.ylabel('Final Variational Energy')
plt.title('Coherent Advantage Tracks VQE Performance\n(Transverse Field Ising Model, n=4)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('vqe_coherent_tracking.pdf', dpi=400, bbox_inches='tight')
plt.show()
