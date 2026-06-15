# ================================================
# FINAL Publication-Ready Regret Benchmark
# Google Colab
# ================================================

import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

def mmw_update_numpy(rho, G, eta):
    log_rho = logm(rho + 1e-12 * np.eye(len(rho)))
    exp_arg = log_rho - eta * G
    rho_new = expm(exp_arg)
    rho_new /= np.trace(rho_new).real
    return rho_new

def classical_mw(p, loss_diag, eta):
    p_new = p * np.exp(-eta * loss_diag)
    return p_new / np.sum(p_new)

def run_benchmark(T=2800, d=6, gap=0.25, eta=0.1, n_trials=20, seed=42):
    np.random.seed(seed)
    regrets_mmw = []
    regrets_class = []
    
    for _ in range(n_trials):
        ps = np.zeros(d)
        ps[0] = (1 + gap) / 2
        ps[1] = (1 - gap) / 2
        if d > 2:
            ps[2:] = (1 - np.sum(ps[:2])) / (d - 2)
        ps /= np.sum(ps)
        
        rho = np.diag(ps).astype(complex)
        p_class = ps.copy()
        comparator = np.diag(ps)
        
        regret_mmw = 0.0
        regret_class = 0.0
        
        for t in range(T):
            diag_part = np.random.randn(d) * 1.1
            offdiag = np.random.randn(d, d) * 0.78
            G = np.diag(diag_part) + (offdiag + offdiag.T.conj()) / 2
            G = G.astype(complex)
            G /= (np.max(np.abs(np.linalg.eigvalsh(G))) + 1e-8)
            
            rho = mmw_update_numpy(rho, G, eta)
            loss_diag = np.real(np.diag(G))
            p_class = classical_mw(p_class, loss_diag, eta)
            
            regret_mmw += np.real(np.trace(G @ (rho - comparator)))
            regret_class += np.dot(loss_diag, p_class - ps)
        
        regrets_mmw.append(regret_mmw)
        regrets_class.append(regret_class)
    
    return (np.mean(regrets_mmw), np.mean(regrets_class), 
            np.std(regrets_mmw), np.std(regrets_class))

def generate_final_figure():
    gaps = [0.01, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6]
    mmw_means, class_means = [], []
    mmw_stds, class_stds = [], []
    improvements = []
    
    print("Running final benchmarks...\n")
    for gap in gaps:
        print(f"δ = {gap:.2f} ... ", end="")
        m_mmw, m_class, s_mmw, s_class = run_benchmark(gap=gap, T=2800, n_trials=20)
        mmw_means.append(m_mmw)
        class_means.append(m_class)
        mmw_stds.append(s_mmw)
        class_stds.append(s_class)
        imp = 100 * (m_class - m_mmw) / abs(m_class)   # Positive improvement %
        improvements.append(imp)
        print(f"MMW better by {imp:.1f}%")
    
    # ==================== PLOTS ====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    x = np.arange(len(gaps))
    width = 0.35
    
    ax1.bar(x - width/2, class_means, width, yerr=class_stds, label='Classical (Shannon)', 
            color='#ff7f7f', alpha=0.9, capsize=6, edgecolor='black')
    ax1.bar(x + width/2, mmw_means, width, yerr=mmw_stds, label='MMW (von Neumann)', 
            color='#7fbf7f', alpha=0.9, capsize=6, edgecolor='black')
    ax1.set_xlabel('Relative Spectral Gap $\\delta$')
    ax1.set_ylabel('Average Cumulative Regret')
    ax1.set_title('Regret Comparison (Lower is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{g:.2f}' for g in gaps])
    ax1.legend(loc='upper right')
    
    ax2.plot(gaps, improvements, 'o-', color='darkgreen', linewidth=3, markersize=10)
    ax2.set_xlabel('Relative Spectral Gap $\\delta$')
    ax2.set_ylabel('Relative Improvement of MMW (%)')
    ax2.set_title('Coherent Advantage in Practice')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--')
    
    plt.suptitle('Numerical Demonstration of Coherent Advantage (Theorems 4.1--4.4)\n'
                 'Matrix Multiplicative Weights vs Classical Entropic Descent', fontsize=14)
    plt.tight_layout()
    plt.savefig('regret_benchmark_final.pdf', dpi=400, bbox_inches='tight')
    plt.savefig('regret_benchmark_final.png', dpi=400, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Final manuscript-ready figures saved!")

if __name__ == "__main__":
    generate_final_figure()
