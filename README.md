# REBeVO: Regret Bounds for Entropy-Guided Variational Optimization

**A Spectral Anisotropy Account of the Coherent Advantage**

[![arXiv](https://img.shields.io/badge/arXiv-Preprint-red.svg)](https://arxiv.org/abs/) <!-- Add arXiv link when available -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About

This repository contains the code to reproduce the numerical results and figures from the paper:

> **Regret Bounds for Entropy-Guided Variational Optimization: A Spectral Anisotropy Account of the Coherent Advantage**  
> Agung Trisetyarso  
> June 15, 2026

The work provides the first exact quantitative account of the *coherent advantage* of matrix multiplicative weights (MMW) over classical entropic mirror descent, localized in the regret bound.

## Key Contributions

- Block decomposition of the MMW stability term into classical (diagonal) and quantum (off-diagonal) parts
- Exact spectral functional for the coherent advantage under isotropic gradients
- Single-pair law showing **quadratic collapse** at spectral degeneracy: $1 - \delta/\arctanh\delta$
- Purity-based asymptotics in the maximally-mixed regime
- Numerical verification of theoretical predictions

## Repository Contents

- `Regret_bench.py` — Main regret benchmarking and advantage computation
- `corrected_regret.py` — Corrected regret analysis
- `decoupling_corrected.py` — Decoupling of advantage and negativity
- `degenarcy_collapse.py` — Spectral degeneracy collapse experiments (Note: typo in filename)
- `invariant.py` — Invariant and structural checks
- `mrta_spect.py` — Spectral analysis on MRTA / variational quantum optimization

## Requirements

```bash
python >= 3.8
numpy
matplotlib
scipy
