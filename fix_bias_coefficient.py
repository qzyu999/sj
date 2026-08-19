"""
Fix the bias coefficient: derive the exact asymptotic bias of our Psi_4 estimator.

Our estimator is:
  Psi_4_hat(h0) = (1/n^2) * sum_ij I_ij
  where I_ij = int [nabla^2 K_h0(x - Yi)] * [nabla^2 K_h0(x - Yj)] dx

This estimates R(nabla^2 f_hat_h0) = int (nabla^2 f_hat_h0)^2 dx.

But we WANT Psi_4 = R(nabla^2 f) = int (nabla^2 f)^2 dx.

The relationship between these two is:
  E[Psi_4_hat(h0)] = R(nabla^2 (f * K_h0))  + O(1/n) diagonal correction
                   = R(nabla^2 f * K_h0)      [derivatives commute with convolution]
                   = int (nabla^2 f * K_h0)^2 dx

Now, f * K_h0 means f convolved with K_h0. In Fourier space:
  F[f * K_h0](w) = F[f](w) * F[K_h0](w) = F[f](w) * exp(-h0^2 ||w||^2 / 2)

So nabla^2 (f * K_h0) has Fourier transform:
  -||w||^2 * F[f](w) * exp(-h0^2 ||w||^2 / 2)

And R(nabla^2 (f * K_h0)) = int ||w||^4 |F[f](w)|^2 exp(-h0^2 ||w||^2) dw   [Plancherel]

Compare to Psi_4 = int ||w||^4 |F[f](w)|^2 dw

So:
  E[Psi_4_hat(h0)] ≈ int ||w||^4 |F[f](w)|^2 * exp(-h0^2 ||w||^2) dw
                    = Psi_4 - h0^2 * int ||w||^6 |F[f](w)|^2 dw + O(h0^4)
                    = Psi_4 - h0^2 * Psi_6 + O(h0^4)

Wait — the exp(-h0^2 ||w||^2) = 1 - h0^2||w||^2 + (h0^4/2)||w||^4 - ...
So the first correction is -h0^2 * int ||w||^6 |F[f]|^2 dw.

But int ||w||^6 |F[f]|^2 dw corresponds to... what in real space?
By Plancherel, int ||w||^{2k} |F[f]|^2 dw relates to derivatives of f.
Specifically: int ||w||^2 |F[g]|^2 dw = int ||nabla g||^2 dx (for scalar g, in d-D).

More precisely, with the standard Fourier convention F[f](w) = int f(x) exp(-iwx) dx:
  int ||w||^{2k} |F[f]|^2 dw = (2pi)^d * int |D^k f|^2 dx

Hmm, the exact relationship depends on conventions. Let me just compute numerically.

KEY INSIGHT: The bias is NEGATIVE (the smoothed version has LESS roughness than
the original). So:
  E[Psi_4_hat(h0)] = Psi_4 - c * h0^2 * Psi_6 + diagonal_term/n

where c > 0 is the coefficient we need. The diagonal adds POSITIVE bias:
  diagonal = P_d(0) / (n * (4pi)^{d/2} * h0^{d+4}) = d(d+2) / (4*n*(4pi)^{d/2}*h0^{d+4})

For the bias cancellation to work, we need the SMOOTHING bias to be NEGATIVE
and cancel the POSITIVE diagonal. But wait — our formula says the smoothing
bias makes Psi_4_hat SMALLER than Psi_4 (since exp(-h0^2||w||^2) < 1).
That means E[Psi_4_hat] < Psi_4 (from smoothing alone), and the diagonal
makes E[Psi_4_hat] > Psi_4 (by adding extra contribution).

So the total bias is:
  E[Psi_4_hat(h0)] - Psi_4 = [diagonal] - [smoothing]
                            = d(d+2)/(4*n*(4pi)^{d/2}*h0^{d+4}) - c*h0^2*Psi_6

For the oracle pilot (where estimate = truth):
  diagonal = smoothing bias
  d(d+2)/(4*n*(4pi)^{d/2}*h0^{d+4}) = c*h0^2*Psi_6

Let me verify this numerically by finding c from the crossing point.
"""

import sys
sys.path.insert(0, 'gsj/src')
import numpy as np
from scipy.optimize import brentq
from gsj._core import _roughness_exact_nd, silverman_pilot
from gsj._utils import whiten

# =============================================================================
# Empirical determination of the bias coefficient c
# =============================================================================

print("=" * 70)
print("EMPIRICAL BIAS COEFFICIENT")
print("=" * 70)

# Use MANY samples to reduce variance and see the EXPECTATION clearly
d = 2
n = 500
psi4_true = d * (d + 2) / (4 * (4 * np.pi) ** (d / 2))
psi6_true = d * (d + 2) * (d + 4) / (8 * (4 * np.pi) ** (d / 2))

print(f"\nd={d}, n={n}")
print(f"Psi_4 true (normal ref) = {psi4_true:.8f}")
print(f"Psi_6 true (normal ref) = {psi6_true:.8f}")

# Average over many realizations to get E[Psi_4_hat(h0)]
n_reps = 200
h_test_values = [0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.80, 1.0]

print(f"\nAveraging over {n_reps} realizations...")
print(f"\n{'h_pilot':<10} {'E[Psi4_hat]':<14} {'Bias':<14} {'Diagonal':<14} {'Smooth bias':<14} {'c_empirical':<12}")
print("-" * 80)

for h0 in h_test_values:
    estimates = []
    for rep in range(n_reps):
        rng = np.random.default_rng(rep)
        X = rng.standard_normal((n, d))
        Y = whiten(X)
        est = _roughness_exact_nd(Y, h0, n, d)
        estimates.append(est)
    
    E_est = np.mean(estimates)
    total_bias = E_est - psi4_true
    
    # Diagonal contribution (deterministic, always present)
    diagonal = d * (d + 2) / (4 * n * (4 * np.pi) ** (d / 2) * h0 ** (d + 4))
    
    # Smoothing bias = diagonal - total_bias (since total = diagonal - smoothing)
    # Wait: total_bias = diagonal - c*h0^2*Psi_6
    # So: c*h0^2*Psi_6 = diagonal - total_bias
    smooth_bias = diagonal - total_bias
    c_emp = smooth_bias / (h0 ** 2 * psi6_true) if psi6_true > 0 else np.nan
    
    print(f"{h0:<10.2f} {E_est:<14.6f} {total_bias:<14.6f} {diagonal:<14.6f} {smooth_bias:<14.6f} {c_emp:<12.4f}")

print()
print("If c is constant across h values, then our bias model is correct.")
print("The value of c tells us the correct smoothing bias coefficient.")
print()

# =============================================================================
# Now do d=3 to check pattern
# =============================================================================
print("\n" + "=" * 70)
print("d=3 CHECK")
print("=" * 70)

d = 3
n = 800
psi4_true_3 = d * (d + 2) / (4 * (4 * np.pi) ** (d / 2))
psi6_true_3 = d * (d + 2) * (d + 4) / (8 * (4 * np.pi) ** (d / 2))
n_reps = 100

print(f"\nd={d}, n={n}")
print(f"Psi_4 true = {psi4_true_3:.8f}")
print(f"Psi_6 true = {psi6_true_3:.8f}")

print(f"\n{'h_pilot':<10} {'E[Psi4_hat]':<14} {'Bias':<14} {'Diagonal':<14} {'c_empirical':<12}")
print("-" * 60)

for h0 in [0.30, 0.35, 0.40, 0.50, 0.60, 0.80]:
    estimates = []
    for rep in range(n_reps):
        rng = np.random.default_rng(rep)
        X = rng.standard_normal((n, d))
        Y = whiten(X)
        est = _roughness_exact_nd(Y, h0, n, d)
        estimates.append(est)
    
    E_est = np.mean(estimates)
    total_bias = E_est - psi4_true_3
    diagonal = d * (d + 2) / (4 * n * (4 * np.pi) ** (d / 2) * h0 ** (d + 4))
    smooth_bias = diagonal - total_bias
    c_emp = smooth_bias / (h0 ** 2 * psi6_true_3)
    
    print(f"{h0:<10.2f} {E_est:<14.6f} {total_bias:<14.6f} {diagonal:<14.6f} {c_emp:<12.4f}")

# =============================================================================
# Theoretical prediction: what should c be?
# =============================================================================
print("\n" + "=" * 70)
print("THEORETICAL ANALYSIS")
print("=" * 70)
print()
print("Our estimator computes R(nabla^2 f_hat_h0) where f_hat_h0 = f * K_h0.")
print()
print("In Fourier space:")
print("  R(nabla^2 (f*K_h0)) = int ||w||^4 |F[f](w)|^2 * |F[K_h0](w)|^2 dw")
print()
print("For Gaussian K_h0: |F[K_h0](w)|^2 = exp(-h0^2 * ||w||^2)")
print()  
print("Taylor expanding exp(-h0^2*||w||^2) = 1 - h0^2*||w||^2 + h0^4*||w||^4/2 - ...")
print()
print("R(nabla^2 (f*K_h0)) = int ||w||^4 |F[f]|^2 dw") 
print("                    - h0^2 * int ||w||^6 |F[f]|^2 dw")
print("                    + (h0^4/2) * int ||w||^8 |F[f]|^2 dw - ...")
print()
print("So: E[Psi_4_hat] = Psi_4 - h0^2 * C_6 + h0^4/2 * C_8 + diagonal/n + ...")
print()
print("where C_6 = int ||w||^6 |F[f]|^2 dw")
print()
print("The relationship between C_6 and our Psi_6 depends on the Fourier convention.")
print("With F[f](w) = int f(x)*exp(-2*pi*i*w.x) dx (standard):")
print("  int ||w||^{2k} |F[f]|^2 dw = (2*pi)^{-2k} * sum of products of 2k-th partials integrated")
print()
print("Actually, for the SYMMETRIC Fourier transform used with Gaussian kernels:")
print("  K_h(x) = (2*pi*h^2)^{-d/2} exp(-||x||^2/(2h^2))")
print("  F[K_h](w) = exp(-h^2*||w||^2/2)")
print("  |F[K_h](w)|^2 = exp(-h^2*||w||^2)")
print()
print("And Parseval: int |f(x)|^2 dx = (2*pi)^{-d} * int |F[f](w)|^2 dw")
print("  OR (if using F[f](w) = int f(x) e^{-iwx} dx)")
print("  int |f|^2 dx = (2*pi)^{-d} int |hat{f}|^2 dw")
print()
print("So: int ||w||^4 |hat{f}|^2 dw = (2*pi)^d * int (nabla^2 f)^2 dx = (2*pi)^d * Psi_4")
print("    int ||w||^6 |hat{f}|^2 dw = (2*pi)^d * int ||nabla(nabla^2 f)||^2 dx = (2*pi)^d * Psi_6")
print()  
print("Wait — that's not right for ||w||^6. Let me be more careful.")
print()
print("For a scalar field g(x) in R^d:")
print("  int ||w||^2 |hat{g}|^2 dw = int ||nabla g||^2 dx * (2*pi)^d  [Parseval + F[nabla g] = iw*hat{g}]")
print()
print("For ||w||^4 |hat{f}|^2:")
print("  F[nabla^2 f](w) = -||w||^2 * hat{f}(w)")
print("  int ||w||^4 |hat{f}|^2 dw = int |F[nabla^2 f]|^2 dw = (2*pi)^d * int (nabla^2 f)^2 dx")
print("  = (2*pi)^d * Psi_4  ✓")
print()
print("For ||w||^6 |hat{f}|^2:")
print("  ||w||^6 = ||w||^2 * ||w||^4")
print("  ||w||^6 |hat{f}|^2 = ||w||^2 * |F[nabla^2 f]|^2")
print("  int ||w||^6 |hat{f}|^2 dw = int ||w||^2 |F[nabla^2 f]|^2 dw")
print("  = (2*pi)^d * int ||nabla(nabla^2 f)||^2 dx = (2*pi)^d * Psi_6  ✓")
print()
print("So the Taylor expansion gives:")
print("  R(nabla^2 (f*K_h0)) = Psi_4 - h0^2 * Psi_6 + O(h0^4)")
print()
print("And the FULL expectation of our estimator is:")
print("  E[Psi_4_hat(h0)] = Psi_4 - h0^2 * Psi_6 + diagonal/n + O(h0^4) + O(1/n)")
print()
print("So the theoretical c = 1.")
print()
print("But WAIT — our kernel has F[K_h](w) = exp(-h^2||w||^2/2), so")
print("|F[K_h]|^2 = exp(-h^2||w||^2). That's what we used above.")
print()
print("However, our PAIRWISE estimator integrates (nabla^2 K_h0(x-a))*(nabla^2 K_h0(x-b)) dx.")
print("This gives the convolution of nabla^2 K_h0 with itself, evaluated at a-b.")
print("The Fourier transform of (nabla^2 K_h0) * (nabla^2 K_h0) is ||w||^4 * |F[K_h0]|^2")
print("= ||w||^4 * exp(-h0^2*||w||^2)")
print()
print("And we're evaluating this at every PAIR of data points (Yi, Yj).")
print("The expected value (for i≠j) is:")
print("  E[I_ij] = int int nabla^2K_h0(x-y1) * nabla^2K_h0(x-y2) * f(y1)*f(y2) dy1 dy2 dx")
print("          = int (nabla^2(f*K_h0)(x))^2 dx")
print("          = R(nabla^2(f*K_h0))")
print("          = Psi_4 - h0^2*Psi_6 + ...  [from above]")
print()
print("So the OFF-DIAGONAL part estimates Psi_4 - h0^2*Psi_6.")
print("The DIAGONAL part (i=j) gives P_d(0)/((4pi)^{d/2} h0^{d+4}) = d(d+2)/(4*(4pi)^{d/2}*h0^{d+4})")
print()
print("Total: E[Psi_4_hat] = (1-1/n)*(Psi_4 - h0^2*Psi_6) + (1/n)*diagonal + higher order")
print("     ≈ Psi_4 - h0^2*Psi_6 + diagonal/n")
print()
print("At the crossing point where E[Psi_4_hat] = Psi_4:")
print("  h0^2 * Psi_6 = diagonal/n = d(d+2)/(4*n*(4pi)^{d/2}*h0^{d+4})")
print("  h0^{d+6} = d(d+2) / (4*n*(4pi)^{d/2} * Psi_6)")
print()
print("For d=2, Psi_6_true = 2*4*6/(8*(4*pi)) = 48/(32*pi) = 3/(2*pi)")
print(f"  = {3/(2*np.pi):.6f}")
print(f"  Previously computed Psi_6_true = {psi4_true * (d+4)/2:.6f}")

psi6_true_d2 = 2*4*6 / (8 * (4*np.pi)**(2/2))
print(f"  Direct formula: {psi6_true_d2:.6f}")

# Predicted crossing point
h_cross_pred = (d*(d+2) / (4*n*(4*np.pi)**(d/2)*psi6_true_d2)) ** (1/(d+6))
print(f"\n  Predicted crossing point: h = {h_cross_pred:.6f}")

# Compare with empirical
# From the earlier run: h_cross ≈ 0.429
print(f"  Empirical crossing (single trial): ~0.429")
print(f"  Note: single trial has variance; let's check with averaged data")

# Use the averaged estimates to find crossing
n_reps_fine = 500
h_scan = np.linspace(0.30, 0.55, 25)
mean_estimates = []
for h0 in h_scan:
    ests = []
    for rep in range(n_reps_fine):
        rng = np.random.default_rng(rep)
        X = rng.standard_normal((n, 2))
        Y = whiten(X)
        ests.append(_roughness_exact_nd(Y, h0, n, 2))
    mean_estimates.append(np.mean(ests))

mean_estimates = np.array(mean_estimates)
# Find where mean = psi4_true
from scipy.interpolate import interp1d
f_interp = interp1d(h_scan, mean_estimates - psi4_true)
h_cross_avg = brentq(f_interp, 0.35, 0.50)
print(f"  Averaged crossing point ({n_reps_fine} reps): h = {h_cross_avg:.6f}")
print(f"  Predicted: {h_cross_pred:.6f}")
print(f"  Ratio: {h_cross_avg / h_cross_pred:.4f}")

# =============================================================================
# THE FIX
# =============================================================================
print("\n" + "=" * 70)
print("THE FIX: CORRECT PILOT BANDWIDTH")
print("=" * 70)
print()
print("The bias cancellation gives:")
print("  h_pilot^{d+6} = d*(d+2) / (4*n*(4*pi)^{d/2} * Psi_6)")
print()
print("This IS the formula we already have in _bandwidth_nd_two_stage!")
print("Using the normal reference Psi_6 = d*(d+2)*(d+4)/(8*(4pi)^{d/2}):")
print("  h_pilot = (2/(n*(d+4)))^{1/(d+6)}")
print()
print("For d=2, n=500: h_pilot = (2/(500*6))^{1/8} = (1/1500)^{1/8}")
h_pilot_formula = (2 / (500 * 6)) ** (1/8)
print(f"  = {h_pilot_formula:.6f}")
print(f"  But we multiply by sigma_hat... if sigma_hat≈1 (whitened data):")
print(f"  Then pilot = {h_pilot_formula:.6f}")
print(f"  This is BELOW Silverman ({silverman_pilot(500, 2):.6f})")
print(f"  And the crossing point is {h_cross_avg:.6f}")
print()
print("The formula SHOULD give the crossing point. Let's check:")
print(f"  (d(d+2)/(4*n*(4pi)^{{d/2}}*Psi_6))^{{1/(d+6)}}")
d = 2; n = 500
val = (d*(d+2) / (4*n*(4*np.pi)**(d/2)*psi6_true_d2)) ** (1/(d+6))
print(f"  = {val:.6f}")
print(f"  Empirical crossing = {h_cross_avg:.6f}")
print(f"  They should match!")
print()
if abs(val - h_cross_avg) / h_cross_avg < 0.05:
    print("  ✓ They match! The formula is correct.")
    print("  The problem was that sigma_hat was estimated incorrectly,")
    print("  or the code has a different formula than what's documented.")
else:
    print(f"  ✗ Mismatch: ratio = {val/h_cross_avg:.4f}")
    print("  The formula needs a correction factor.")
    print(f"  Correction needed: multiply by {h_cross_avg/val:.4f}")
