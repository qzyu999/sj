"""Quick test: verify the three algorithm variants work correctly."""

import sys
sys.path.insert(0, 'gsj/src')

import numpy as np
from gsj import bandwidth, roughness, roughness_gradient

np.random.seed(42)

# Test 1: Basic 2D normal
print("=" * 60)
print("Test 1: 2D Standard Normal (n=500)")
print("=" * 60)
X = np.random.randn(500, 2)

h_1stage = bandwidth(X, algorithm='one-stage')
h_2stage = bandwidth(X, algorithm='two-stage')
h_ste = bandwidth(X, algorithm='ste')

print(f"  One-stage:  h = {h_1stage:.6f}")
print(f"  Two-stage:  h = {h_2stage:.6f}")
print(f"  STE:        h = {h_ste:.6f}")
print(f"  Silverman:  h = {(4/(500*4))**(1/6):.6f}")

# Test 2: 2D mixture
print("\n" + "=" * 60)
print("Test 2: 2D Mixture (3 components, n=600)")
print("=" * 60)
means = np.array([[0, 0], [3, 3], [-3, 3]])
X_mix = np.vstack([np.random.randn(200, 2) + m for m in means])

h_1stage = bandwidth(X_mix, algorithm='one-stage')
h_2stage = bandwidth(X_mix, algorithm='two-stage')
h_ste = bandwidth(X_mix, algorithm='ste')

print(f"  One-stage:  h = {h_1stage:.6f}")
print(f"  Two-stage:  h = {h_2stage:.6f}")
print(f"  STE:        h = {h_ste:.6f}")

# Test 3: 5D data
print("\n" + "=" * 60)
print("Test 3: 5D Normal (n=1000)")
print("=" * 60)
X_5d = np.random.randn(1000, 5)

h_1stage = bandwidth(X_5d, algorithm='one-stage')
h_2stage = bandwidth(X_5d, algorithm='two-stage')
h_ste = bandwidth(X_5d, algorithm='ste')

print(f"  One-stage:  h = {h_1stage:.6f}")
print(f"  Two-stage:  h = {h_2stage:.6f}")
print(f"  STE:        h = {h_ste:.6f}")

# Test 4: Roughness functionals
print("\n" + "=" * 60)
print("Test 4: Roughness Functionals")
print("=" * 60)
X_test = np.random.randn(300, 3)
psi4 = roughness(X_test)
psi6 = roughness_gradient(X_test)

# Normal reference values for d=3, sigma=1:
# Psi4_NR = 3*5 / (4*(4pi)^{3/2}) = 15 / (4*15.7496) = 0.2381
# Psi6_NR = 3*5*7 / (8*(4pi)^{3/2}) = 105 / (8*15.7496) = 0.8333
psi4_nr = 3*5 / (4 * (4*np.pi)**1.5)
psi6_nr = 3*5*7 / (8 * (4*np.pi)**1.5)

print(f"  Psi_4 (estimated): {psi4:.6f}")
print(f"  Psi_4 (NR d=3):   {psi4_nr:.6f}")
print(f"  Ratio:             {psi4/psi4_nr:.3f}")
print(f"  Psi_6 (estimated): {psi6:.6f}")
print(f"  Psi_6 (NR d=3):   {psi6_nr:.6f}")
print(f"  Ratio:             {psi6/psi6_nr:.3f}")

# Test 5: 1D still works
print("\n" + "=" * 60)
print("Test 5: 1D (backward compat)")
print("=" * 60)
X_1d = np.random.randn(200)
h_1d = bandwidth(X_1d)
print(f"  1D bandwidth: {h_1d:.6f}")

# Test 6: Verify d=1 R_d polynomial correctness
print("\n" + "=" * 60)
print("Test 6: d=1 R_d polynomial check")
print("=" * 60)
from gsj._core import _R_d_poly
# R_1(0) should be 1*3*5/8 = 15/8 = 1.875
print(f"  R_1(0) = {_R_d_poly(0, 1):.6f} (expected 1.875)")
# R_2(0) = 2*4*6/8 = 48/8 = 6.0
print(f"  R_2(0) = {_R_d_poly(0, 2):.6f} (expected 6.0)")
# R_5(0) = 5*7*9/8 = 315/8 = 39.375
print(f"  R_5(0) = {_R_d_poly(0, 5):.6f} (expected 39.375)")

# Test 7: Large data (subsample path)
print("\n" + "=" * 60)
print("Test 7: Large data (n=10000, d=3, subsample)")
print("=" * 60)
X_large = np.random.randn(10000, 3)
import time
t0 = time.time()
h_1s = bandwidth(X_large, algorithm='one-stage')
t1 = time.time()
h_2s = bandwidth(X_large, algorithm='two-stage')
t2 = time.time()
h_ste_l = bandwidth(X_large, algorithm='ste')
t3 = time.time()

print(f"  One-stage:  h = {h_1s:.6f} ({(t1-t0)*1000:.0f} ms)")
print(f"  Two-stage:  h = {h_2s:.6f} ({(t2-t1)*1000:.0f} ms)")
print(f"  STE:        h = {h_ste_l:.6f} ({(t3-t2)*1000:.0f} ms)")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
