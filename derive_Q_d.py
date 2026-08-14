"""
Derive Q_d: the bi-Laplacian polynomial for Psi_6 estimation.

Goal: Find Q_d(t) such that the pairwise contribution to the
bi-Laplacian roughness integral is:

  I_ij^(6) = (4*pi*b^2)^{-d/2} * exp(-r^2/4) * Q_d(r^2)

where r^2 = ||Y_i - Y_j||^2 / b^2.

This is the SAME calculation as for P_d, but with (nabla^2)^2 K_b
instead of nabla^2 K_b.

Strategy:
1. Compute (nabla^2)^2 K_h(t) in closed form
2. Form the product integral int [(nabla^2)^2 K_h(x-a)][(nabla^2)^2 K_h(x-b)] dx
3. Use moment calculations to get Q_d as a polynomial in r^2 = ||a-b||^2/h^2
4. Verify: at d=1, Q_1(t) should give phi^(6)(z) behavior

We use SymPy for exact symbolic computation.
"""

import sympy as sp
from sympy import symbols, sqrt, pi, exp, Rational, simplify, expand, factor
from sympy import gamma as Gamma, oo, integrate, cos, sin

# =============================================================================
# APPROACH: Direct moment calculation (same as P_d derivation)
# =============================================================================

# The key identity:
#
# nabla^2 K_h(t) = K_h(t) * h^{-2} * (||t||^2/h^2 - d)
#                = K_h(t) * h^{-2} * L(t)
#
# where L(t) = ||t||^2/h^2 - d
#
# (nabla^2)^2 K_h(t) = nabla^2 [K_h(t) * h^{-2} * L(t)]
#
# Using the product rule for Laplacian: nabla^2(fg) = f*nabla^2(g) + g*nabla^2(f) + 2*nabla(f).nabla(g)
#
# Let f = K_h(t), g = h^{-2} * L(t) = h^{-2} * (||t||^2/h^2 - d)
#
# nabla g = h^{-2} * 2t/h^2 = 2t/h^4
# nabla^2 g = h^{-2} * 2d/h^2 = 2d/h^4
# nabla f = K_h(t) * (-t/h^2)
# nabla^2 f = K_h(t) * h^{-2} * L(t)  [this is what we started with]
#
# So:
# nabla^2(fg) = f * (2d/h^4) + (h^{-2}*L) * K_h*h^{-2}*L + 2*K_h*(-t/h^2) . (2t/h^4)
#             = K_h * [2d/h^4 + L^2/h^4 - 4||t||^2/h^6]
#             = K_h/h^4 * [2d + L^2 - 4||t||^2/h^2]
#
# Now L = ||t||^2/h^2 - d, so L^2 = (||t||^2/h^2)^2 - 2d*||t||^2/h^2 + d^2
# and 4||t||^2/h^2 = 4*||t||^2/h^2
#
# 2d + L^2 - 4||t||^2/h^2 
# = 2d + (||t||^2/h^2)^2 - 2d*(||t||^2/h^2) + d^2 - 4*(||t||^2/h^2)
# = (||t||^2/h^2)^2 - (2d+4)*(||t||^2/h^2) + (d^2 + 2d)
# = s^2 - 2(d+2)*s + d(d+2)
#
# where s = ||t||^2/h^2.
#
# So: (nabla^2)^2 K_h(t) = K_h(t) / h^4 * [s^2 - 2(d+2)s + d(d+2)]
#
# Define B_d(s) = s^2 - 2(d+2)s + d(d+2)  [the bi-Laplacian polynomial of the kernel]
#
# NOTE: This is EXACTLY the polynomial we called Q_d before in paper_v12!
# Q_d(s) = s^2 - 2(d+2)s + d(d+2)
#
# Verification at d=1: Q_1(s) = s^2 - 6s + 3
# And phi^(4)(z) = (z^4 - 6z^2 + 3)*phi(z), so at z -> kernel argument,
# (nabla^2)^2 K_h evaluated gives the He_4 polynomial. ✓

print("=" * 70)
print("BI-LAPLACIAN OF GAUSSIAN KERNEL")
print("=" * 70)
print()
print("(nabla^2)^2 K_h(t) = K_h(t) * h^{-4} * B_d(s)")
print("where s = ||t||^2/h^2 and")
print("B_d(s) = s^2 - 2(d+2)s + d(d+2)")
print()

d = symbols('d', positive=True, integer=True)
s = symbols('s', nonneg=True)

B_d = s**2 - 2*(d+2)*s + d*(d+2)
print(f"B_d(s) = {expand(B_d)}")
print(f"B_1(s) = {B_d.subs(d, 1)} = s^2 - 6s + 3")
print(f"Check He_4(z) = z^4 - 6z^2 + 3 at z^2=s: matches ✓")
print()

# =============================================================================
# NOW: Compute the pairwise integral for Psi_6
# =============================================================================
# 
# Psi_6 = int [(nabla^2)^2 f_hat(x)]^2 dx
#        = (1/n^2) sum_ij int [(nabla^2)^2 K_b(x-Yi)] * [(nabla^2)^2 K_b(x-Yj)] dx
#
# Each pairwise term:
# I_ij = int [K_b(x-a)/b^4 * B_d(||x-a||^2/b^2)] * [K_b(x-b)/b^4 * B_d(||x-b||^2/b^2)] dx
#       = b^{-8} * int K_b(x-a)*K_b(x-b) * B_d(||x-a||^2/b^2) * B_d(||x-b||^2/b^2) dx
#
# Using the Gaussian product lemma:
# K_b(x-a)*K_b(x-b) = C_ab * phi_{b/sqrt(2)}(x - (a+b)/2)
# where C_ab = (4*pi*b^2)^{-d/2} * exp(-||a-b||^2/(4b^2))
#
# After the substitution x -> midpoint + b/sqrt(2) * W, W ~ N(0,I):
# ||x-a||^2/b^2 = ||W - delta/(sqrt(2)*b)||^2 ... etc
#
# This is the SAME moment calculation as for P_d, but with B_d*B_d instead of L*L.
#
# Let me parameterize:
#   delta = a - b (the difference vector)
#   r^2 = ||delta||^2/b^2 (scaled squared distance)
#   U = (x - midpoint)/(b/sqrt(2)) ~ N(0, I_d)
#
# Then:
#   (x - a)/b = U*sqrt(1/2) - delta/(2b) = sigma*U - mu  with sigma=1/sqrt(2), mu=delta/(2b)
#   (x - b)/b = U*sqrt(1/2) + delta/(2b) = sigma*U + mu
#
#   ||x-a||^2/b^2 = ||sigma*U - mu||^2 = sigma^2*||U||^2 - 2*sigma*(U.mu) + ||mu||^2
#                 = (1/2)*||U||^2 - sqrt(2)*(U.mu_hat)*||mu|| + ||mu||^2
#
# Actually, let's use the same notation as the P_d derivation:
#   Let V = sigma*U + mu, W = sigma*U - mu, where sigma = 1/sqrt(2), mu = delta/(2b)
#   Then ||V||^2 = ||x-b||^2/b^2, ||W||^2 = ||x-a||^2/b^2
#
# Wait, let me just use the abstract moment approach directly.

print("=" * 70)
print("PAIRWISE INTEGRAL FOR Psi_6")
print("=" * 70)
print()
print("I_ij^(6) = b^{-8} * C_ab * E[B_d(||U||^2) * B_d(||V||^2)]")
print("where U = sigma*W - mu, V = sigma*W + mu")
print("with sigma = 1/sqrt(2), mu = delta/(2b), ||mu||^2 = r^2/4")
print("W ~ N(0, I_d)")
print()

# So we need E[B_d(||sigma*W + mu||^2) * B_d(||sigma*W - mu||^2)]
# 
# Let A = ||sigma*W + mu||^2 = sigma^2*||W||^2 + 2*sigma*(W.mu) + ||mu||^2
#       = (1/2)*||W||^2 + sqrt(2)*(W.mu) + ||mu||^2
# Let B = ||sigma*W - mu||^2 = (1/2)*||W||^2 - sqrt(2)*(W.mu) + ||mu||^2
#
# Note: A + B = ||W||^2 + 2*||mu||^2
#       A - B = 2*sqrt(2)*(W.mu)
#       A*B = [(1/2)*||W||^2 + ||mu||^2]^2 - 2*(W.mu)^2
#
# With sigma = 1/sqrt(2) and ||mu||^2 = r^2/4:
#
# A = (1/2)*S + sqrt(2)*T + r^2/4
# B = (1/2)*S - sqrt(2)*T + r^2/4
#
# where S = ||W||^2 ~ chi^2_d and T = W.mu_hat * ||mu|| (mu_hat = unit vector)
# T | S ~ N(0, ||mu||^2) ... actually T = (W . mu) and W ~ N(0,I_d)
# so T ~ N(0, ||mu||^2) = N(0, r^2/4)... no wait.
# mu = delta/(2b), ||mu|| = ||delta||/(2b) = r/2 (where r = ||delta||/b)
# T = W . mu = (W . mu_hat) * ||mu|| = Z * r/2 where Z = W.mu_hat ~ N(0,1)
#
# Actually let me just work with:
# Let S = ||W||^2 (chi-squared d)
# Let Z = W . e_1 (standard normal, independent of the orthogonal part)
# Then W.mu = Z * ||mu|| = Z * r/2
#
# A = S/2 + sqrt(2) * Z * r/2 + r^2/4 = S/2 + r*Z/sqrt(2) + r^2/4
# B = S/2 - r*Z/sqrt(2) + r^2/4
#
# Now B_d(A) = A^2 - 2(d+2)*A + d(d+2)
#     B_d(B) = B^2 - 2(d+2)*B + d(d+2)
#
# E[B_d(A)*B_d(B)] = E[(A^2 - 2(d+2)A + d(d+2)) * (B^2 - 2(d+2)B + d(d+2))]
#
# This expands to:
# E[A^2*B^2] - 2(d+2)*E[A^2*B] - 2(d+2)*E[A*B^2] + 4(d+2)^2*E[A*B]
# + d(d+2)*E[A^2] + d(d+2)*E[B^2] - 2d(d+2)^2*E[A] - 2d(d+2)^2*E[B] + d^2(d+2)^2
#
# By symmetry (A and B are symmetric under Z -> -Z): E[A^k] = E[B^k], E[A^2*B] = E[A*B^2]
#
# = E[A^2*B^2] - 4(d+2)*E[A^2*B] + 4(d+2)^2*E[AB] + 2d(d+2)*E[A^2] - 4d(d+2)^2*E[A] + d^2(d+2)^2

# Let me compute this with SymPy using the chi-squared moments approach.
# Actually, let me use a DIRECT numerical approach for specific d values,
# then match the polynomial pattern.

import numpy as np

def compute_Q_d_numerically(d_val, r_sq_val, n_samples=10_000_000):
    """
    Monte Carlo computation of E[B_d(A)*B_d(B)] for verification.
    """
    rng = np.random.default_rng(42)
    
    # W ~ N(0, I_d)
    # S = ||W||^2, Z = W_1 (first component)
    S = rng.chisquare(d_val, n_samples)
    Z = rng.standard_normal(n_samples)
    
    r = np.sqrt(r_sq_val)
    
    # A = S/2 + r*Z/sqrt(2) + r^2/4
    # B = S/2 - r*Z/sqrt(2) + r^2/4
    A = S/2 + r*Z/np.sqrt(2) + r_sq_val/4
    B = S/2 - r*Z/np.sqrt(2) + r_sq_val/4
    
    # B_d(x) = x^2 - 2(d+2)x + d(d+2)
    def Bd(x):
        return x**2 - 2*(d_val+2)*x + d_val*(d_val+2)
    
    return np.mean(Bd(A) * Bd(B))


# Now let's do the SYMBOLIC calculation
# We need moments of A and B jointly.
# A = S/2 + r*Z/sqrt(2) + r^2/4
# B = S/2 - r*Z/sqrt(2) + r^2/4
# S ~ chi^2_d (independent of Z ~ N(0,1))

# Moments of S (chi^2_d):
# E[S] = d
# E[S^2] = d(d+2)
# E[S^3] = d(d+2)(d+4)
# E[S^4] = d(d+2)(d+4)(d+6)

# Moments of Z (standard normal):
# E[Z] = 0, E[Z^2] = 1, E[Z^3] = 0, E[Z^4] = 3

# Let c = r/sqrt(2), m = r^2/4. Then:
# A = S/2 + c*Z + m
# B = S/2 - c*Z + m
# 
# A*B = (S/2 + m)^2 - c^2*Z^2
# A+B = S + 2m
# A-B = 2*c*Z
#
# A^2 = (S/2)^2 + c^2*Z^2 + m^2 + S*c*Z + S*m + 2*c*m*Z
#      = S^2/4 + c^2*Z^2 + m^2 + c*S*Z + m*S + 2*c*m*Z
# B^2 = S^2/4 + c^2*Z^2 + m^2 - c*S*Z + m*S - 2*c*m*Z
#
# By S, Z independence and E[Z]=0, E[Z^3]=0:
# E[A] = E[S]/2 + m = d/2 + r^2/4
# E[A^2] = E[S^2]/4 + c^2 + m^2 + m*E[S]
#         = d(d+2)/4 + r^2/2 + r^4/16 + r^2*d/4
#         = d(d+2)/4 + r^2*(d+2)/4 + r^4/16   ... wait let me be more careful

# c = r/sqrt(2), c^2 = r^2/2
# m = r^2/4, m^2 = r^4/16

# E[A^2] = E[S^2]/4 + c^2*E[Z^2] + m^2 + 0 + m*E[S] + 0
#         = d(d+2)/4 + r^2/2 + r^4/16 + (r^2/4)*d
#         = d(d+2)/4 + r^2/2 + r^2*d/4 + r^4/16
#         = d(d+2)/4 + r^2*(2+d)/4 + r^4/16    ... hmm wait
#         r^2/2 = 2r^2/4, so: d(d+2)/4 + r^2*(2 + d)/4 + r^4/16
#         Hmm: r^2/2 + r^2*d/4 = r^2*(2/4 + d/4) = r^2*(d+2)/4. Yes.
# E[A^2] = d(d+2)/4 + (d+2)*r^2/4 + r^4/16

# Hmm - that's exactly the P_d polynomial! P_d(r^2) = r^4/16 - (d+2)*r^2/4 + d(d+2)/4
# And E[A^2] = P_d(r^2) + 2*(d+2)*r^2/4 = P_d(r^2) + (d+2)*r^2/2... 
# Actually E[A^2] = r^4/16 + (d+2)*r^2/4 + d(d+2)/4, which differs in sign from P_d.
# P_d has -(d+2)*r^2/4 while E[A^2] has +(d+2)*r^2/4. Different things.

# Let me just compute everything symbolically with SymPy for correctness.

print("\n" + "=" * 70)
print("SYMBOLIC DERIVATION OF Q_d")
print("=" * 70)

d_sym, r_sq, t = symbols('d r_sq t', positive=True)
c_sq = r_sq / 2  # c^2 = r^2/2
m = r_sq / 4     # m = r^2/4

# Moments of S ~ chi^2_d:
ES1 = d_sym
ES2 = d_sym * (d_sym + 2)
ES3 = d_sym * (d_sym + 2) * (d_sym + 4)
ES4 = d_sym * (d_sym + 2) * (d_sym + 4) * (d_sym + 6)

# Moments of Z ~ N(0,1):
EZ2 = 1
EZ4 = 3

# Now we need to compute E[B_d(A) * B_d(B)]
# B_d(x) = x^2 - 2(d+2)x + d(d+2)
# 
# E[B_d(A)*B_d(B)] = E[A^2*B^2] - 2(d+2)*E[A^2*B] - 2(d+2)*E[A*B^2] 
#                    + 4(d+2)^2*E[AB] + d(d+2)*E[A^2] + d(d+2)*E[B^2]
#                    - 2d(d+2)^2*E[A] - 2d(d+2)^2*E[B] + d^2*(d+2)^2
#
# By symmetry (A <-> B under Z -> -Z): E[A^k] = E[B^k], E[A^j*B^k] = E[A^k*B^j]
#
# = E[A^2*B^2] - 4(d+2)*E[A^2*B] + 4(d+2)^2*E[AB] + 2d(d+2)*E[A^2] - 4d(d+2)^2*E[A] + d^2*(d+2)^2

# Step by step: compute each moment
# Using A = S/2 + c*Z + m, B = S/2 - c*Z + m, with S indep of Z

# Let p = S/2 + m (the common part), q = c*Z (the signed part)
# A = p + q, B = p - q

# E[A] = E[p] = E[S]/2 + m = d/2 + r^2/4
EA = ES1/2 + m

# E[AB] = E[(p+q)(p-q)] = E[p^2 - q^2] = E[p^2] - E[q^2]
# E[p^2] = E[(S/2+m)^2] = E[S^2]/4 + m*E[S] + m^2 = d(d+2)/4 + m*d + m^2
Ep2 = ES2/4 + m*ES1 + m**2
# E[q^2] = c^2*E[Z^2] = c^2 = r^2/2
Eq2 = c_sq * EZ2
EAB = Ep2 - Eq2

# E[A^2] = E[(p+q)^2] = E[p^2] + 2*E[p*q] + E[q^2]
# E[p*q] = E[(S/2+m)*c*Z] = c*E[(S/2+m)]*E[Z] = 0 (since E[Z]=0 and S,Z indep)
EA2 = Ep2 + Eq2

# E[A^2*B] = E[(p+q)^2*(p-q)] = E[(p^2+2pq+q^2)(p-q)]
#           = E[p^3 - p^2*q + 2p^2*q - 2pq^2 + pq^2 - q^3]
#           = E[p^3 + p^2*q - pq^2 - q^3]
#           = E[p^3] + E[p^2]*E[q] - E[p]*E[q^2] - E[q^3]
# Since E[q] = c*E[Z] = 0 and E[q^3] = c^3*E[Z^3] = 0:
#           = E[p^3] - E[p]*E[q^2]  ... wait let me redo
# 
# Actually: E[p^2*q] = E[p^2]*E[q] = 0 (independence, E[q]=0)
# E[pq^2] = E[p]*E[q^2] = (d/2 + m) * c^2
# E[q^3] = c^3 * E[Z^3] = 0
#
# E[A^2*B] = E[(p+q)^2*(p-q)]
#           = E[p^3 + p^2*q - p^2*q - pq^2 + pq^2 + q^2*p - q^2*p - q^3]
# Hmm let me just expand carefully:
# (p+q)^2 = p^2 + 2pq + q^2
# (p+q)^2 * (p-q) = p^3 - p^2*q + 2p^2*q - 2pq^2 + q^2*p - q^3
#                  = p^3 + p^2*q - 2pq^2 + q^2*p - q^3
#                  = p^3 + p^2*q - pq^2 - q^3

# E[A^2*B] = E[p^3] + E[p^2*q] - E[p*q^2] - E[q^3]
# E[p^2*q] = E[p^2]*E[q] = 0
# E[p*q^2] = E[p]*E[q^2] (independence) = EA * c_sq
# E[q^3] = 0
# E[p^3] = E[(S/2+m)^3] = E[S^3]/8 + 3*E[S^2]*m/4 + 3*E[S]*m^2/2 + m^3
Ep3 = ES3/8 + 3*ES2*m/4 + 3*ES1*m**2/2 + m**3

EA2B = Ep3 - EA * Eq2

# E[A^2*B^2] = E[(p+q)^2*(p-q)^2] = E[(p^2-q^2)^2] = E[p^4 - 2p^2*q^2 + q^4]
# E[p^4] = E[(S/2+m)^4]
# E[p^2*q^2] = E[p^2]*E[q^2] (independence)
# E[q^4] = c^4*E[Z^4] = (r^2/2)^2 * 3 = 3*r^4/4
Ep4 = ES4/16 + 4*ES3*m/8 + 6*ES2*m**2/4 + 4*ES1*m**3/2 + m**4
# Simplify: ES4/16 + ES3*m/2 + 3*ES2*m^2/2 + 2*ES1*m^3 + m^4
# Actually let me use binomial:
# (S/2+m)^4 = sum_{k=0}^4 C(4,k) (S/2)^k * m^{4-k}
# E[(S/2+m)^4] = E[S^4]/16 + 4*E[S^3]*m/8 + 6*E[S^2]*m^2/4 + 4*E[S]*m^3/2 + m^4
#              = ES4/16 + ES3*m/2 + 3*ES2*m^2/2 + 2*ES1*m^3 + m^4
Ep4 = ES4/16 + ES3*m/2 + Rational(3,2)*ES2*m**2 + 2*ES1*m**3 + m**4

Ep2q2 = Ep2 * Eq2
Eq4 = c_sq**2 * EZ4  # = (r^2/2)^2 * 3 = 3*r^4/4

EA2B2 = Ep4 - 2*Ep2q2 + Eq4

# Now assemble E[B_d(A)*B_d(B)]:
# = E[A^2*B^2] - 4(d+2)*E[A^2*B] + 4(d+2)^2*E[AB] + 2d(d+2)*E[A^2] - 4d(d+2)^2*E[A] + d^2*(d+2)^2

D2 = d_sym + 2  # shorthand

result = (EA2B2 
          - 4*D2*EA2B 
          + 4*D2**2*EAB 
          + 2*d_sym*D2*EA2 
          - 4*d_sym*D2**2*EA 
          + d_sym**2*D2**2)

# Substitute m = r_sq/4, c_sq = r_sq/2
result_expanded = expand(result)

print(f"\nE[B_d(A)*B_d(B)] = ")
print(f"  {result_expanded}")

# Now substitute specific d values and express as polynomial in r_sq
print("\n" + "-" * 70)
print("CHECKING SPECIFIC DIMENSIONS:")
print("-" * 70)

for d_val in [1, 2, 3, 5, 10]:
    expr = result_expanded.subs(d_sym, d_val)
    expr_simplified = expand(expr)
    # Collect in powers of r_sq
    poly = sp.Poly(expr_simplified, r_sq)
    coeffs = poly.all_coeffs()
    print(f"\nd = {d_val}:")
    print(f"  Q_{d_val}(r^2) = {expr_simplified}")
    print(f"  Coefficients (highest power first): {coeffs}")
    
    # Verify numerically
    for r_test in [0.0, 1.0, 4.0, 9.0]:
        symbolic_val = float(expr_simplified.subs(r_sq, r_test))
        numeric_val = compute_Q_d_numerically(d_val, r_test, n_samples=5_000_000)
        print(f"  r^2={r_test:.1f}: symbolic={symbolic_val:.6f}, MC={numeric_val:.6f}, "
              f"rel_err={abs(symbolic_val-numeric_val)/(abs(symbolic_val)+1e-10):.2e}")

# Now let's find the general pattern
print("\n" + "=" * 70)
print("GENERAL POLYNOMIAL FORM")
print("=" * 70)

# The result should be a polynomial in r_sq of degree 4 (since B_d is degree 2 in A,B
# and A,B are linear in S which is degree 1 in the "r_sq polynomial" sense)
# Actually A and B contain r_sq/4 terms, so the product B_d(A)*B_d(B) can go up to
# degree 4 in r_sq.

# Let's collect as polynomial in r_sq with d as parameter
poly_general = sp.Poly(result_expanded, r_sq)
print(f"\nDegree in r_sq: {poly_general.degree()}")
print(f"\nCoefficients (from highest to lowest power of r_sq):")
for i, c in enumerate(poly_general.all_coeffs()):
    power = poly_general.degree() - i
    c_simplified = factor(expand(c))
    print(f"  r_sq^{power}: {c_simplified}")

print("\n" + "=" * 70)
print("FINAL Q_d POLYNOMIAL")
print("=" * 70)
print()
print("The full pairwise contribution to Psi_6 estimation is:")
print()
print("  I_ij^(6) = (4*pi*b^2)^{-d/2} * exp(-r^2/4) * Q_d(r^2) / b^8")
print()
print("where Q_d(r^2) = E[B_d(A)*B_d(B)] as computed above.")
print()
print("The COMPLETE Psi_6 estimator is:")
print()
print("  Psi_6_hat(b) = [1/(n^2 * (4pi)^{d/2} * b^{d+8})] * sum_ij exp(-r_ij^2/4) * Q_d(r_ij^2)")
print()

# Verify d=1 case against known phi^(6) result
print("\n" + "=" * 70)
print("VERIFICATION: d=1 vs KNOWN phi^(6)")
print("=" * 70)
print()
print("In 1D, the estimator of psi_6 = R(f''') uses phi^(6)(z) = (z^6 - 15z^4 + 45z^2 - 15)*phi(z)")
print()
print("The pairwise formula in 1D should give, for the SJ estimator:")
print("  T_D(b) = -1/(n^2 * b^7) * sum_ij phi^(6)((Xi-Xj)/b)")
print()
print("Our formula gives:")
print("  Psi_6_hat(b) = 1/(n^2 * (4pi)^{1/2} * b^9) * sum_ij exp(-r_ij^2/4) * Q_1(r_ij^2)")
print()
print("where r_ij^2 = (Xi-Xj)^2/b^2")
print()

# For d=1, let's verify that exp(-r^2/4)*Q_1(r^2) / (4pi)^{1/2} * b^{-9}
# matches phi^(6)(z)/b^7 where z = (Xi-Xj)/b
# 
# phi^(6)(z) = (z^6 - 15z^4 + 45z^2 - 15) * (2pi)^{-1/2} * exp(-z^2/2)
#
# In OUR framework:
# The integral of (nabla^2)^2 K_b(x-a) * (nabla^2)^2 K_b(x-b) dx
# = (4*pi*b^2)^{-1/2} * exp(-z^2/4) * Q_1(z^2) * b^{-8}
#
# But in SJ's notation:
# int phi^(4)}(x-a)/b) * phi^{(4)}((x-b)/b) / b^2 dx  (the b^{-2} from each phi^(4)/b)
# = ... this gets complicated. Let me just verify numerically.

print("\nNumerical verification for d=1:")
print("Compare our Q_1 formula against direct integration of phi^(6)")

from scipy import integrate as quad_integrate
from scipy.stats import norm

def phi_6(z):
    """6th derivative of standard normal pdf."""
    return (z**6 - 15*z**4 + 45*z**2 - 15) * norm.pdf(z)

def psi6_formula_1d(X, b):
    """Our Q_d formula for Psi_6 at d=1."""
    n = len(X)
    total = 0.0
    for i in range(n):
        for j in range(n):
            r_sq_val = (X[i] - X[j])**2 / b**2
            Q1 = float(result_expanded.subs(d_sym, 1).subs(r_sq, r_sq_val))
            total += np.exp(-r_sq_val/4) * Q1
    return total / (n**2 * np.sqrt(4*np.pi) * b**9)

def psi6_sj_1d(X, b):
    """SJ's T_D formula: -1/(n^2 * b^7) * sum_ij phi^(6)((Xi-Xj)/b)."""
    n = len(X)
    total = 0.0
    for i in range(n):
        for j in range(n):
            z = (X[i] - X[j]) / b
            total += phi_6(z)
    return -total / (n**2 * b**7)

# Test with a small dataset
np.random.seed(123)
X_test = np.random.randn(20)
b_test = 0.5

val_ours = psi6_formula_1d(X_test, b_test)
val_sj = psi6_sj_1d(X_test, b_test)
print(f"\n  b = {b_test}")
print(f"  Our formula (Psi_6_hat): {val_ours:.10f}")
print(f"  SJ formula (T_D):        {val_sj:.10f}")
print(f"  Ratio (should be 1.0):   {val_ours/val_sj:.10f}")

# Try another b
b_test2 = 1.0
val_ours2 = psi6_formula_1d(X_test, b_test2)
val_sj2 = psi6_sj_1d(X_test, b_test2)
print(f"\n  b = {b_test2}")
print(f"  Our formula (Psi_6_hat): {val_ours2:.10f}")
print(f"  SJ formula (T_D):        {val_sj2:.10f}")
print(f"  Ratio (should be 1.0):   {val_ours2/val_sj2:.10f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
