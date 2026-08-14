"""
Investigate the discrepancy between our Q_1 formula and SJ's phi^(6) formula.

The issue: our formula computes the integral of products of bi-Laplacians of the kernel,
while SJ's formula computes the integral of products of 3rd derivatives.

In 1D:
  - Our Psi_4 uses: int (f'')^2 dx  (Laplacian squared, coincides with 2nd deriv in 1D)
  - Our Psi_6 should be: int (f''')^2 dx  (the NEXT functional in the chain)
  
BUT what we computed is: int [(f'')^2]'' dx  (the bi-Laplacian = 4th derivative in 1D)
That gives psi_8 = int (f'''')^2 dx, NOT psi_6 = int (f''')^2 dx!

The confusion: in the multivariate case, what's the correct "next order" functional?

In 1D:
  - psi_4 = R(f'') = int (f'')^2     ← 2nd derivative squared
  - psi_6 = R(f''') = int (f''')^2    ← 3rd derivative squared  
  - psi_8 = R(f'''') = int (f'''')^2  ← 4th derivative squared

In d-D:
  - Psi_4 = int (nabla^2 f)^2         ← Laplacian squared (order 2 operator)
  - Psi_6 = ???                        ← what goes here?
  - Psi_8 = int ((nabla^2)^2 f)^2     ← bi-Laplacian squared (order 4 operator)

The key question: in the AMISE of the Psi_4 estimator, which functional appears in
the bias? Let's work this out.

From the 1D case:
  E[psi_4_hat(g)] - psi_4 = (1/2) * g^2 * sigma_K^2 * psi_6 + O(g^4)
  
  The bias of order g^2 involves psi_6 = R(f'''), one derivative order higher.

In d-D, for our estimator:
  E[Psi_4_hat(g)] - Psi_4 = ???

The bias comes from the smoothing: convolving f with K_g blurs the true f.
  (nabla^2 f_g)^2 ≈ (nabla^2 f)^2 + g^2 * [correction involving higher derivatives]

The correction involves: int (nabla^2 f)(nabla^4 f) dx  (by Taylor expansion)

Actually, let me derive this carefully for 1D first to understand the pattern,
then generalize.
"""

import numpy as np
from scipy.stats import norm

# =============================================================================
# 1D: Understand what functional appears in the bias of psi_4_hat
# =============================================================================

# In 1D, psi_4_hat(g) = (1/n^2) * sum_ij int f_g''(x)^2 dx
# where f_g = f * K_g (smoothed density)
#
# But in the SJ framework, they estimate psi_4 directly from the sample:
# S_D(g) = (1/n^2) * g^{-5} * sum_ij phi^(4)((Xi-Xj)/g)
#
# The bias of S_D(g) as an estimator of psi_4 is:
# E[S_D(g)] - psi_4 = diagonal_term + smoothing_bias
#
# Diagonal term: phi^(4)(0)/(n*g^5) = 3/(n*g^5*sqrt(2*pi))
# Smoothing bias: (1/2)*g^2 * mu_2(K) * psi_6  [Jones & Sheather 1991]
#   where mu_2(K) = int z^2 K(z) dz = 1 for Gaussian kernel
#
# So: E[S_D(g)] - psi_4 ≈ 3/(n*g^5*sqrt(2*pi)) + (1/2)*g^2*psi_6
#
# Setting the two terms equal for optimal bias cancellation:
# 3/(n*g^5*sqrt(2*pi)) = -(1/2)*g^2*psi_6  ... but signs!
#
# Actually the smoothing bias is NEGATIVE:
# E[S_D(g)] ≈ psi_4 + 3/(n*g^5*sqrt(2*pi)) + (1/2)*g^2*(-|psi_6|)
#
# For normal densities, psi_6 = R(f''') is POSITIVE (integral of a square)
# but in SJ's notation, psi_6 relates to f''' differently...
# Actually T_D estimates psi_6 which they define with a sign convention.
# In the AMISE expansion, the bias term has coefficient that involves
# psi_6 = int (f''')^2 dx > 0 always.

# The key: the SMOOTHING BIAS of the Laplacian roughness estimator involves
# the GRADIENT of the Laplacian, not the bi-Laplacian.
#
# In d-D: when you smooth f with K_g, the Laplacian roughness of the smoothed version is:
# int (nabla^2 (f*K_g))^2 dx = int (nabla^2 f)^2 dx + g^2 * [bias involving nabla^3 f]
#
# The relevant functional is:
# Psi_6 = int ||nabla(nabla^2 f)||^2 dx  (gradient of Laplacian squared)
#
# NOT int ((nabla^2)^2 f)^2 dx  (bi-Laplacian squared)!
#
# In 1D: ||nabla(nabla^2 f)||^2 = (f''')^2, so Psi_6 = psi_6 = R(f'''). ✓
# In d-D: ||nabla(nabla^2 f)||^2 = ||grad(Lap f)||^2

# =============================================================================
# So we need the GRADIENT-OF-LAPLACIAN kernel, not the bi-Laplacian kernel!
# =============================================================================

# In 1D: grad of Laplacian of K_h = d^3/dx^3 K_h = h^{-3} K_h * He_3(x/h)
# where He_3(z) = z^3 - 3z (3rd Hermite polynomial)
#
# The pairwise integral: int [K_h'''(x-a)] * [K_h'''(x-b)] dx
# This gives: the 1D version with phi^(6) because:
# int f'''(x)^2 dx via KDE estimation = (1/n^2) sum_ij int K_g'''(x-Xi)*K_g'''(x-Xj) dx
# = (1/n^2) sum_ij [convolution of 3rd derivatives evaluated at Xi-Xj]
# = (1/n^2) sum_ij [h^{-7} * phi^(6)((Xi-Xj)/a)] where a = g*sqrt(2)
#
# Actually SJ uses a DIFFERENT bandwidth for this estimation.
# Let me just verify the relationship directly.

# The pairwise integral for d=1:
# int K_g'''(x-a) * K_g'''(x-b) dx
# = g^{-6} * int K_g(x-a) * He_3((x-a)/g) * K_g(x-b) * He_3((x-b)/g) dx
# 
# Using Gaussian product: K_g(x-a)*K_g(x-b) = C_ab * phi_{g/sqrt(2)}(x - mid)
# where C_ab = (4*pi*g^2)^{-1/2} * exp(-delta^2/(4g^2))
#
# Then the integral becomes C_ab * E[He_3(U/g) * He_3(V/g)]
# where U = x-a, V = x-b with x ~ N(mid, g^2/2)

# Let me verify: what polynomial do we get for the 3rd-derivative pairwise integral?

def verify_3rd_derivative_integral_1d():
    """
    Compute int K_g'''(x-a) * K_g'''(x-b) dx numerically for various delta = a-b.
    """
    g = 1.0  # bandwidth
    
    def K_g_triple_prime(x, center):
        z = (x - center) / g
        # K_g(x) = (2*pi*g^2)^{-1/2} * exp(-z^2/2)
        # K_g'''(x) = g^{-3} * K_g(x) * (-z^3 + 3z) = -g^{-3} * K_g(x) * He_3(z)
        # He_3(z) = z^3 - 3z
        return -g**(-3) * norm.pdf(z)/g * (z**3 - 3*z)
    
    print("\n1D: Pairwise integral of K_g''' * K_g''' vs formula")
    print(f"g = {g}")
    
    for delta in [0.0, 0.5, 1.0, 2.0, 3.0]:
        a, b = delta/2, -delta/2
        
        # Numerical integration
        def integrand(x):
            return K_g_triple_prime(x, a) * K_g_triple_prime(x, b)
        
        from scipy.integrate import quad
        val_num, err = quad(integrand, -20, 20, limit=200)
        
        # Our formula would give:
        # C_ab * g^{-6} * E[He_3(U) * He_3(V)]
        # where U = (x-a)/g = sigma*W - mu, V = (x-b)/g = sigma*W + mu
        # sigma = 1/sqrt(2), mu = delta/(2g)
        # He_3(z) = z^3 - 3z
        
        r_sq_val = delta**2 / g**2
        C_ab = (4*np.pi*g**2)**(-0.5) * np.exp(-r_sq_val/4)
        
        # Monte Carlo for E[He_3(U)*He_3(V)]
        rng = np.random.default_rng(42)
        W = rng.standard_normal(2_000_000)
        sigma = 1/np.sqrt(2)
        mu = delta/(2*g)
        U = sigma*W + mu  # (x-b)/g when x ~ N(mid, g^2/2)
        V = sigma*W - mu  # (x-a)/g
        
        He3_U = U**3 - 3*U
        He3_V = V**3 - 3*V
        
        E_product = np.mean(He3_U * He3_V)
        
        val_formula = C_ab * g**(-6) * E_product
        
        print(f"  delta={delta:.1f}: numerical={val_num:.8f}, formula={val_formula:.8f}, "
              f"ratio={val_num/val_formula if abs(val_formula) > 1e-15 else 'inf':.6f}")

verify_3rd_derivative_integral_1d()

# =============================================================================
# Now let's derive the d-D formula for Psi_6 = int ||grad(Lap f)||^2 dx
# =============================================================================
#
# The gradient of the Laplacian of K_h is a VECTOR:
# grad(nabla^2 K_h(t)) = grad[K_h(t) * h^{-2} * (||t||^2/h^2 - d)]
#                       = h^{-2} * [K_h(t)*grad(L) + L*grad(K_h(t))]
#                       = h^{-2} * [K_h * (2t/h^2) + L * K_h * (-t/h^2)]
#                       = K_h * h^{-2} * t/h^2 * [2 - L]
#                       = K_h * t/(h^4) * [2 - (||t||^2/h^2 - d)]
#                       = K_h * t/(h^4) * [(d+2) - ||t||^2/h^2]
#
# So: grad(nabla^2 K_h(t)) = K_h(t) * t * h^{-4} * [(d+2) - s]
#     where s = ||t||^2/h^2
#
# ||grad(nabla^2 K_h(t))||^2 = K_h(t)^2 * ||t||^2 * h^{-8} * [(d+2) - s]^2
#                             = K_h(t)^2 * s * h^{-6} * [(d+2) - s]^2
#
# Wait, let me be more careful:
# ||grad(nabla^2 K_h(t))||^2 = [K_h(t)]^2 * ||t||^2/h^8 * [(d+2) - s]^2
#                             = [K_h(t)]^2 * (s*h^2)/h^8 * [(d+2) - s]^2
#                             = [K_h(t)]^2 * s * h^{-6} * [(d+2) - s]^2
#
# Hmm, but this is for the Psi_6 of the DENSITY, not the kernel estimator product.
# What we actually need is the PAIRWISE integral:
#
# int [grad(nabla^2 K_b(x-a))] . [grad(nabla^2 K_b(x-b))] dx
#
# Since grad(nabla^2 K_b(t)) = K_b(t) * (t/b^4) * [(d+2) - ||t||^2/b^2]:
#
# [grad(nabla^2 K_b(x-a))] . [grad(nabla^2 K_b(x-b))]
# = K_b(x-a)*K_b(x-b) * [(x-a)/b^4 * ((d+2) - ||x-a||^2/b^2)] . [(x-b)/b^4 * ((d+2) - ||x-b||^2/b^2)]
# = K_b(x-a)*K_b(x-b) / b^8 * [(x-a).(x-b)] * [(d+2) - ||x-a||^2/b^2] * [(d+2) - ||x-b||^2/b^2]
#
# After Gaussian product substitution (x = mid + (b/sqrt(2))*W, W ~ N(0,I_d)):
# This becomes C_ab / b^8 * E[dot_product * scalar1 * scalar2]
#
# Where:
# (x-a) = (b/sqrt(2))*W - delta/2
# (x-b) = (b/sqrt(2))*W + delta/2
# (x-a).(x-b) = b^2/2 * ||W||^2 - ||delta||^2/4 = b^2/2 * (||W||^2 - r^2/2)
#   where r^2 = ||delta||^2/b^2
#
# ||x-a||^2/b^2 = ||(b/sqrt(2))*W - delta/2||^2 / b^2
#               = ||W||^2/2 - (W.delta)/(sqrt(2)*b) + ||delta||^2/(4b^2)
#               = S/2 - sqrt(2)*(W.mu) + r^2/4   (with mu = delta/(2b))
# Wait: (b/sqrt(2))*W - delta/2, divided by b:
# = W/sqrt(2) - delta/(2b)
# ||...||^2 = ||W||^2/2 - sqrt(2)*(W.delta/(2b)) + ||delta/(2b)||^2
#           = S/2 - (W.delta)/(sqrt(2)*b) + r^2/4
# Hmm, (W . delta/(2b)) * sqrt(2)... let me be more careful.
# Let mu_vec = delta/(2b), ||mu_vec|| = r/2
# ||x-a||^2/b^2 = ||W/sqrt(2) - mu_vec||^2 = ||W||^2/2 - sqrt(2)*(W.mu_vec) + ||mu_vec||^2
#               = S/2 - sqrt(2)*T + r^2/4
# where T = W . mu_vec_hat * ||mu_vec|| = W . (delta_hat/2) * r ... 
# Actually T = W . mu_vec where mu_vec = delta/(2b), so T is scalar with E[T]=0, E[T^2] = ||mu_vec||^2 = r^2/4
# Actually T = W . mu_vec = sum_k W_k * mu_vec_k, has variance ||mu_vec||^2 = r^2/4
# Let Z = T / ||mu_vec|| = W . mu_hat ~ N(0,1), so T = Z * r/2

# OK this is getting complex. Let me just compute it numerically for d=1 to verify
# that the gradient-of-Laplacian approach gives the correct 1D result.

print("\n" + "=" * 70)
print("d-D: Pairwise integral of grad(Lap K) . grad(Lap K)")
print("=" * 70)

def pairwise_grad_lap_integral_mc(d_val, r_sq_val, b=1.0, n_samples=5_000_000):
    """
    Monte Carlo: int [grad(nabla^2 K_b(x-a))] . [grad(nabla^2 K_b(x-b))] dx
    
    After Gaussian product substitution:
    = C_ab / b^8 * E[(U.V) * ((d+2) - ||U||^2/b^2) * ((d+2) - ||V||^2/b^2)]
    
    Wait, let me be precise.
    
    U = x - a, V = x - b, with x ~ N(mid, (b^2/2)*I) after the Gaussian product.
    In scaled coords: U/b and V/b.
    
    grad(nabla^2 K_b(x-a)) = K_b(x-a) * (x-a)/b^4 * [(d+2) - ||x-a||^2/b^2]
    
    After substitution x = mid + (b/sqrt(2))*W:
    (x-a)/b = W/sqrt(2) - delta/(2b) = W/sqrt(2) - mu_vec
    (x-b)/b = W/sqrt(2) + mu_vec
    
    Let u = W/sqrt(2) - mu_vec, v = W/sqrt(2) + mu_vec (these are (x-a)/b and (x-b)/b)
    
    The integrand (divided by K_b(x-a)*K_b(x-b)/b^8) is:
    (u . v) * b^2 * [(d+2) - ||u||^2*b^2/b^2] * [(d+2) - ||v||^2*b^2/b^2]
    
    Wait I'm confusing myself. Let me restart carefully.
    
    The formula for the GRADIENT of the Laplacian of a Gaussian kernel:
    
    nabla^2 K_h(t) = K_h(t) * h^{-2} * (||t||^2/h^2 - d)
    grad(nabla^2 K_h(t)) = grad[K_h(t) * h^{-2} * (||t||^2/h^2 - d)]
    
    Using product rule: grad(fg) = f*grad(g) + g*grad(f)
    
    f = K_h(t), g = h^{-2}(||t||^2/h^2 - d) = ||t||^2/h^4 - d/h^2
    grad(f) = K_h(t) * (-t/h^2)
    grad(g) = 2t/h^4
    
    grad(nabla^2 K_h(t)) = K_h(t)*2t/h^4 + (||t||^2/h^4 - d/h^2)*K_h(t)*(-t/h^2)
                         = K_h(t) * t * [2/h^4 - ||t||^2/h^6 + d/h^4]
                         = K_h(t) * t/h^4 * [(d+2) - ||t||^2/h^2]
    
    So: grad(nabla^2 K_b(x-a)) = K_b(x-a) * (x-a)/b^4 * [(d+2) - ||x-a||^2/b^2]
    
    The dot product of two such vectors:
    [grad(nabla^2 K_b(x-a))] . [grad(nabla^2 K_b(x-b))]
    = K_b(x-a)*K_b(x-b) / b^8 * [(x-a).(x-b)] * [(d+2)-||x-a||^2/b^2]*[(d+2)-||x-b||^2/b^2]
    
    Integrating and using the Gaussian product:
    = C_ab * b^{-8} * E[((x-a).(x-b)) * ((d+2)-||x-a||^2/b^2) * ((d+2)-||x-b||^2/b^2)]
    
    where x ~ N(mid, (b^2/2)*I_d), so x-a = (b/sqrt(2))*W - delta/2, x-b = (b/sqrt(2))*W + delta/2
    with W ~ N(0,I_d).
    
    Let u = (x-a)/b = W/sqrt(2) - mu, v = (x-b)/b = W/sqrt(2) + mu
    where mu = delta/(2b), ||mu||^2 = r^2/4
    
    Then:
    (x-a).(x-b) = b^2 * (u.v)
    ||x-a||^2/b^2 = ||u||^2
    ||x-b||^2/b^2 = ||v||^2
    
    So the integral = C_ab * b^{-8} * b^2 * E[(u.v) * ((d+2)-||u||^2) * ((d+2)-||v||^2)]
                    = C_ab * b^{-6} * E[(u.v) * ((d+2)-||u||^2) * ((d+2)-||v||^2)]
    
    Hmm wait, that gives b^{-6} not b^{-8}. Let me recheck.
    Actually C_ab already has b^{-d} in it (from (4*pi*b^2)^{-d/2}), so the total is:
    
    (4*pi*b^2)^{-d/2} * exp(-r^2/4) * b^{-6} * E[...]
    = (4*pi)^{-d/2} * b^{-(d+6)} * exp(-r^2/4) * E[...]
    
    So: Psi_6_hat(b) = 1/(n^2 * (4*pi)^{d/2} * b^{d+6}) * sum_ij exp(-r_ij^2/4) * R_d(r_ij^2)
    
    where R_d(r^2) = E[(u.v) * ((d+2)-||u||^2) * ((d+2)-||v||^2)]
    
    Let me compute this numerically.
    """
    rng = np.random.default_rng(42)
    
    r = np.sqrt(r_sq_val)
    mu_vec = np.zeros(d_val)
    if r > 0:
        mu_vec[0] = r / 2  # mu = delta/(2b), ||mu|| = r/2
    
    W = rng.standard_normal((n_samples, d_val))
    
    # u = W/sqrt(2) - mu, v = W/sqrt(2) + mu
    u = W / np.sqrt(2) - mu_vec
    v = W / np.sqrt(2) + mu_vec
    
    # (u.v) = dot product along axis 1
    u_dot_v = np.sum(u * v, axis=1)
    
    # ||u||^2, ||v||^2
    u_sq = np.sum(u**2, axis=1)
    v_sq = np.sum(v**2, axis=1)
    
    # The expectation
    integrand = u_dot_v * ((d_val + 2) - u_sq) * ((d_val + 2) - v_sq)
    
    return np.mean(integrand)


# Test d=1: should match phi^(6) pairwise formula
print("\nd=1 verification:")
print("Our R_1(r^2) should relate to phi^(6) formula")

for r_sq_val in [0.0, 1.0, 4.0, 9.0]:
    R1 = pairwise_grad_lap_integral_mc(1, r_sq_val, n_samples=10_000_000)
    
    # The SJ formula in 1D: int K_b'''(x-a)*K_b'''(x-b) dx = b^{-7} * phi^(6)(delta/b_eff)
    # Actually, from the convolution: int K_b'''(x-a)*K_b'''(x-b) dx = (-1)^3 * (K_b*K_b)^(6)(delta)
    # = K_{b*sqrt(2)}^(6)(delta) [convolution of derivatives]
    # = (b*sqrt(2))^{-7} * phi^(6)(delta/(b*sqrt(2)))
    # 
    # Let b=1: = (sqrt(2))^{-7} * phi^(6)(delta/sqrt(2))
    # With r^2 = delta^2/b^2 = delta^2: delta = sqrt(r_sq)
    # z = delta/sqrt(2) = sqrt(r_sq/2)
    
    delta = np.sqrt(r_sq_val)
    b = 1.0
    z = delta / (b * np.sqrt(2))
    # phi^(6)(z) = (z^6 - 15z^4 + 45z^2 - 15) * phi(z)
    phi6_val = (z**6 - 15*z**4 + 45*z**2 - 15) * norm.pdf(z)
    conv_formula = phi6_val / (b*np.sqrt(2))**7
    
    # Our integral = C_ab * b^{-6} * R_1(r^2)
    # C_ab = (4*pi*b^2)^{-1/2} * exp(-r^2/4)
    C_ab = (4*np.pi*b**2)**(-0.5) * np.exp(-r_sq_val/4)
    our_integral = C_ab * b**(-6) * R1
    
    print(f"  r^2={r_sq_val:.1f}: R_1={R1:.6f}, our_int={our_integral:.8f}, "
          f"conv_formula={conv_formula:.8f}, ratio={our_integral/conv_formula if abs(conv_formula)>1e-15 else 'nan':.6f}")

# Now check d=1 with the full estimator comparison
print("\n\nFull estimator comparison (d=1):")
print("Our Psi_6_hat vs SJ's T_D")

np.random.seed(123)
X_test = np.random.randn(30)
b_test = 0.8
n = len(X_test)

# SJ's formula: T_D(b) = -1/(n^2*b^7) * sum_ij phi^(6)((Xi-Xj)/b)
total_sj = 0.0
for i in range(n):
    for j in range(n):
        z = (X_test[i] - X_test[j]) / b_test
        total_sj += (z**6 - 15*z**4 + 45*z**2 - 15) * norm.pdf(z)
T_D = -total_sj / (n**2 * b_test**7)

# Our formula using R_d:
# Psi_6_hat = 1/(n^2 * (4pi)^{d/2} * b^{d+6}) * sum_ij exp(-r^2/4) * R_d(r^2)
# But R_d comes from MC, which is slow for a sum. Let me derive R_1 symbolically.

# For d=1, u = W/sqrt(2) - mu, v = W/sqrt(2) + mu (scalars, W~N(0,1))
# u*v = W^2/2 - mu^2
# (3 - u^2) = 3 - (W^2/2 - sqrt(2)*W*mu + mu^2)
# (3 - v^2) = 3 - (W^2/2 + sqrt(2)*W*mu + mu^2)
#
# With d=1: (d+2) = 3
# u.v = (W/sqrt(2) - mu)*(W/sqrt(2) + mu) = W^2/2 - mu^2
# (3 - ||u||^2) = 3 - u^2 = 3 - W^2/2 + sqrt(2)*W*mu - mu^2
# (3 - ||v||^2) = 3 - v^2 = 3 - W^2/2 - sqrt(2)*W*mu - mu^2
#
# Let S = W^2/2, T = sqrt(2)*W*mu = sqrt(2)*W*r/2 = W*r/sqrt(2)
# u.v = S - mu^2 = S - r^2/4
# (3-u^2) = 3 - S + T - r^2/4 = (3 - r^2/4) - S + T
# (3-v^2) = 3 - S - T - r^2/4 = (3 - r^2/4) - S - T
#
# Product: (3-u^2)(3-v^2) = [(3-r^2/4) - S]^2 - T^2
# Full: u.v * (3-u^2)*(3-v^2) = (S - r^2/4) * {[(3-r^2/4) - S]^2 - T^2}
#
# E[...] = E[(S - r^2/4) * ((3-r^2/4-S)^2 - T^2)]
# where S = W^2/2, T = W*r/sqrt(2), so T^2 = W^2*r^2/2 = S*r^2
#
# = E[(S-r^2/4)*((3-r^2/4-S)^2 - S*r^2)]
#
# Let a = r^2/4, b_val = 3 - r^2/4 (NOTE: for d=1, d+2=3 - a = 3 - r^2/4)
# = E[(S-a)*((b_val-S)^2 - S*r^2)]
# = E[(S-a)*(b_val^2 - 2*b_val*S + S^2 - 4*a*S)]  [since r^2 = 4a]
# = E[(S-a)*(S^2 - (2*b_val+4a)*S + b_val^2)]
#
# With S = W^2/2 where W~N(0,1): E[S^k] = E[(W^2/2)^k] = (1/2^k)*E[W^{2k}]
# E[W^2] = 1, E[W^4] = 3, E[W^6] = 15, E[W^8] = 105
# E[S] = 1/2, E[S^2] = 3/4, E[S^3] = 15/8, E[S^4] = 105/16

import sympy as sp

S_sym, a_sym, r2_sym = sp.symbols('S a r2', real=True)
b_sym = 3 - r2_sym/4  # (d+2) - r^2/4 for d=1
a_sym_expr = r2_sym/4

# The expression to take expectation of:
expr = (S_sym - a_sym_expr) * (S_sym**2 - (2*b_sym + r2_sym)*S_sym + b_sym**2)

# Expand
expr_expanded = sp.expand(expr)
print(f"\nExpanded expression (before expectation):")
print(f"  {expr_expanded}")

# Collect by powers of S
poly_S = sp.Poly(expr_expanded, S_sym)
print(f"\n  As polynomial in S: {poly_S}")

# Take expectation: replace S^k with E[S^k]
# S = W^2/2, W~N(0,1)
# E[S^k] = (2k-1)!! / 2^k = (1/2^k) * (2k)! / (2^k * k!) = (2k)!/(4^k * k!)
# E[S] = 1/2, E[S^2] = 3/4, E[S^3] = 15/8

ES_moments = {0: 1, 1: sp.Rational(1,2), 2: sp.Rational(3,4), 3: sp.Rational(15,8)}

expectation = sp.S(0)
for (power,), coeff in poly_S.as_dict().items():
    expectation += coeff * ES_moments[power]

R1_symbolic = sp.expand(expectation)
print(f"\n  R_1(r^2) = E[...] = {R1_symbolic}")
print(f"  Simplified: {sp.factor(R1_symbolic)}")

# Check values
for r2_test in [0, 1, 4, 9]:
    val = float(R1_symbolic.subs(r2_sym, r2_test))
    print(f"  R_1({r2_test}) = {val:.6f}")

# Now compare: does our formula (4pi)^{-1/2} * b^{-(1+6)} * sum exp(-r^2/4)*R_1(r^2)
# equal SJ's -b^{-7} * sum phi^(6)(z/b) ?
#
# Our single-pair contribution (without the 1/n^2):
# (4pi)^{-1/2} * b^{-7} * exp(-r^2/4) * R_1(r^2)
#
# SJ's single-pair contribution:
# -b^{-7} * phi^(6)((Xi-Xj)/b) = -b^{-7} * (z^6 - 15z^4 + 45z^2 - 15) * (2pi)^{-1/2} * exp(-z^2/2)
# where z = (Xi-Xj)/b, so z^2 = r^2
#
# = -b^{-7} * (r^6 - 15r^4 + 45r^2 - 15) * (2pi)^{-1/2} * exp(-r^2/2)
#
# Our pair: (4pi)^{-1/2} * b^{-7} * exp(-r^2/4) * R_1(r^2)
#
# For these to be equal:
# (4pi)^{-1/2} * exp(-r^2/4) * R_1(r^2) = -(2pi)^{-1/2} * (r^6-15r^4+45r^2-15) * exp(-r^2/2)
#
# (4pi)^{-1/2} / (2pi)^{-1/2} = 1/sqrt(2)
#
# So: exp(-r^2/4) * R_1(r^2) / sqrt(2) = -(r^6-15r^4+45r^2-15) * exp(-r^2/2)
# R_1(r^2) = -sqrt(2) * (r^6-15r^4+45r^2-15) * exp(-r^2/4)
#
# Hmm that has an exp(-r^2/4) which doesn't work as a polynomial identity.
# The issue is the different Gaussian widths: our formula convolves K_b with K_b (width b*sqrt(2)),
# while SJ evaluates phi^(6) at (Xi-Xj)/b directly.
#
# SJ's estimator uses the kernel K evaluated at the PAIR DIFFERENCE divided by bandwidth:
# T_D(b) = -1/(n^2*b^7) * sum_ij phi^(6)((Xi-Xj)/b)
#
# This is NOT the integral of products. It's a direct evaluation. The connection:
# int K_b^(3)(x-a) * K_b^(3)(x-b) dx = [K_b * K_b]^(6)|_{delta} 
#                                       = K_{b*sqrt(2)}^(6)(delta)
# Because convolution of 3rd derivatives = 6th derivative of convolution.
#
# K_{b*sqrt(2)}^(6)(delta) = (b*sqrt(2))^{-7} * phi^(6)(delta/(b*sqrt(2)))
#
# But SJ evaluates phi^(6)(delta/b), NOT phi^(6)(delta/(b*sqrt(2))).
# This is because SJ's estimator uses a DIFFERENT normalization.
# Their S_D(alpha) estimates R(f'') by plugging in the KDE and using the identity:
# R(f'') = int f''(x) * f(x) dx  (integration by parts identity)
#
# This gives: R_hat(f'') = (1/n) sum_i f_hat''(X_i) = (1/n^2*alpha^5) sum_ij phi^(4)((Xi-Xj)/alpha)
# NOT the integral of (f_hat'')^2.
#
# So SJ's estimator is DIFFERENT from int (nabla^2 f_hat)^2.
# It's the "V-statistic" estimator, not the "square-integral" estimator.

print("\n" + "=" * 70)
print("KEY INSIGHT: SJ uses a V-STATISTIC estimator, not square-integral!")
print("=" * 70)
print()
print("SJ's estimator: S_D(alpha) = (1/n^2*alpha^5) * sum_ij phi^(4)((Xi-Xj)/alpha)")
print()
print("This estimates R(f'') via the identity R(f'') = int f''(x)*f(x) dx")
print("which leads to: R_hat = (1/n) * sum_i f_hat''(X_i)")
print("              = (1/n^2*alpha^5) * sum_ij phi^(4)((Xi-Xj)/alpha)")
print()
print("OUR estimator: Psi_4_hat = int (nabla^2 f_hat)^2 dx")
print("             = (1/n^2*(4pi)^{d/2}*h^{d+4}) * sum_ij exp(-r^2/4)*P_d(r^2)")
print()
print("In 1D, these are DIFFERENT:")
print("  SJ: sum_ij (2pi)^{-1/2} * exp(-z^2/2) * He_4(z) / alpha^5")
print("  Ours: sum_ij (4pi)^{-1/2} * exp(-r^2/4) * P_1(r^2) / h^5")
print("  where z = (Xi-Xj)/alpha and r^2 = (Xi-Xj)^2/h^2")
print()
print("The exp(-z^2/2) vs exp(-r^2/4) shows these use DIFFERENT Gaussian widths.")
print("  SJ: exp(-(Xi-Xj)^2 / (2*alpha^2))")
print("  Ours: exp(-(Xi-Xj)^2 / (4*h^2))")
print()
print("CONCLUSION: Our formula estimates R(nabla^2 f) via the INTEGRAL of squares")
print("(convolution of two kernels -> effective width sqrt(2)*h), while SJ estimates")
print("it via the V-statistic (single kernel evaluation -> width alpha).")
print()
print("BOTH are valid estimators of the same functional Psi_4 = R(f'').")
print("The difference is in the effective bandwidth relationship:")
print("  Our h corresponds to SJ's alpha/sqrt(2).")
print()

# Let's verify this relationship
print("Verification: is our formula = SJ's formula with alpha = h*sqrt(2)?")
print()

np.random.seed(42)
X_v = np.random.randn(50)
h_val = 0.6

# Our formula
n_v = len(X_v)
total_ours = 0.0
for i in range(n_v):
    for j in range(n_v):
        r_sq_v = (X_v[i] - X_v[j])**2 / h_val**2
        P1 = r_sq_v**2/16 - 3*r_sq_v/4 + 3/4
        total_ours += np.exp(-r_sq_v/4) * P1
psi4_ours = total_ours / (n_v**2 * np.sqrt(4*np.pi) * h_val**5)

# SJ's formula with alpha = h*sqrt(2)
alpha_sj = h_val * np.sqrt(2)
total_sj2 = 0.0
for i in range(n_v):
    for j in range(n_v):
        z = (X_v[i] - X_v[j]) / alpha_sj
        total_sj2 += (z**4 - 6*z**2 + 3) * norm.pdf(z)
psi4_sj = total_sj2 / (n_v**2 * alpha_sj**5)

# SJ's formula with alpha = h (direct)
total_sj3 = 0.0
for i in range(n_v):
    for j in range(n_v):
        z = (X_v[i] - X_v[j]) / h_val
        total_sj3 += (z**4 - 6*z**2 + 3) * norm.pdf(z)
psi4_sj_direct = total_sj3 / (n_v**2 * h_val**5)

print(f"  Our Psi4_hat(h={h_val}):           {psi4_ours:.10f}")
print(f"  SJ S_D(alpha=h*sqrt(2)={alpha_sj:.4f}): {psi4_sj:.10f}")
print(f"  SJ S_D(alpha=h={h_val}):           {psi4_sj_direct:.10f}")
print(f"  Ratio ours/SJ(h*sqrt2):     {psi4_ours/psi4_sj:.10f}")
print(f"  Ratio ours/SJ(h):           {psi4_ours/psi4_sj_direct:.10f}")

print("\n" + "=" * 70)
print("FINAL UNDERSTANDING")
print("=" * 70)
print()
print("Our estimator and SJ's estimator compute the SAME quantity (roughness R(f''))")
print("but through different routes:")
print("  - SJ: V-statistic using phi^(r) at pair differences / alpha")
print("  - Ours: U-statistic via integral of product kernels / h")
print()
print("The relationship: our formula at bandwidth h = SJ's formula at bandwidth h*sqrt(2)")
print("(because our formula arises from convolving two copies of K_h, giving effective width h*sqrt(2))")
print()
print("For the TWO-STAGE PILOT generalization to d-D:")
print("We can EITHER:")
print("  (a) Use OUR framework consistently (integral-of-products) for both Psi_4 and Psi_6")
print("  (b) Use SJ's V-statistic framework (evaluate kernel derivative at pairs)")
print()
print("Option (a) is what we should do — use our Q_d polynomial for Psi_6 estimation")
print("in the same integral-of-products framework. The bias structure will be the same")
print("(positive diagonal + negative smoothing), just with our bandwidth convention.")
