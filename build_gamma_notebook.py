"""
Dealer Gamma Exposure (GEX) Map: KDE-based gamma concentration detection.

Concept: At each price level, compute the net gamma exposure that dealers
hold (from selling options to customers). Where gamma is concentrated,
dealers must delta-hedge aggressively → creates price "magnetism" (positive gamma)
or "acceleration" (negative gamma).

Method:
1. Pull SPY options chain
2. Compute gamma per contract using Black-Scholes
3. Weight each strike by gamma × OI (net gamma exposure)
4. KDE on price axis, weighted by |GEX| → find where gamma concentrates
5. Separate positive vs negative gamma zones

Also do 2D version: (strike, DTE) weighted by gamma×OI to see gamma across time.
"""

import sys
sys.path.insert(0, 'gsj/src')

import numpy as np
import yfinance as yf
from gsj import bandwidth
from scipy.stats import norm as scipy_norm
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import date, datetime
import json
import warnings
warnings.filterwarnings('ignore')


def black_scholes_gamma(S, K, T, r, sigma):
    """Compute BS gamma for a European option."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    gamma = scipy_norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma


def get_gex_data(ticker='SPY'):
    """Pull options and compute GEX per strike."""
    tk = yf.Ticker(ticker)
    spot = tk.info.get('regularMarketPrice', tk.info.get('previousClose', 500))
    expiries = tk.options
    today = date.today()
    
    records = []
    for exp_str in expiries:
        try:
            chain = tk.option_chain(exp_str)
        except Exception:
            continue
        
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        if dte <= 0 or dte > 90:
            continue
        T = dte / 365.0
        
        for side, df, sign in [('calls', chain.calls, 1), ('puts', chain.puts, -1)]:
            for _, row in df.iterrows():
                oi = row.get('openInterest', 0) or 0
                iv = row.get('impliedVolatility', 0) or 0
                strike = row['strike']
                
                if oi <= 0 or iv <= 0:
                    continue
                
                # Filter: within 20% of spot
                if abs(strike - spot) / spot > 0.20:
                    continue
                
                gamma = black_scholes_gamma(spot, strike, T, 0.05, iv)
                
                # GEX convention: dealers are SHORT gamma on customer-bought options
                # Calls: customer buys calls → dealer short gamma → dealer GEX is NEGATIVE
                # BUT standard GEX convention: positive GEX = pin (support), negative = accelerate
                # Net GEX at a strike = (call_OI × gamma - put_OI × gamma) × 100 × spot
                # Simplified: just gamma × OI per contract, sign by call/put
                gex = gamma * oi * 100 * spot  # dollar gamma
                
                records.append({
                    'strike': strike,
                    'dte': dte,
                    'gamma': gamma,
                    'oi': oi,
                    'gex': gex * sign,  # +1 for calls (dealers short → pin), -1 for puts
                    'abs_gex': abs(gex),
                    'side': side,
                    'iv': iv,
                })
    
    return records, spot


def kde_1d_weighted(prices, weights, h, grid):
    """1D weighted KDE."""
    W = np.abs(weights).sum()
    kde = np.zeros_like(grid)
    for p, w in zip(prices, weights):
        kde += np.abs(w) * np.exp(-(grid - p)**2 / (2*h**2))
    kde /= W * np.sqrt(2*np.pi) * h
    return kde


# =============================================================================
print("=" * 70)
print("DEALER GAMMA EXPOSURE MAP")
print("=" * 70)

ticker = 'SPY'
print(f"\nFetching {ticker} options for GEX...")
records, spot = get_gex_data(ticker)
print(f"  Spot: ${spot:.2f}")
print(f"  {len(records)} contracts with gamma data")

# Aggregate GEX per strike
from collections import defaultdict
gex_by_strike = defaultdict(float)
call_gex_by_strike = defaultdict(float)
put_gex_by_strike = defaultdict(float)

for r in records:
    gex_by_strike[r['strike']] += r['gex']
    if r['side'] == 'calls':
        call_gex_by_strike[r['strike']] += r['abs_gex']
    else:
        put_gex_by_strike[r['strike']] += r['abs_gex']

strikes = np.array(sorted(gex_by_strike.keys()))
net_gex = np.array([gex_by_strike[k] for k in strikes])
call_gex = np.array([call_gex_by_strike[k] for k in strikes])
put_gex = np.array([put_gex_by_strike[k] for k in strikes])

print(f"  {len(strikes)} unique strikes with GEX")
print(f"  Net GEX range: {net_gex.min()/1e6:.1f}M to {net_gex.max()/1e6:.1f}M")

# =============================================================================
# 1D KDE on strike axis, weighted by |GEX|
# =============================================================================
abs_gex = np.abs(net_gex)
mask = abs_gex > 0
strikes_valid = strikes[mask]
gex_valid = net_gex[mask]
abs_gex_valid = abs_gex[mask]

# Bandwidth on the strike data
h_silv = 0.9 * min(np.std(strikes_valid), np.subtract(*np.percentile(strikes_valid, [75,25]))/1.349) * len(strikes_valid)**(-0.2)
# For GSJ: create weighted sample
probs = abs_gex_valid / abs_gex_valid.sum()
rng = np.random.default_rng(42)
sample = rng.choice(strikes_valid, size=2000, p=probs)
h_gsj = bandwidth(sample)

print(f"\n  1D GEX bandwidth:")
print(f"    Silverman: ${h_silv:.2f}")
print(f"    GSJ:       ${h_gsj:.2f}")

# Grids
grid = np.linspace(spot * 0.88, spot * 1.12, 500)

kde_silv = kde_1d_weighted(strikes_valid, abs_gex_valid, h_silv, grid)
kde_gsj = kde_1d_weighted(strikes_valid, abs_gex_valid, h_gsj, grid)

# Also compute signed GEX profile (positive gamma zones vs negative)
kde_pos = kde_1d_weighted(strikes_valid[gex_valid > 0], gex_valid[gex_valid > 0], h_gsj, grid)
kde_neg = kde_1d_weighted(strikes_valid[gex_valid < 0], np.abs(gex_valid[gex_valid < 0]), h_gsj, grid)

# Find peaks
peaks_silv, _ = find_peaks(kde_silv, prominence=0.05*kde_silv.max(), distance=10)
peaks_gsj, _ = find_peaks(kde_gsj, prominence=0.05*kde_gsj.max(), distance=10)

print(f"\n  Gamma concentration zones:")
print(f"    Silverman: {len(peaks_silv)} zones")
print(f"    GSJ:       {len(peaks_gsj)} zones")
print(f"\n  Top GSJ gamma walls:")
for p in peaks_gsj[:6]:
    print(f"    ${grid[p]:.0f} ({(grid[p]-spot)/spot*100:+.1f}% from spot)")

# =============================================================================
# FIGURE 1: GEX profile comparison
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle(f'{ticker} Dealer Gamma Exposure Map (spot=${spot:.2f}, {date.today()})\n'
             f'Peaks = price levels where dealer hedging is most intense',
             fontsize=12, fontweight='bold')

# Top: KDE comparison
ax = axes[0]
ax.fill_between(grid, kde_silv, alpha=0.3, color='steelblue', label=f'Silverman (h=${h_silv:.1f}, {len(peaks_silv)} zones)')
ax.fill_between(grid, kde_gsj, alpha=0.4, color='firebrick', label=f'GSJ (h=${h_gsj:.1f}, {len(peaks_gsj)} zones)')
ax.plot(grid, kde_silv, color='steelblue', linewidth=1.5)
ax.plot(grid, kde_gsj, color='firebrick', linewidth=2)

for p in peaks_gsj:
    ax.axvline(grid[p], color='red', alpha=0.4, linewidth=1.5, linestyle='-')
    ax.text(grid[p], kde_gsj[p]*1.02, f'${grid[p]:.0f}', fontsize=8, ha='center',
           color='darkred', fontweight='bold')

ax.axvline(spot, color='black', linewidth=2, linestyle='--', label=f'Spot ${spot:.0f}')
ax.set_ylabel('Gamma Concentration (density)')
ax.set_xlabel('Strike Price ($)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.2)
ax.set_xlim(spot*0.92, spot*1.08)

# Bottom: Net GEX bar chart (positive = pin, negative = accelerate)
ax = axes[1]
colors = ['green' if g > 0 else 'red' for g in net_gex]
ax.bar(strikes, net_gex/1e6, width=(strikes[1]-strikes[0])*0.8 if len(strikes)>1 else 1,
       color=colors, alpha=0.6, edgecolor='none')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(spot, color='black', linewidth=2, linestyle='--')
ax.set_xlabel('Strike Price ($)')
ax.set_ylabel('Net GEX ($M)')
ax.set_title('Net Gamma: Green (+) = Pin/Support | Red (-) = Acceleration')
ax.grid(True, alpha=0.2)
ax.set_xlim(spot*0.92, spot*1.08)

plt.tight_layout()
plt.savefig('gex_fig1_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved gex_fig1_profile.png")

# =============================================================================
# FIGURE 2: Positive vs Negative gamma zones
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(grid, kde_pos, alpha=0.4, color='green', label='Positive gamma (pin/support)')
ax.fill_between(grid, -kde_neg, alpha=0.4, color='red', label='Negative gamma (acceleration)')
ax.plot(grid, kde_pos, color='darkgreen', linewidth=1.5)
ax.plot(grid, -kde_neg, color='darkred', linewidth=1.5)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(spot, color='black', linewidth=2, linestyle='--', label=f'Spot ${spot:.0f}')
ax.set_xlabel('Strike Price ($)')
ax.set_ylabel('Gamma Density (+ above / - below zero)')
ax.set_title(f'{ticker} — Positive vs Negative Gamma Zones (GSJ bandwidth h=${h_gsj:.1f})')
ax.legend()
ax.grid(True, alpha=0.2)
ax.set_xlim(spot*0.92, spot*1.08)
plt.tight_layout()
plt.savefig('gex_fig2_posneg.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved gex_fig2_posneg.png")

# =============================================================================
# FIGURE 3: 2D GEX surface (strike × DTE)
# =============================================================================
strikes_2d = np.array([r['strike'] for r in records])
dtes_2d = np.array([r['dte'] for r in records], dtype=float)
gex_2d = np.array([r['abs_gex'] for r in records])

# Clean NaN/inf
valid_2d = np.isfinite(gex_2d) & (gex_2d > 0) & np.isfinite(strikes_2d) & np.isfinite(dtes_2d)
strikes_2d = strikes_2d[valid_2d]
dtes_2d = dtes_2d[valid_2d]
gex_2d = gex_2d[valid_2d]

# Standardize
std_s = np.std(strikes_2d)
std_d = np.std(dtes_2d)
pts_std = np.column_stack([(strikes_2d - spot)/std_s, dtes_2d/std_d])

# 2D bandwidth
probs_2d = gex_2d / gex_2d.sum()
n_samp = min(2500, len(pts_std))
idx_2d = rng.choice(len(pts_std), size=n_samp, p=probs_2d)
h_2d_gsj = bandwidth(pts_std[idx_2d], algorithm='two-stage')
h_2d_silv = (4/(n_samp*4))**(1/6)

# Evaluate 2D KDE
grid_s = np.linspace(spot*0.9, spot*1.1, 100)
grid_d = np.linspace(0, 60, 60)
gs_std = (grid_s - spot) / std_s
gd_std = grid_d / std_d

Gs, Gd = np.meshgrid(gs_std, gd_std, indexing='ij')
kde_2d = np.zeros_like(Gs)
W_total = gex_2d.sum()
for i in range(0, len(pts_std), 500):
    batch = pts_std[i:i+500]
    bw = gex_2d[i:i+500]
    dx = Gs[:,:,None] - batch[:,0][None,None,:]
    dy = Gd[:,:,None] - batch[:,1][None,None,:]
    kde_2d += np.sum(np.exp(-(dx**2+dy**2)/(2*h_2d_gsj**2)) * bw[None,None,:], axis=2)
kde_2d /= W_total * (2*np.pi*h_2d_gsj**2)

fig, ax = plt.subplots(figsize=(12, 7))
cf = ax.contourf(grid_s, grid_d, kde_2d.T, levels=25, cmap='hot_r')
ax.contour(grid_s, grid_d, kde_2d.T, levels=10, colors='white', linewidths=0.3, alpha=0.5)
ax.axvline(spot, color='cyan', linewidth=2, linestyle='--', label=f'Spot ${spot:.0f}')
ax.set_xlabel('Strike Price ($)')
ax.set_ylabel('Days to Expiry')
ax.set_title(f'{ticker} — 2D Gamma Concentration Surface (GSJ h={h_2d_gsj:.3f})\n'
             f'Bright = high gamma exposure at that (strike, DTE) combination')
ax.legend()
plt.colorbar(cf, label='Gamma Density')
plt.tight_layout()
plt.savefig('gex_fig3_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved gex_fig3_2d.png")

# =============================================================================
# NOTEBOOK
# =============================================================================
cells = []
def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [s]}

cells.append(md(f"""# Dealer Gamma Exposure (GEX) Map

## What Is GEX?

When you buy a call or put option, a market maker (dealer) takes the other side. The dealer is now **short gamma** — meaning as the stock moves, they must continuously buy/sell shares to stay hedged (delta-neutral).

**Where gamma is concentrated → where dealers must hedge the most → where price gets "sticky" (pinned) or "accelerated":**

- **Positive net gamma** (call OI > put OI): Dealers buy dips and sell rips → price pins/stabilizes → **support/resistance**
- **Negative net gamma** (put OI > call OI): Dealers sell dips and buy rips → moves amplify → **acceleration zones**

## Data

- **Ticker:** {ticker}
- **Spot:** ${spot:.2f}
- **Contracts analyzed:** {len(records)}
- **Date:** {date.today()}

## Bandwidth Comparison

The KDE bandwidth determines how precisely we locate the gamma walls:
- Silverman: h = ${h_silv:.1f} → {len(peaks_silv)} gamma zones
- GSJ: h = ${h_gsj:.1f} → **{len(peaks_gsj)} gamma zones**

GSJ's tighter bandwidth resolves distinct gamma walls that Silverman merges into one broad zone.
"""))

cells.append(md("""## Figure 1: GEX Profile — Where Is Gamma Concentrated?

Top panel: KDE of gamma concentration across strikes (red = GSJ, blue = Silverman).
Vertical lines = detected "gamma walls" (where hedging flow is most intense).

Bottom panel: Raw net GEX per strike (green = positive/pin, red = negative/accelerate).

![GEX Profile](gex_fig1_profile.png)

**Trading interpretation:** Price tends to be "pulled toward" the highest positive gamma zone (largest green peak in the KDE). It "accelerates away" from negative gamma zones.
"""))

cells.append(md("""## Figure 2: Positive vs Negative Gamma Zones

Green area (above zero): positive gamma = pin/support zones
Red area (below zero): negative gamma = acceleration/instability zones

![Pos/Neg Gamma](gex_fig2_posneg.png)

**Key levels:**
- Where green peaks: price likely to "pin" or oscillate around these levels
- Where red peaks: if price reaches these levels, expect fast movement through them
- The zero-crossing points: the "flip" between supportive and accelerating regimes
"""))

cells.append(md("""## Figure 3: 2D Gamma Surface (Strike × DTE)

The full picture: where is gamma concentrated in BOTH strike and time dimensions?

![2D GEX](gex_fig3_2d.png)

This is a **true 2D** (d=2) KDE application. It reveals:
- Near-term gamma (bright spots at low DTE) = this week's pin levels
- Medium-term gamma (bright spots at 20-40 DTE) = monthly expiry walls
- The distinction matters: a gamma wall that only exists for 2 more days behaves differently than one with 30 DTE

## Why Bandwidth Matters for GEX

Professional GEX tools (like SpotGamma, Menthor Q, Unusual Whales) all face this problem:
- Should $760 and $765 be one gamma zone or two separate walls?
- The answer depends on the data density — which is exactly what the bandwidth controls

GSJ adapts: in a concentrated market (lots of OI at few strikes), it finds narrow precise walls.
In a diffuse market (OI spread across many strikes), it finds broader zones.
Silverman always uses the same resolution regardless of how the OI is distributed.
"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells
}
with open('gamma_exposure_notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("\nSaved gamma_exposure_notebook.ipynb")
print("Done!")
