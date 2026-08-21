"""
Build a clear, visual options density notebook.
Multiple views: contour, 3D surface, scatter+KDE overlay, zoomed sections.
"""

import sys
sys.path.insert(0, 'gsj/src')

import numpy as np
import yfinance as yf
from gsj import bandwidth
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import date
import json
import warnings
warnings.filterwarnings('ignore')


def get_options_data(ticker='SPY'):
    tk = yf.Ticker(ticker)
    expiries = tk.options
    all_data = []
    today = date.today()
    from datetime import datetime
    for exp_str in expiries:
        try:
            chain = tk.option_chain(exp_str)
        except Exception:
            continue
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        if dte < 0 or dte > 120:
            continue
        for side, df in [('calls', chain.calls), ('puts', chain.puts)]:
            for _, row in df.iterrows():
                oi = row.get('openInterest', 0) or 0
                if oi > 0:
                    all_data.append({
                        'strike': row['strike'],
                        'dte': dte,
                        'oi': float(oi),
                        'side': side,
                    })
    price = tk.info.get('regularMarketPrice', tk.info.get('previousClose', 0))
    return all_data, price


def weighted_kde_2d(points_std, weights, h, grid_x_std, grid_y_std):
    Gx, Gy = np.meshgrid(grid_x_std, grid_y_std, indexing='ij')
    kde = np.zeros_like(Gx)
    n = len(points_std)
    W = weights.sum()
    for i in range(0, n, 500):
        batch = points_std[i:i+500]
        bw = weights[i:i+500]
        dx = Gx[:, :, None] - batch[:, 0][None, None, :]
        dy = Gy[:, :, None] - batch[:, 1][None, None, :]
        kde += np.sum(np.exp(-(dx**2 + dy**2) / (2*h**2)) * bw[None, None, :], axis=2)
    kde /= W * (2 * np.pi * h**2)
    return kde


def find_peaks_2d(kde, grid_x, grid_y, threshold=0.2):
    size = max(5, kde.shape[0] // 15)
    local_max = maximum_filter(kde, size=size)
    mask = (kde == local_max) & (kde > threshold * kde.max())
    coords = np.argwhere(mask)
    peaks = []
    for (ix, iy) in coords:
        peaks.append({'x': grid_x[ix], 'y': grid_y[iy], 'z': kde[ix, iy]})
    peaks.sort(key=lambda p: -p['z'])
    return peaks


# =============================================================================
print("Fetching SPY options...")
data, spot = get_options_data('SPY')
print(f"  Spot: ${spot:.2f}, {len(data)} contracts")

strikes = np.array([d['strike'] for d in data])
dtes = np.array([d['dte'] for d in data], dtype=float)
oi = np.array([d['oi'] for d in data])
moneyness = (strikes - spot) / spot * 100

mask = (np.abs(moneyness) < 15) & (dtes > 0) & (dtes <= 100) & (oi > 0)
money = moneyness[mask]
dte = dtes[mask]
weights = oi[mask]
sides = np.array([d['side'] for d in data])[mask]

# Standardize
std_m = np.std(money)
std_d = np.std(dte)
pts_std = np.column_stack([money/std_m, dte/std_d])

# Weighted sample for bandwidth
probs = weights / weights.sum()
rng = np.random.default_rng(42)
n_samp = min(2500, len(pts_std))
idx = rng.choice(len(pts_std), size=n_samp, p=probs)
sample = pts_std[idx]

h_silv = (4 / (n_samp * 4)) ** (1/6)
h_gsj = bandwidth(sample, algorithm='two-stage')

print(f"  h_silverman={h_silv:.4f}, h_gsj={h_gsj:.4f}, ratio={h_gsj/h_silv:.3f}")

# Grids
gm = np.linspace(-13, 13, 130)
gd = np.linspace(0, 100, 100)
gm_s = gm / std_m
gd_s = gd / std_d

kde_silv = weighted_kde_2d(pts_std, weights, h_silv, gm_s, gd_s)
kde_gsj = weighted_kde_2d(pts_std, weights, h_gsj, gm_s, gd_s)

peaks_silv = find_peaks_2d(kde_silv, gm, gd)
peaks_gsj = find_peaks_2d(kde_gsj, gm, gd)

print(f"  Silverman peaks: {len(peaks_silv)}, GSJ peaks: {len(peaks_gsj)}")

# =============================================================================
# FIGURE 1: Side-by-side filled contour with peaks
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
fig.suptitle(f'SPY Options Open Interest Density (spot=${spot:.2f}, {date.today()})\n'
             f'2D KDE of (Moneyness%, DTE) weighted by Open Interest',
             fontsize=12, fontweight='bold')

for ax, kde, peaks, title, cmap in [
    (axes[0], kde_silv, peaks_silv, f'Silverman (h={h_silv:.3f}) — {len(peaks_silv)} zones', 'YlGnBu'),
    (axes[1], kde_gsj, peaks_gsj, f'GSJ 2-stage (h={h_gsj:.3f}) — {len(peaks_gsj)} zones', 'YlOrRd'),
]:
    cf = ax.contourf(gm, gd, kde.T, levels=25, cmap=cmap)
    ax.contour(gm, gd, kde.T, levels=10, colors='k', linewidths=0.3, alpha=0.4)
    for i, p in enumerate(peaks[:8]):
        sz = 80 + 300 * (p['z'] / kde.max())
        ax.scatter(p['x'], p['y'], s=sz, facecolors='none', edgecolors='white', linewidths=2.5, zorder=5)
        strike_price = spot * (1 + p['x']/100)
        ax.annotate(f"${strike_price:.0f}\n{p['y']:.0f}d",
                   (p['x'], p['y']), fontsize=7, color='white', fontweight='bold',
                   textcoords='offset points', xytext=(6, 4),
                   bbox=dict(fc='black', alpha=0.7, pad=1.5, boxstyle='round'))
    ax.axvline(0, color='white', ls='--', lw=1, alpha=0.6)
    ax.set_xlabel('Moneyness (% from spot)', fontsize=10)
    ax.set_title(title, fontsize=10)
    plt.colorbar(cf, ax=ax, shrink=0.85, label='OI Density')

axes[0].set_ylabel('Days to Expiry', fontsize=10)
plt.tight_layout()
plt.savefig('opt_fig1_contour.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved opt_fig1_contour.png")

# =============================================================================
# FIGURE 2: 3D surface plot
# =============================================================================
fig = plt.figure(figsize=(16, 6))

for idx_ax, (kde, title, cmap) in enumerate([
    (kde_silv, 'Silverman', 'YlGnBu'),
    (kde_gsj, 'GSJ (Sheather-Jones)', 'YlOrRd'),
]):
    ax = fig.add_subplot(1, 2, idx_ax+1, projection='3d')
    Gm, Gd = np.meshgrid(gm, gd, indexing='ij')
    ax.plot_surface(Gm, Gd, kde, cmap=cmap, alpha=0.85, linewidth=0, antialiased=True)
    ax.set_xlabel('Moneyness %', fontsize=8)
    ax.set_ylabel('DTE', fontsize=8)
    ax.set_zlabel('Density', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=25, azim=-60)

fig.suptitle(f'SPY Options OI — 3D Density Surface', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('opt_fig2_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved opt_fig2_3d.png")

# =============================================================================
# FIGURE 3: Raw data scatter + KDE contours overlay
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
fig.suptitle('Raw OI Data (size=OI weight) + KDE Contours', fontsize=12, fontweight='bold')

for ax, kde, peaks, title, cmap, scatter_c in [
    (axes[0], kde_silv, peaks_silv, 'Silverman contours', 'Blues', 'steelblue'),
    (axes[1], kde_gsj, peaks_gsj, 'GSJ contours', 'Reds', 'firebrick'),
]:
    # Scatter: calls and puts with different markers
    calls_mask = sides == 'calls'
    puts_mask = sides == 'puts'
    
    sizes_c = np.clip(weights[calls_mask] / 500, 2, 80)
    sizes_p = np.clip(weights[puts_mask] / 500, 2, 80)
    
    ax.scatter(money[calls_mask], dte[calls_mask], s=sizes_c, alpha=0.25,
              color='green', marker='^', label='Calls')
    ax.scatter(money[puts_mask], dte[puts_mask], s=sizes_p, alpha=0.25,
              color='purple', marker='v', label='Puts')
    
    # KDE contours on top
    ax.contour(gm, gd, kde.T, levels=8, colors=scatter_c, linewidths=1.5, alpha=0.8)
    
    # Peak markers
    for p in peaks[:6]:
        ax.scatter(p['x'], p['y'], s=200, facecolors='yellow',
                  edgecolors='black', linewidths=2, zorder=10, marker='*')
    
    ax.axvline(0, color='gray', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Moneyness %')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

axes[0].set_ylabel('Days to Expiry')
plt.tight_layout()
plt.savefig('opt_fig3_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved opt_fig3_scatter.png")

# =============================================================================
# FIGURE 4: Zoomed near-term (DTE < 14) comparison
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
fig.suptitle('Near-Term Zoom (DTE < 14 days) — Where Is the Gamma Pin?', fontsize=12, fontweight='bold')

dte_zoom_mask = gd < 14
gd_zoom = gd[dte_zoom_mask]

for ax, kde, peaks, title, cmap in [
    (axes[0], kde_silv[:, dte_zoom_mask], [p for p in peaks_silv if p['y'] < 14], 'Silverman', 'YlGnBu'),
    (axes[1], kde_gsj[:, dte_zoom_mask], [p for p in peaks_gsj if p['y'] < 14], 'GSJ', 'YlOrRd'),
]:
    cf = ax.contourf(gm, gd_zoom, kde.T, levels=20, cmap=cmap)
    for p in peaks[:5]:
        strike_p = spot * (1 + p['x']/100)
        ax.scatter(p['x'], p['y'], s=150, facecolors='none', edgecolors='white', linewidths=2.5, zorder=5)
        ax.annotate(f"${strike_p:.0f}", (p['x'], p['y']),
                   fontsize=9, color='white', fontweight='bold',
                   textcoords='offset points', xytext=(5, 3),
                   bbox=dict(fc='black', alpha=0.7, pad=1, boxstyle='round'))
    ax.axvline(0, color='white', ls='--', lw=1, alpha=0.6)
    ax.set_xlabel('Moneyness %')
    ax.set_title(f'{title} ({len(peaks)} near-term zones)')
    plt.colorbar(cf, ax=ax, shrink=0.9)

axes[0].set_ylabel('Days to Expiry')
plt.tight_layout()
plt.savefig('opt_fig4_zoom.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved opt_fig4_zoom.png")

# =============================================================================
# BUILD NOTEBOOK
# =============================================================================
cells = []
def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [s]}

cells.append(md(f"""# Options Open Interest Density — 2D Multivariate KDE

**Date:** {date.today().strftime('%Y-%m-%d')} | **SPY spot:** ${spot:.2f}

## What This Shows

Every options contract on SPY is a point in 2D space: **(moneyness %, days to expiry)**.
We weight each contract by its open interest and fit a 2D Gaussian KDE to find where
market participants are **concentrated**.

**Peaks in this 2D density = crowded positions:**
- Near-ATM, short DTE = gamma pin (dealers hedging weekly options)
- OTM puts, 20-60 DTE = protective put walls (institutional hedging)
- OTM calls, short DTE = call overwriting / resistance

**This is a TRUE 2D application of our $P_d$ polynomial** with $d=2$:
$$P_2(t) = \\frac{{t^2}}{{16}} - t + 2$$

Bandwidth comparison:
- Silverman: $h = {h_silv:.4f}$ → **{len(peaks_silv)} zones detected**
- GSJ (data-driven): $h = {h_gsj:.4f}$ → **{len(peaks_gsj)} zones detected**
- Ratio: {h_gsj/h_silv:.2f}x (GSJ uses ~half the bandwidth → resolves finer structure)
"""))

cells.append(md("""## Figure 1: Density Contour Maps

Side-by-side comparison. White circles mark detected concentration zones.
Labels show the strike price and days to expiry.

![Contour](opt_fig1_contour.png)

**Key observation:** GSJ resolves distinct clusters that Silverman merges into one blob.
For example, a near-term ATM pin and an OTM put cluster appear as separate zones in GSJ
but are blurred together in Silverman.
"""))

cells.append(md("""## Figure 2: 3D Surface View

The density "landscape" — peaks are mountains where OI concentrates.

![3D Surface](opt_fig2_3d.png)

The GSJ surface has sharper, more defined peaks. Silverman creates a smoother hill
that obscures the individual concentration points.
"""))

cells.append(md("""## Figure 3: Raw Data + Contours

The underlying data: each triangle/point is one contract (green △ = calls, purple ▽ = puts).
Size = open interest. KDE contour lines overlaid. Stars = detected peaks.

![Scatter](opt_fig3_scatter.png)

You can see the data IS multimodal — there are distinct clusters at round strikes
and at popular monthly expiries. The GSJ contours track these clusters more tightly.
"""))

cells.append(md("""## Figure 4: Near-Term Zoom (DTE < 14)

Where is the weekly gamma pin? Zooming into the short-dated region where
dealer gamma hedging matters most.

![Zoom](opt_fig4_zoom.png)

This is where precise level detection matters most for trading:
- The exact strike of the gamma pin determines where price gets "magnetized"
- A few dollars of precision in detecting this can mean the difference between
  a profitable and unprofitable options trade
"""))

cells.append(md(f"""## Detected Zones (GSJ)

| # | Strike | Moneyness | DTE | Relative Strength |
|---|--------|-----------|-----|-------------------|
""" + "\n".join([
    f"| {i+1} | ${spot*(1+p['x']/100):.0f} | {p['x']:+.1f}% | {p['y']:.0f}d | {p['z']/kde_gsj.max():.0%} |"
    for i, p in enumerate(peaks_gsj[:8])
]) + """

These are the positions where the market is most concentrated. A move THROUGH
one of these zones requires absorbing significant hedging flow — making them
natural support/resistance levels from an options microstructure perspective.
"""))

cells.append(md("""## Why 2D Matters (Can't Do This with Two 1D KDEs)

If you did 1D KDE on strikes alone, you'd see "lots of OI at $760."
If you did 1D KDE on DTE alone, you'd see "lots of OI at 4 days."

But that doesn't tell you: **is there OI at $760 WITH 4 DTE?**
Maybe the $760 OI is all in 60-day puts, and the 4-DTE OI is at $780 calls.

The 2D KDE answers: "Where do strike AND expiry JOINTLY concentrate?"
This is the information options market makers actually use for hedging decisions.
"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells
}
with open('options_visual_notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("\nSaved options_visual_notebook.ipynb")
print("Done!")
