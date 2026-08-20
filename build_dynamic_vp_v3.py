"""
Dynamic Volume Profile v3:
- NO resampling: use raw bars directly, weighted by volume in the KDE
- Line opacity/thickness encodes peak prominence (relative to max density so far)
- Stronger levels = brighter/thicker; weaker levels = faint/thin
"""

import sys
sys.path.insert(0, 'gsj/src')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import find_peaks
from gsj import bandwidth
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from datetime import datetime, time
import pytz
import json
import warnings
warnings.filterwarnings('ignore')

EST = pytz.timezone('US/Eastern')


def get_data(ticker, period='5d'):
    df = yf.download(ticker, period=period, interval='1m', progress=False, prepost=True)
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)
    df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert(EST)
    else:
        df.index = df.index.tz_convert(EST)
    return df


def split_trading_days(df):
    days = []
    for date, group in df.groupby(df.index.date):
        mask = (group.index.time >= time(9, 30)) & (group.index.time <= time(16, 0))
        day_df = group[mask]
        if len(day_df) > 60:
            days.append((date, day_df))
    return days


def weighted_kde_eval(prices, volumes, h, x_grid):
    """
    Volume-weighted KDE: no resampling.
    f(x) = (1/V_total) * sum_i v_i * K_h(x - p_i)
    where V_total = sum(v_i)
    """
    n = len(prices)
    V_total = volumes.sum()
    if V_total == 0:
        return np.zeros_like(x_grid)
    
    # Vectorized computation
    diff = x_grid[:, None] - prices[None, :]  # (grid, n)
    kernels = np.exp(-diff**2 / (2*h**2)) / (np.sqrt(2*np.pi) * h)  # (grid, n)
    # Weight by volume
    kde_y = (kernels * volumes[None, :]).sum(axis=1) / V_total
    return kde_y


def detect_levels_weighted(prices, volumes, h, min_prominence=0.04):
    """
    Detect peaks in volume-weighted KDE.
    Returns (levels, prominences, kde_max) where prominences are normalized 0-1.
    """
    if len(prices) < 20:
        return np.array([]), np.array([]), 0.0
    
    x_grid = np.linspace(prices.min(), prices.max(), 500)
    kde_y = weighted_kde_eval(prices, volumes, h, x_grid)
    
    kde_max = kde_y.max()
    if kde_max == 0:
        return np.array([]), np.array([]), 0.0
    
    prominence_threshold = min_prominence * kde_max
    peaks, properties = find_peaks(kde_y, prominence=prominence_threshold, distance=10)
    
    if len(peaks) == 0:
        return np.array([]), np.array([]), kde_max
    
    levels = x_grid[peaks]
    # Normalize prominences: relative to the highest peak's density
    peak_heights = kde_y[peaks]
    prominences = peak_heights / kde_max  # 0 to 1
    
    return levels, prominences, kde_max


def silverman_bw_weighted(prices, volumes):
    """Silverman bandwidth for volume-weighted data."""
    # Effective sample size for weighted data
    V = volumes.sum()
    V2 = (volumes**2).sum()
    n_eff = V**2 / V2 if V2 > 0 else len(prices)
    
    # Weighted std
    w = volumes / V
    mu = np.sum(w * prices)
    sigma = np.sqrt(np.sum(w * (prices - mu)**2))
    
    # Weighted IQR approximation
    sorted_idx = np.argsort(prices)
    cum_w = np.cumsum(w[sorted_idx])
    q25_idx = np.searchsorted(cum_w, 0.25)
    q75_idx = np.searchsorted(cum_w, 0.75)
    iqr = prices[sorted_idx[min(q75_idx, len(prices)-1)]] - prices[sorted_idx[min(q25_idx, len(prices)-1)]]
    
    A = min(sigma, iqr / 1.349) if iqr > 0 else sigma
    if A <= 0:
        A = sigma
    return 0.9 * A * n_eff**(-1/5)


def gsj_bw_weighted(prices, volumes):
    """
    GSJ bandwidth using volume-weighted samples.
    We pass volume-weighted resampled data to bandwidth() since the
    roughness formula expects unweighted data.
    """
    # Create weighted sample for bandwidth estimation
    V = volumes.sum()
    if V == 0:
        return silverman_bw_weighted(prices, volumes)
    probs = volumes / V
    rng = np.random.default_rng(0)
    n_eff = int(min(2000, V**2 / (volumes**2).sum()))
    n_eff = max(n_eff, 100)
    indices = rng.choice(len(prices), size=n_eff, p=probs)
    return bandwidth(prices[indices])


def compute_rolling_levels_v3(prices, volumes, timestamps, window=60, step=4, method='gsj'):
    """
    Rolling weighted KDE with prominence info.
    Returns: [(timestamp, levels, prominences, running_max_density), ...]
    """
    n = len(prices)
    results = []
    running_max_density = 0.0
    
    for end in range(window, n, step):
        start = end - window
        p_w = prices[start:end]
        v_w = volumes[start:end]
        
        mask = (v_w > 0) & np.isfinite(p_w)
        if mask.sum() < 20:
            results.append((timestamps[end-1], np.array([]), np.array([]), running_max_density))
            continue
        
        p_valid = p_w[mask]
        v_valid = v_w[mask]
        
        # Bandwidth
        if method == 'gsj':
            h = gsj_bw_weighted(p_valid, v_valid)
        else:
            h = silverman_bw_weighted(p_valid, v_valid)
        
        # Detect levels with prominences
        levels, prominences, kde_max = detect_levels_weighted(p_valid, v_valid, h)
        
        # Update running max
        running_max_density = max(running_max_density, kde_max)
        
        # Normalize prominences relative to running max density
        if running_max_density > 0 and len(prominences) > 0:
            # Scale: prominence is already 0-1 relative to window max,
            # further scale by window_max / running_max to get global relative strength
            global_prominences = prominences * (kde_max / running_max_density)
        else:
            global_prominences = prominences
        
        results.append((timestamps[end-1], levels, global_prominences, running_max_density))
    
    return results


def plot_day_v3(ticker, date, day_df, window=60, step=4):
    """Plot with prominence-encoded level lines."""
    
    prices = day_df['TypicalPrice'].values
    volumes = day_df['Volume'].values
    close = day_df['Close'].values
    timestamps = day_df.index
    
    mask = np.isfinite(prices) & np.isfinite(volumes) & (volumes > 0)
    prices = prices[mask]
    volumes = volumes[mask]
    close = close[mask]
    timestamps = timestamps[mask]
    
    if len(prices) < window + 30:
        return None
    
    print(f"    {date}: {len(prices)} bars")
    
    levels_gsj = compute_rolling_levels_v3(prices, volumes, timestamps, window, step, 'gsj')
    levels_silv = compute_rolling_levels_v3(prices, volumes, timestamps, window, step, 'silverman')
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    date_str = date.strftime('%Y-%m-%d (%A)')
    fig.suptitle(f'{ticker} — {date_str}\n'
                 f'Line brightness = level strength (volume density relative to session max)',
                 fontsize=11, fontweight='bold')
    
    for ax, levels_data, title, cmap_name in [
        (axes[0], levels_silv, 'Silverman', 'Blues'),
        (axes[1], levels_gsj, 'GSJ (Sheather-Jones)', 'Reds')
    ]:
        # Price line
        ax.plot(timestamps, close, color='#1a1a1a', linewidth=0.8, alpha=0.95, zorder=10)
        
        cmap = plt.get_cmap(cmap_name)
        
        # Draw level segments with prominence-encoded appearance
        for i, (ts, levels, proms, _) in enumerate(levels_data):
            if len(levels) == 0:
                continue
            if i > 0:
                ts_start = levels_data[i-1][0]
            else:
                ts_start = ts
            
            for level, prom in zip(levels, proms):
                # prom is 0-1: 0=weakest, 1=strongest
                alpha = 0.15 + 0.7 * prom  # range: 0.15 to 0.85
                lw = 1.0 + 3.0 * prom      # range: 1.0 to 4.0
                color = cmap(0.3 + 0.6 * prom)  # lighter for weak, darker for strong
                
                ax.plot([ts_start, ts], [level, level],
                       color=color, linewidth=lw, alpha=alpha,
                       solid_capstyle='round', zorder=3)
        
        # Stats
        all_proms = [p for _, _, proms, _ in levels_data for p in proms]
        avg_levels = np.mean([len(l) for _, l, _, _ in levels_data if len(l) > 0]) if levels_data else 0
        
        ax.set_title(f'{title} (avg {avg_levels:.1f} levels/window)', fontsize=10)
        ax.set_xlabel('Time (EST)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=EST))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=EST))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.15)
        
        # Market open line
        open_dt = EST.localize(datetime.combine(date, time(9, 30)))
        if timestamps[0] <= open_dt <= timestamps[-1]:
            ax.axvline(x=open_dt, color='green', linewidth=0.8, alpha=0.4, linestyle=':')
    
    axes[0].set_ylabel('Price ($)')
    
    # Add a legend for prominence
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=plt.get_cmap('Reds')(0.9), lw=4, alpha=0.85, label='Strong level'),
        Line2D([0], [0], color=plt.get_cmap('Reds')(0.4), lw=1.5, alpha=0.3, label='Weak level'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    fname = f'vp3_{ticker}_{date.strftime("%Y%m%d")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    return fname


# =============================================================================
# MAIN
# =============================================================================

tickers = ['NVDA', 'TSLA', 'SPY']

print("=" * 70)
print("VOLUME PROFILE v3: No Resampling + Prominence-Encoded Levels")
print("=" * 70)
print("- Direct volume-weighted KDE (no resampling)")
print("- Line brightness/thickness = peak prominence (relative to max density)")
print()

all_files = []

for ticker in tickers:
    print(f"\n{ticker}:")
    df = get_data(ticker)
    if df.empty:
        continue
    days = split_trading_days(df)
    for date, day_df in days[-3:]:
        f = plot_day_v3(ticker, date, day_df, window=60, step=4)
        if f:
            all_files.append(f)

# Notebook
cells = []
def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [s]}

cells.append(md("""# Volume Profile v3: Prominence-Encoded Dynamic Levels

## Changes from v2:
- **No resampling** — uses raw price bars directly, weighted by volume in the KDE
- **Line brightness/thickness encodes level strength** — stronger levels (higher density peak relative to session max) are brighter and thicker; weak levels are faint and thin
- This mimics what a real-time volume profile display would show: prominent nodes stand out visually

## Reading the chart:
- **Thick/bright lines** = High Volume Nodes (HVN) — strong support/resistance where significant volume traded
- **Thin/faint lines** = weaker levels that may or may not hold
- Price line in black; level strength accumulates through the session as more data arrives

## Key difference between methods:
- **Silverman** (left): fewer, broader levels. Even the "strong" ones are imprecise — they cover a wide price band
- **GSJ** (right): more levels, each more precisely located. You can distinguish between a strong single level and two nearby weak levels that Silverman would merge
"""))

for f in all_files:
    ticker = f.split('_')[1]
    date = f.split('_')[2].replace('.png', '')
    cells.append(md(f"### {ticker} — {date[:4]}-{date[4:6]}-{date[6:]}\n![{f}]({f})"))

cells.append(md("""## Trading Implications

| What you see | Silverman interpretation | GSJ interpretation |
|-------------|------------------------|-------------------|
| Thick bright band | "General support zone" | "Exact level at $X.XX" |
| Two thin nearby lines | Not visible (merged) | "Two levels $0.50 apart — trade the bounce between them" |
| Fading level | Still shows (too smooth) | Disappears → level is broken, don't trade it |
| New level forming | Slow to appear (high inertia) | Quick to detect (adapts to new clustering) |

The data-driven bandwidth means GSJ **adapts to the current market regime**:
- Tight range → smaller h → more levels (correct: price is bouncing between close levels)
- Wide trend → larger h → fewer levels (correct: no meaningful S/R within the trend)
"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells
}
with open('volume_profile_v3.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\nSaved volume_profile_v3.ipynb ({len(all_files)} images)")
print("Done!")
