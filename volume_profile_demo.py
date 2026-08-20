"""
Volume Profile / Market Profile Demo
=====================================

Use yfinance 1-min bars to build volume profiles for in-play stocks.
Compare bandwidth selection methods on their ability to identify
meaningful support/resistance levels (modes of the volume-weighted KDE).

Stocks chosen: high-volume, recently active names likely to have 
clear intraday structure.
"""

import sys
sys.path.insert(0, 'gsj/src')

import numpy as np
import yfinance as yf
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from gsj import bandwidth
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA COLLECTION
# =============================================================================

def get_volume_profile_data(ticker, period='5d', interval='1m'):
    """
    Get 1-min OHLCV data and create volume-weighted price samples.
    Each bar's VWAP (approx: (H+L+C)/3) repeated by volume gives
    a proxy for tick-level volume profile.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return None, None
    
    # Flatten multi-level columns if present
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)
    
    # Typical price as VWAP proxy
    df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # For volume profile: create weighted samples
    # Each bar contributes its typical price, weighted by volume
    prices = df['TypicalPrice'].values
    volumes = df['Volume'].values
    
    # Remove NaN/zero
    mask = np.isfinite(prices) & np.isfinite(volumes) & (volumes > 0)
    prices = prices[mask]
    volumes = volumes[mask]
    
    return prices, volumes


def build_volume_profile(prices, volumes, n_samples=5000):
    """
    Create a volume-weighted sample for KDE.
    Sample prices proportional to their volume weight.
    """
    # Normalize volumes to probabilities
    probs = volumes / volumes.sum()
    
    # Resample prices according to volume weights
    rng = np.random.default_rng(42)
    indices = rng.choice(len(prices), size=n_samples, p=probs)
    weighted_prices = prices[indices]
    
    return weighted_prices


# =============================================================================
# KDE + LEVEL DETECTION
# =============================================================================

def detect_levels(data, h, min_prominence=0.05):
    """
    Fit Gaussian KDE at bandwidth h, find modes (peaks) and antimodes (valleys).
    Returns (levels, kde_x, kde_y) where levels are the peak prices.
    """
    # Evaluate KDE on a fine grid
    x_min, x_max = data.min(), data.max()
    margin = (x_max - x_min) * 0.05
    x_grid = np.linspace(x_min - margin, x_max + margin, 1000)
    
    # Build KDE manually at given h
    n = len(data)
    kde_y = np.zeros_like(x_grid)
    for i in range(0, n, 100):  # batch for speed
        batch = data[i:i+100]
        diff = x_grid[:, None] - batch[None, :]
        kde_y += np.sum(np.exp(-diff**2 / (2*h**2)), axis=1)
    kde_y /= n * np.sqrt(2 * np.pi) * h
    
    # Find peaks
    prominence = min_prominence * kde_y.max()
    peaks, properties = find_peaks(kde_y, prominence=prominence, distance=20)
    
    levels = x_grid[peaks]
    return levels, x_grid, kde_y


def silverman_bw_1d(data):
    """Silverman 1D rule of thumb."""
    n = len(data)
    sigma = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    # Silverman's robust estimate
    A = min(sigma, iqr / 1.349)
    return 0.9 * A * n ** (-1/5)


# =============================================================================
# MAIN DEMO
# =============================================================================

# In-play stocks: high volume, likely ranging or with clear levels
tickers = ['NVDA', 'TSLA', 'AAPL', 'SPY', 'AMD']

print("=" * 70)
print("VOLUME PROFILE: BANDWIDTH COMPARISON ON REAL MARKET DATA")
print("=" * 70)
print()
print("Method: yfinance 1-min bars → volume-weighted resampling → KDE → peak detection")
print("Compare: Silverman vs GSJ (data-driven) bandwidth on number and")
print("         location of detected support/resistance levels.")
print()

results_all = []

for ticker in tickers:
    print(f"\n{'='*70}")
    print(f"  {ticker}")
    print(f"{'='*70}")
    
    prices, volumes = get_volume_profile_data(ticker, period='5d', interval='1m')
    if prices is None or len(prices) < 100:
        print(f"  Insufficient data for {ticker}, skipping.")
        continue
    
    # Build volume-weighted sample
    vp_data = build_volume_profile(prices, volumes, n_samples=5000)
    n = len(vp_data)
    
    price_range = vp_data.max() - vp_data.min()
    print(f"  Price range: ${vp_data.min():.2f} - ${vp_data.max():.2f} (range: ${price_range:.2f})")
    print(f"  Bars: {len(prices)}, Resampled: {n}")
    
    # Method 1: Silverman
    h_silv = silverman_bw_1d(vp_data)
    levels_silv, x_silv, y_silv = detect_levels(vp_data, h_silv)
    
    # Method 2: GSJ (1D SJ)
    h_gsj = bandwidth(vp_data)
    levels_gsj, x_gsj, y_gsj = detect_levels(vp_data, h_gsj)
    
    # Method 3: Very tight (h = Silverman/3) — overfit baseline
    h_tight = h_silv / 3
    levels_tight, _, _ = detect_levels(vp_data, h_tight)
    
    print(f"\n  {'Method':<16} {'Bandwidth':<12} {'# Levels':<10} {'Levels'}")
    print(f"  {'-'*70}")
    print(f"  {'Silverman':<16} {h_silv:<12.4f} {len(levels_silv):<10} {', '.join(f'${p:.2f}' for p in levels_silv[:8])}")
    print(f"  {'GSJ (SJ)':<16} {h_gsj:<12.4f} {len(levels_gsj):<10} {', '.join(f'${p:.2f}' for p in levels_gsj[:8])}")
    print(f"  {'Tight (h/3)':<16} {h_tight:<12.4f} {len(levels_tight):<10} {', '.join(f'${p:.2f}' for p in levels_tight[:8])}")
    
    print(f"\n  GSJ bandwidth is {h_gsj/h_silv:.2f}x Silverman")
    if len(levels_gsj) > len(levels_silv):
        print(f"  → GSJ reveals {len(levels_gsj) - len(levels_silv)} more levels (finer structure)")
    elif len(levels_gsj) < len(levels_silv):
        print(f"  → GSJ merges {len(levels_silv) - len(levels_gsj)} levels (cleaner picture)")
    else:
        print(f"  → Same number of levels (positions may differ)")
    
    results_all.append({
        'ticker': ticker,
        'h_silv': h_silv,
        'h_gsj': h_gsj,
        'n_levels_silv': len(levels_silv),
        'n_levels_gsj': len(levels_gsj),
        'levels_silv': levels_silv,
        'levels_gsj': levels_gsj,
    })

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Ticker':<8} {'h_Silv':<10} {'h_GSJ':<10} {'Ratio':<8} {'Levels(S)':<12} {'Levels(G)':<12}")
print("-" * 60)
for r in results_all:
    ratio = r['h_gsj'] / r['h_silv']
    print(f"{r['ticker']:<8} {r['h_silv']:<10.4f} {r['h_gsj']:<10.4f} {ratio:<8.3f} {r['n_levels_silv']:<12} {r['n_levels_gsj']:<12}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
Volume profile in trading uses KDE to identify:
  - Peaks (modes) = High Volume Nodes (HVN) = support/resistance levels
  - Valleys (antimodes) = Low Volume Nodes (LVN) = breakout zones

Bandwidth choice directly determines how many levels are detected:
  - Too wide (Silverman on multimodal data): merges nearby S/R levels
  - Too narrow: every tick cluster becomes a "level" (noise)
  - Data-driven (GSJ/SJ): adapts to the session's actual structure

For ranging/consolidating days: GSJ should find MORE levels (correct —
  the price is oscillating between tight levels).
For trending days: GSJ should find FEWER levels (correct —
  the density is spread out, not concentrated at specific prices).
""")
