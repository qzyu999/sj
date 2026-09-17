#include "sierrachart.h"
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

SCDLLName("Sheather-Jones Volume Profile")

static const double SJ_PI = 3.14159265358979323846;

// Safe min/max/clamp — immune to Windows min/max macros
template<typename T> static inline T sjMin(T a, T b) { return (a < b) ? a : b; }
template<typename T> static inline T sjMax(T a, T b) { return (a > b) ? a : b; }
template<typename T> static inline T sjClamp(T v, T lo, T hi) { return (v < lo) ? lo : ((v > hi) ? hi : v); }

// =========================================================================
// 1. HISTOGRAM COLLECTION — chart-type-agnostic
//    Reads VP API for a bar range, returns price→volume map.
// =========================================================================

struct s_Histogram
{
    std::map<int, double> TickVolume; // PriceInTicks → total volume
    double TotalVolume;
};

static s_Histogram CollectHistogram(SCStudyInterfaceRef sc, int startBar, int endBar)
{
    s_Histogram hist;
    hist.TotalVolume = 0.0;

    if (sc.VolumeAtPriceForBars == nullptr)
        return hist;

    for (int b = startBar; b <= endBar; ++b)
    {
        const int n = sc.VolumeAtPriceForBars->GetSizeAtBarIndex(b);
        for (int v = 0; v < n; ++v)
        {
            const s_VolumeAtPriceV2* p = nullptr;
            if (!sc.VolumeAtPriceForBars->GetVAPElementAtIndex(b, v, &p) || p == nullptr)
                break;
            if (p->Volume > 0)
            {
                hist.TickVolume[p->PriceInTicks] += static_cast<double>(p->Volume);
                hist.TotalVolume += static_cast<double>(p->Volume);
            }
        }
    }
    return hist;
}

// =========================================================================
// 2. SHEATHER-JONES BANDWIDTH — pure math, no chart concepts
//
//    Input:  sorted prices[], weights[] (= volume_i / totalVolume), N = totalVolume
//    Output: optimal bandwidth h
//
//    This is the standard 1D SJ Direct Plug-In:
//      (a) Silverman pilot h_0  = (4/(3N))^(1/5) * scale
//      (b) Roughness estimate   S = sum_{i,j} w_i w_j exp(-r^2/4) P(r^2)
//                                   / (sqrt(4*pi) * h_0^5)
//      (c) Optimal bandwidth    h = (R(K) / (N * S))^(1/5)
//
//    where R(K) = 1/(2*sqrt(pi)), P(t) = t^2/16 - 3t/4 + 3/4
// =========================================================================

struct s_SJResult
{
    double Bandwidth;
    std::vector<double> Prices;  // sorted, in price units
    std::vector<double> Weights; // normalized by total volume
    double N;                    // total volume (= effective sample size)
};

static s_SJResult ComputeSJBandwidth(const s_Histogram& hist, double tickSize)
{
    s_SJResult result;
    result.Bandwidth = tickSize; // fallback

    const int M = static_cast<int>(hist.TickVolume.size());
    if (M < 3 || hist.TotalVolume <= 0.0)
        return result;

    result.N = hist.TotalVolume;
    result.Prices.resize(M);
    result.Weights.resize(M);

    // Build sorted price/weight arrays from the histogram
    double weightedMean = 0.0;
    int idx = 0;
    for (auto it = hist.TickVolume.begin(); it != hist.TickVolume.end(); ++it, ++idx)
    {
        result.Prices[idx]  = it->first * tickSize;
        result.Weights[idx] = it->second / hist.TotalVolume;
        weightedMean += result.Weights[idx] * result.Prices[idx];
    }

    // Weighted variance
    double weightedVar = 0.0;
    for (int i = 0; i < M; ++i)
    {
        double d = result.Prices[i] - weightedMean;
        weightedVar += result.Weights[i] * d * d;
    }
    double sigma = sqrt(sjMax(weightedVar, 1e-12));

    // Weighted IQR for robust scale
    double cumW = 0.0, Q25 = result.Prices.front(), Q75 = result.Prices.back();
    bool foundQ25 = false;
    for (int i = 0; i < M; ++i)
    {
        cumW += result.Weights[i];
        if (!foundQ25 && cumW >= 0.25) { Q25 = result.Prices[i]; foundQ25 = true; }
        if (cumW >= 0.75)              { Q75 = result.Prices[i]; break; }
    }
    double iqr = Q75 - Q25;
    double scale = (iqr > 0.0) ? sjMin(sigma, iqr / 1.349) : sigma;
    if (scale <= 0.0) scale = sigma;

    // (a) Silverman pilot
    double h0 = pow(4.0 / (3.0 * result.N), 0.2) * scale;
    h0 = sjMax(h0, tickSize);

    // (b) Roughness: S = sum_{i,j} w_i w_j exp(-r^2/4) P_1(r^2)
    //     P_1(t) = t^2/16 - 3t/4 + 3/4    (the d=1 polynomial)
    //     Diagonal (i=j): r=0, exp(0)=1, P(0)=3/4
    //     Off-diagonal: exploit sorted prices for early exit
    double S = 0.0;
    const double h0sq = h0 * h0;

    for (int i = 0; i < M; ++i)
    {
        // Diagonal contribution
        S += result.Weights[i] * result.Weights[i] * 0.75;

        // Off-diagonal (j > i only, multiply by 2)
        for (int j = i + 1; j < M; ++j)
        {
            double diff = result.Prices[j] - result.Prices[i];
            double rsq = (diff * diff) / h0sq;
            if (rsq > 36.0) break; // sorted, so all further j are farther

            double P = (rsq * rsq) / 16.0 - (3.0 * rsq) / 4.0 + 0.75;
            double W = exp(-rsq / 4.0);
            S += 2.0 * result.Weights[i] * result.Weights[j] * W * P;
        }
    }

    double roughness = S / (sqrt(4.0 * SJ_PI) * pow(h0, 5.0));

    // (c) Optimal bandwidth
    const double RK = 1.0 / (2.0 * sqrt(SJ_PI));
    if (roughness > 1e-15)
        result.Bandwidth = pow(RK / (result.N * roughness), 0.2);
    else
        result.Bandwidth = h0;

    result.Bandwidth = sjMax(result.Bandwidth, tickSize);
    return result;
}

// =========================================================================
// 3. KDE EVALUATION + PEAK/VALLEY DETECTION — pure math
// =========================================================================

struct s_Level
{
    float Price;
    float Density;
    float Prominence;
    bool  IsHVN; // true = peak, false = valley
};

struct s_KDEResult
{
    std::vector<float> GridPrices;
    std::vector<float> GridDensity;
    float MaxDensity;
    std::vector<s_Level> HVNs; // sorted by prominence descending
    std::vector<s_Level> LVNs; // sorted by prominence descending
};

static s_KDEResult EvaluateKDE(
    const std::vector<double>& prices,
    const std::vector<double>& weights,
    double h,
    double tickSize,
    float minProminencePct)
{
    s_KDEResult res;
    res.MaxDensity = 0.0f;

    const int M = static_cast<int>(prices.size());
    if (M < 3 || h <= 0.0)
        return res;

    // Grid with 4h margin on each side
    double dataMin = prices.front();
    double dataMax = prices.back();
    double gridMin = dataMin - 4.0 * h;
    double gridMax = dataMax + 4.0 * h;

    double gridStep = sjMin((dataMax - dataMin) / 1499.0, h / 2.5);
    gridStep = sjMax(gridStep, tickSize);

    int numPts = sjClamp(static_cast<int>((gridMax - gridMin) / gridStep) + 1, 20, 2000);
    res.GridPrices.resize(numPts);
    res.GridDensity.resize(numPts, 0.0f);

    const double inv2hsq = 1.0 / (2.0 * h * h);
    const double norm    = 1.0 / (sqrt(2.0 * SJ_PI) * h);

    for (int k = 0; k < numPts; ++k)
    {
        double xk = gridMin + k * gridStep;
        res.GridPrices[k] = static_cast<float>(xk);

        double sum = 0.0;
        for (int i = 0; i < M; ++i)
        {
            double d = xk - prices[i];
            double arg = d * d * inv2hsq;
            if (arg < 18.0)
                sum += weights[i] * exp(-arg);
        }
        res.GridDensity[k] = static_cast<float>(sum * norm);
        if (res.GridDensity[k] > res.MaxDensity)
            res.MaxDensity = res.GridDensity[k];
    }

    if (res.MaxDensity <= 0.0f)
        return res;

    // --- Peak detection with minimum spacing ---
    const int minDist = sjMax(2, static_cast<int>(round((h * 0.7) / gridStep)));

    struct RawPeak { int idx; float price; float density; };
    std::vector<RawPeak> candidates;

    for (int k = 1; k < numPts - 1; ++k)
    {
        if (res.GridDensity[k] > res.GridDensity[k - 1] &&
            res.GridDensity[k] > res.GridDensity[k + 1])
        {
            // Parabolic interpolation for sub-grid resolution
            float interpPrice = res.GridPrices[k];
            double denom = res.GridDensity[k-1] - 2.0*res.GridDensity[k] + res.GridDensity[k+1];
            if (fabs(denom) > 1e-15)
            {
                double off = 0.5 * (res.GridDensity[k-1] - res.GridDensity[k+1]) / denom;
                off = sjClamp(off, -0.5, 0.5);
                interpPrice = static_cast<float>(res.GridPrices[k] + off * gridStep);
            }
            candidates.push_back({k, interpPrice, res.GridDensity[k]});
        }
    }

    // Fallback: if no local max found, use global max
    if (candidates.empty())
    {
        int best = 0;
        for (int k = 1; k < numPts; ++k)
            if (res.GridDensity[k] > res.GridDensity[best]) best = k;
        candidates.push_back({best, res.GridPrices[best], res.GridDensity[best]});
    }

    // Suppress micro-ripples: keep dominant peak within minDist
    std::vector<RawPeak> filtered;
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        bool dominant = true;
        for (size_t j = 0; j < candidates.size(); ++j)
        {
            if (i == j) continue;
            if (abs(candidates[i].idx - candidates[j].idx) <= minDist &&
                candidates[j].density > candidates[i].density)
            {
                dominant = false;
                break;
            }
        }
        if (dominant) filtered.push_back(candidates[i]);
    }

    // --- Topographic prominence for each peak ---
    const float minPromThreshold = (minProminencePct / 100.0f) * res.MaxDensity;

    for (size_t p = 0; p < filtered.size(); ++p)
    {
        const int k = filtered[p].idx;
        const float peakD = filtered[p].density;

        // Left saddle: deepest point before encountering higher ground
        float leftMin = peakD;
        for (int l = k - 1; l >= 0; --l)
        {
            if (res.GridDensity[l] > peakD) break;
            if (res.GridDensity[l] < leftMin) leftMin = res.GridDensity[l];
        }
        // Right saddle
        float rightMin = peakD;
        for (int r = k + 1; r < numPts; ++r)
        {
            if (res.GridDensity[r] > peakD) break;
            if (res.GridDensity[r] < rightMin) rightMin = res.GridDensity[r];
        }

        float keyCol = sjMax(leftMin, rightMin);
        float prominence = peakD - keyCol;
        float promRatio = sjClamp(prominence / res.MaxDensity, 0.0f, 1.0f);

        if (prominence >= minPromThreshold ||
            (peakD >= 0.30f * res.MaxDensity && promRatio >= 0.015f))
        {
            s_Level lvl;
            lvl.Price = filtered[p].price;
            lvl.Density = peakD;
            lvl.Prominence = sjMax(promRatio, peakD / res.MaxDensity * 0.35f);
            lvl.Prominence = sjClamp(lvl.Prominence, 0.05f, 1.0f);
            lvl.IsHVN = true;
            res.HVNs.push_back(lvl);
        }
    }

    // --- LVN detection: deepest valley between adjacent HVNs ---
    std::vector<s_Level> hvnByPrice = res.HVNs;
    std::sort(hvnByPrice.begin(), hvnByPrice.end(),
              [](const s_Level& a, const s_Level& b) { return a.Price < b.Price; });

    for (size_t i = 0; i + 1 < hvnByPrice.size(); ++i)
    {
        int idxA = sjClamp(static_cast<int>(round((hvnByPrice[i].Price - gridMin) / gridStep)),
                           0, numPts - 1);
        int idxB = sjClamp(static_cast<int>(round((hvnByPrice[i+1].Price - gridMin) / gridStep)),
                           0, numPts - 1);
        if (idxA > idxB) std::swap(idxA, idxB);

        int minIdx = idxA;
        float minD = res.GridDensity[idxA];
        for (int m = idxA + 1; m < idxB; ++m)
        {
            if (res.GridDensity[m] < minD) { minD = res.GridDensity[m]; minIdx = m; }
        }

        float flanking = sjMin(hvnByPrice[i].Density, hvnByPrice[i+1].Density);
        float depth = flanking - minD;

        if (depth > 0.0f && minIdx > idxA && minIdx < idxB)
        {
            s_Level lvn;
            lvn.Price = res.GridPrices[minIdx];
            lvn.Density = minD;
            lvn.Prominence = sjClamp(depth / res.MaxDensity, 0.05f, 1.0f);
            lvn.IsHVN = false;
            res.LVNs.push_back(lvn);
        }
    }

    // Sort by prominence descending
    std::sort(res.HVNs.begin(), res.HVNs.end(),
              [](const s_Level& a, const s_Level& b) { return a.Prominence > b.Prominence; });
    std::sort(res.LVNs.begin(), res.LVNs.end(),
              [](const s_Level& a, const s_Level& b) { return a.Prominence > b.Prominence; });

    return res;
}

// =========================================================================
// 4. COLOR HELPERS
// =========================================================================

static uint32_t GradientColor(COLORREF base, float prominence, int mode)
{
    if (mode == 2) return base;

    uint8_t r = base & 0xFF;
    uint8_t g = (base >> 8) & 0xFF;
    uint8_t b = (base >> 16) & 0xFF;
    float t = sjClamp(prominence, 0.0f, 1.0f);

    if (mode == 0) // Darker for strongest
    {
        float dark  = 0.70f + 0.30f * (1.0f - t);
        float light = (1.0f - t) * 0.60f;
        auto mix = [&](uint8_t c) {
            return static_cast<uint8_t>(sjClamp(c * dark * (1.0f - light) + 245.0f * light, 0.0f, 255.0f));
        };
        return RGB(mix(r), mix(g), mix(b));
    }
    else // Brighter for strongest
    {
        float f = 0.35f + 0.65f * t;
        return RGB(static_cast<uint8_t>(sjClamp(r*f, 0.0f, 255.0f)),
                   static_cast<uint8_t>(sjClamp(g*f, 0.0f, 255.0f)),
                   static_cast<uint8_t>(sjClamp(b*f, 0.0f, 255.0f)));
    }
}

// =========================================================================
// 5. DYNAMIC SUBGRAPH HELPER — runs SJ per bar for evolving levels
// =========================================================================

static const int MAX_TRACKS = 5;

struct s_DynOutput
{
    float HVN[MAX_TRACKS];
    float LVN[MAX_TRACKS];
    float Bandwidth;
};

static s_DynOutput ComputeDynamic(SCStudyInterfaceRef sc, int sIdx, int eIdx,
                                  float bwMult, int maxTracks)
{
    s_DynOutput out = {};

    s_Histogram hist = CollectHistogram(sc, sIdx, eIdx);
    if (hist.TickVolume.size() < 3 || hist.TotalVolume <= 0.0)
        return out;

    s_SJResult sj = ComputeSJBandwidth(hist, static_cast<double>(sc.TickSize));
    sj.Bandwidth *= bwMult;
    sj.Bandwidth = sjMax(sj.Bandwidth, static_cast<double>(sc.TickSize));
    out.Bandwidth = static_cast<float>(sj.Bandwidth);

    s_KDEResult kde = EvaluateKDE(sj.Prices, sj.Weights, sj.Bandwidth,
                                  static_cast<double>(sc.TickSize), 3.0f);

    // Top HVNs by density, then assign to tracks by price order
    struct Cand { double price; double density; };
    std::vector<Cand> topPeaks;
    for (size_t i = 0; i < kde.HVNs.size() && static_cast<int>(i) < maxTracks; ++i)
        topPeaks.push_back({kde.HVNs[i].Price, kde.HVNs[i].Density});
    std::sort(topPeaks.begin(), topPeaks.end(),
              [](const Cand& a, const Cand& b) { return a.price < b.price; });
    for (int t = 0; t < static_cast<int>(topPeaks.size()); ++t)
        out.HVN[t] = static_cast<float>(topPeaks[t].price);

    // Top LVNs
    std::vector<Cand> topValleys;
    for (size_t i = 0; i < kde.LVNs.size() && static_cast<int>(i) < maxTracks; ++i)
        topValleys.push_back({kde.LVNs[i].Price, kde.LVNs[i].Density});
    std::sort(topValleys.begin(), topValleys.end(),
              [](const Cand& a, const Cand& b) { return a.price < b.price; });
    for (int t = 0; t < static_cast<int>(topValleys.size()); ++t)
        out.LVN[t] = static_cast<float>(topValleys[t].price);

    return out;
}

// =========================================================================
// 6. BAR RANGE SELECTION — the ONLY chart-type-aware logic
// =========================================================================

static bool IsMacroChart(SCStudyInterfaceRef sc)
{
    if (sc.ArraySize < 2) return false;
    double totalDays = sc.BaseDateTimeIn[sc.ArraySize - 1].GetAsDouble()
                     - sc.BaseDateTimeIn[0].GetAsDouble();
    double avgDaysPerBar = totalDays / (sc.ArraySize - 1);
    return (sc.SecondsPerBar >= 86400 || avgDaysPerBar >= 0.70 ||
            sc.ArraySize < 60 || (totalDays > 25.0 && sc.ArraySize < 250));
}

// Returns [startBar, endBar] for the selected scope
static void SelectBarRange(SCStudyInterfaceRef sc, int scope, int numBarsBack,
                           bool isMacro, int& startBar, int& endBar)
{
    const int last = sc.ArraySize - 1;
    endBar = last;
    startBar = 0;

    if (scope == 0) // Current Session / Day
    {
        if (isMacro)
        {
            startBar = 0; // aggregate entire chart on macro timeframes
        }
        else
        {
            const int today = sc.GetTradingDayDate(sc.BaseDateTimeIn[last]);
            startBar = last;
            while (startBar > 0 &&
                   sc.GetTradingDayDate(sc.BaseDateTimeIn[startBar - 1]) == today)
                --startBar;
        }
    }
    else if (scope == 1) // Number of Bars Back
    {
        startBar = sjMax(0, last - numBarsBack + 1);
    }
    else if (scope == 2) // Entire Chart
    {
        startBar = 0;
    }
    else if (scope == 3) // Prior Completed Session
    {
        const int today = sc.GetTradingDayDate(sc.BaseDateTimeIn[last]);
        int priorEnd = last;
        while (priorEnd >= 0 &&
               sc.GetTradingDayDate(sc.BaseDateTimeIn[priorEnd]) == today)
            --priorEnd;

        if (priorEnd >= 0)
        {
            endBar = priorEnd;
            const int priorDay = sc.GetTradingDayDate(sc.BaseDateTimeIn[priorEnd]);
            startBar = priorEnd;
            while (startBar > 0 &&
                   sc.GetTradingDayDate(sc.BaseDateTimeIn[startBar - 1]) == priorDay)
                --startBar;
        }
    }
}

// =========================================================================
// 7. MAIN STUDY FUNCTION
// =========================================================================

SCSFExport scsf_SheatherJonesVolumeProfile(SCStudyInterfaceRef sc)
{
    // --- Subgraphs: 0-4 = Dynamic HVNs, 5-9 = Dynamic LVNs, 10 = Bandwidth ---
    SCSubgraphRef Sub_Bandwidth = sc.Subgraph[10];

    // --- Inputs ---
    SCInputRef In_ProfileScope           = sc.Input[0];
    SCInputRef In_NumBars                = sc.Input[1];
    SCInputRef In_MinProminencePct       = sc.Input[2];
    SCInputRef In_MaxHVNCount            = sc.Input[3];
    SCInputRef In_MaxLVNCount            = sc.Input[4];
    SCInputRef In_LineType               = sc.Input[5];
    SCInputRef In_HVNColor               = sc.Input[6];
    SCInputRef In_LVNColor               = sc.Input[7];
    SCInputRef In_BaseLineWidth          = sc.Input[8];
    SCInputRef In_ScaleWidthByProminence = sc.Input[9];
    SCInputRef In_ColorGradientMode      = sc.Input[10];
    SCInputRef In_EnableTransparency     = sc.Input[11];
    SCInputRef In_CalcIntervalSec        = sc.Input[12];
    SCInputRef In_BandwidthMultiplier    = sc.Input[13];
    SCInputRef In_PlotKDE                = sc.Input[14];
    SCInputRef In_KDEColor               = sc.Input[15];
    SCInputRef In_KDETransparency        = sc.Input[16];
    SCInputRef In_KDEWidthBars           = sc.Input[17];
    SCInputRef In_BandwidthMode          = sc.Input[18];
    SCInputRef In_FixedBandwidthPts      = sc.Input[19];
    SCInputRef In_KDEStyle               = sc.Input[20];
    SCInputRef In_KDEPlacement           = sc.Input[21];
    SCInputRef In_KDERightOffset         = sc.Input[22];
    SCInputRef In_KDELineWidth           = sc.Input[23];
    SCInputRef In_LabelPosition          = sc.Input[24];
    SCInputRef In_LabelFontSize          = sc.Input[25];
    SCInputRef In_DynamicMode            = sc.Input[26];
    SCInputRef In_DynamicWindowBars      = sc.Input[27];
    SCInputRef In_DynamicMaxHistory      = sc.Input[28];
    SCInputRef In_DynamicMaxLevels       = sc.Input[29];
    SCInputRef In_DrawStaticLevels       = sc.Input[30];

    int& r_LastDrawnCount = sc.GetPersistentInt(1);
    SCDateTime& r_LastCalcTime = sc.GetPersistentSCDateTime(2);

    // -----------------------------------------------------------------
    // DEFAULTS
    // -----------------------------------------------------------------
    if (sc.SetDefaults)
    {
        sc.GraphName = "Sheather-Jones Volume Profile";
        sc.AutoLoop = 0;
        sc.GraphRegion = 0;
        sc.CalculationPrecedence = LOW_PREC_LEVEL;

        const char* hvnNames[5] = {
            "Dynamic HVN 1 (Primary)", "Dynamic HVN 2",
            "Dynamic HVN 3", "Dynamic HVN 4", "Dynamic HVN 5"
        };
        const COLORREF hvnColors[5] = {
            RGB(0,160,75), RGB(0,200,95), RGB(46,204,113),
            RGB(72,220,150), RGB(120,230,180)
        };
        for (int t = 0; t < 5; ++t)
        {
            sc.Subgraph[t].Name = hvnNames[t];
            sc.Subgraph[t].DrawStyle = DRAWSTYLE_LINE_SKIP_ZEROS;
            sc.Subgraph[t].PrimaryColor = hvnColors[t];
            sc.Subgraph[t].LineWidth = (t < 2) ? 2 : 1;
            sc.Subgraph[t].DrawZeros = false;
        }

        const char* lvnNames[5] = {
            "Dynamic LVN 1 (Primary)", "Dynamic LVN 2",
            "Dynamic LVN 3", "Dynamic LVN 4", "Dynamic LVN 5"
        };
        const COLORREF lvnColors[5] = {
            RGB(220,30,30), RGB(245,50,50), RGB(255,99,71),
            RGB(240,128,128), RGB(250,160,160)
        };
        for (int t = 0; t < 5; ++t)
        {
            sc.Subgraph[5+t].Name = lvnNames[t];
            sc.Subgraph[5+t].DrawStyle = DRAWSTYLE_DASH;
            sc.Subgraph[5+t].PrimaryColor = lvnColors[t];
            sc.Subgraph[5+t].LineWidth = (t < 2) ? 2 : 1;
            sc.Subgraph[5+t].DrawZeros = false;
        }

        Sub_Bandwidth.Name = "Dynamic Bandwidth (h)";
        Sub_Bandwidth.DrawStyle = DRAWSTYLE_IGNORE;

        In_ProfileScope.Name = "Profile Scope";
        In_ProfileScope.SetCustomInputStrings(
            "Current Session / Day;Number of Bars Back;Entire Chart;Prior Completed Session");
        In_ProfileScope.SetCustomInputIndex(0);

        In_NumBars.Name = "Number of Bars Back (if Scope = Bars Back)";
        In_NumBars.SetInt(390);
        In_NumBars.SetIntLimits(10, 50000);

        In_MinProminencePct.Name = "Minimum Peak Prominence % (0-100)";
        In_MinProminencePct.SetFloat(3.0f);
        In_MinProminencePct.SetFloatLimits(0.1f, 100.0f);

        In_MaxHVNCount.Name = "Maximum HVN Lines";
        In_MaxHVNCount.SetInt(6);
        In_MaxHVNCount.SetIntLimits(1, 25);

        In_MaxLVNCount.Name = "Maximum LVN Lines";
        In_MaxLVNCount.SetInt(4);
        In_MaxLVNCount.SetIntLimits(0, 25);

        In_LineType.Name = "Line Drawing Type";
        In_LineType.SetCustomInputStrings("Horizontal Line;Horizontal Ray");
        In_LineType.SetCustomInputIndex(0);

        In_HVNColor.Name = "HVN Base Color";
        In_HVNColor.SetColor(RGB(0, 185, 90));

        In_LVNColor.Name = "LVN Base Color";
        In_LVNColor.SetColor(RGB(225, 45, 45));

        In_BaseLineWidth.Name = "Base Line Width";
        In_BaseLineWidth.SetInt(2);
        In_BaseLineWidth.SetIntLimits(1, 10);

        In_ScaleWidthByProminence.Name = "Scale Width by Prominence";
        In_ScaleWidthByProminence.SetYesNo(1);

        In_ColorGradientMode.Name = "Color Gradient Style";
        In_ColorGradientMode.SetCustomInputStrings(
            "Darker for Strongest;Brighter for Strongest;Solid Color");
        In_ColorGradientMode.SetCustomInputIndex(0);

        In_EnableTransparency.Name = "Transparency Gradient";
        In_EnableTransparency.SetYesNo(1);

        In_CalcIntervalSec.Name = "Recalculation Interval (seconds)";
        In_CalcIntervalSec.SetInt(15);
        In_CalcIntervalSec.SetIntLimits(1, 300);

        In_BandwidthMultiplier.Name = "Bandwidth Multiplier (<1 sharper, >1 smoother)";
        In_BandwidthMultiplier.SetFloat(1.0f);
        In_BandwidthMultiplier.SetFloatLimits(0.1f, 5.0f);

        In_PlotKDE.Name = "Plot KDE Profile on Chart";
        In_PlotKDE.SetYesNo(1);

        In_KDEColor.Name = "KDE Profile Color";
        In_KDEColor.SetColor(RGB(70, 130, 180));

        In_KDETransparency.Name = "KDE Transparency % (0-95)";
        In_KDETransparency.SetInt(50);
        In_KDETransparency.SetIntLimits(0, 95);

        In_KDEWidthBars.Name = "KDE Profile Width (bars)";
        In_KDEWidthBars.SetInt(35);
        In_KDEWidthBars.SetIntLimits(5, 300);

        In_BandwidthMode.Name = "Bandwidth Mode";
        In_BandwidthMode.SetCustomInputStrings("Sheather-Jones;Fixed Points");
        In_BandwidthMode.SetCustomInputIndex(0);

        In_FixedBandwidthPts.Name = "Fixed Bandwidth (points)";
        In_FixedBandwidthPts.SetFloat(5.0f);
        In_FixedBandwidthPts.SetFloatLimits(0.25f, 500.0f);

        In_KDEStyle.Name = "KDE Display Style";
        In_KDEStyle.SetCustomInputStrings(
            "Smooth Envelope;Filled Bars;Both");
        In_KDEStyle.SetCustomInputIndex(0);

        In_KDEPlacement.Name = "KDE Placement";
        In_KDEPlacement.SetCustomInputStrings(
            "Right Margin;Right Edge (VP aligned);Over Price Action");
        In_KDEPlacement.SetCustomInputIndex(0);

        In_KDERightOffset.Name = "KDE Right Offset (bars)";
        In_KDERightOffset.SetInt(6);
        In_KDERightOffset.SetIntLimits(0, 300);

        In_KDELineWidth.Name = "KDE Line Width";
        In_KDELineWidth.SetInt(2);
        In_KDELineWidth.SetIntLimits(1, 10);

        In_LabelPosition.Name = "Label Position";
        In_LabelPosition.SetCustomInputStrings("Right Side;Left Side;Hidden");
        In_LabelPosition.SetCustomInputIndex(0);

        In_LabelFontSize.Name = "Label Font Size";
        In_LabelFontSize.SetInt(8);
        In_LabelFontSize.SetIntLimits(6, 24);

        In_DynamicMode.Name = "Dynamic Mode";
        In_DynamicMode.SetCustomInputStrings(
            "Rolling Window;Session Developing;Entire Chart Developing;Disabled");
        In_DynamicMode.SetCustomInputIndex(1);

        In_DynamicWindowBars.Name = "Dynamic Window (bars)";
        In_DynamicWindowBars.SetInt(60);
        In_DynamicWindowBars.SetIntLimits(5, 2000);

        In_DynamicMaxHistory.Name = "Dynamic History Depth (bars)";
        In_DynamicMaxHistory.SetInt(500);
        In_DynamicMaxHistory.SetIntLimits(20, 10000);

        In_DynamicMaxLevels.Name = "Dynamic Levels (1-5)";
        In_DynamicMaxLevels.SetInt(5);
        In_DynamicMaxLevels.SetIntLimits(1, 5);

        In_DrawStaticLevels.Name = "Draw Static Levels & Profile";
        In_DrawStaticLevels.SetYesNo(1);

        return;
    }

    // -----------------------------------------------------------------
    // LIFECYCLE
    // -----------------------------------------------------------------
    const int MaxDrawn = 280;
    const int BaseLine = 800000 + (sc.StudyGraphInstanceID * 300);

    if (sc.LastCallToFunction)
    {
        for (int i = 0; i < MaxDrawn; ++i)
            sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, BaseLine + i);
        r_LastDrawnCount = 0;
        return;
    }

    // Rate-limit
    const int interval = sjMax(1, In_CalcIntervalSec.GetInt());
    SCDateTime now = sc.CurrentSystemDateTime;
    if (r_LastCalcTime.GetTimeInSeconds() > 0 &&
        (now.GetTimeInSeconds() - r_LastCalcTime.GetTimeInSeconds()) < interval &&
        sc.UpdateStartIndex > 0)
        return;
    r_LastCalcTime = now;

    if (sc.ArraySize < 1) return;

    if (sc.VolumeAtPriceForBars == nullptr)
    {
        sc.AddMessageToLog("SJ Volume Profile: No Volume-At-Price data. "
            "Use an Intraday chart (.scid) for tick-level volume.", 1);
        return;
    }

    const bool isMacro = IsMacroChart(sc);
    const float bwMult = sjClamp(In_BandwidthMultiplier.GetFloat(), 0.1f, 5.0f);

    // -----------------------------------------------------------------
    // A. DYNAMIC EVOLVING LEVELS (subgraphs 0-9)
    // -----------------------------------------------------------------
    const int dynMode = In_DynamicMode.GetIndex();
    if (dynMode != 3) // 0=Rolling, 1=Session, 2=EntireChart, 3=Disabled
    {
        const int windowBars = sjMax(5, In_DynamicWindowBars.GetInt());
        const int maxHist    = sjMax(20, In_DynamicMaxHistory.GetInt());
        const int maxTracks  = sjClamp(In_DynamicMaxLevels.GetInt(), 1, 5);

        int calcStart = 0;
        if (sc.UpdateStartIndex == 0)
        {
            calcStart = sjMax(0, sc.ArraySize - maxHist);
            for (int i = 0; i < calcStart; ++i)
            {
                for (int t = 0; t < 5; ++t)
                {
                    sc.Subgraph[t][i] = 0.0f;
                    sc.Subgraph[5+t][i] = 0.0f;
                }
                Sub_Bandwidth[i] = 0.0f;
            }
        }
        else
        {
            calcStart = sjMax(0, sc.UpdateStartIndex);
        }

        for (int bar = calcStart; bar < sc.ArraySize; ++bar)
        {
            int sIdx = 0;

            if (dynMode == 0) // Rolling Window
            {
                sIdx = sjMax(0, bar - windowBars + 1);
            }
            else if (dynMode == 1) // Session Developing
            {
                if (isMacro)
                {
                    sIdx = 0; // macro: accumulate entire chart
                }
                else
                {
                    const int day = sc.GetTradingDayDate(sc.BaseDateTimeIn[bar]);
                    sIdx = bar;
                    while (sIdx > 0 &&
                           sc.GetTradingDayDate(sc.BaseDateTimeIn[sIdx - 1]) == day)
                        --sIdx;
                }
            }
            // dynMode == 2: sIdx stays 0 (entire chart developing)

            s_DynOutput out = ComputeDynamic(sc, sIdx, bar, bwMult, maxTracks);

            for (int t = 0; t < 5; ++t)
            {
                sc.Subgraph[t][bar]   = (t < maxTracks) ? out.HVN[t] : 0.0f;
                sc.Subgraph[5+t][bar] = (t < maxTracks) ? out.LVN[t] : 0.0f;
            }
            Sub_Bandwidth[bar] = out.Bandwidth;
        }
    }
    else if (sc.UpdateStartIndex == 0)
    {
        for (int i = 0; i < sc.ArraySize; ++i)
        {
            for (int t = 0; t < 5; ++t)
            {
                sc.Subgraph[t][i] = 0.0f;
                sc.Subgraph[5+t][i] = 0.0f;
            }
            Sub_Bandwidth[i] = 0.0f;
        }
    }

    // -----------------------------------------------------------------
    // B. STATIC PROFILE + LEVELS (horizontal lines + KDE drawing)
    // -----------------------------------------------------------------
    if (!In_DrawStaticLevels.GetYesNo())
    {
        for (int i = 0; i < r_LastDrawnCount && i < MaxDrawn; ++i)
            sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, BaseLine + i);
        r_LastDrawnCount = 0;
        return;
    }

    // B1. Select bar range
    int startBar = 0, endBar = sc.ArraySize - 1;
    SelectBarRange(sc, In_ProfileScope.GetIndex(), In_NumBars.GetInt(),
                   isMacro, startBar, endBar);

    // Thin-session fallback: if scope=CurrentSession and < 5 ticks, expand to prior session
    s_Histogram hist = CollectHistogram(sc, startBar, endBar);
    if (In_ProfileScope.GetIndex() == 0 && hist.TickVolume.size() < 5 && startBar > 0)
    {
        const int priorDay = sc.GetTradingDayDate(sc.BaseDateTimeIn[startBar - 1]);
        int expanded = startBar - 1;
        while (expanded > 0 &&
               sc.GetTradingDayDate(sc.BaseDateTimeIn[expanded - 1]) == priorDay)
            --expanded;
        startBar = expanded;
        hist = CollectHistogram(sc, startBar, endBar);
    }

    if (hist.TickVolume.size() < 5 || hist.TotalVolume <= 0.0)
    {
        SCString msg;
        msg.Format("SJ: Insufficient data (ticks=%d, volume=%.0f)",
                   static_cast<int>(hist.TickVolume.size()), hist.TotalVolume);
        sc.AddMessageToLog(msg, 0);
        return;
    }

    // B2. Compute bandwidth
    s_SJResult sj = ComputeSJBandwidth(hist, static_cast<double>(sc.TickSize));

    double h_final;
    const int bwMode = In_BandwidthMode.GetIndex();
    if (bwMode == 1) // Fixed Points
        h_final = static_cast<double>(In_FixedBandwidthPts.GetFloat());
    else
        h_final = sj.Bandwidth;

    h_final *= bwMult;
    h_final = sjMax(h_final, static_cast<double>(sc.TickSize));

    // B3. KDE + peak detection
    s_KDEResult kde = EvaluateKDE(sj.Prices, sj.Weights, h_final,
                                  static_cast<double>(sc.TickSize),
                                  In_MinProminencePct.GetFloat());

    if (kde.MaxDensity <= 0.0f) return;

    const int numHVN = sjMin(In_MaxHVNCount.GetInt(), static_cast<int>(kde.HVNs.size()));
    const int numLVN = sjMin(In_MaxLVNCount.GetInt(), static_cast<int>(kde.LVNs.size()));

    // Diagnostic
    SCString diag;
    diag.Format("SJ: N=%.0f M=%d h=%.4f HVNs=%d LVNs=%d bars=[%d..%d]",
                sj.N, static_cast<int>(sj.Prices.size()), h_final,
                numHVN, numLVN, startBar, endBar);
    sc.AddMessageToLog(diag, 0);

    // -----------------------------------------------------------------
    // B4. DRAWING — chart-type-aware time projection
    // -----------------------------------------------------------------
    int lineIdx = 0;

    // Time scaling
    int firstVis = sjClamp(sc.IndexOfFirstVisibleBar, 0, sc.ArraySize - 1);
    int lastVis  = sjClamp(sc.IndexOfLastVisibleBar, 0, sc.ArraySize - 1);
    if (lastVis <= firstVis)
    {
        firstVis = sjMax(0, sc.ArraySize - 80);
        lastVis  = sc.ArraySize - 1;
    }
    int numVis = sjMax(1, lastVis - firstVis);
    double visSpan = sc.BaseDateTimeIn[lastVis].GetAsDouble()
                   - sc.BaseDateTimeIn[firstVis].GetAsDouble();
    double daysPerBar = (visSpan > 1e-7) ? (visSpan / numVis) : (1.0 / 1440.0);
    if (daysPerBar <= 1e-7)
        daysPerBar = (sc.SecondsPerBar > 0) ? (sc.SecondsPerBar / 86400.0) : (1.0/1440.0);

    // Forward projection: volume/range charts need a floor
    double fwdDaysPerBar = daysPerBar;
    if (sc.SecondsPerBar == 0)
        fwdDaysPerBar = sjMax(fwdDaysPerBar, 15.0 / 1440.0);

    auto OffsetDT = [&](const SCDateTime& anchor, double barOff) -> SCDateTime {
        double step = (barOff >= 0.0) ? fwdDaysPerBar : daysPerBar;
        return SCDateTime(anchor.GetAsDouble() + barOff * step);
    };

    // KDE profile width capped at 35% of visible bars
    int userWidth   = sjClamp(In_KDEWidthBars.GetInt(), 5, 250);
    double maxWidth = sjMax(4.0, numVis * 0.35);
    double effWidth = sjMin(static_cast<double>(userWidth), maxWidth);
    int marginOff   = sjClamp(In_KDERightOffset.GetInt(), 0, 100);

    auto KDECoords = [&](float ratio, const SCDateTime& lastDT,
                         SCDateTime& baseDT, SCDateTime& tipDT)
    {
        int placement = sjClamp(In_KDEPlacement.GetIndex(), 0, 2);
        if (placement == 0) // Right Margin
        {
            baseDT = OffsetDT(lastDT, marginOff);
            tipDT  = OffsetDT(lastDT, marginOff + ratio * effWidth);
        }
        else if (placement == 1) // Right Edge
        {
            baseDT = OffsetDT(lastDT, marginOff + effWidth);
            tipDT  = OffsetDT(lastDT, marginOff + effWidth - ratio * effWidth);
        }
        else // Over Price Action
        {
            baseDT = lastDT;
            tipDT  = OffsetDT(lastDT, -ratio * effWidth);
        }
    };

    // --- Draw KDE profile ---
    if (In_PlotKDE.GetYesNo() && kde.MaxDensity > 0.0f)
    {
        const int style   = sjClamp(In_KDEStyle.GetIndex(), 0, 2);
        const COLORREF kc = In_KDEColor.GetColor();
        const int kTrans  = sjClamp(In_KDETransparency.GetInt(), 0, 95);
        const int kLineW  = sjClamp(In_KDELineWidth.GetInt(), 1, 10);
        const SCDateTime lastDT = sc.BaseDateTimeIn[endBar];
        const int numGrid = static_cast<int>(kde.GridPrices.size());

        // Envelope curve
        if (style == 0 || style == 2)
        {
            const int segs = sjMin(180, numGrid);
            SCDateTime prevTip;
            float prevPrice = 0.0f;
            bool hasPrev = false;
            SCDateTime firstBase, firstTip, lastBase, lastTip;
            float firstP = 0.0f, lastP = 0.0f;

            for (int s = 0; s < segs && lineIdx < 200; ++s)
            {
                int gi = sjClamp(static_cast<int>(round(
                    s * (numGrid - 1) / static_cast<double>(segs - 1))), 0, numGrid - 1);

                float ratio = sjClamp(kde.GridDensity[gi] / kde.MaxDensity, 0.0f, 1.0f);
                float price = kde.GridPrices[gi];

                SCDateTime baseDT, tipDT;
                KDECoords(ratio, lastDT, baseDT, tipDT);

                if (s == 0) { firstBase = baseDT; firstTip = tipDT; firstP = price; }
                lastBase = baseDT; lastTip = tipDT; lastP = price;

                if (hasPrev)
                {
                    s_UseTool T; T.Clear();
                    T.ChartNumber = sc.ChartNumber;
                    T.DrawingType = DRAWING_LINE;
                    T.LineNumber  = BaseLine + lineIdx++;
                    T.BeginDateTime = prevTip; T.BeginValue = prevPrice;
                    T.EndDateTime   = tipDT;   T.EndValue   = price;
                    T.Color = kc; T.TransparencyLevel = kTrans;
                    T.LineWidth = kLineW; T.LineStyle = LINESTYLE_SOLID;
                    T.AddMethod = UTAM_ADD_OR_ADJUST;
                    sc.UseTool(T);
                }
                prevTip = tipDT; prevPrice = price; hasPrev = true;
            }

            // Baseline + caps
            if (hasPrev && lineIdx + 3 < 200)
            {
                int capTrans = sjMin(95, kTrans + 15);
                // Spine
                s_UseTool T; T.Clear();
                T.ChartNumber = sc.ChartNumber; T.DrawingType = DRAWING_LINE;
                T.LineNumber = BaseLine + lineIdx++;
                T.BeginDateTime = firstBase; T.BeginValue = firstP;
                T.EndDateTime   = lastBase;  T.EndValue   = lastP;
                T.Color = kc; T.TransparencyLevel = capTrans;
                T.LineWidth = 1; T.LineStyle = LINESTYLE_DASH;
                T.AddMethod = UTAM_ADD_OR_ADJUST;
                sc.UseTool(T);
                // Bottom cap
                T.Clear(); T.ChartNumber = sc.ChartNumber; T.DrawingType = DRAWING_LINE;
                T.LineNumber = BaseLine + lineIdx++;
                T.BeginDateTime = firstBase; T.BeginValue = firstP;
                T.EndDateTime   = firstTip;  T.EndValue   = firstP;
                T.Color = kc; T.TransparencyLevel = capTrans;
                T.LineWidth = 1; T.LineStyle = LINESTYLE_SOLID;
                T.AddMethod = UTAM_ADD_OR_ADJUST;
                sc.UseTool(T);
                // Top cap
                T.Clear(); T.ChartNumber = sc.ChartNumber; T.DrawingType = DRAWING_LINE;
                T.LineNumber = BaseLine + lineIdx++;
                T.BeginDateTime = lastBase; T.BeginValue = lastP;
                T.EndDateTime   = lastTip;  T.EndValue   = lastP;
                T.Color = kc; T.TransparencyLevel = capTrans;
                T.LineWidth = 1; T.LineStyle = LINESTYLE_SOLID;
                T.AddMethod = UTAM_ADD_OR_ADJUST;
                sc.UseTool(T);
            }
        }

        // Filled bars
        if (style == 1 || style == 2)
        {
            const int slices = (style == 2) ? sjMin(60, numGrid) : sjMin(160, numGrid);
            for (int s = 0; s < slices && lineIdx < 230; ++s)
            {
                int gi = sjClamp(static_cast<int>(round(
                    s * (numGrid - 1) / static_cast<double>(slices - 1))), 0, numGrid - 1);

                float ratio = kde.GridDensity[gi] / kde.MaxDensity;
                if (ratio < 0.02f) continue;

                SCDateTime baseDT, tipDT;
                KDECoords(ratio, lastDT, baseDT, tipDT);
                if (baseDT > tipDT) std::swap(baseDT, tipDT);

                s_UseTool T; T.Clear();
                T.ChartNumber = sc.ChartNumber; T.DrawingType = DRAWING_LINE;
                T.LineNumber = BaseLine + lineIdx++;
                T.BeginDateTime = baseDT; T.BeginValue = kde.GridPrices[gi];
                T.EndDateTime   = tipDT;  T.EndValue   = kde.GridPrices[gi];
                T.Color = kc; T.TransparencyLevel = kTrans;
                T.LineWidth = 3; T.LineStyle = LINESTYLE_SOLID;
                T.AddMethod = UTAM_ADD_OR_ADJUST;
                sc.UseTool(T);
            }
        }
    }

    // --- Draw HVN lines ---
    const DrawingTypeEnum drawType =
        (In_LineType.GetIndex() == 0) ? DRAWING_HORIZONTALLINE : DRAWING_RAY;
    const int gradMode  = sjClamp(In_ColorGradientMode.GetIndex(), 0, 2);
    const bool useTrans = In_EnableTransparency.GetYesNo() != 0;
    const int labelPos  = sjClamp(In_LabelPosition.GetIndex(), 0, 2);
    const int labelFont = sjClamp(In_LabelFontSize.GetInt(), 6, 24);

    for (int i = 0; i < numHVN && lineIdx < MaxDrawn; ++i, ++lineIdx)
    {
        const s_Level& h = kde.HVNs[i];
        s_UseTool T; T.Clear();
        T.ChartNumber = sc.ChartNumber;
        T.DrawingType = drawType;
        T.LineNumber  = BaseLine + lineIdx;
        T.BeginValue  = h.Price; T.EndValue = h.Price;
        T.BeginDateTime = sc.BaseDateTimeIn[startBar];
        T.EndDateTime   = sc.BaseDateTimeIn[endBar];
        T.Color = GradientColor(In_HVNColor.GetColor(), h.Prominence, gradMode);

        if (useTrans)
            T.TransparencyLevel = sjClamp(static_cast<int>((1.0f - h.Prominence) * 65.0f), 0, 80);

        int w = In_BaseLineWidth.GetInt();
        if (In_ScaleWidthByProminence.GetYesNo())
            w = sjClamp(static_cast<int>(round(w * (0.6f + 0.8f * h.Prominence))), 1, 6);
        T.LineWidth = w;
        T.LineStyle = LINESTYLE_SOLID;

        if (labelPos != 2)
        {
            T.TransparentLabelBackground = 1;
            T.FontSize = labelFont;
            T.TextAlignment = (labelPos == 0) ? (DT_RIGHT|DT_BOTTOM) : (DT_LEFT|DT_BOTTOM);
            T.DisplayHorizontalLineValue = 0;
            T.Text.Format("HVN %.2f (%.0f%%)", h.Price, h.Prominence * 100.0f);
        }
        else
        {
            T.Text = ""; T.DisplayHorizontalLineValue = 0;
        }
        T.AddMethod = UTAM_ADD_OR_ADJUST;
        sc.UseTool(T);
    }

    // --- Draw LVN lines ---
    for (int i = 0; i < numLVN && lineIdx < MaxDrawn; ++i, ++lineIdx)
    {
        const s_Level& l = kde.LVNs[i];
        s_UseTool T; T.Clear();
        T.ChartNumber = sc.ChartNumber;
        T.DrawingType = drawType;
        T.LineNumber  = BaseLine + lineIdx;
        T.BeginValue  = l.Price; T.EndValue = l.Price;
        T.BeginDateTime = sc.BaseDateTimeIn[startBar];
        T.EndDateTime   = sc.BaseDateTimeIn[endBar];
        T.Color = GradientColor(In_LVNColor.GetColor(), l.Prominence, gradMode);

        if (useTrans)
            T.TransparencyLevel = sjClamp(static_cast<int>((1.0f - l.Prominence) * 65.0f), 0, 80);

        int w = sjMax(1, In_BaseLineWidth.GetInt() - 1);
        if (In_ScaleWidthByProminence.GetYesNo())
            w = sjClamp(static_cast<int>(round(w * (0.6f + 0.6f * l.Prominence))), 1, 5);
        T.LineWidth = w;
        T.LineStyle = LINESTYLE_DASH;

        if (labelPos != 2)
        {
            T.TransparentLabelBackground = 1;
            T.FontSize = labelFont;
            T.TextAlignment = (labelPos == 0) ? (DT_RIGHT|DT_BOTTOM) : (DT_LEFT|DT_BOTTOM);
            T.DisplayHorizontalLineValue = 0;
            T.Text.Format("LVN %.2f (%.0f%%)", l.Price, l.Prominence * 100.0f);
        }
        else
        {
            T.Text = ""; T.DisplayHorizontalLineValue = 0;
        }
        T.AddMethod = UTAM_ADD_OR_ADJUST;
        sc.UseTool(T);
    }

    // Clean up stale lines from prior calculation
    for (int i = lineIdx; i < r_LastDrawnCount && i < MaxDrawn; ++i)
        sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, BaseLine + i);
    r_LastDrawnCount = lineIdx;
}
