#include "sierrachart.h"
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

SCDLLName("Sheather-Jones Volume Profile")

static const double SJ_PI = 3.14159265358979323846;

template<typename T> static inline T sjMin(T a, T b) { return (a < b) ? a : b; }
template<typename T> static inline T sjMax(T a, T b) { return (a > b) ? a : b; }
template<typename T> static inline T sjClamp(T v, T lo, T hi) { return (v < lo) ? lo : ((v > hi) ? hi : v); }
static inline int sjClampI(unsigned int v, int lo, int hi) {
    int iv = static_cast<int>(v); return (iv < lo) ? lo : ((iv > hi) ? hi : iv);
}

// =========================================================================
// 1. HISTOGRAM — chart-type-agnostic
// =========================================================================

struct s_Histogram
{
    std::map<int, double> TickVol;
    double Total;
};

static s_Histogram CollectHistogram(SCStudyInterfaceRef sc, int s, int e)
{
    s_Histogram h;
    h.Total = 0.0;
    if (!sc.VolumeAtPriceForBars) return h;
    for (int b = s; b <= e; ++b)
    {
        const int n = sc.VolumeAtPriceForBars->GetSizeAtBarIndex(b);
        for (int v = 0; v < n; ++v)
        {
            const s_VolumeAtPriceV2* p = nullptr;
            if (!sc.VolumeAtPriceForBars->GetVAPElementAtIndex(b, v, &p) || !p) break;
            if (p->Volume > 0)
            {
                h.TickVol[p->PriceInTicks] += static_cast<double>(p->Volume);
                h.Total += static_cast<double>(p->Volume);
            }
        }
    }
    return h;
}

// =========================================================================
// 2. SHEATHER-JONES BANDWIDTH
// =========================================================================

struct s_SJResult
{
    double Bandwidth;
    std::vector<double> Prices;
    std::vector<double> Weights;
    double N;
};

static s_SJResult ComputeSJBandwidth(const s_Histogram& hist, double tickSize)
{
    s_SJResult r;
    r.Bandwidth = tickSize;
    const int M = static_cast<int>(hist.TickVol.size());
    if (M < 3 || hist.Total <= 0.0) return r;

    r.N = hist.Total;
    r.Prices.resize(M);
    r.Weights.resize(M);

    double wMean = 0.0;
    int idx = 0;
    for (auto it = hist.TickVol.begin(); it != hist.TickVol.end(); ++it, ++idx)
    {
        r.Prices[idx]  = it->first * tickSize;
        r.Weights[idx] = it->second / hist.Total;
        wMean += r.Weights[idx] * r.Prices[idx];
    }

    double wVar = 0.0;
    for (int i = 0; i < M; ++i) { double d = r.Prices[i]-wMean; wVar += r.Weights[i]*d*d; }
    double sigma = sqrt(sjMax(wVar, 1e-12));

    double cumW = 0.0, Q25 = r.Prices.front(), Q75 = r.Prices.back();
    bool fQ25 = false;
    for (int i = 0; i < M; ++i)
    {
        cumW += r.Weights[i];
        if (!fQ25 && cumW >= 0.25) { Q25 = r.Prices[i]; fQ25 = true; }
        if (cumW >= 0.75)          { Q75 = r.Prices[i]; break; }
    }
    double iqr = Q75 - Q25;
    double scale = (iqr > 0.0) ? sjMin(sigma, iqr / 1.349) : sigma;
    if (scale <= 0.0) scale = sigma;

    double h0 = pow(4.0/(3.0*r.N), 0.2) * scale;
    h0 = sjMax(h0, tickSize);

    double S = 0.0;
    const double h0sq = h0*h0;
    for (int i = 0; i < M; ++i)
    {
        S += r.Weights[i]*r.Weights[i]*0.75;
        for (int j = i+1; j < M; ++j)
        {
            double d = r.Prices[j]-r.Prices[i];
            double rsq = d*d/h0sq;
            if (rsq > 36.0) break;
            S += 2.0*r.Weights[i]*r.Weights[j]*exp(-rsq/4.0)*((rsq*rsq)/16.0 - 3.0*rsq/4.0 + 0.75);
        }
    }
    double rough = S / (sqrt(4.0*SJ_PI)*pow(h0, 5.0));

    const double RK = 1.0/(2.0*sqrt(SJ_PI));
    if (rough > 1e-15)
        r.Bandwidth = pow(RK/(r.N*rough), 0.2);
    else
        r.Bandwidth = h0;
    r.Bandwidth = sjMax(r.Bandwidth, tickSize);
    return r;
}

// =========================================================================
// 3. KDE + PEAK/VALLEY + POC
// =========================================================================

struct s_Level
{
    float Price;
    float Density;
    float Prominence;
    bool  IsHVN;
};

struct s_KDEResult
{
    std::vector<float> GridPrices;
    std::vector<float> GridDensity;
    float MaxDensity;
    float POC;
    std::vector<s_Level> HVNs;
    std::vector<s_Level> LVNs;
    double GridMin;
    double GridStep;
};

static s_KDEResult EvaluateKDE(
    const std::vector<double>& prices,
    const std::vector<double>& weights,
    double h, double tickSize, float minPromPct)
{
    s_KDEResult res;
    res.MaxDensity = 0.0f;
    res.POC = 0.0f;
    res.GridMin = 0.0; res.GridStep = tickSize;

    const int M = static_cast<int>(prices.size());
    if (M < 3 || h <= 0.0) return res;

    double dMin = prices.front(), dMax = prices.back();
    res.GridMin = dMin - 4.0*h;
    double gridMax = dMax + 4.0*h;
    res.GridStep = sjMin((dMax-dMin)/1499.0, h/2.5);
    res.GridStep = sjMax(res.GridStep, tickSize);

    int nPts = sjClamp(static_cast<int>((gridMax-res.GridMin)/res.GridStep)+1, 20, 2000);
    res.GridPrices.resize(nPts);
    res.GridDensity.resize(nPts, 0.0f);

    const double inv2hsq = 1.0/(2.0*h*h);
    const double norm = 1.0/(sqrt(2.0*SJ_PI)*h);
    int pocIdx = 0;

    for (int k = 0; k < nPts; ++k)
    {
        double xk = res.GridMin + k*res.GridStep;
        res.GridPrices[k] = static_cast<float>(xk);
        double sum = 0.0;
        for (int i = 0; i < M; ++i)
        {
            double d = xk - prices[i];
            double arg = d*d*inv2hsq;
            if (arg < 18.0) sum += weights[i]*exp(-arg);
        }
        res.GridDensity[k] = static_cast<float>(sum*norm);
        if (res.GridDensity[k] > res.MaxDensity)
        {
            res.MaxDensity = res.GridDensity[k];
            pocIdx = k;
        }
    }
    if (res.MaxDensity <= 0.0f) return res;

    res.POC = res.GridPrices[pocIdx];

    // Peak detection
    const int minDist = sjMax(2, static_cast<int>(round((h*0.7)/res.GridStep)));
    struct RawPk { int idx; float price; float density; };
    std::vector<RawPk> cands;

    for (int k = 1; k < nPts-1; ++k)
    {
        if (res.GridDensity[k] > res.GridDensity[k-1] && res.GridDensity[k] > res.GridDensity[k+1])
        {
            float ip = res.GridPrices[k];
            double den = res.GridDensity[k-1] - 2.0*res.GridDensity[k] + res.GridDensity[k+1];
            if (fabs(den) > 1e-15)
            {
                double off = 0.5*(res.GridDensity[k-1]-res.GridDensity[k+1])/den;
                off = sjClamp(off, -0.5, 0.5);
                ip = static_cast<float>(res.GridPrices[k] + off*res.GridStep);
            }
            cands.push_back({k, ip, res.GridDensity[k]});
        }
    }
    if (cands.empty())
        cands.push_back({pocIdx, res.GridPrices[pocIdx], res.GridDensity[pocIdx]});

    // Suppress micro-ripples
    std::vector<RawPk> filtered;
    for (size_t i = 0; i < cands.size(); ++i)
    {
        bool dom = true;
        for (size_t j = 0; j < cands.size(); ++j)
        {
            if (i != j && abs(cands[i].idx-cands[j].idx) <= minDist &&
                cands[j].density > cands[i].density) { dom = false; break; }
        }
        if (dom) filtered.push_back(cands[i]);
    }

    // Prominence
    const float minPT = (minPromPct/100.0f)*res.MaxDensity;
    for (size_t p = 0; p < filtered.size(); ++p)
    {
        const int k = filtered[p].idx;
        const float pD = filtered[p].density;
        float lMin = pD;
        for (int l = k-1; l >= 0; --l) { if (res.GridDensity[l]>pD) break; if (res.GridDensity[l]<lMin) lMin=res.GridDensity[l]; }
        float rMin = pD;
        for (int r = k+1; r < nPts; ++r) { if (res.GridDensity[r]>pD) break; if (res.GridDensity[r]<rMin) rMin=res.GridDensity[r]; }
        float prom = pD - sjMax(lMin, rMin);
        float pr = sjClamp(prom/res.MaxDensity, 0.0f, 1.0f);
        if (prom >= minPT || (pD >= 0.30f*res.MaxDensity && pr >= 0.015f))
        {
            s_Level lv;
            lv.Price = filtered[p].price;
            lv.Density = pD;
            lv.Prominence = sjClamp(sjMax(pr, pD/res.MaxDensity*0.35f), 0.05f, 1.0f);
            lv.IsHVN = true;
            res.HVNs.push_back(lv);
        }
    }

    // LVN detection
    std::vector<s_Level> byP = res.HVNs;
    std::sort(byP.begin(), byP.end(), [](const s_Level& a, const s_Level& b){ return a.Price<b.Price; });
    for (size_t i = 0; i+1 < byP.size(); ++i)
    {
        int iA = sjClamp(static_cast<int>(round((byP[i].Price-res.GridMin)/res.GridStep)), 0, nPts-1);
        int iB = sjClamp(static_cast<int>(round((byP[i+1].Price-res.GridMin)/res.GridStep)), 0, nPts-1);
        if (iA > iB) std::swap(iA, iB);
        int mI = iA; float mD = res.GridDensity[iA];
        for (int m = iA+1; m < iB; ++m) if (res.GridDensity[m]<mD) { mD=res.GridDensity[m]; mI=m; }
        float fl = sjMin(byP[i].Density, byP[i+1].Density);
        float dp = fl - mD;
        if (dp > 0.0f && mI > iA && mI < iB)
        {
            s_Level lv; lv.Price = res.GridPrices[mI]; lv.Density = mD;
            lv.Prominence = sjClamp(dp/res.MaxDensity, 0.05f, 1.0f);
            lv.IsHVN = false;
            res.LVNs.push_back(lv);
        }
    }

    std::sort(res.HVNs.begin(), res.HVNs.end(), [](const s_Level& a, const s_Level& b){ return a.Prominence>b.Prominence; });
    std::sort(res.LVNs.begin(), res.LVNs.end(), [](const s_Level& a, const s_Level& b){ return a.Prominence>b.Prominence; });
    return res;
}

// =========================================================================
// 4. COLOR HELPER
// =========================================================================

static uint32_t GradientColor(COLORREF base, float t, int mode)
{
    if (mode == 2) return base;
    uint8_t r = base&0xFF, g = (base>>8)&0xFF, b = (base>>16)&0xFF;
    t = sjClamp(t, 0.0f, 1.0f);
    if (mode == 0)
    {
        float dk = 0.70f+0.30f*(1.0f-t), lt = (1.0f-t)*0.60f;
        auto mx = [&](uint8_t c){ return static_cast<uint8_t>(sjClamp(c*dk*(1.0f-lt)+245.0f*lt, 0.0f, 255.0f)); };
        return RGB(mx(r),mx(g),mx(b));
    }
    float f = 0.35f+0.65f*t;
    return RGB(static_cast<uint8_t>(sjClamp(r*f,0.0f,255.0f)),
               static_cast<uint8_t>(sjClamp(g*f,0.0f,255.0f)),
               static_cast<uint8_t>(sjClamp(b*f,0.0f,255.0f)));
}

// =========================================================================
// 5. DYNAMIC LEVELS — price-ordered slot assignment
// =========================================================================

static const int MAX_TRACKS = 5;

struct s_DynOutput { float HVN[MAX_TRACKS]; float LVN[MAX_TRACKS]; float Bandwidth; float POC; };

// Simple price-ordered slot assignment: sort peaks by price, assign to slots 0..N-1.
// No track continuity needed — uniform colors mean slot shifts are invisible.
static s_DynOutput ComputeDynamic(SCStudyInterfaceRef sc, int sIdx, int eIdx,
                                  float bwMult, int maxTracks)
{
    s_DynOutput out = {};
    s_Histogram hist = CollectHistogram(sc, sIdx, eIdx);
    if (hist.TickVol.size() < 3 || hist.Total <= 0.0) return out;

    s_SJResult sj = ComputeSJBandwidth(hist, static_cast<double>(sc.TickSize));
    sj.Bandwidth *= bwMult;
    sj.Bandwidth = sjMax(sj.Bandwidth, static_cast<double>(sc.TickSize));
    out.Bandwidth = static_cast<float>(sj.Bandwidth);

    s_KDEResult kde = EvaluateKDE(sj.Prices, sj.Weights, sj.Bandwidth,
                                  static_cast<double>(sc.TickSize), 3.0f);

    out.POC = kde.POC;

    // HVNs: top by prominence, then assign to slots by price order
    int nH = sjMin(maxTracks, static_cast<int>(kde.HVNs.size()));
    std::vector<float> hp;
    for (int i = 0; i < nH; ++i) hp.push_back(kde.HVNs[i].Price);
    std::sort(hp.begin(), hp.end());
    for (int i = 0; i < nH; ++i) out.HVN[i] = hp[i];

    // LVNs: same approach
    int nL = sjMin(maxTracks, static_cast<int>(kde.LVNs.size()));
    std::vector<float> lp;
    for (int i = 0; i < nL; ++i) lp.push_back(kde.LVNs[i].Price);
    std::sort(lp.begin(), lp.end());
    for (int i = 0; i < nL; ++i) out.LVN[i] = lp[i];

    return out;
}

// =========================================================================
// 6. BAR RANGE SELECTION
// =========================================================================

static void SelectBarRange(SCStudyInterfaceRef sc, int scope, int nBarsBack,
                           int& sBar, int& eBar)
{
    const int last = sc.ArraySize - 1;
    eBar = last; sBar = 0;
    if (scope == 0) // Current Session — walks back to session boundary, or bar 0 if none found
    {
        const int today = sc.GetTradingDayDate(sc.BaseDateTimeIn[last]);
        sBar = last;
        while (sBar > 0 && sc.GetTradingDayDate(sc.BaseDateTimeIn[sBar-1]) == today) --sBar;
    }
    else if (scope == 1) sBar = sjMax(0, last - nBarsBack + 1);
    else if (scope == 2) sBar = 0;
    else if (scope == 3)
    {
        const int today = sc.GetTradingDayDate(sc.BaseDateTimeIn[last]);
        int pe = last;
        while (pe >= 0 && sc.GetTradingDayDate(sc.BaseDateTimeIn[pe]) == today) --pe;
        if (pe >= 0)
        {
            eBar = pe;
            const int pd = sc.GetTradingDayDate(sc.BaseDateTimeIn[pe]);
            sBar = pe;
            while (sBar > 0 && sc.GetTradingDayDate(sc.BaseDateTimeIn[sBar-1]) == pd) --sBar;
        }
    }
}

// Find prior session bar range for prior-session levels overlay
static bool FindPriorSession(SCStudyInterfaceRef sc, int curSessionStart,
                             int& priorStart, int& priorEnd)
{
    if (curSessionStart <= 0) return false;
    priorEnd = curSessionStart - 1;
    const int pd = sc.GetTradingDayDate(sc.BaseDateTimeIn[priorEnd]);
    priorStart = priorEnd;
    while (priorStart > 0 && sc.GetTradingDayDate(sc.BaseDateTimeIn[priorStart-1]) == pd)
        --priorStart;
    return true;
}

// =========================================================================
// 7. MAIN STUDY
// =========================================================================

SCSFExport scsf_SheatherJonesVolumeProfile(SCStudyInterfaceRef sc)
{
    SCSubgraphRef Sub_BW = sc.Subgraph[10];
    SCSubgraphRef Sub_POC = sc.Subgraph[11];

    SCInputRef In_Scope        = sc.Input[0];
    SCInputRef In_NumBars      = sc.Input[1];
    SCInputRef In_MinProm      = sc.Input[2];
    SCInputRef In_MaxHVN       = sc.Input[3];
    SCInputRef In_MaxLVN       = sc.Input[4];
    SCInputRef In_LineType     = sc.Input[5];
    SCInputRef In_HVNColor     = sc.Input[6];
    SCInputRef In_LVNColor     = sc.Input[7];
    SCInputRef In_LineWidth    = sc.Input[8];
    SCInputRef In_ScaleWidth   = sc.Input[9];
    SCInputRef In_GradMode     = sc.Input[10];
    SCInputRef In_UseTrans     = sc.Input[11];
    SCInputRef In_CalcSec      = sc.Input[12];
    SCInputRef In_BWMult       = sc.Input[13];
    SCInputRef In_PlotKDE      = sc.Input[14];
    SCInputRef In_KDEColor     = sc.Input[15];
    SCInputRef In_KDETrans     = sc.Input[16];
    SCInputRef In_KDEWidth     = sc.Input[17];
    SCInputRef In_BWMode       = sc.Input[18];
    SCInputRef In_FixedBW      = sc.Input[19];
    SCInputRef In_KDEStyle     = sc.Input[20];
    SCInputRef In_KDEPlace     = sc.Input[21];
    SCInputRef In_KDEOffset    = sc.Input[22];
    SCInputRef In_KDELineW     = sc.Input[23];
    SCInputRef In_LblPos       = sc.Input[24];
    SCInputRef In_LblFont      = sc.Input[25];
    SCInputRef In_DynMode      = sc.Input[26];
    SCInputRef In_DynWindow    = sc.Input[27];
    SCInputRef In_DynHistory   = sc.Input[28];
    SCInputRef In_DynLevels    = sc.Input[29];
    SCInputRef In_DrawStatic   = sc.Input[30];
    SCInputRef In_ShowPrior    = sc.Input[31];
    SCInputRef In_KDEWidthMode = sc.Input[32];

    int& r_LastDrawn = sc.GetPersistentInt(1);
    SCDateTime& r_LastTime = sc.GetPersistentSCDateTime(2);

    if (sc.SetDefaults)
    {
        sc.GraphName = "Sheather-Jones Volume Profile";
        sc.AutoLoop = 0;
        sc.GraphRegion = 0;
        sc.CalculationPrecedence = LOW_PREC_LEVEL;

        const char* hn[5] = {"Dynamic HVN 1","Dynamic HVN 2","Dynamic HVN 3","Dynamic HVN 4","Dynamic HVN 5"};
        for (int t = 0; t < 5; ++t)
        {
            sc.Subgraph[t].Name = hn[t]; sc.Subgraph[t].DrawStyle = DRAWSTYLE_LINE_SKIP_ZEROS;
            sc.Subgraph[t].PrimaryColor = RGB(0,185,90); sc.Subgraph[t].LineWidth = 2;
            sc.Subgraph[t].DrawZeros = false;
        }
        const char* ln[5] = {"Dynamic LVN 1","Dynamic LVN 2","Dynamic LVN 3","Dynamic LVN 4","Dynamic LVN 5"};
        for (int t = 0; t < 5; ++t)
        {
            sc.Subgraph[5+t].Name = ln[t]; sc.Subgraph[5+t].DrawStyle = DRAWSTYLE_DASH;
            sc.Subgraph[5+t].PrimaryColor = RGB(225,45,45); sc.Subgraph[5+t].LineWidth = 1;
            sc.Subgraph[5+t].DrawZeros = false;
        }
        Sub_BW.Name = "Bandwidth"; Sub_BW.DrawStyle = DRAWSTYLE_IGNORE;

        Sub_POC.Name = "POC"; Sub_POC.DrawStyle = DRAWSTYLE_LINE_SKIP_ZEROS;
        Sub_POC.PrimaryColor = RGB(255,215,0); Sub_POC.LineWidth = 3; Sub_POC.DrawZeros = false;

        In_Scope.Name = "Profile Scope";
        In_Scope.SetCustomInputStrings("Current Session;Bars Back;Entire Chart;Prior Session");
        In_Scope.SetCustomInputIndex(0);
        In_NumBars.Name = "Bars Back"; In_NumBars.SetInt(390); In_NumBars.SetIntLimits(10,50000);
        In_MinProm.Name = "Min Prominence %"; In_MinProm.SetFloat(3.0f); In_MinProm.SetFloatLimits(0.1f,100.0f);
        In_MaxHVN.Name = "Max HVN Lines"; In_MaxHVN.SetInt(6); In_MaxHVN.SetIntLimits(1,25);
        In_MaxLVN.Name = "Max LVN Lines"; In_MaxLVN.SetInt(4); In_MaxLVN.SetIntLimits(0,25);
        In_LineType.Name = "Line Type (unused)"; In_LineType.SetCustomInputStrings("Reserved"); In_LineType.SetCustomInputIndex(0); // Preserved to keep Input[] indices stable
        In_HVNColor.Name = "HVN Color"; In_HVNColor.SetColor(RGB(0,185,90));
        In_LVNColor.Name = "LVN Color"; In_LVNColor.SetColor(RGB(225,45,45));
        In_LineWidth.Name = "Line Width"; In_LineWidth.SetInt(2); In_LineWidth.SetIntLimits(1,10);
        In_ScaleWidth.Name = "Scale Width by Prominence"; In_ScaleWidth.SetYesNo(1);
        In_GradMode.Name = "Color Gradient"; In_GradMode.SetCustomInputStrings("Darker=Strongest;Brighter=Strongest;Solid"); In_GradMode.SetCustomInputIndex(0);
        In_UseTrans.Name = "Transparency Gradient"; In_UseTrans.SetYesNo(1);
        In_CalcSec.Name = "Recalc Interval (s)"; In_CalcSec.SetInt(15); In_CalcSec.SetIntLimits(1,300);
        In_BWMult.Name = "Bandwidth Multiplier"; In_BWMult.SetFloat(1.0f); In_BWMult.SetFloatLimits(0.1f,5.0f);
        In_PlotKDE.Name = "Plot KDE Profile"; In_PlotKDE.SetYesNo(1);
        In_KDEColor.Name = "KDE Color"; In_KDEColor.SetColor(RGB(70,130,180));
        In_KDETrans.Name = "KDE Transparency %"; In_KDETrans.SetInt(50); In_KDETrans.SetIntLimits(0,95);
        In_KDEWidth.Name = "KDE Width (% or bars)"; In_KDEWidth.SetInt(15); In_KDEWidth.SetIntLimits(1,300);
        In_BWMode.Name = "Bandwidth Mode"; In_BWMode.SetCustomInputStrings("Sheather-Jones;Fixed"); In_BWMode.SetCustomInputIndex(0);
        In_FixedBW.Name = "Fixed Bandwidth (pts)"; In_FixedBW.SetFloat(5.0f); In_FixedBW.SetFloatLimits(0.25f,500.0f);
        In_KDEStyle.Name = "KDE Style"; In_KDEStyle.SetCustomInputStrings("Envelope;Filled;Both"); In_KDEStyle.SetCustomInputIndex(0);
        In_KDEPlace.Name = "KDE Placement"; In_KDEPlace.SetCustomInputStrings("Right Margin;Right Edge;Over Price"); In_KDEPlace.SetCustomInputIndex(0);
        In_KDEOffset.Name = "KDE Offset (bars)"; In_KDEOffset.SetInt(6); In_KDEOffset.SetIntLimits(0,300);
        In_KDELineW.Name = "KDE Line Width"; In_KDELineW.SetInt(2); In_KDELineW.SetIntLimits(1,10);
        In_LblPos.Name = "Labels"; In_LblPos.SetCustomInputStrings("Right;Left;Hidden"); In_LblPos.SetCustomInputIndex(0);
        In_LblFont.Name = "Label Font Size"; In_LblFont.SetInt(8); In_LblFont.SetIntLimits(6,24);
        In_DynMode.Name = "Dynamic Mode"; In_DynMode.SetCustomInputStrings("Rolling;Session;Entire Chart;Disabled"); In_DynMode.SetCustomInputIndex(1);
        In_DynWindow.Name = "Dynamic Window (bars)"; In_DynWindow.SetInt(60); In_DynWindow.SetIntLimits(5,2000);
        In_DynHistory.Name = "Dynamic History (bars)"; In_DynHistory.SetInt(500); In_DynHistory.SetIntLimits(20,10000);
        In_DynLevels.Name = "Dynamic Levels (1-5)"; In_DynLevels.SetInt(3); In_DynLevels.SetIntLimits(1,5);
        In_DrawStatic.Name = "Draw Static Levels"; In_DrawStatic.SetYesNo(1);
        In_ShowPrior.Name = "Show Prior Session POC"; In_ShowPrior.SetYesNo(1);
        In_KDEWidthMode.Name = "KDE Width Mode"; In_KDEWidthMode.SetCustomInputStrings("% of Visible Chart;Fixed Bars"); In_KDEWidthMode.SetCustomInputIndex(0);
        return;
    }

    const int MaxDrawn = 300;
    const int BaseLn = 800000 + (sc.StudyGraphInstanceID * 310);

    if (sc.LastCallToFunction)
    {
        for (int i = 0; i < MaxDrawn; ++i)
            sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, BaseLn+i);
        r_LastDrawn = 0;
        return;
    }

    // Rate-limit
    const int interval = sjMax(1, In_CalcSec.GetInt());
    SCDateTime now = sc.CurrentSystemDateTime;
    if (r_LastTime.GetTimeInSeconds() > 0 &&
        (now.GetTimeInSeconds() - r_LastTime.GetTimeInSeconds()) < interval &&
        sc.UpdateStartIndex > 0) return;
    r_LastTime = now;

    if (sc.ArraySize < 1) return;
    if (!sc.VolumeAtPriceForBars)
    {
        sc.AddMessageToLog("SJ: No Volume-At-Price data. Use an Intraday chart.", 1);
        return;
    }

    const float bwMult = sjClamp(In_BWMult.GetFloat(), 0.1f, 5.0f);

    // Early exit if both dynamic and static are disabled
    const int dynMode = In_DynMode.GetIndex();
    if (dynMode == 3 && !In_DrawStatic.GetYesNo())
    {
        for (int i = 0; i < r_LastDrawn && i < MaxDrawn; ++i)
            sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, BaseLn+i);
        r_LastDrawn = 0;
        return;
    }

    // =================================================================
    // A. DYNAMIC LEVELS
    // =================================================================
    if (dynMode != 3)
    {
        const int winBars  = sjMax(5, In_DynWindow.GetInt());
        const int maxHist  = sjMax(20, In_DynHistory.GetInt());
        const int maxT     = sjClamp(In_DynLevels.GetInt(), 1, 5);

        int calcStart = 0;
        if (sc.UpdateStartIndex == 0)
        {
            calcStart = sjMax(0, sc.ArraySize - maxHist);
            for (int i = 0; i < calcStart; ++i)
            {
                for (int t = 0; t < 5; ++t) { sc.Subgraph[t][i]=0; sc.Subgraph[5+t][i]=0; }
                Sub_BW[i]=0; Sub_POC[i]=0;
            }
        }
        else calcStart = sjMax(0, sc.UpdateStartIndex);

        for (int bar = calcStart; bar < sc.ArraySize; ++bar)
        {
            int sIdx = 0;
            if (dynMode == 0) sIdx = sjMax(0, bar-winBars+1);
            else if (dynMode == 1) // Session Developing
            {
                const int day = sc.GetTradingDayDate(sc.BaseDateTimeIn[bar]);
                sIdx = bar;
                while (sIdx > 0 && sc.GetTradingDayDate(sc.BaseDateTimeIn[sIdx-1]) == day) --sIdx;
                // If session search found only this bar (e.g. daily/weekly chart), expand to chart start
                if (sIdx == bar) sIdx = 0;
            }

            s_DynOutput out = ComputeDynamic(sc, sIdx, bar, bwMult, maxT);

            for (int t = 0; t < 5; ++t)
            {
                sc.Subgraph[t][bar]   = (t<maxT) ? out.HVN[t] : 0.0f;
                sc.Subgraph[5+t][bar] = (t<maxT) ? out.LVN[t] : 0.0f;
            }
            Sub_BW[bar] = out.Bandwidth;
            Sub_POC[bar] = out.POC;
        }
    }
    else if (sc.UpdateStartIndex == 0)
    {
        for (int i = 0; i < sc.ArraySize; ++i)
        {
            for (int t = 0; t < 5; ++t) { sc.Subgraph[t][i]=0; sc.Subgraph[5+t][i]=0; }
            Sub_BW[i]=0; Sub_POC[i]=0;
        }
    }

    // =================================================================
    // B. STATIC LEVELS + KDE PROFILE
    // =================================================================
    if (!In_DrawStatic.GetYesNo())
    {
        for (int i = 0; i < r_LastDrawn && i < MaxDrawn; ++i)
            sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, BaseLn+i);
        r_LastDrawn = 0;
        return;
    }

    int sBar = 0, eBar = sc.ArraySize - 1;
    SelectBarRange(sc, In_Scope.GetIndex(), In_NumBars.GetInt(), sBar, eBar);

    s_Histogram hist = CollectHistogram(sc, sBar, eBar);
    if (In_Scope.GetIndex() == 0 && hist.TickVol.size() < 5 && sBar > 0)
    {
        const int pd = sc.GetTradingDayDate(sc.BaseDateTimeIn[sBar-1]);
        int ex = sBar-1;
        while (ex > 0 && sc.GetTradingDayDate(sc.BaseDateTimeIn[ex-1]) == pd) --ex;
        sBar = ex;
        hist = CollectHistogram(sc, sBar, eBar);
    }
    if (hist.TickVol.size() < 5 || hist.Total <= 0.0) return;

    s_SJResult sj = ComputeSJBandwidth(hist, static_cast<double>(sc.TickSize));
    double hFinal = (In_BWMode.GetIndex() == 1) ? static_cast<double>(In_FixedBW.GetFloat()) : sj.Bandwidth;
    hFinal *= bwMult;
    hFinal = sjMax(hFinal, static_cast<double>(sc.TickSize));

    s_KDEResult kde = EvaluateKDE(sj.Prices, sj.Weights, hFinal,
                                  static_cast<double>(sc.TickSize), In_MinProm.GetFloat());
    if (kde.MaxDensity <= 0.0f) return;

    const int nH = sjMin(In_MaxHVN.GetInt(), static_cast<int>(kde.HVNs.size()));
    const int nL = sjMin(In_MaxLVN.GetInt(), static_cast<int>(kde.LVNs.size()));

    // =================================================================
    // B4. DRAWING
    // =================================================================
    int li = 0; // line index

    // Time projection — dual mode for KDE width
    int uW = sjClamp(In_KDEWidth.GetInt(), 1, 300);
    double profileSpan = 0.0;
    double eW = 35.0; // effective width in "drawing units" for KC lambda

    if (In_KDEWidthMode.GetIndex() == 0) // % of Visible Chart
    {
        int fV = sjClamp(sc.IndexOfFirstVisibleBar, 0, sc.ArraySize-1);
        int lV = sjClamp(sc.IndexOfLastVisibleBar, 0, sc.ArraySize-1);
        if (lV <= fV) { fV = sjMax(0, sc.ArraySize-80); lV = sc.ArraySize-1; }
        double visSpan = sc.BaseDateTimeIn[lV].GetAsDouble() - sc.BaseDateTimeIn[fV].GetAsDouble();
        if (visSpan <= 1e-7)
        {
            int nV = sjMax(1, lV-fV);
            visSpan = (sc.SecondsPerBar > 0) ? (nV * sc.SecondsPerBar / 86400.0) : (nV / 1440.0);
        }
        double pct = sjClamp(uW / 100.0, 0.01, 1.0);
        profileSpan = visSpan * pct;
    }
    else // Fixed Bars — look up actual bar datetimes
    {
        int lookbackBar = sjMax(0, eBar - uW);
        profileSpan = sc.BaseDateTimeIn[eBar].GetAsDouble() - sc.BaseDateTimeIn[lookbackBar].GetAsDouble();
        int actualBars = eBar - lookbackBar;
        if (actualBars < 1 || profileSpan <= 1e-7)
        {
            double fb = (sc.SecondsPerBar > 0) ? (sc.SecondsPerBar / 86400.0) : (1.0 / 1440.0);
            profileSpan = uW * fb;
        }
    }

    // dpb: one "eW unit" of time. eW is fixed at 35 internal units for the KC lambda math.
    eW = 35.0;
    double fDpb = profileSpan / eW;
    if (fDpb <= 1e-10) fDpb = 1.0 / 1440.0;

    auto OD = [&](const SCDateTime& a, double off) -> SCDateTime {
        return SCDateTime(a.GetAsDouble() + off * fDpb);
    };

    int mOff = sjClamp(In_KDEOffset.GetInt(), 0, 100);

    // Diagnostic
    {
        SCString diag;
        diag.Format("SJ: N=%.0f M=%d h=%.4f POC=%.2f HVNs=%d LVNs=%d bars=[%d..%d] ArraySize=%d widthMode=%d profileSpan=%.4f fDpb=%.6f gridPts=%d priceRange=[%.2f,%.2f]",
                    sj.N, static_cast<int>(sj.Prices.size()), hFinal,
                    kde.POC, nH, nL, sBar, eBar, sc.ArraySize,
                    In_KDEWidthMode.GetIndex(), profileSpan, fDpb,
                    static_cast<int>(kde.GridPrices.size()),
                    sj.Prices.empty() ? 0.0 : sj.Prices.front(),
                    sj.Prices.empty() ? 0.0 : sj.Prices.back());
        sc.AddMessageToLog(diag, 0);
    }

    // Compute the KDE baseline datetime — this is where static lines will originate from
    const SCDateTime lastBarDT = sc.BaseDateTimeIn[eBar];
    SCDateTime kdeBaseDT = lastBarDT; // fallback
    {
        int pl = sjClampI(In_KDEPlace.GetIndex(), 0, 2);
        if (pl == 0)      kdeBaseDT = OD(lastBarDT, mOff);
        else if (pl == 1) kdeBaseDT = OD(lastBarDT, mOff + eW);
        else              kdeBaseDT = lastBarDT;
    }

    // Right edge for static lines: extend far into forward space to reach chart edge
    SCDateTime lineRightDT = OD(lastBarDT, mOff + eW + 1000);

    auto KC = [&](float ratio, const SCDateTime& last, SCDateTime& bDT, SCDateTime& tDT)
    {
        int pl = sjClampI(In_KDEPlace.GetIndex(), 0, 2);
        if (pl==0) { bDT=OD(last,mOff); tDT=OD(last,mOff+ratio*eW); }
        else if (pl==1) { bDT=OD(last,mOff+eW); tDT=OD(last,mOff+eW-ratio*eW); }
        else { bDT=last; tDT=OD(last,-ratio*eW); }
    };

    // --- KDE Profile ---
    if (In_PlotKDE.GetYesNo() && kde.MaxDensity > 0.0f)
    {
        const int st = sjClampI(In_KDEStyle.GetIndex(), 0, 2);
        const COLORREF kc = In_KDEColor.GetColor();
        const int kT = sjClamp(In_KDETrans.GetInt(), 0, 95);
        const int kW = sjClamp(In_KDELineW.GetInt(), 1, 10);
        const SCDateTime lastDT = sc.BaseDateTimeIn[eBar];
        const int nG = static_cast<int>(kde.GridPrices.size());

        if (st==0 || st==2)
        {
            const int segs = sjMin(180, nG);
            SCDateTime pTip; float pPr=0; bool hP=false;
            SCDateTime fB,fT,lB,lT; float fP=0,lP=0;
            for (int s = 0; s < segs && li < 200; ++s)
            {
                int gi = sjClamp(static_cast<int>(round(s*(nG-1)/static_cast<double>(segs-1))), 0, nG-1);
                float r = sjClamp(kde.GridDensity[gi]/kde.MaxDensity, 0.0f, 1.0f);
                float pr = kde.GridPrices[gi];
                SCDateTime bD,tD; KC(r, lastDT, bD, tD);
                if (s==0){fB=bD;fT=tD;fP=pr;} lB=bD;lT=tD;lP=pr;
                if (hP)
                {
                    s_UseTool T; T.Clear(); T.ChartNumber=sc.ChartNumber; T.DrawingType=DRAWING_LINE;
                    T.LineNumber=BaseLn+li++; T.BeginDateTime=pTip; T.BeginValue=pPr;
                    T.EndDateTime=tD; T.EndValue=pr; T.Color=kc; T.TransparencyLevel=kT;
                    T.LineWidth=kW; T.LineStyle=LINESTYLE_SOLID; T.AddMethod=UTAM_ADD_OR_ADJUST;
                    sc.UseTool(T);
                }
                pTip=tD; pPr=pr; hP=true;
            }
            if (hP && li+3 < 200)
            {
                int cT = sjMin(95, kT+15);
                s_UseTool T; T.Clear(); T.ChartNumber=sc.ChartNumber; T.DrawingType=DRAWING_LINE;
                T.LineNumber=BaseLn+li++; T.BeginDateTime=fB; T.BeginValue=fP;
                T.EndDateTime=lB; T.EndValue=lP; T.Color=kc; T.TransparencyLevel=cT;
                T.LineWidth=1; T.LineStyle=LINESTYLE_DASH; T.AddMethod=UTAM_ADD_OR_ADJUST;
                sc.UseTool(T);
                T.Clear(); T.ChartNumber=sc.ChartNumber; T.DrawingType=DRAWING_LINE;
                T.LineNumber=BaseLn+li++; T.BeginDateTime=fB; T.BeginValue=fP;
                T.EndDateTime=fT; T.EndValue=fP; T.Color=kc; T.TransparencyLevel=cT;
                T.LineWidth=1; T.LineStyle=LINESTYLE_SOLID; T.AddMethod=UTAM_ADD_OR_ADJUST;
                sc.UseTool(T);
                T.Clear(); T.ChartNumber=sc.ChartNumber; T.DrawingType=DRAWING_LINE;
                T.LineNumber=BaseLn+li++; T.BeginDateTime=lB; T.BeginValue=lP;
                T.EndDateTime=lT; T.EndValue=lP; T.Color=kc; T.TransparencyLevel=cT;
                T.LineWidth=1; T.LineStyle=LINESTYLE_SOLID; T.AddMethod=UTAM_ADD_OR_ADJUST;
                sc.UseTool(T);
            }
        }
        if (st==1 || st==2)
        {
            const int sl = (st==2) ? sjMin(60,nG) : sjMin(160,nG);
            for (int s = 0; s < sl && li < 240; ++s)
            {
                int gi = sjClamp(static_cast<int>(round(s*(nG-1)/static_cast<double>(sl-1))), 0, nG-1);
                float r = kde.GridDensity[gi]/kde.MaxDensity;
                if (r < 0.02f) continue;
                SCDateTime bD,tD; KC(r, lastDT, bD, tD);
                if (bD > tD) std::swap(bD,tD);
                s_UseTool T; T.Clear(); T.ChartNumber=sc.ChartNumber; T.DrawingType=DRAWING_LINE;
                T.LineNumber=BaseLn+li++; T.BeginDateTime=bD; T.BeginValue=kde.GridPrices[gi];
                T.EndDateTime=tD; T.EndValue=kde.GridPrices[gi]; T.Color=kc; T.TransparencyLevel=kT;
                T.LineWidth=3; T.LineStyle=LINESTYLE_SOLID; T.AddMethod=UTAM_ADD_OR_ADJUST;
                sc.UseTool(T);
            }
        }

        // Bandwidth label on KDE profile
        if (li < MaxDrawn - 1)
        {
            SCDateTime bD, tD;
            KC(1.0f, lastDT, bD, tD);
            s_UseTool T; T.Clear(); T.ChartNumber = sc.ChartNumber;
            T.DrawingType = DRAWING_TEXT;
            T.LineNumber = BaseLn + li++;
            T.BeginDateTime = tD;
            T.BeginValue = kde.POC;
            T.Color = kc;
            T.FontSize = 7;
            T.Text.Format("h=%.3f", hFinal);
            T.AddMethod = UTAM_ADD_OR_ADJUST;
            sc.UseTool(T);
        }
    }

    // --- Static POC line ---
    const int gM = sjClampI(In_GradMode.GetIndex(), 0, 2);
    const bool uT = In_UseTrans.GetYesNo() != 0;
    const int lPos = sjClampI(In_LblPos.GetIndex(), 0, 2);
    const int lFnt = sjClamp(In_LblFont.GetInt(), 6, 24);

    if (li < MaxDrawn)
    {
        s_UseTool T; T.Clear(); T.ChartNumber=sc.ChartNumber; T.DrawingType=DRAWING_LINE;
        T.LineNumber=BaseLn+li++; T.BeginValue=kde.POC; T.EndValue=kde.POC;
        T.BeginDateTime=lineRightDT; T.EndDateTime=kdeBaseDT;
        T.Color=RGB(255,215,0); T.LineWidth=3; T.LineStyle=LINESTYLE_SOLID;
        if (lPos!=2) { T.TransparentLabelBackground=1; T.FontSize=lFnt;
            T.TextAlignment=(lPos==0)?(DT_RIGHT|DT_TOP):(DT_LEFT|DT_TOP);
            T.DisplayHorizontalLineValue=0; T.Text.Format("POC %.2f", kde.POC); }
        T.AddMethod=UTAM_ADD_OR_ADJUST; sc.UseTool(T);
    }

    // --- Prior session POC overlay ---
    if (In_ShowPrior.GetYesNo() && In_Scope.GetIndex() == 0)
    {
        int pS, pE;
        if (FindPriorSession(sc, sBar, pS, pE))
        {
            s_Histogram ph = CollectHistogram(sc, pS, pE);
            if (ph.TickVol.size() >= 5 && ph.Total > 0.0)
            {
                s_SJResult psj = ComputeSJBandwidth(ph, static_cast<double>(sc.TickSize));
                double pbw = psj.Bandwidth * bwMult;
                pbw = sjMax(pbw, static_cast<double>(sc.TickSize));
                s_KDEResult pk = EvaluateKDE(psj.Prices, psj.Weights, pbw,
                                             static_cast<double>(sc.TickSize), In_MinProm.GetFloat());
                if (pk.MaxDensity > 0.0f)
                {
                    // Prior POC
                    if (li < MaxDrawn) {
                        s_UseTool T; T.Clear(); T.ChartNumber=sc.ChartNumber; T.DrawingType=DRAWING_LINE;
                        T.LineNumber=BaseLn+li++; T.BeginValue=pk.POC; T.EndValue=pk.POC;
                        T.BeginDateTime=lineRightDT; T.EndDateTime=kdeBaseDT;
                        T.Color=RGB(200,180,60); T.TransparencyLevel=40; T.LineWidth=2; T.LineStyle=LINESTYLE_DOT;
                        if (lPos!=2) { T.TransparentLabelBackground=1; T.FontSize=lFnt-1;
                            T.TextAlignment=(lPos==0)?(DT_RIGHT|DT_TOP):(DT_LEFT|DT_TOP);
                            T.DisplayHorizontalLineValue=0; T.Text.Format("pPOC %.2f", pk.POC); }
                        T.AddMethod=UTAM_ADD_OR_ADJUST; sc.UseTool(T);
                    }
                }
            }
        }
    }

    // --- HVN lines ---
    for (int i = 0; i < nH && li < MaxDrawn; ++i, ++li)
    {
        const s_Level& h = kde.HVNs[i];
        s_UseTool T; T.Clear(); T.ChartNumber=sc.ChartNumber; T.DrawingType=DRAWING_LINE;
        T.LineNumber=BaseLn+li; T.BeginValue=h.Price; T.EndValue=h.Price;
        T.BeginDateTime=lineRightDT; T.EndDateTime=kdeBaseDT;
        T.Color=GradientColor(In_HVNColor.GetColor(), h.Prominence, gM);
        if (uT) T.TransparencyLevel=sjClamp(static_cast<int>((1.0f-h.Prominence)*65.0f), 0, 80);
        int w = In_LineWidth.GetInt();
        if (In_ScaleWidth.GetYesNo()) w=sjClamp(static_cast<int>(round(w*(0.6f+0.8f*h.Prominence))), 1, 6);
        T.LineWidth=w; T.LineStyle=LINESTYLE_SOLID;
        if (lPos!=2) { T.TransparentLabelBackground=1; T.FontSize=lFnt;
            T.TextAlignment=(lPos==0)?(DT_RIGHT|DT_BOTTOM):(DT_LEFT|DT_BOTTOM);
            T.DisplayHorizontalLineValue=0; T.Text.Format("HVN %.2f (%.0f%%)", h.Price, h.Prominence*100); }
        else { T.Text=""; T.DisplayHorizontalLineValue=0; }
        T.AddMethod=UTAM_ADD_OR_ADJUST; sc.UseTool(T);
    }

    // --- LVN lines ---
    for (int i = 0; i < nL && li < MaxDrawn; ++i, ++li)
    {
        const s_Level& l = kde.LVNs[i];
        s_UseTool T; T.Clear(); T.ChartNumber=sc.ChartNumber; T.DrawingType=DRAWING_LINE;
        T.LineNumber=BaseLn+li; T.BeginValue=l.Price; T.EndValue=l.Price;
        T.BeginDateTime=lineRightDT; T.EndDateTime=kdeBaseDT;
        T.Color=GradientColor(In_LVNColor.GetColor(), l.Prominence, gM);
        if (uT) T.TransparencyLevel=sjClamp(static_cast<int>((1.0f-l.Prominence)*65.0f), 0, 80);
        int w = sjMax(1, In_LineWidth.GetInt()-1);
        if (In_ScaleWidth.GetYesNo()) w=sjClamp(static_cast<int>(round(w*(0.6f+0.6f*l.Prominence))), 1, 5);
        T.LineWidth=w; T.LineStyle=LINESTYLE_DASH;
        if (lPos!=2) { T.TransparentLabelBackground=1; T.FontSize=lFnt;
            T.TextAlignment=(lPos==0)?(DT_RIGHT|DT_BOTTOM):(DT_LEFT|DT_BOTTOM);
            T.DisplayHorizontalLineValue=0; T.Text.Format("LVN %.2f (%.0f%%)", l.Price, l.Prominence*100); }
        else { T.Text=""; T.DisplayHorizontalLineValue=0; }
        T.AddMethod=UTAM_ADD_OR_ADJUST; sc.UseTool(T);
    }

    // Cleanup stale
    for (int i = li; i < r_LastDrawn && i < MaxDrawn; ++i)
        sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, BaseLn+i);
    r_LastDrawn = li;
}
