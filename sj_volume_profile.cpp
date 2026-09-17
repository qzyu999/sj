#include "sierrachart.h"
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <iterator>

SCDLLName("Sheather-Jones Volume Profile")

static const double SJ_PI = 3.14159265358979323846;

/*==========================================================================*/
// Safe min/max/clamp helpers immune to Windows/SierraChart min/max macros and mixed types
template<typename T1, typename T2>
static inline auto SJ_Min(T1 a, T2 b) -> typename std::common_type<T1, T2>::type
{
    return (a < b) ? a : b;
}

template<typename T1, typename T2>
static inline auto SJ_Max(T1 a, T2 b) -> typename std::common_type<T1, T2>::type
{
    return (a > b) ? a : b;
}

template<typename T, typename LowT, typename HighT>
static inline T SJ_Clamp(T val, LowT low, HighT high)
{
    T l = static_cast<T>(low);
    T h = static_cast<T>(high);
    return (val < l) ? l : ((val > h) ? h : val);
}

/*==========================================================================*/
// Structure to store detected levels (HVNs and LVNs)
struct s_SJLevel
{
    float Price;
    float Density;
    float Prominence; // Normalized 0.0 to 1.0
    bool IsHVN;       // true = HVN (peak), false = LVN (valley)
};

/*==========================================================================*/
// Helper function to calculate prominence-based color gradient
static uint32_t CalculateGradientColor(COLORREF BaseColor, float Prominence, int GradientMode)
{
    // GradientMode: 0 = Darker for Strongest, Lighter for Weakest
    //               1 = Brighter for Strongest, Dimmer for Weakest
    //               2 = Solid Base Color
    if (GradientMode == 2)
        return BaseColor;

    uint8_t r = BaseColor & 0xFF;
    uint8_t g = (BaseColor >> 8) & 0xFF;
    uint8_t b = (BaseColor >> 16) & 0xFF;

    float t = SJ_Clamp(Prominence, 0.0f, 1.0f);

    if (GradientMode == 0) // Darker & richer for Strongest, Lighter & softer for Weakest
    {
        float dark_scale = 0.70f + 0.30f * (1.0f - t); // 0.70 when t=1, 1.00 when t=0
        float light_blend = (1.0f - t) * 0.60f;         // 0.0 when t=1, 0.60 when t=0

        float new_r = (r * dark_scale) * (1.0f - light_blend) + 245.0f * light_blend;
        float new_g = (g * dark_scale) * (1.0f - light_blend) + 245.0f * light_blend;
        float new_b = (b * dark_scale) * (1.0f - light_blend) + 245.0f * light_blend;

        uint8_t final_r = static_cast<uint8_t>(SJ_Clamp(new_r, 0.0f, 255.0f));
        uint8_t final_g = static_cast<uint8_t>(SJ_Clamp(new_g, 0.0f, 255.0f));
        uint8_t final_b = static_cast<uint8_t>(SJ_Clamp(new_b, 0.0f, 255.0f));

        return RGB(final_r, final_g, final_b);
    }
    else // Brighter for Strongest, Dimmer for Weakest
    {
        float factor = 0.35f + 0.65f * t;
        uint8_t final_r = static_cast<uint8_t>(SJ_Clamp(r * factor, 0.0f, 255.0f));
        uint8_t final_g = static_cast<uint8_t>(SJ_Clamp(g * factor, 0.0f, 255.0f));
        uint8_t final_b = static_cast<uint8_t>(SJ_Clamp(b * factor, 0.0f, 255.0f));
        return RGB(final_r, final_g, final_b);
    }
}

// -------------------------------------------------------------------------
// Structure and helper for Dynamic Rolling / Developing Sheather-Jones KDE
// -------------------------------------------------------------------------
const int MAX_DYNAMIC_TRACKS = 5;

struct s_DynamicKDEOutput
{
    float HVN[MAX_DYNAMIC_TRACKS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float LVN[MAX_DYNAMIC_TRACKS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float Bandwidth = 0.0f;
};

static s_DynamicKDEOutput ComputeDynamicKDE(
    SCStudyInterfaceRef sc,
    int sIdx,
    int eIdx,
    float BandwidthMultiplier,
    int MaxTracks,
    std::vector<double>& PricesBuf,
    std::vector<double>& WeightsBuf,
    std::vector<double>& GridXBuf,
    std::vector<double>& DensityBuf
)
{
    s_DynamicKDEOutput fallback;

    if (sc.VolumeAtPriceForBars == nullptr || sIdx > eIdx)
        return fallback;

    std::map<int, double> VolMap;
    double TotalVol = 0.0;

    for (int b = sIdx; b <= eIdx; ++b)
    {
        const int VAPSize = sc.VolumeAtPriceForBars->GetSizeAtBarIndex(b);
        for (int v = 0; v < VAPSize; ++v)
        {
            const s_VolumeAtPriceV2* p_VAP = nullptr;
            if (!sc.VolumeAtPriceForBars->GetVAPElementAtIndex(b, v, &p_VAP) || p_VAP == nullptr)
                break;
            if (p_VAP->Volume > 0)
            {
                VolMap[p_VAP->PriceInTicks] += static_cast<double>(p_VAP->Volume);
                TotalVol += static_cast<double>(p_VAP->Volume);
            }
        }
    }

    if (VolMap.size() < 3 || TotalVol <= 0.0)
        return fallback;

    const int M = static_cast<int>(VolMap.size());
    PricesBuf.resize(M);
    WeightsBuf.resize(M);
    double WeightedMean = 0.0;
    int idx = 0;
    for (auto it = VolMap.begin(); it != VolMap.end(); ++it, ++idx)
    {
        PricesBuf[idx] = it->first * static_cast<double>(sc.TickSize);
        WeightsBuf[idx] = it->second / TotalVol;
        WeightedMean += WeightsBuf[idx] * PricesBuf[idx];
    }

    double WeightedVar = 0.0;
    double SumSqW = 0.0;
    for (int i = 0; i < M; ++i)
    {
        double diff = PricesBuf[i] - WeightedMean;
        WeightedVar += WeightsBuf[i] * diff * diff;
        SumSqW += WeightsBuf[i] * WeightsBuf[i];
    }
    double Sigma = sqrt(SJ_Max(WeightedVar, 1e-6));
    double n_eff = (SumSqW > 1e-12) ? (1.0 / SumSqW) : static_cast<double>(M);

    double CumW = 0.0;
    double Q25 = PricesBuf.front();
    double Q75 = PricesBuf.back();
    bool FoundQ25 = false;
    for (int i = 0; i < M; ++i)
    {
        CumW += WeightsBuf[i];
        if (!FoundQ25 && CumW >= 0.25)
        {
            Q25 = PricesBuf[i];
            FoundQ25 = true;
        }
        if (CumW >= 0.75)
        {
            Q75 = PricesBuf[i];
            break;
        }
    }
    double IQR = Q75 - Q25;
    double ScaleParam = (IQR > 0.0) ? SJ_Min(Sigma, IQR / 1.349) : Sigma;
    if (ScaleParam <= 0.0) ScaleParam = Sigma;

    double h_0 = pow(4.0 / (3.0 * n_eff), 0.2) * ScaleParam;
    h_0 = SJ_Max(h_0, static_cast<double>(sc.TickSize));

    double S = 0.0;
    const double h_0_sq = h_0 * h_0;
    for (int i = 0; i < M; ++i)
    {
        S += WeightsBuf[i] * WeightsBuf[i] * 0.75;
        for (int j = i + 1; j < M; ++j)
        {
            double diff = PricesBuf[i] - PricesBuf[j];
            double r_sq = (diff * diff) / h_0_sq;
            if (r_sq > 36.0) break; // data is sorted (from std::map), so all j>current are farther
            double W = exp(-r_sq / 4.0);
            double P = (r_sq * r_sq) / 16.0 - (3.0 * r_sq) / 4.0 + 0.75;
            S += 2.0 * WeightsBuf[i] * WeightsBuf[j] * W * P;
        }
    }
    S = S / (sqrt(4.0 * SJ_PI) * pow(h_0, 5));
    if (S <= 1e-12) S = 1e-12;

    double R_K = 1.0 / (2.0 * sqrt(SJ_PI));
    double h_SJ = pow(R_K / (n_eff * S), 0.2) * BandwidthMultiplier;
    h_SJ = SJ_Max(h_SJ, static_cast<double>(sc.TickSize));

    // Adaptive grid matching static profile: ensures adequate samples across each peak
    double TickSz = static_cast<double>(sc.TickSize);
    double DataMin = PricesBuf.front();
    double DataMax = PricesBuf.back();
    double DataRange = SJ_Max(DataMax - DataMin, TickSz);
    double GridMin = DataMin - 4.0 * h_SJ;
    double GridMax = DataMax + 4.0 * h_SJ;
    double GridStep = SJ_Min(DataRange / 1499.0, h_SJ / 2.5);
    GridStep = SJ_Max(GridStep, TickSz);
    int NumGridPoints = static_cast<int>((GridMax - GridMin) / GridStep) + 1;
    NumGridPoints = SJ_Clamp(NumGridPoints, 20, 2000);

    GridXBuf.resize(NumGridPoints);
    DensityBuf.assign(NumGridPoints, 0.0);
    double MaxDensity = 0.0;
    const double InvTwoHSq = 1.0 / (2.0 * h_SJ * h_SJ);
    const double NormConst = 1.0 / (sqrt(2.0 * SJ_PI) * h_SJ);

    for (int k = 0; k < NumGridPoints; ++k)
    {
        GridXBuf[k] = GridMin + k * GridStep;
        double sum = 0.0;
        for (int i = 0; i < M; ++i)
        {
            double d = GridXBuf[k] - PricesBuf[i];
            double arg = d * d * InvTwoHSq;
            if (arg < 18.0)
                sum += WeightsBuf[i] * exp(-arg);
        }
        DensityBuf[k] = sum * NormConst;
        if (DensityBuf[k] > MaxDensity) MaxDensity = DensityBuf[k];
    }

    struct Candidate {
        double Price;
        double Density;
    };
    std::vector<Candidate> RawPeaks;
    std::vector<Candidate> RawValleys;

    for (int k = 1; k < NumGridPoints - 1; ++k)
    {
        if (DensityBuf[k] > DensityBuf[k - 1] && DensityBuf[k] >= DensityBuf[k + 1])
        {
            // Parabolic interpolation for sub-grid peak resolution
            double interpPrice = GridXBuf[k];
            double interpDensity = DensityBuf[k];
            double denom = DensityBuf[k - 1] - 2.0 * DensityBuf[k] + DensityBuf[k + 1];
            if (fabs(denom) > 1e-15)
            {
                double offset = 0.5 * (DensityBuf[k - 1] - DensityBuf[k + 1]) / denom;
                offset = SJ_Clamp(offset, -0.5, 0.5);
                interpPrice = GridXBuf[k] + offset * GridStep;
                interpDensity = DensityBuf[k] - 0.25 * (DensityBuf[k - 1] - DensityBuf[k + 1]) * offset;
            }
            RawPeaks.push_back({interpPrice, interpDensity});
        }
        else if (DensityBuf[k] < DensityBuf[k - 1] && DensityBuf[k] <= DensityBuf[k + 1])
        {
            // Parabolic interpolation for sub-grid valley resolution
            double interpPrice = GridXBuf[k];
            double interpDensity = DensityBuf[k];
            double denom = DensityBuf[k - 1] - 2.0 * DensityBuf[k] + DensityBuf[k + 1];
            if (fabs(denom) > 1e-15)
            {
                double offset = 0.5 * (DensityBuf[k - 1] - DensityBuf[k + 1]) / denom;
                offset = SJ_Clamp(offset, -0.5, 0.5);
                interpPrice = GridXBuf[k] + offset * GridStep;
                interpDensity = DensityBuf[k] - 0.25 * (DensityBuf[k - 1] - DensityBuf[k + 1]) * offset;
            }
            RawValleys.push_back({interpPrice, interpDensity});
        }
    }

    // Fallback: if KDE has no local maximum, use the global max
    if (RawPeaks.empty())
    {
        int BestK = 0;
        for (int k = 1; k < NumGridPoints; ++k)
            if (DensityBuf[k] > DensityBuf[BestK]) BestK = k;
        double interpPrice = GridXBuf[BestK];
        if (BestK > 0 && BestK < NumGridPoints - 1)
        {
            double denom = DensityBuf[BestK - 1] - 2.0 * DensityBuf[BestK] + DensityBuf[BestK + 1];
            if (fabs(denom) > 1e-15)
            {
                double offset = 0.5 * (DensityBuf[BestK - 1] - DensityBuf[BestK + 1]) / denom;
                offset = SJ_Clamp(offset, -0.5, 0.5);
                interpPrice = GridXBuf[BestK] + offset * GridStep;
            }
        }
        RawPeaks.push_back({interpPrice, DensityBuf[BestK]});
    }

    // --- OUTPUT: Top K by density, assigned to tracks by price order ---
    s_DynamicKDEOutput out;
    out.Bandwidth = static_cast<float>(h_SJ);

    // HVNs: sort by density descending, take top MaxTracks, then sort by price ascending for stable track identity
    std::sort(RawPeaks.begin(), RawPeaks.end(), [](const Candidate& a, const Candidate& b) {
        return a.Density > b.Density;
    });
    int nPeaks = SJ_Min(MaxTracks, static_cast<int>(RawPeaks.size()));
    std::vector<Candidate> TopPeaks(RawPeaks.begin(), RawPeaks.begin() + nPeaks);
    std::sort(TopPeaks.begin(), TopPeaks.end(), [](const Candidate& a, const Candidate& b) {
        return a.Price < b.Price;
    });
    for (int t = 0; t < nPeaks; ++t)
        out.HVN[t] = static_cast<float>(TopPeaks[t].Price);

    // LVNs: sort by density ascending (deepest valleys first), take top MaxTracks, then sort by price ascending
    std::sort(RawValleys.begin(), RawValleys.end(), [](const Candidate& a, const Candidate& b) {
        return a.Density < b.Density;
    });
    int nValleys = SJ_Min(MaxTracks, static_cast<int>(RawValleys.size()));
    std::vector<Candidate> TopValleys(RawValleys.begin(), RawValleys.begin() + nValleys);
    std::sort(TopValleys.begin(), TopValleys.end(), [](const Candidate& a, const Candidate& b) {
        return a.Price < b.Price;
    });
    for (int t = 0; t < nValleys; ++t)
        out.LVN[t] = static_cast<float>(TopValleys[t].Price);

    return out;
}

/*==========================================================================*/
SCSFExport scsf_SheatherJonesVolumeProfile(SCStudyInterfaceRef sc)
{
    // Subgraphs for Dynamic Moving Average Style Levels (Up to K=5 HVNs and K=5 LVNs)
    SCSubgraphRef Sub_DynamicBandwidth = sc.Subgraph[10];

    // Inputs
    SCInputRef In_ProfileScope             = sc.Input[0];
    SCInputRef In_NumBars                  = sc.Input[1];
    SCInputRef In_MinProminencePct         = sc.Input[2];
    SCInputRef In_MaxHVNCount              = sc.Input[3];
    SCInputRef In_MaxLVNCount              = sc.Input[4];
    SCInputRef In_LineType                 = sc.Input[5];
    SCInputRef In_HVNColor                 = sc.Input[6];
    SCInputRef In_LVNColor                 = sc.Input[7];
    SCInputRef In_BaseLineWidth            = sc.Input[8];
    SCInputRef In_ScaleWidthByProminence   = sc.Input[9];
    SCInputRef In_ColorGradientMode        = sc.Input[10];
    SCInputRef In_EnableTransparency       = sc.Input[11];
    SCInputRef In_CalculationIntervalSec   = sc.Input[12];
    SCInputRef In_BandwidthMultiplier      = sc.Input[13];
    SCInputRef In_PlotKDEProfile           = sc.Input[14];
    SCInputRef In_KDEProfileColor          = sc.Input[15];
    SCInputRef In_KDEProfileTransparency   = sc.Input[16];
    SCInputRef In_KDEProfileWidthBars      = sc.Input[17];
    SCInputRef In_BandwidthMode            = sc.Input[18];
    SCInputRef In_FixedBandwidthPoints     = sc.Input[19];
    SCInputRef In_KDEProfileStyle          = sc.Input[20];
    SCInputRef In_KDEProfilePlacement      = sc.Input[21];
    SCInputRef In_KDERightOffsetBars       = sc.Input[22];
    SCInputRef In_KDELineWidth             = sc.Input[23];
    SCInputRef In_LevelLabelPosition       = sc.Input[24];
    SCInputRef In_LevelLabelFontSize       = sc.Input[25];
    SCInputRef In_DynamicMode              = sc.Input[26];
    SCInputRef In_DynamicWindowBars        = sc.Input[27];
    SCInputRef In_DynamicMaxHistoryBars    = sc.Input[28];
    SCInputRef In_DynamicMaxLevels         = sc.Input[29];
    SCInputRef In_DrawStaticLevels         = sc.Input[30];

    // Persistent storage for tracking drawn lines
    int& r_LastDrawnCount = sc.GetPersistentInt(1);
    SCDateTime& r_LastCalcTime = sc.GetPersistentSCDateTime(2);

    if (sc.SetDefaults)
    {
        sc.GraphName = "Sheather-Jones Volume Profile Levels";
        sc.AutoLoop = 0; // Manual loop: whole profile calculation
        sc.GraphRegion = 0; // Main price chart
        sc.CalculationPrecedence = LOW_PREC_LEVEL;

        // Dynamic HVNs (Green palette)
        const char* HVNNames[5] = {
            "Dynamic HVN 1 (Primary Peak)",
            "Dynamic HVN 2",
            "Dynamic HVN 3",
            "Dynamic HVN 4",
            "Dynamic HVN 5"
        };
        const COLORREF HVNColors[5] = {
            RGB(0, 160, 75),   // Dark Forest Green
            RGB(0, 200, 95),   // Rich Green
            RGB(46, 204, 113), // Emerald Green
            RGB(72, 220, 150), // Mint Green
            RGB(120, 230, 180) // Soft Sage Green
        };
        const int HVNWidths[5] = { 2, 2, 1, 1, 1 };

        for (int t = 0; t < 5; ++t)
        {
            sc.Subgraph[t].Name = HVNNames[t];
            sc.Subgraph[t].DrawStyle = DRAWSTYLE_LINE_SKIP_ZEROS;
            sc.Subgraph[t].PrimaryColor = HVNColors[t];
            sc.Subgraph[t].LineWidth = HVNWidths[t];
            sc.Subgraph[t].DrawZeros = false;
        }

        // Dynamic LVNs (Crimson/Red palette)
        const char* LVNNames[5] = {
            "Dynamic LVN 1 (Primary Valley)",
            "Dynamic LVN 2",
            "Dynamic LVN 3",
            "Dynamic LVN 4",
            "Dynamic LVN 5"
        };
        const COLORREF LVNColors[5] = {
            RGB(220, 30, 30),  // Crimson Red
            RGB(245, 50, 50),  // Bright Red
            RGB(255, 99, 71),  // Coral Red
            RGB(240, 128, 128),// Light Coral
            RGB(250, 160, 160) // Soft Rose
        };
        const int LVNWidths[5] = { 2, 2, 1, 1, 1 };

        for (int t = 0; t < 5; ++t)
        {
            sc.Subgraph[5 + t].Name = LVNNames[t];
            sc.Subgraph[5 + t].DrawStyle = DRAWSTYLE_DASH;
            sc.Subgraph[5 + t].PrimaryColor = LVNColors[t];
            sc.Subgraph[5 + t].LineWidth = LVNWidths[t];
            sc.Subgraph[5 + t].DrawZeros = false;
        }

        Sub_DynamicBandwidth.Name = "Dynamic Bandwidth (h)";
        Sub_DynamicBandwidth.DrawStyle = DRAWSTYLE_IGNORE;

        In_ProfileScope.Name = "Profile Scope";
        In_ProfileScope.SetCustomInputStrings("Current Session / Day;Number of Bars Back;Entire Chart;Prior Completed Session");
        In_ProfileScope.SetCustomInputIndex(0);

        In_NumBars.Name = "Number of Bars Back (if Scope = Number of Bars Back)";
        In_NumBars.SetInt(390); // 390 1-min bars = standard 6.5h RTH session
        In_NumBars.SetIntLimits(10, 50000);

        In_MinProminencePct.Name = "Minimum Peak Prominence % (0-100)";
        In_MinProminencePct.SetFloat(3.0f);
        In_MinProminencePct.SetFloatLimits(0.1f, 100.0f);

        In_MaxHVNCount.Name = "Maximum HVN (Peak) Lines to Draw";
        In_MaxHVNCount.SetInt(6);
        In_MaxHVNCount.SetIntLimits(1, 25);

        In_MaxLVNCount.Name = "Maximum LVN (Valley) Lines to Draw";
        In_MaxLVNCount.SetInt(4);
        In_MaxLVNCount.SetIntLimits(0, 25);

        In_LineType.Name = "Line Drawing Type";
        In_LineType.SetCustomInputStrings("Horizontal Line;Horizontal Ray");
        In_LineType.SetCustomInputIndex(0);

        In_HVNColor.Name = "HVN (High Volume Node) Base Color";
        In_HVNColor.SetColor(RGB(0, 185, 90)); // Rich Green

        In_LVNColor.Name = "LVN (Low Volume Node) Base Color";
        In_LVNColor.SetColor(RGB(225, 45, 45));  // Rich Crimson Red

        In_BaseLineWidth.Name = "Base Line Width";
        In_BaseLineWidth.SetInt(2);
        In_BaseLineWidth.SetIntLimits(1, 10);

        In_ScaleWidthByProminence.Name = "Scale Line Width by Prominence";
        In_ScaleWidthByProminence.SetYesNo(1);

        In_ColorGradientMode.Name = "Color Gradient Style";
        In_ColorGradientMode.SetCustomInputStrings("Darker for Strongest, Lighter for Weakest;Brighter for Strongest, Dimmer for Weakest;Solid Base Color Only");
        In_ColorGradientMode.SetCustomInputIndex(0);

        In_EnableTransparency.Name = "Apply Transparency Gradient (Strong=Solid, Weak=Faint)";
        In_EnableTransparency.SetYesNo(1);

        In_CalculationIntervalSec.Name = "Recalculation Interval in Seconds";
        In_CalculationIntervalSec.SetInt(15);
        In_CalculationIntervalSec.SetIntLimits(1, 300);

        In_BandwidthMultiplier.Name = "Bandwidth Multiplier (1.0 = Default, <1.0 = Sharper, >1.0 = Smoother)";
        In_BandwidthMultiplier.SetFloat(1.0f);
        In_BandwidthMultiplier.SetFloatLimits(0.1f, 5.0f);

        In_PlotKDEProfile.Name = "Plot KDE Density Profile on Chart";
        In_PlotKDEProfile.SetYesNo(1);

        In_KDEProfileColor.Name = "KDE Profile Color";
        In_KDEProfileColor.SetColor(RGB(70, 130, 180)); // Steel Blue

        In_KDEProfileTransparency.Name = "KDE Profile Transparency % (0-95)";
        In_KDEProfileTransparency.SetInt(50);
        In_KDEProfileTransparency.SetIntLimits(0, 95);

        In_KDEProfileWidthBars.Name = "KDE Profile Display Width (in Bars)";
        In_KDEProfileWidthBars.SetInt(35);
        In_KDEProfileWidthBars.SetIntLimits(5, 300);

        In_BandwidthMode.Name = "Bandwidth Selection Mode";
        In_BandwidthMode.SetCustomInputStrings("Sheather-Jones (Pure);Sheather-Jones (Pure);Fixed Points");
        In_BandwidthMode.SetCustomInputIndex(0);

        In_FixedBandwidthPoints.Name = "Fixed Bandwidth in Points (if Mode = Fixed Points)";
        In_FixedBandwidthPoints.SetFloat(5.0f);
        In_FixedBandwidthPoints.SetFloatLimits(0.25f, 500.0f);

        In_KDEProfileStyle.Name = "KDE Display Style";
        In_KDEProfileStyle.SetCustomInputStrings("Smooth Envelope Curve;Filled Density Bars;Both (Curve + Bars)");
        In_KDEProfileStyle.SetCustomInputIndex(0);

        In_KDEProfilePlacement.Name = "KDE Profile Placement";
        In_KDEProfilePlacement.SetCustomInputStrings("Right Margin (Right of Price Bars);Right Edge (Aligned with Native VP);Over Price Action (Leftward from Last Bar)");
        In_KDEProfilePlacement.SetCustomInputIndex(0);

        In_KDERightOffsetBars.Name = "KDE Right Margin Offset (in Bars)";
        In_KDERightOffsetBars.SetInt(6);
        In_KDERightOffsetBars.SetIntLimits(0, 300);

        In_KDELineWidth.Name = "KDE Envelope Line Width";
        In_KDELineWidth.SetInt(2);
        In_KDELineWidth.SetIntLimits(1, 10);

        In_LevelLabelPosition.Name = "Level Label Position";
        In_LevelLabelPosition.SetCustomInputStrings("Right Side (Clear of Bars);Left Side;Hide Labels (Lines Only)");
        In_LevelLabelPosition.SetCustomInputIndex(0);

        In_LevelLabelFontSize.Name = "Level Label Font Size";
        In_LevelLabelFontSize.SetInt(8);
        In_LevelLabelFontSize.SetIntLimits(6, 24);

        In_DynamicMode.Name = "Dynamic Evolution Mode (Moving Average Style)";
        In_DynamicMode.SetCustomInputStrings("Rolling Window (Moving Average Style);Session Developing (Cumulative);Entire Chart Developing;Disabled");
        In_DynamicMode.SetCustomInputIndex(1);

        In_DynamicWindowBars.Name = "Dynamic Rolling Window (in Bars)";
        In_DynamicWindowBars.SetInt(60);
        In_DynamicWindowBars.SetIntLimits(5, 2000);

        In_DynamicMaxHistoryBars.Name = "Dynamic History Depth (in Bars)";
        In_DynamicMaxHistoryBars.SetInt(500);
        In_DynamicMaxHistoryBars.SetIntLimits(20, 10000);

        In_DynamicMaxLevels.Name = "Number of Dynamic Levels to Track (1-5)";
        In_DynamicMaxLevels.SetInt(5);
        In_DynamicMaxLevels.SetIntLimits(1, 5);

        In_DrawStaticLevels.Name = "Draw Static S/R Horizontal Lines & Labels";
        In_DrawStaticLevels.SetYesNo(1);

        return;
    }

    const int MaxAllowedDrawnLines = 280;
    const int BaseLineNumber = 800000 + (sc.StudyGraphInstanceID * 300);

    // Clean up all drawings when study is removed or reloaded
    if (sc.LastCallToFunction)
    {
        for (int i = 0; i < MaxAllowedDrawnLines; ++i)
        {
            sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, BaseLineNumber + i);
        }
        r_LastDrawnCount = 0;
        return;
    }

    // Rate-limit calculations to avoid unnecessary recalculation on every single tick
    const int CalcInterval = SJ_Max(1, In_CalculationIntervalSec.GetInt());
    SCDateTime CurrentDateTime = sc.CurrentSystemDateTime;
    if (r_LastCalcTime.GetTimeInSeconds() > 0 && 
        (CurrentDateTime.GetTimeInSeconds() - r_LastCalcTime.GetTimeInSeconds()) < CalcInterval &&
        sc.UpdateStartIndex > 0)
    {
        return;
    }
    r_LastCalcTime = CurrentDateTime;

    if (sc.ArraySize < 1)
        return;

    if (sc.VolumeAtPriceForBars == nullptr)
    {
        sc.AddMessageToLog("Sheather-Jones Study Notice: No Volume-At-Price data found on this chart. If this is a Historical Daily/Yearly chart (.dly), open an Intraday Chart (.scid) with daily bars to access tick volume profile.", 1);
        return;
    }

    // -------------------------------------------------------------------------
    // 0. Dynamic Evolving Levels (Moving Average Style Subgraphs)
    // -------------------------------------------------------------------------
    const int DynMode = In_DynamicMode.GetIndex();
    if (DynMode != 3) // 0 = Rolling Window, 1 = Session Developing, 2 = Entire Chart, 3 = Disabled
    {
        const int WindowBars = SJ_Max(5, In_DynamicWindowBars.GetInt());
        const int MaxHistBars = SJ_Max(20, In_DynamicMaxHistoryBars.GetInt());
        const int MaxTracks = SJ_Clamp(In_DynamicMaxLevels.GetInt(), 1, 5);
        const float BwMultiplier = In_BandwidthMultiplier.GetFloat();

        // Auto-detect macro chart (daily, weekly, monthly, yearly bars)
        double TotalChartDaysDyn = 0.0;
        const int LastBar = sc.ArraySize - 1;
        if (LastBar > 0)
            TotalChartDaysDyn = sc.BaseDateTimeIn[LastBar].GetAsDouble() - sc.BaseDateTimeIn[0].GetAsDouble();
        double AvgDaysPerBarDyn = (LastBar > 0) ? (TotalChartDaysDyn / LastBar) : 0.0;
        const bool IsMacroChartDyn = (sc.SecondsPerBar >= 86400 || AvgDaysPerBarDyn >= 0.70 || sc.ArraySize < 60 || (TotalChartDaysDyn > 25.0 && sc.ArraySize < 250));

        int StartCalcBar = 0;
        if (sc.UpdateStartIndex == 0)
        {
            StartCalcBar = SJ_Max(0, sc.ArraySize - MaxHistBars);
            for (int t = 0; t < 5; ++t)
            {
                for (int i = 0; i < StartCalcBar; ++i)
                {
                    sc.Subgraph[t][i] = 0.0f;
                    sc.Subgraph[5 + t][i] = 0.0f;
                }
            }
            for (int i = 0; i < StartCalcBar; ++i)
                Sub_DynamicBandwidth[i] = 0.0f;
        }
        else
        {
            StartCalcBar = SJ_Max(0, sc.UpdateStartIndex);
        }

        std::vector<double> DynPrices;
        std::vector<double> DynWeights;
        std::vector<double> DynGridX;
        std::vector<double> DynDensity;

        for (int BarIdx = StartCalcBar; BarIdx < sc.ArraySize; ++BarIdx)
        {
            int sIdx = 0;
            const int eIdx = BarIdx;

            if (DynMode == 0) // Rolling Window
            {
                sIdx = SJ_Max(0, BarIdx - WindowBars + 1);
            }
            else if (DynMode == 1) // Session Developing
            {
                if (IsMacroChartDyn)
                {
                    // On macro charts (daily/weekly/monthly/yearly bars), each bar is its own session.
                    // Fall back to Entire Chart Developing so levels accumulate properly.
                    sIdx = 0;
                }
                else
                {
                    const int BarDay = sc.GetTradingDayDate(sc.BaseDateTimeIn[BarIdx]);
                    sIdx = BarIdx;
                    while (sIdx > 0 && sc.GetTradingDayDate(sc.BaseDateTimeIn[sIdx - 1]) == BarDay)
                    {
                        --sIdx;
                    }
                }
            }
            else if (DynMode == 2) // Entire Chart Developing
            {
                sIdx = 0;
            }

            s_DynamicKDEOutput out = ComputeDynamicKDE(
                sc, sIdx, eIdx, BwMultiplier,
                MaxTracks,
                DynPrices, DynWeights, DynGridX, DynDensity
            );

            for (int t = 0; t < 5; ++t)
            {
                sc.Subgraph[t][BarIdx] = (t < MaxTracks) ? out.HVN[t] : 0.0f;
                sc.Subgraph[5 + t][BarIdx] = (t < MaxTracks) ? out.LVN[t] : 0.0f;
            }
            Sub_DynamicBandwidth[BarIdx] = out.Bandwidth;
        }
    }
    else if (sc.UpdateStartIndex == 0)
    {
        for (int t = 0; t < 5; ++t)
        {
            for (int i = 0; i < sc.ArraySize; ++i)
            {
                sc.Subgraph[t][i] = 0.0f;
                sc.Subgraph[5 + t][i] = 0.0f;
            }
        }
        for (int i = 0; i < sc.ArraySize; ++i)
            Sub_DynamicBandwidth[i] = 0.0f;
    }

    // If static levels & right-margin profile are disabled, clean up any previous lines and finish
    if (!In_DrawStaticLevels.GetYesNo())
    {
        for (int i = 0; i < r_LastDrawnCount && i < MaxAllowedDrawnLines; ++i)
        {
            sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, BaseLineNumber + i);
        }
        r_LastDrawnCount = 0;
        return;
    }

    // -------------------------------------------------------------------------
    // 1. Determine Bar Range & Aggregate Volume (Static Profile)
    // -------------------------------------------------------------------------
    int StartBarIndex = 0;
    const int EndBarIndex = sc.ArraySize - 1;
    int EffectiveEndBarIndex = EndBarIndex;
    const int Scope = In_ProfileScope.GetIndex();

    std::map<int, double> VolumeAtTick;
    double TotalVolume = 0.0;

    auto AggregateRange = [&](int sIdx, int eIdx) {
        VolumeAtTick.clear();
        TotalVolume = 0.0;
        for (int BarIdx = sIdx; BarIdx <= eIdx; ++BarIdx)
        {
            const int VAPSize = sc.VolumeAtPriceForBars->GetSizeAtBarIndex(BarIdx);
            for (int VAPIdx = 0; VAPIdx < VAPSize; ++VAPIdx)
            {
                const s_VolumeAtPriceV2* p_VAP = nullptr;
                if (!sc.VolumeAtPriceForBars->GetVAPElementAtIndex(BarIdx, VAPIdx, &p_VAP) || p_VAP == nullptr)
                    break;

                if (p_VAP->Volume > 0)
                {
                    VolumeAtTick[p_VAP->PriceInTicks] += static_cast<double>(p_VAP->Volume);
                    TotalVolume += static_cast<double>(p_VAP->Volume);
                }
            }
        }
    };

    // Auto-detect Macro chart timeframe (Daily, Weekly, Monthly, Quarterly, Yearly, or multi-day volume bars)
    double TotalChartDays = 0.0;
    if (EndBarIndex > 0)
    {
        TotalChartDays = sc.BaseDateTimeIn[EndBarIndex].GetAsDouble() - sc.BaseDateTimeIn[0].GetAsDouble();
    }
    double AvgDaysPerBar = (EndBarIndex > 0) ? (TotalChartDays / EndBarIndex) : 0.0;

    const bool IsMacroChart = (sc.SecondsPerBar >= 86400 || AvgDaysPerBar >= 0.70 || sc.ArraySize < 60 || (TotalChartDays > 25.0 && sc.ArraySize < 250));

    if (Scope == 0) // Current Session / Day (Trading Day aware)
    {
        if (IsMacroChart)
        {
            // On macro charts, automatically aggregate the entire chart so levels appear immediately
            StartBarIndex = 0;
            AggregateRange(StartBarIndex, EndBarIndex);
        }
        else
        {
            const int CurrentTradingDay = sc.GetTradingDayDate(sc.BaseDateTimeIn[EndBarIndex]);
            StartBarIndex = EndBarIndex;
            while (StartBarIndex > 0 && sc.GetTradingDayDate(sc.BaseDateTimeIn[StartBarIndex - 1]) == CurrentTradingDay)
            {
                --StartBarIndex;
            }

            AggregateRange(StartBarIndex, EndBarIndex);

            // If the developing session has insufficient price levels (< 5 ticks, e.g. right after 5 PM rollover),
            // automatically expand back to include the prior trading session so levels remain active on chart.
            if (VolumeAtTick.size() < 5 && StartBarIndex > 0)
            {
                const int PriorTradingDay = sc.GetTradingDayDate(sc.BaseDateTimeIn[StartBarIndex - 1]);
                int ExpandedStart = StartBarIndex - 1;
                while (ExpandedStart > 0 && sc.GetTradingDayDate(sc.BaseDateTimeIn[ExpandedStart - 1]) == PriorTradingDay)
                {
                    --ExpandedStart;
                }
                StartBarIndex = ExpandedStart;
                AggregateRange(StartBarIndex, EndBarIndex);
            }
        }
    }
    else if (Scope == 1) // Number of Bars Back
    {
        StartBarIndex = SJ_Max(0, EndBarIndex - In_NumBars.GetInt() + 1);
        AggregateRange(StartBarIndex, EndBarIndex);
    }
    else if (Scope == 2) // Entire Chart
    {
        StartBarIndex = 0;
        AggregateRange(StartBarIndex, EndBarIndex);
    }
    else if (Scope == 3) // Prior Completed Session
    {
        const int CurrentTradingDay = sc.GetTradingDayDate(sc.BaseDateTimeIn[EndBarIndex]);
        int PriorEnd = EndBarIndex;
        while (PriorEnd >= 0 && sc.GetTradingDayDate(sc.BaseDateTimeIn[PriorEnd]) == CurrentTradingDay)
        {
            --PriorEnd;
        }
        if (PriorEnd >= 0)
        {
            EffectiveEndBarIndex = PriorEnd;
            const int PriorTradingDay = sc.GetTradingDayDate(sc.BaseDateTimeIn[PriorEnd]);
            StartBarIndex = PriorEnd;
            while (StartBarIndex > 0 && sc.GetTradingDayDate(sc.BaseDateTimeIn[StartBarIndex - 1]) == PriorTradingDay)
            {
                --StartBarIndex;
            }
            AggregateRange(StartBarIndex, EffectiveEndBarIndex);
        }
        else
        {
            StartBarIndex = 0;
            AggregateRange(StartBarIndex, EndBarIndex);
        }
    }

    if (VolumeAtTick.size() < 5 || TotalVolume <= 0.0)
    {
        SCString msg;
        msg.Format("Sheather-Jones Volume Profile: Insufficient volume data to compute profile (unique price ticks: %d, total volume: %.0f).", static_cast<int>(VolumeAtTick.size()), TotalVolume);
        sc.AddMessageToLog(msg, 0);
        return;
    }

    // -------------------------------------------------------------------------
    // 2. Statistical Metrics & Scale Parameter
    // -------------------------------------------------------------------------
    const int M = static_cast<int>(VolumeAtTick.size());
    std::vector<double> Prices(M);
    std::vector<double> Weights(M);

    double WeightedMean = 0.0;
    int idx = 0;
    for (auto it = VolumeAtTick.begin(); it != VolumeAtTick.end(); ++it, ++idx)
    {
        Prices[idx] = it->first * static_cast<double>(sc.TickSize);
        Weights[idx] = it->second / TotalVolume;
        WeightedMean += Weights[idx] * Prices[idx];
    }

    // Weighted variance & std dev
    double WeightedVar = 0.0;
    double SumSqWeights = 0.0;
    for (int i = 0; i < M; ++i)
    {
        double diff = Prices[i] - WeightedMean;
        WeightedVar += Weights[i] * diff * diff;
        SumSqWeights += Weights[i] * Weights[i];
    }
    double Sigma = sqrt(SJ_Max(WeightedVar, 1e-6));

    // Kish's Effective Sample Size for volume-weighted data
    // In Sheather-Jones AMISE theory, N represents the effective number of observations.
    // For Volume Profile, volume represents aggregate trade executions.
    // We scale Kish's diversity metric so N reflects the true statistical information scale
    // of the volume distribution (N in [2000, 8000]), allowing pure SJ to resolve all specific peaks
    // without over-smoothing them into a single lump.
    double n_eff = (SumSqWeights > 1e-12) ? (1.0 / SumSqWeights) : static_cast<double>(M);
    n_eff = SJ_Clamp(n_eff * 50.0, 2000.0, 8000.0);

    // Weighted IQR for robust scale estimation across long-horizon / macro charts
    double CumWeight = 0.0;
    double Q25 = Prices.front();
    double Q75 = Prices.back();
    bool FoundQ25 = false;
    for (int i = 0; i < M; ++i)
    {
        CumWeight += Weights[i];
        if (!FoundQ25 && CumWeight >= 0.25)
        {
            Q25 = Prices[i];
            FoundQ25 = true;
        }
        if (CumWeight >= 0.75)
        {
            Q75 = Prices[i];
            break;
        }
    }
    double IQR = Q75 - Q25;
    double ScaleParam = (IQR > 0.0) ? SJ_Min(Sigma, IQR / 1.349) : Sigma;
    if (ScaleParam <= 0.0)
        ScaleParam = Sigma;

    // -------------------------------------------------------------------------
    // 3. Sheather-Jones 1D Bandwidth Calculation
    // -------------------------------------------------------------------------
    double h_0 = pow(4.0 / (3.0 * n_eff), 0.2) * ScaleParam;
    h_0 = SJ_Max(h_0, static_cast<double>(sc.TickSize));

    // Sheather-Jones 1D Roughness: S = sum_i sum_j w_i w_j W(r^2) P(r^2)
    double S = 0.0;
    const double h_0_sq = h_0 * h_0;
    const double Cutoff_r_sq = 36.0;

    for (int i = 0; i < M; ++i)
    {
        S += Weights[i] * Weights[i] * 0.75;
        for (int j = i + 1; j < M; ++j)
        {
            double diff = Prices[i] - Prices[j];
            double r_sq = (diff * diff) / h_0_sq;
            if (r_sq > Cutoff_r_sq)
                break;

            double P = (r_sq * r_sq) / 16.0 - (3.0 * r_sq) / 4.0 + 0.75;
            double W = exp(-r_sq / 4.0);
            double term = 2.0 * Weights[i] * Weights[j] * W * P;
            S += term;
        }
    }

    double Roughness = S / (sqrt(4.0 * SJ_PI) * pow(h_0, 5.0));

    // Optimal bandwidth h_Target
    const double R_K = 1.0 / (2.0 * sqrt(SJ_PI));
    const double MinPrice = Prices.front();
    const double MaxPrice = Prices.back();
    const double PriceRange = SJ_Max(1.0, MaxPrice - MinPrice);

    const int BwMode = SJ_Clamp(static_cast<int>(In_BandwidthMode.GetIndex()), 0, 2);
    double h_Target = h_0;

    if (BwMode == 2) // Fixed Points
    {
        h_Target = static_cast<double>(In_FixedBandwidthPoints.GetFloat());
    }
    else // Both Mode 0 (Pure Sheather-Jones) and Mode 1 (SJ Exact) use the same pure SJ formula
    {
        if (Roughness > 1e-12)
            h_Target = pow(R_K / (n_eff * Roughness), 0.2);
        else
            h_Target = h_0;
    }

    // Apply user-defined Bandwidth Multiplier (<1.0 sharper, >1.0 smoother)
    const float BwMult = SJ_Clamp(In_BandwidthMultiplier.GetFloat(), 0.1f, 5.0f);
    double h_SJ = h_Target * static_cast<double>(BwMult);

    // Only safe guard: bandwidth cannot be sub-tick
    h_SJ = SJ_Max(h_SJ, static_cast<double>(sc.TickSize));

    // --- Diagnostic logging ---
    SCString DiagMsg;
    DiagMsg.Format("SJ Diag: M=%d n_eff=%.1f Sigma=%.2f IQR=%.2f ScaleParam=%.2f h_0=%.3f "
                   "Roughness=%.6f h_raw=%.3f h_final=%.3f PriceRange=%.1f BwMode=%d BwMult=%.2f",
                   M, n_eff, Sigma, IQR, ScaleParam, h_0,
                   Roughness, h_Target, h_SJ, PriceRange, BwMode, static_cast<double>(BwMult));
    sc.AddMessageToLog(DiagMsg, 0);

    // -------------------------------------------------------------------------
    // 4. Evaluate KDE along a Dynamic Price Grid
    // -------------------------------------------------------------------------
    const int TargetGridPoints = 1500;
    double GridStep = (MaxPrice - MinPrice) / static_cast<double>(TargetGridPoints - 1);
    GridStep = SJ_Min(GridStep, h_SJ / 2.5);
    GridStep = SJ_Max(GridStep, static_cast<double>(sc.TickSize));
    const int NumGridPoints = SJ_Clamp(static_cast<int>((MaxPrice - MinPrice) / GridStep) + 1, 20, 2000);

    std::vector<float> GridPrices(NumGridPoints);
    std::vector<float> GridDensity(NumGridPoints, 0.0f);
    const double two_h_sq = 2.0 * h_SJ * h_SJ;
    const double norm_factor = 1.0 / (sqrt(2.0 * SJ_PI) * h_SJ);

    float MaxDensity = 0.0f;

    for (int k = 0; k < NumGridPoints; ++k)
    {
        double x_k = MinPrice + k * GridStep;
        GridPrices[k] = static_cast<float>(x_k);

        double density_k = 0.0;
        double min_p = x_k - 4.0 * h_SJ;
        double max_p = x_k + 4.0 * h_SJ;

        auto it_start = std::lower_bound(Prices.begin(), Prices.end(), min_p);
        int start_i = static_cast<int>(std::distance(Prices.begin(), it_start));

        for (int i = start_i; i < M; ++i)
        {
            if (Prices[i] > max_p)
                break;

            double diff = x_k - Prices[i];
            density_k += Weights[i] * exp(-(diff * diff) / two_h_sq);
        }
        density_k *= norm_factor;
        GridDensity[k] = static_cast<float>(density_k);

        if (GridDensity[k] > MaxDensity)
            MaxDensity = GridDensity[k];
    }

    if (MaxDensity <= 0.0f)
        return;

    // -------------------------------------------------------------------------
    // 5. Detect Peaks (HVNs) and Valleys (LVNs) with True Topographic Prominence
    // -------------------------------------------------------------------------
    struct s_PeakCandidate
    {
        int GridIdx;
        float Price;
        float Density;
    };

    std::vector<s_PeakCandidate> CandidatePeaks;

    // Minimum distance between distinct peaks to prevent sub-tick noise ripples
    const int MinPeakDistSteps = SJ_Max(2, static_cast<int>(round((h_SJ * 0.70) / GridStep)));

    for (int k = 1; k < NumGridPoints - 1; ++k)
    {
        if (GridDensity[k] >= GridDensity[k - 1] && GridDensity[k] > GridDensity[k + 1])
        {
            s_PeakCandidate cand;
            cand.GridIdx = k;
            cand.Price = GridPrices[k];
            cand.Density = GridDensity[k];
            CandidatePeaks.push_back(cand);
        }
    }

    // Filter candidate peaks: suppress micro-ripples within MinPeakDistSteps of a higher neighbor
    std::vector<s_PeakCandidate> FilteredPeaks;
    for (size_t i = 0; i < CandidatePeaks.size(); ++i)
    {
        bool is_dominant = true;
        for (size_t j = 0; j < CandidatePeaks.size(); ++j)
        {
            if (i == j)
                continue;

            int dist = std::abs(CandidatePeaks[i].GridIdx - CandidatePeaks[j].GridIdx);
            if (dist <= MinPeakDistSteps)
            {
                if (CandidatePeaks[j].Density > CandidatePeaks[i].Density)
                {
                    is_dominant = false;
                    break;
                }
                else if (CandidatePeaks[j].Density == CandidatePeaks[i].Density && j < i)
                {
                    is_dominant = false;
                    break;
                }
            }
        }
        if (is_dominant)
        {
            FilteredPeaks.push_back(CandidatePeaks[i]);
        }
    }

    // Compute True Topographic Prominence for each filtered peak
    // Prominence = Height - KeyCol (highest saddle point connecting to higher ground)
    const float MinProminenceThreshold = (In_MinProminencePct.GetFloat() / 100.0f) * MaxDensity;
    std::vector<s_SJLevel> DetectedHVNs;

    for (size_t p = 0; p < FilteredPeaks.size(); ++p)
    {
        const int k = FilteredPeaks[p].GridIdx;
        const float peak_density = FilteredPeaks[p].Density;

        // Search left: find minimum density before encountering a point higher than this peak
        float left_min = peak_density;
        for (int l = k - 1; l >= 0; --l)
        {
            if (GridDensity[l] > peak_density)
                break;
            if (GridDensity[l] < left_min)
                left_min = GridDensity[l];
        }

        // Search right: find minimum density before encountering a point higher than this peak
        float right_min = peak_density;
        for (int r = k + 1; r < NumGridPoints; ++r)
        {
            if (GridDensity[r] > peak_density)
                break;
            if (GridDensity[r] < right_min)
                right_min = GridDensity[r];
        }

        // Col is the highest of the two flanking saddle points
        float key_col = SJ_Max(left_min, right_min);
        float raw_prominence = peak_density - key_col;
        float rel_prominence = SJ_Clamp(raw_prominence / MaxDensity, 0.0f, 1.0f);

        // Also check prominence relative to adjacent peaks in FilteredPeaks
        float adj_prominence = raw_prominence;
        if (FilteredPeaks.size() > 1)
        {
            float valley_min = peak_density;
            if (p > 0)
            {
                int prev_k = FilteredPeaks[p - 1].GridIdx;
                for (int m = prev_k; m <= k; ++m)
                {
                    if (GridDensity[m] < valley_min)
                        valley_min = GridDensity[m];
                }
            }
            if (p + 1 < FilteredPeaks.size())
            {
                int next_k = FilteredPeaks[p + 1].GridIdx;
                for (int m = k; m <= next_k; ++m)
                {
                    if (GridDensity[m] < valley_min)
                        valley_min = GridDensity[m];
                }
            }
            adj_prominence = SJ_Max(raw_prominence, peak_density - valley_min);
        }

        float effective_prominence = SJ_Max(raw_prominence, adj_prominence);
        float effective_ratio = SJ_Clamp(effective_prominence / MaxDensity, 0.0f, 1.0f);

        if (effective_prominence >= MinProminenceThreshold || 
            (peak_density >= 0.30f * MaxDensity && effective_ratio >= 0.015f))
        {
            s_SJLevel lvl;
            lvl.Price = FilteredPeaks[p].Price;
            lvl.Density = peak_density;
            lvl.Prominence = SJ_Max(effective_ratio, peak_density / MaxDensity * 0.35f);
            lvl.Prominence = SJ_Clamp(lvl.Prominence, 0.05f, 1.0f);
            lvl.IsHVN = true;
            DetectedHVNs.push_back(lvl);
        }
    }

    // Detect LVNs (Valleys) between adjacent detected HVNs
    std::vector<s_SJLevel> DetectedLVNs;
    std::vector<s_SJLevel> HVNsByPrice = DetectedHVNs;
    std::sort(HVNsByPrice.begin(), HVNsByPrice.end(), [](const s_SJLevel& a, const s_SJLevel& b) {
        return a.Price < b.Price;
    });

    for (size_t i = 0; i + 1 < HVNsByPrice.size(); ++i)
    {
        float pA = HVNsByPrice[i].Price;
        float pB = HVNsByPrice[i + 1].Price;
        
        int idxA = SJ_Clamp(static_cast<int>(round((pA - MinPrice) / GridStep)), 0, NumGridPoints - 1);
        int idxB = SJ_Clamp(static_cast<int>(round((pB - MinPrice) / GridStep)), 0, NumGridPoints - 1);
        if (idxA > idxB) std::swap(idxA, idxB);

        int min_idx = idxA;
        float min_density = GridDensity[idxA];

        for (int m = idxA + 1; m < idxB; ++m)
        {
            if (GridDensity[m] < min_density)
            {
                min_density = GridDensity[m];
                min_idx = m;
            }
        }

        float flanking_min = SJ_Min(HVNsByPrice[i].Density, HVNsByPrice[i + 1].Density);
        float vacuum_depth = flanking_min - min_density;

        if (vacuum_depth > 0.0f && min_idx > idxA && min_idx < idxB)
        {
            s_SJLevel lvn;
            lvn.Price = GridPrices[min_idx];
            lvn.Density = min_density;
            lvn.Prominence = SJ_Clamp(vacuum_depth / MaxDensity, 0.05f, 1.0f);
            lvn.IsHVN = false;
            DetectedLVNs.push_back(lvn);
        }
    }

    // Sort HVNs by prominence descending
    std::sort(DetectedHVNs.begin(), DetectedHVNs.end(), [](const s_SJLevel& a, const s_SJLevel& b) {
        return a.Prominence > b.Prominence;
    });

    // Sort LVNs by depth descending
    std::sort(DetectedLVNs.begin(), DetectedLVNs.end(), [](const s_SJLevel& a, const s_SJLevel& b) {
        return a.Prominence > b.Prominence;
    });

    // -------------------------------------------------------------------------
    // 6. Draw Levels and KDE Profile using sc.UseTool
    // -------------------------------------------------------------------------
    int LineIdx = 0;
    const int TargetHVNCount = SJ_Min(In_MaxHVNCount.GetInt(), static_cast<int>(DetectedHVNs.size()));
    const int TargetLVNCount = SJ_Min(In_MaxLVNCount.GetInt(), static_cast<int>(DetectedLVNs.size()));
    const DrawingTypeEnum SelectedDrawingType = (In_LineType.GetIndex() == 0) ? DRAWING_HORIZONTALLINE : DRAWING_RAY;
    const int GradientMode = SJ_Clamp(static_cast<int>(In_ColorGradientMode.GetIndex()), 0, 2);
    const bool UseTransparency = In_EnableTransparency.GetYesNo() != 0;

    SCString calcMsg;
    calcMsg.Format("SJ Volume Profile: Bandwidth h=%.2f. Detected %d HVNs and %d LVNs across bars [%d..%d].",
                   h_SJ, TargetHVNCount, TargetLVNCount, StartBarIndex, EffectiveEndBarIndex);
    sc.AddMessageToLog(calcMsg, 0);

    // Determine screen-accurate time scaling for drawings
    int FirstVis = SJ_Clamp(sc.IndexOfFirstVisibleBar, 0, sc.ArraySize - 1);
    int LastVis  = SJ_Clamp(sc.IndexOfLastVisibleBar, 0, sc.ArraySize - 1);
    if (LastVis <= FirstVis)
    {
        FirstVis = SJ_Max(0, sc.ArraySize - 80);
        LastVis  = sc.ArraySize - 1;
    }

    int NumVisBars = SJ_Max(1, LastVis - FirstVis);
    double VisTimeSpan = sc.BaseDateTimeIn[LastVis].GetAsDouble() - sc.BaseDateTimeIn[FirstVis].GetAsDouble();
    
    // Time per visible bar on the user's active screen (in SCDateTime double units / days)
    double DaysPerBar = (VisTimeSpan > 1e-7) ? (VisTimeSpan / NumVisBars) : (1.0 / 1440.0);
    
    if (DaysPerBar <= 1e-7)
    {
        if (sc.SecondsPerBar > 0)
            DaysPerBar = sc.SecondsPerBar / 86400.0;
        else
            DaysPerBar = 1.0 / 1440.0; // 1 minute
    }

    // Determine proportional profile display width
    // Capped at 35% of visible chart bars so it never stretches across or off the screen on macro charts
    int UserWidthBars = SJ_Clamp(In_KDEProfileWidthBars.GetInt(), 5, 250);
    double MaxAllowedBars = SJ_Max(4.0, NumVisBars * 0.35);
    double EffectiveWidthBars = SJ_Min(static_cast<double>(UserWidthBars), MaxAllowedBars);

    // Margin offset in bars
    int MarginOffsetBars = SJ_Clamp(In_KDERightOffsetBars.GetInt(), 0, 100);

    // Time per bar for forward projection:
    // In Sierra Chart, volume/range charts (SecondsPerBar == 0) use session steps in forward space.
    // Ensure forward projection uses at least 15 minutes (0.0104 days) per bar so the profile
    // spans real visual columns across forward space without collapsing in session breaks.
    double ForwardDaysPerBar = DaysPerBar;
    if (sc.SecondsPerBar == 0)
    {
        ForwardDaysPerBar = SJ_Max(ForwardDaysPerBar, 15.0 / 1440.0);
    }

    auto GetOffsetDateTime = [&](const SCDateTime& anchor, double barOffset) -> SCDateTime {
        SCDateTime dt = anchor;
        double step = (barOffset >= 0.0) ? ForwardDaysPerBar : DaysPerBar;
        double addDays = barOffset * step;
        return SCDateTime(dt.GetAsDouble() + addDays);
    };

    // Draw KDE Density Profile (Envelope Curve and/or Filled Bars)
    if (In_PlotKDEProfile.GetYesNo() != 0 && MaxDensity > 0.0f)
    {
        const int Style = SJ_Clamp(In_KDEProfileStyle.GetIndex(), 0, 2);
        const int Placement = SJ_Clamp(In_KDEProfilePlacement.GetIndex(), 0, 2);
        const COLORREF KDEColor = In_KDEProfileColor.GetColor();
        const int KDETrans = SJ_Clamp(In_KDEProfileTransparency.GetInt(), 0, 95);
        const int KDELineW = SJ_Clamp(In_KDELineWidth.GetInt(), 1, 10);
        const SCDateTime LastBarDT = sc.BaseDateTimeIn[EffectiveEndBarIndex];

        auto GetKDECoordinates = [&](float ratio, SCDateTime& r_BaseDT, SCDateTime& r_TipDT) {
            if (Placement == 0) // Right Margin (Right of Price Bars)
            {
                r_BaseDT = GetOffsetDateTime(LastBarDT, MarginOffsetBars);
                r_TipDT  = GetOffsetDateTime(LastBarDT, MarginOffsetBars + ratio * EffectiveWidthBars);
            }
            else if (Placement == 1) // Right Edge (Aligned with Native VP)
            {
                r_BaseDT = GetOffsetDateTime(LastBarDT, MarginOffsetBars + EffectiveWidthBars);
                r_TipDT  = GetOffsetDateTime(LastBarDT, MarginOffsetBars + EffectiveWidthBars - ratio * EffectiveWidthBars);
            }
            else // Over Price Action (Leftward from Last Bar)
            {
                r_BaseDT = LastBarDT;
                r_TipDT  = GetOffsetDateTime(LastBarDT, -ratio * EffectiveWidthBars);
            }
        };

        // Draw Smooth Envelope Curve (High resolution along grid points — no sub-sampling)
        if (Style == 0 || Style == 2)
        {
            const int NumCurveSegments = SJ_Min(180, NumGridPoints);
            SCDateTime prev_tip_dt;
            float prev_price = 0.0f;
            bool has_prev = false;
            SCDateTime first_base_dt, first_tip_dt, last_base_dt, last_tip_dt;
            float first_price = 0.0f, last_price = 0.0f;

            for (int s = 0; s < NumCurveSegments && LineIdx < 200; ++s)
            {
                int grid_idx = static_cast<int>(round(s * (NumGridPoints - 1) / static_cast<double>(NumCurveSegments - 1)));
                grid_idx = SJ_Clamp(grid_idx, 0, NumGridPoints - 1);

                float density = GridDensity[grid_idx];
                float ratio = SJ_Clamp(density / MaxDensity, 0.0f, 1.0f);
                float cur_price = GridPrices[grid_idx];

                SCDateTime base_dt, tip_dt;
                GetKDECoordinates(ratio, base_dt, tip_dt);

                if (s == 0)
                {
                    first_base_dt = base_dt;
                    first_tip_dt = tip_dt;
                    first_price = cur_price;
                }
                last_base_dt = base_dt;
                last_tip_dt = tip_dt;
                last_price = cur_price;

                if (has_prev)
                {
                    s_UseTool Tool;
                    Tool.Clear();
                    Tool.ChartNumber = sc.ChartNumber;
                    Tool.DrawingType = DRAWING_LINE;
                    Tool.LineNumber = BaseLineNumber + LineIdx++;
                    Tool.BeginDateTime = prev_tip_dt;
                    Tool.BeginValue = prev_price;
                    Tool.EndDateTime = tip_dt;
                    Tool.EndValue = cur_price;
                    Tool.Color = KDEColor;
                    Tool.TransparencyLevel = KDETrans;
                    Tool.LineWidth = KDELineW;
                    Tool.LineStyle = LINESTYLE_SOLID;
                    Tool.AddMethod = UTAM_ADD_OR_ADJUST;
                    sc.UseTool(Tool);
                }

                prev_tip_dt = tip_dt;
                prev_price = cur_price;
                has_prev = true;
            }

            // Draw baseline spine and end caps to form a complete distribution envelope
            if (has_prev && LineIdx + 3 < 150)
            {
                // Baseline spine
                s_UseTool SpineTool;
                SpineTool.Clear();
                SpineTool.ChartNumber = sc.ChartNumber;
                SpineTool.DrawingType = DRAWING_LINE;
                SpineTool.LineNumber = BaseLineNumber + LineIdx++;
                SpineTool.BeginDateTime = first_base_dt;
                SpineTool.BeginValue = first_price;
                SpineTool.EndDateTime = last_base_dt;
                SpineTool.EndValue = last_price;
                SpineTool.Color = KDEColor;
                SpineTool.TransparencyLevel = SJ_Min(95, KDETrans + 15);
                SpineTool.LineWidth = 1;
                SpineTool.LineStyle = LINESTYLE_DASH;
                SpineTool.AddMethod = UTAM_ADD_OR_ADJUST;
                sc.UseTool(SpineTool);

                // Bottom cap
                s_UseTool BotCap;
                BotCap.Clear();
                BotCap.ChartNumber = sc.ChartNumber;
                BotCap.DrawingType = DRAWING_LINE;
                BotCap.LineNumber = BaseLineNumber + LineIdx++;
                BotCap.BeginDateTime = first_base_dt;
                BotCap.BeginValue = first_price;
                BotCap.EndDateTime = first_tip_dt;
                BotCap.EndValue = first_price;
                BotCap.Color = KDEColor;
                BotCap.TransparencyLevel = SJ_Min(95, KDETrans + 15);
                BotCap.LineWidth = 1;
                BotCap.LineStyle = LINESTYLE_SOLID;
                BotCap.AddMethod = UTAM_ADD_OR_ADJUST;
                sc.UseTool(BotCap);

                // Top cap
                s_UseTool TopCap;
                TopCap.Clear();
                TopCap.ChartNumber = sc.ChartNumber;
                TopCap.DrawingType = DRAWING_LINE;
                TopCap.LineNumber = BaseLineNumber + LineIdx++;
                TopCap.BeginDateTime = last_base_dt;
                TopCap.BeginValue = last_price;
                TopCap.EndDateTime = last_tip_dt;
                TopCap.EndValue = last_price;
                TopCap.Color = KDEColor;
                TopCap.TransparencyLevel = SJ_Min(95, KDETrans + 15);
                TopCap.LineWidth = 1;
                TopCap.LineStyle = LINESTYLE_SOLID;
                TopCap.AddMethod = UTAM_ADD_OR_ADJUST;
                sc.UseTool(TopCap);
            }
        }

        // Draw Filled Density Bars (High resolution along price grid)
        if (Style == 1 || Style == 2)
        {
            const int NumSlices = (Style == 2) ? SJ_Min(60, NumGridPoints) : SJ_Min(160, NumGridPoints);
            for (int s = 0; s < NumSlices && LineIdx < 230; ++s)
            {
                int grid_idx = static_cast<int>(round(s * (NumGridPoints - 1) / static_cast<double>(NumSlices - 1)));
                grid_idx = SJ_Clamp(grid_idx, 0, NumGridPoints - 1);

                float density = GridDensity[grid_idx];
                float ratio = density / MaxDensity;
                if (ratio < 0.02f)
                    continue;

                SCDateTime base_dt, tip_dt;
                GetKDECoordinates(ratio, base_dt, tip_dt);

                SCDateTime bar_start = base_dt;
                SCDateTime bar_end = tip_dt;
                if (bar_start > bar_end)
                    std::swap(bar_start, bar_end);

                s_UseTool Tool;
                Tool.Clear();
                Tool.ChartNumber = sc.ChartNumber;
                Tool.DrawingType = DRAWING_LINE;
                Tool.LineNumber = BaseLineNumber + LineIdx++;
                Tool.BeginDateTime = bar_start;
                Tool.BeginValue = GridPrices[grid_idx];
                Tool.EndDateTime = bar_end;
                Tool.EndValue = GridPrices[grid_idx];
                Tool.Color = KDEColor;
                Tool.TransparencyLevel = KDETrans;
                Tool.LineWidth = 3;
                Tool.LineStyle = LINESTYLE_SOLID;
                Tool.AddMethod = UTAM_ADD_OR_ADJUST;
                sc.UseTool(Tool);
            }
        }
    }

    const int LabelPos = SJ_Clamp(In_LevelLabelPosition.GetIndex(), 0, 2);
    const int LabelFontSize = SJ_Clamp(In_LevelLabelFontSize.GetInt(), 6, 24);

    // Draw Top HVNs
    for (int i = 0; i < TargetHVNCount && LineIdx < MaxAllowedDrawnLines; ++i, ++LineIdx)
    {
        const s_SJLevel& hvn = DetectedHVNs[i];
        s_UseTool Tool;
        Tool.Clear();
        Tool.ChartNumber = sc.ChartNumber;
        Tool.DrawingType = SelectedDrawingType;
        Tool.LineNumber = BaseLineNumber + LineIdx;
        Tool.BeginValue = hvn.Price;
        Tool.EndValue = hvn.Price;
        Tool.BeginDateTime = sc.BaseDateTimeIn[StartBarIndex];
        Tool.EndDateTime = sc.BaseDateTimeIn[EffectiveEndBarIndex];

        // Apply color gradient based on prominence
        Tool.Color = CalculateGradientColor(In_HVNColor.GetColor(), hvn.Prominence, GradientMode);

        // Apply transparency gradient (stronger levels are solid/opaque; weaker levels fade out)
        if (UseTransparency)
        {
            int trans = static_cast<int>(round((1.0f - hvn.Prominence) * 65.0f));
            Tool.TransparencyLevel = SJ_Clamp(trans, 0, 80);
        }
        else
        {
            Tool.TransparencyLevel = 0;
        }

        int Width = In_BaseLineWidth.GetInt();
        if (In_ScaleWidthByProminence.GetYesNo())
        {
            Width = SJ_Clamp(static_cast<int>(round(Width * (0.6f + 0.8f * hvn.Prominence))), 1, 6);
        }
        Tool.LineWidth = Width;
        Tool.LineStyle = LINESTYLE_SOLID;

        if (LabelPos != 2) // 0 = Right Side, 1 = Left Side
        {
            Tool.TransparentLabelBackground = 1;
            Tool.FontSize = LabelFontSize;
            Tool.TextAlignment = (LabelPos == 0) ? (DT_RIGHT | DT_BOTTOM) : (DT_LEFT | DT_BOTTOM);
            Tool.DisplayHorizontalLineValue = 0;
            Tool.Text.Format("HVN %.2f (%.0f%%)", hvn.Price, hvn.Prominence * 100.0f);
        }
        else
        {
            Tool.Text = "";
            Tool.DisplayHorizontalLineValue = 0;
        }
        Tool.AddMethod = UTAM_ADD_OR_ADJUST;

        sc.UseTool(Tool);
    }

    // Draw Top LVNs
    for (int i = 0; i < TargetLVNCount && LineIdx < MaxAllowedDrawnLines; ++i, ++LineIdx)
    {
        const s_SJLevel& lvn = DetectedLVNs[i];
        s_UseTool Tool;
        Tool.Clear();
        Tool.ChartNumber = sc.ChartNumber;
        Tool.DrawingType = SelectedDrawingType;
        Tool.LineNumber = BaseLineNumber + LineIdx;
        Tool.BeginValue = lvn.Price;
        Tool.EndValue = lvn.Price;
        Tool.BeginDateTime = sc.BaseDateTimeIn[StartBarIndex];
        Tool.EndDateTime = sc.BaseDateTimeIn[EffectiveEndBarIndex];

        // Apply color gradient based on vacuum depth
        Tool.Color = CalculateGradientColor(In_LVNColor.GetColor(), lvn.Prominence, GradientMode);

        // Apply transparency gradient
        if (UseTransparency)
        {
            int trans = static_cast<int>(round((1.0f - lvn.Prominence) * 65.0f));
            Tool.TransparencyLevel = SJ_Clamp(trans, 0, 80);
        }
        else
        {
            Tool.TransparencyLevel = 0;
        }

        int Width = SJ_Max(1, In_BaseLineWidth.GetInt() - 1);
        if (In_ScaleWidthByProminence.GetYesNo())
        {
            Width = SJ_Clamp(static_cast<int>(round(Width * (0.6f + 0.6f * lvn.Prominence))), 1, 5);
        }
        Tool.LineWidth = Width;
        Tool.LineStyle = LINESTYLE_DASH;

        if (LabelPos != 2) // 0 = Right Side, 1 = Left Side
        {
            Tool.TransparentLabelBackground = 1;
            Tool.FontSize = LabelFontSize;
            Tool.TextAlignment = (LabelPos == 0) ? (DT_RIGHT | DT_BOTTOM) : (DT_LEFT | DT_BOTTOM);
            Tool.DisplayHorizontalLineValue = 0;
            Tool.Text.Format("LVN %.2f (%.0f%%)", lvn.Price, lvn.Prominence * 100.0f);
        }
        else
        {
            Tool.Text = "";
            Tool.DisplayHorizontalLineValue = 0;
        }
        Tool.AddMethod = UTAM_ADD_OR_ADJUST;

        sc.UseTool(Tool);
    }

    // Clear any obsolete lines from previous calculation
    for (int i = LineIdx; i < r_LastDrawnCount && i < MaxAllowedDrawnLines; ++i)
    {
        sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, BaseLineNumber + i);
    }

    r_LastDrawnCount = LineIdx;
}
