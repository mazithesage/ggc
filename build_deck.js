const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10" x 5.625"
pres.author = "Quantitative Strategy Research";
pres.title = "Regime-Aware Mean Reversion — Crypto Spot Markets";

// ── Paths ──
const IMG = path.join(__dirname, "output");
const EDA = path.join(IMG, "eda");

// ── Colors ──
const BG       = "0D1117";
const BG2      = "161B22";
const ACCENT   = "58A6FF";
const GREEN    = "3FB950";
const RED      = "F85149";
const PURPLE   = "D2A8FF";
const GOLD     = "D29922";
const TXT      = "F0F6FC";
const MUTED    = "8B949E";
const DIVIDER  = "30363D";

// ── Helpers ──
function addFooter(slide, text) {
  slide.addText(text, {
    x: 0.5, y: 5.2, w: 9, h: 0.3,
    fontSize: 8, color: MUTED, fontFace: "Calibri",
  });
}

function addSectionTag(slide, label) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.35, w: 1.6, h: 0.28,
    fill: { color: ACCENT, transparency: 85 },
  });
  slide.addText(label.toUpperCase(), {
    x: 0.55, y: 0.35, w: 1.5, h: 0.28,
    fontSize: 8, fontFace: "Calibri", color: ACCENT,
    bold: true, margin: 0,
  });
}

function addSlideTitle(slide, title) {
  slide.addText(title, {
    x: 0.5, y: 0.7, w: 9, h: 0.5,
    fontSize: 24, fontFace: "Arial", color: TXT,
    bold: true, margin: 0,
  });
}

function addSubtitle(slide, text) {
  slide.addText(text, {
    x: 0.5, y: 1.2, w: 9, h: 0.35,
    fontSize: 11, fontFace: "Calibri", color: MUTED,
    margin: 0,
  });
}

// Fit image to maxW x maxH preserving aspect ratio, return {w, h}
function fitImage(origW, origH, maxW, maxH) {
  const scale = Math.min(maxW / origW, maxH / origH);
  return { w: origW * scale, h: origH * scale };
}

// ════════════════════════════════════════════════
// SLIDE 1: TITLE
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };

  // Top accent bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT },
  });

  // Title
  s.addText("Regime-Aware\nMean Reversion Strategy", {
    x: 0.8, y: 1.0, w: 8.4, h: 1.6,
    fontSize: 38, fontFace: "Arial", color: TXT,
    bold: true, margin: 0, lineSpacingMultiple: 1.1,
  });

  // Subtitle
  s.addText("Quantitative Crypto Spot Trading System  |  BTC  ETH  SOL  BNB  XRP", {
    x: 0.8, y: 2.7, w: 8.4, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: MUTED,
    margin: 0,
  });

  // Divider line
  s.addShape(pres.shapes.LINE, {
    x: 0.8, y: 3.2, w: 3.5, h: 0,
    line: { color: ACCENT, width: 2 },
  });

  // Key stats row
  const stats = [
    { label: "Backtest Period", value: "5 Years" },
    { label: "Assets", value: "5 Crypto" },
    { label: "Strategy Versions", value: "V2 vs V3" },
    { label: "Validation", value: "Walk-Forward" },
  ];
  stats.forEach((st, i) => {
    const x = 0.8 + i * 2.2;
    s.addText(st.value, {
      x, y: 3.55, w: 2.0, h: 0.35,
      fontSize: 16, fontFace: "Arial", color: ACCENT,
      bold: true, margin: 0,
    });
    s.addText(st.label, {
      x, y: 3.9, w: 2.0, h: 0.25,
      fontSize: 9, fontFace: "Calibri", color: MUTED,
      margin: 0,
    });
  });

  // Bottom info
  s.addText("March 2021 \u2013 March 2026  |  $100K Initial Capital  |  Long-Only", {
    x: 0.8, y: 4.8, w: 8.4, h: 0.3,
    fontSize: 10, fontFace: "Calibri", color: MUTED,
    margin: 0,
  });
}

// ════════════════════════════════════════════════
// SLIDE 2: PROJECT OVERVIEW
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSectionTag(s, "OVERVIEW");
  addSlideTitle(s, "Full-Pipeline Quantitative Research");
  addSubtitle(s, "From raw data exploration through statistically validated trading signals");

  // Pipeline steps as cards
  const steps = [
    { num: "01", title: "Exploratory Analysis", desc: "Return distributions, correlation structure, regime detection, tail risk, PCA", color: ACCENT },
    { num: "02", title: "Signal Engineering", desc: "OU-process regime scoring, Z-score entries, RSI confirmation, trend filter", color: GREEN },
    { num: "03", title: "Backtesting", desc: "Event-driven engine, multi-layer entries, correlation-aware sizing, circuit breakers", color: PURPLE },
    { num: "04", title: "Validation", desc: "Walk-forward OOS testing, bootstrap Sharpe CIs, regime dependence tests", color: GOLD },
  ];

  steps.forEach((step, i) => {
    const x = 0.5 + i * 2.35;
    const y = 1.85;
    // Card bg
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 2.15, h: 3.1,
      fill: { color: BG2 },
      line: { color: DIVIDER, width: 0.5 },
    });
    // Number
    s.addText(step.num, {
      x: x + 0.15, y: y + 0.15, w: 1.8, h: 0.5,
      fontSize: 28, fontFace: "Arial", color: step.color,
      bold: true, margin: 0,
    });
    // Title
    s.addText(step.title, {
      x: x + 0.15, y: y + 0.7, w: 1.8, h: 0.4,
      fontSize: 13, fontFace: "Arial", color: TXT,
      bold: true, margin: 0,
    });
    // Desc
    s.addText(step.desc, {
      x: x + 0.15, y: y + 1.15, w: 1.8, h: 1.7,
      fontSize: 10, fontFace: "Calibri", color: MUTED,
      margin: 0, valign: "top",
    });
  });

  addFooter(s, "Tools: Python  |  pandas, numpy, scipy, statsmodels, matplotlib, seaborn, sklearn  |  Custom event-driven backtester");
}

// ════════════════════════════════════════════════
// SLIDE 3: DATA & MARKET OVERVIEW
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSectionTag(s, "EDA");
  addSlideTitle(s, "Market Understanding: 5-Year Price Evolution");
  addSubtitle(s, "All assets normalized to base 100 on log scale \u2014 reveals relative performance across different price magnitudes");

  // Price evolution image (2085x885 -> fit into ~9x3.5)
  const pe = fitImage(2085, 885, 9.0, 3.2);
  s.addImage({
    path: path.join(EDA, "02_price_evolution.png"),
    x: (10 - pe.w) / 2, y: 1.75, w: pe.w, h: pe.h,
  });

  addFooter(s, "Data: Yahoo Finance OHLCV daily bars  |  Period: March 2021 \u2013 March 2026  |  Cached locally for reproducibility");
}

// ════════════════════════════════════════════════
// SLIDE 4: SUMMARY STATS + RETURN DISTRIBUTIONS
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSectionTag(s, "EDA");
  addSlideTitle(s, "Return Properties & Distributional Analysis");
  addSubtitle(s, "Fat tails and non-normality justify ATR-based position sizing over fixed percentage models");

  // Summary stats (2385x585 -> left side)
  const ss = fitImage(2385, 585, 9.0, 1.6);
  s.addImage({
    path: path.join(EDA, "01_summary_stats.png"),
    x: (10 - ss.w) / 2, y: 1.7, w: ss.w, h: ss.h,
  });

  // Return histograms (2985x619 -> bottom)
  const rh = fitImage(2985, 619, 9.0, 1.7);
  s.addImage({
    path: path.join(EDA, "03_return_histograms.png"),
    x: (10 - rh.w) / 2, y: 3.5, w: rh.w, h: rh.h,
  });

  addFooter(s, "All assets reject normality at p < 0.001 (Jarque-Bera, Shapiro-Wilk)  |  Excess kurtosis confirms heavy tails");
}

// ════════════════════════════════════════════════
// SLIDE 5: VOLATILITY CLUSTERING & CORRELATION
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSectionTag(s, "EDA");
  addSlideTitle(s, "Volatility Clustering & Correlation Structure");
  addSubtitle(s, "Persistent vol regimes + unstable correlations motivate adaptive regime detection and cluster-capping");

  // Rolling volatility (2085x885 -> top, ~4.4x1.85)
  const rv = fitImage(2085, 885, 4.4, 1.85);
  s.addImage({
    path: path.join(EDA, "05_rolling_volatility.png"),
    x: 0.4, y: 1.7, w: rv.w, h: rv.h,
  });

  // Correlation heatmap (1002x885 -> right side, ~4.3x3.8)
  const ch = fitImage(1002, 885, 4.4, 1.85);
  s.addImage({
    path: path.join(EDA, "07_correlation_heatmap.png"),
    x: 5.2, y: 1.7, w: ch.w, h: ch.h,
  });

  // ACF of squared returns (2985x619 -> bottom)
  const acf = fitImage(2985, 619, 9.0, 1.4);
  s.addImage({
    path: path.join(EDA, "06_squared_returns_acf.png"),
    x: (10 - acf.w) / 2, y: 3.8, w: acf.w, h: acf.h,
  });

  addFooter(s, "Significant ACF in r\u00B2 confirms GARCH-like clustering  |  BTC-ETH correlation fluctuates 0.5\u20130.95  |  Cluster threshold: \u03C1 \u2265 0.90");
}

// ════════════════════════════════════════════════
// SLIDE 6: REGIME ANALYSIS & PCA
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSectionTag(s, "EDA");
  addSlideTitle(s, "Market Regimes & Factor Structure");
  addSubtitle(s, "PC1 explains ~75%+ of variance \u2014 a single crypto market factor dominates, limiting diversification");

  // Regime analysis (2085x1485 -> left, tall)
  const ra = fitImage(2085, 1485, 4.6, 3.6);
  s.addImage({
    path: path.join(EDA, "10_regime_analysis.png"),
    x: 0.3, y: 1.7, w: ra.w, h: ra.h,
  });

  // PCA analysis (1785x735 -> right)
  const pca = fitImage(1785, 735, 4.6, 1.65);
  s.addImage({
    path: path.join(EDA, "09_pca_analysis.png"),
    x: 5.2, y: 1.7, w: pca.w, h: pca.h,
  });

  // Tail risk (2085x885 -> right bottom)
  const tr = fitImage(2085, 885, 4.6, 1.65);
  s.addImage({
    path: path.join(EDA, "11_tail_risk.png"),
    x: 5.2, y: 3.6, w: tr.w, h: tr.h,
  });

  addFooter(s, "Regime detection: rolling 60-day return + 30-day volatility  |  VaR/CVaR at 95% and 99% confidence levels");
}

// ════════════════════════════════════════════════
// SLIDE 7: STRATEGY ARCHITECTURE
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSectionTag(s, "STRATEGY");
  addSlideTitle(s, "Signal Architecture: V3 Multi-Layer System");
  addSubtitle(s, "Ornstein-Uhlenbeck regime scoring \u00D7 Z-score mean reversion \u00D7 correlation clustering");

  // Three-column layout for key components
  const cols = [
    {
      title: "Signal Generation",
      color: ACCENT,
      items: [
        "Z = (ln(P) \u2212 EMA) / \u03C3",
        "3 entry levels: Z=1.8, 2.0, 2.5",
        "RSI confirmation on weak signals",
        "200-day EMA trend filter",
        "ATR-based stop-loss (2.8\u00D7)",
      ],
    },
    {
      title: "Regime Detection",
      color: PURPLE,
      items: [
        "Rolling AR(1) \u2192 OU calibration",
        "\u03B8 = \u2212ln(1+b), T\u00BD = ln2/\u03B8",
        "Score = 0.4\u00D7halflife + 0.3\u00D7R\u00B2 + 0.3\u00D7pval",
        "Continuous sizing: 0.15\u00D7 \u2192 1.4\u00D7",
        "Hard floor at score < 0.1",
      ],
    },
    {
      title: "Risk Management",
      color: GREEN,
      items: [
        "ATR position sizing: risk/ATR\u00D7k",
        "Max 18% per layer, 38% total",
        "DD breaker: \u221250% at 15% DD",
        "Correlation cluster cap: 35%",
        "Max 15 concurrent positions",
      ],
    },
  ];

  cols.forEach((col, i) => {
    const x = 0.5 + i * 3.15;
    const y = 1.75;
    // Card
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 2.95, h: 3.3,
      fill: { color: BG2 },
      line: { color: DIVIDER, width: 0.5 },
    });
    // Left accent bar
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.06, h: 3.3,
      fill: { color: col.color },
    });
    // Title
    s.addText(col.title, {
      x: x + 0.2, y: y + 0.12, w: 2.6, h: 0.35,
      fontSize: 13, fontFace: "Arial", color: col.color,
      bold: true, margin: 0,
    });
    // Items
    const textItems = col.items.map((item, j) => ({
      text: item,
      options: {
        bullet: true,
        breakLine: j < col.items.length - 1,
        fontSize: 10,
        fontFace: "Calibri",
        color: MUTED,
        paraSpaceAfter: 6,
      },
    }));
    s.addText(textItems, {
      x: x + 0.2, y: y + 0.55, w: 2.55, h: 2.5,
      valign: "top", margin: 0,
    });
  });

  addFooter(s, "V3 improvements over V2: continuous regime scaling (vs binary), multi-layer entries, correlation-aware allocation, RSI confirmation");
}

// ════════════════════════════════════════════════
// SLIDE 8: V2 vs V3 EQUITY COMPARISON
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSectionTag(s, "RESULTS");
  addSlideTitle(s, "Before & After: V2 vs V3 Equity Curves");
  addSubtitle(s, "V3 (blue) achieves higher returns with better capital utilization \u2014 walk-forward OOS in gold");

  // Main comparison chart (2084x1783 -> fit to ~9x4.0)
  const eq = fitImage(2084, 1783, 9.0, 3.7);
  s.addImage({
    path: path.join(IMG, "v2_vs_v3_comparison.png"),
    x: (10 - eq.w) / 2, y: 1.65, w: eq.w, h: eq.h,
  });

  addFooter(s, "Red = V2 baseline (single entry, binary regime)  |  Blue = V3 (multi-layer, continuous regime, correlation-aware)  |  Gold = Walk-forward OOS");
}

// ════════════════════════════════════════════════
// SLIDE 9: TRADE ANALYTICS
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSectionTag(s, "RESULTS");
  addSlideTitle(s, "Trade Distribution & Capital Utilization");
  addSubtitle(s, "V3 generates significantly more trades across all assets with tighter holding periods");

  // Trade distribution (2083x1481 -> left)
  const td = fitImage(2083, 1481, 5.0, 3.5);
  s.addImage({
    path: path.join(IMG, "trade_distribution.png"),
    x: 0.2, y: 1.7, w: td.w, h: td.h,
  });

  // Exposure timeline (2084x1182 -> right, slightly smaller)
  const et = fitImage(2084, 1182, 4.5, 3.5);
  s.addImage({
    path: path.join(IMG, "exposure_timeline.png"),
    x: 5.3, y: 1.7, w: et.w, h: et.h,
  });

  addFooter(s, "V3 multi-layer system: trades per asset, return distribution, holding period, PnL by entry layer  |  Exposure target: 30%");
}

// ════════════════════════════════════════════════
// SLIDE 10: BTC DEEP DIVE
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSectionTag(s, "DEEP DIVE");
  addSlideTitle(s, "BTC-USD: Price, Z-Score, Regime & RSI");
  addSubtitle(s, "Four-panel signal decomposition showing exactly when and why the strategy trades");

  // BTC detail (2084x2083 -> fit to center)
  const btc = fitImage(2084, 2083, 5.5, 3.7);
  s.addImage({
    path: path.join(IMG, "v3_detail_BTC_USD.png"),
    x: 0.3, y: 1.7, w: btc.w, h: btc.h,
  });

  // Annotation text on the right
  const annotations = [
    { label: "Panel 1: Price", desc: "Trade arrows show entry \u2192 exit. Green = profit, red = loss. Visual confirmation of mean-reversion behavior." },
    { label: "Panel 2: Z-Score", desc: "Multi-level thresholds at Z = 1.8, 2.0, 2.5. Entries fire when price deviates significantly from rolling EMA." },
    { label: "Panel 3: Regime", desc: "OU-calibrated score. Positions scale continuously with regime quality. Red dashed line = hard floor (0.1)." },
    { label: "Panel 4: RSI", desc: "Momentum confirmation for weak (Layer 0) signals only. Oversold < 40 required for long entries." },
  ];

  annotations.forEach((ann, i) => {
    const y = 1.75 + i * 0.9;
    s.addText(ann.label, {
      x: 5.8, y, w: 3.8, h: 0.25,
      fontSize: 11, fontFace: "Arial", color: ACCENT,
      bold: true, margin: 0,
    });
    s.addText(ann.desc, {
      x: 5.8, y: y + 0.25, w: 3.8, h: 0.55,
      fontSize: 9, fontFace: "Calibri", color: MUTED,
      margin: 0, valign: "top",
    });
  });

  addFooter(s, "Same 4-panel analysis generated for all 5 assets: BTC, ETH, SOL, BNB, XRP");
}

// ════════════════════════════════════════════════
// SLIDE 11: WALK-FORWARD + SENSITIVITY
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSectionTag(s, "VALIDATION");
  addSlideTitle(s, "Walk-Forward Validation & Parameter Sensitivity");
  addSubtitle(s, "Out-of-sample Sharpe ratios by fold + robustness across Z-threshold and window parameter space");

  // WF fold sharpes (1484x732 -> left)
  const wf = fitImage(1484, 732, 4.6, 3.2);
  s.addImage({
    path: path.join(IMG, "v3_wf_fold_sharpes.png"),
    x: 0.3, y: 1.75, w: wf.w, h: wf.h,
  });

  // Sensitivity heatmap (1377x882 -> right)
  const sh = fitImage(1377, 882, 4.6, 3.2);
  s.addImage({
    path: path.join(IMG, "sensitivity_sharpe_ratio.png"),
    x: 5.2, y: 1.75, w: sh.w, h: sh.h,
  });

  addFooter(s, "Walk-forward: 252d train / 126d test, stepping 126d  |  Grid search: Z \u2208 {1.5, 1.8, 2.0, 2.5}, window \u2208 {20, 30, 40, 50, 60}");
}

// ════════════════════════════════════════════════
// SLIDE 12: STATISTICAL RIGOR
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };
  addSectionTag(s, "STATISTICS");
  addSlideTitle(s, "Statistical Significance & Regime Dependence");
  addSubtitle(s, "Bootstrap confidence intervals on Sharpe ratios + hypothesis test on regime score predictive power");

  // Bootstrap Sharpe (1183x484 -> left)
  const bs = fitImage(1183, 484, 4.5, 2.0);
  s.addImage({
    path: path.join(EDA, "stat_02_sharpe_bootstrap.png"),
    x: 0.3, y: 1.75, w: bs.w, h: bs.h,
  });

  // Regime dependence (784x484 -> right top)
  const rd = fitImage(784, 484, 4.3, 2.0);
  s.addImage({
    path: path.join(EDA, "stat_03_regime_dependence.png"),
    x: 5.3, y: 1.75, w: rd.w, h: rd.h,
  });

  // V2 vs V3 diff (1384x784 -> bottom center)
  const diff = fitImage(1384, 784, 9.0, 1.7);
  s.addImage({
    path: path.join(EDA, "stat_04_v2_v3_diff.png"),
    x: (10 - diff.w) / 2, y: 3.9, w: diff.w, h: diff.h,
  });

  addFooter(s, "Bootstrap: 10,000 resamples  |  Regime split: Mann-Whitney U + Welch\u2019s t-test  |  V2 vs V3: paired t-test + Wilcoxon signed-rank");
}

// ════════════════════════════════════════════════
// SLIDE 13: CLOSING
// ════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: BG };

  // Top accent bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT },
  });

  s.addText("Key Takeaways", {
    x: 0.8, y: 0.5, w: 8.4, h: 0.6,
    fontSize: 30, fontFace: "Arial", color: TXT,
    bold: true, margin: 0,
  });

  // Takeaway cards in 2x2 grid
  const takeaways = [
    { title: "End-to-End Pipeline", desc: "Built complete research workflow: data ingestion \u2192 EDA \u2192 signal engineering \u2192 backtesting \u2192 walk-forward validation \u2192 statistical testing", color: ACCENT },
    { title: "Statistically Validated", desc: "Bootstrap Sharpe CIs, regime dependence tests, Ljung-Box serial correlation checks, paired V2/V3 significance tests", color: GREEN },
    { title: "Robust Design", desc: "Walk-forward OOS confirms out-of-sample predictive power; sensitivity heatmaps show stability across parameter space", color: PURPLE },
    { title: "Professional Engineering", desc: "Modular Python codebase, interactive Plotly dashboard, frozen dataclass configs, event-driven backtest engine", color: GOLD },
  ];

  takeaways.forEach((tk, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.8 + col * 4.3;
    const y = 1.4 + row * 1.8;

    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 3.9, h: 1.5,
      fill: { color: BG2 },
      line: { color: DIVIDER, width: 0.5 },
    });
    // Left accent
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.06, h: 1.5,
      fill: { color: tk.color },
    });
    s.addText(tk.title, {
      x: x + 0.2, y: y + 0.12, w: 3.5, h: 0.3,
      fontSize: 14, fontFace: "Arial", color: tk.color,
      bold: true, margin: 0,
    });
    s.addText(tk.desc, {
      x: x + 0.2, y: y + 0.5, w: 3.5, h: 0.85,
      fontSize: 10, fontFace: "Calibri", color: MUTED,
      margin: 0, valign: "top",
    });
  });

  // Bottom
  s.addShape(pres.shapes.LINE, {
    x: 0.8, y: 5.0, w: 3.5, h: 0,
    line: { color: ACCENT, width: 2 },
  });
  s.addText("github.com \u2014 Full source code, interactive dashboard, and detailed documentation available", {
    x: 0.8, y: 5.1, w: 8.4, h: 0.3,
    fontSize: 10, fontFace: "Calibri", color: MUTED,
    margin: 0,
  });
}

// ── Write ──
const outPath = path.join(__dirname, "output", "presentation.pptx");
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("Presentation saved to: " + outPath);
}).catch(err => {
  console.error("Error:", err);
});
