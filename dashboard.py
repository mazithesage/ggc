"""
Interactive HTML dashboard for V3 Regime-Aware Mean Reversion Strategy.
Before/After comparison: V2 (low exposure) vs V3 (capital-efficient multi-entry).
Embeds all data inline — no server needed beyond static file serving.
All HTML content is generated from trusted internal backtest data only.
"""
import json
import os
import numpy as np
import pandas as pd
from config import StrategyConfig, get_v2_config
from data.loader import load_all
from strategy.signals import generate_signals
from strategy.correlation import compute_rolling_correlations
from backtest.engine import BacktestEngine
from backtest.walkforward import walk_forward_test
from analysis.metrics import compute_metrics


def run_comparison_for_dashboard():
    """Run V2 + V3 strategies and return all data for the comparison dashboard."""
    v3_cfg = StrategyConfig()
    v2_cfg = get_v2_config()
    data = load_all(v3_cfg)

    # Correlations for V3
    pair_corr = compute_rolling_correlations(data, v3_cfg.correlation_window)

    # V2 baseline
    print("  Running V2 baseline...")
    v2_signals = {t: generate_signals(df, v2_cfg, long_only=True) for t, df in data.items()}
    v2_engine = BacktestEngine(v2_cfg)
    v2_eq = v2_engine.run(v2_signals)
    v2_trades = v2_engine.get_trades_df()
    v2_metrics = compute_metrics(v2_eq, v2_trades, v2_cfg)

    v2_per_asset = {}
    for t, sdf in v2_signals.items():
        eng = BacktestEngine(v2_cfg)
        eq = eng.run({t: sdf})
        trd = eng.get_trades_df()
        v2_per_asset[t] = {"equity": eq, "trades": trd, "metrics": compute_metrics(eq, trd, v2_cfg)}

    # V3 capital-efficient
    print("  Running V3 capital-efficient system...")
    v3_signals = {t: generate_signals(df, v3_cfg, long_only=True) for t, df in data.items()}
    v3_engine = BacktestEngine(v3_cfg)
    v3_eq = v3_engine.run(v3_signals, pair_corr)
    v3_trades = v3_engine.get_trades_df()
    v3_metrics = compute_metrics(v3_eq, v3_trades, v3_cfg)

    v3_per_asset = {}
    for t, sdf in v3_signals.items():
        eng = BacktestEngine(v3_cfg)
        eq = eng.run({t: sdf}, pair_corr)
        trd = eng.get_trades_df()
        v3_per_asset[t] = {"equity": eq, "trades": trd, "metrics": compute_metrics(eq, trd, v3_cfg)}

    # Walk-forward
    print("  Running walk-forward validation...")
    wf = walk_forward_test(data, v3_cfg, long_only=True, pair_correlations=pair_corr)
    wf_metrics = {}
    if len(wf["oos_equity"]) > 0:
        wf_metrics = compute_metrics(wf["oos_equity"], wf["oos_trades"], v3_cfg)

    return {
        "data": data, "v2_cfg": v2_cfg, "v3_cfg": v3_cfg,
        "v2_signals": v2_signals, "v3_signals": v3_signals,
        "v2_eq": v2_eq, "v2_trades": v2_trades, "v2_metrics": v2_metrics,
        "v3_eq": v3_eq, "v3_trades": v3_trades, "v3_metrics": v3_metrics,
        "v2_per_asset": v2_per_asset, "v3_per_asset": v3_per_asset,
        "wf_eq": wf["oos_equity"], "wf_trades": wf["oos_trades"],
        "wf_metrics": wf_metrics,
        "fold_results": wf["fold_results"], "best_params": wf["best_params_per_fold"],
    }


def to_js(series):
    """Convert pandas series to JSON-safe list."""
    vals = series.tolist()
    return [None if (isinstance(v, float) and np.isnan(v)) else v for v in vals]


def fmt_m(m):
    """Format metrics dict for JS."""
    if not m:
        return {}
    return {
        "CAGR": f"{m.get('CAGR', 0):.2%}",
        "Sharpe": f"{m.get('Sharpe Ratio', 0):.3f}",
        "Sortino": f"{m.get('Sortino Ratio', 0):.3f}",
        "Calmar": f"{m.get('Calmar Ratio', 0):.3f}",
        "MaxDD": f"{m.get('Max Drawdown', 0):.2%}",
        "WinRate": f"{m.get('Win Rate', 0):.1%}",
        "PF": f"{m.get('Profit Factor', 0):.3f}",
        "Trades": str(m.get("Total Trades", 0)),
        "Return": f"{m.get('Total Return', 0):.2%}",
        "Exposure": f"{m.get('Avg Exposure', 0):.1%}",
        "ExpAdj": f"{m.get('Exposure-Adj Return', 0):.2%}",
    }


def fmt_asset(m):
    return {
        "Sharpe": f"{m.get('Sharpe Ratio', 0):.3f}",
        "CAGR": f"{m.get('CAGR', 0):.2%}",
        "MaxDD": f"{m.get('Max Drawdown', 0):.2%}",
        "WinRate": f"{m.get('Win Rate', 0):.1%}",
        "Trades": str(m.get("Total Trades", 0)),
        "Exposure": f"{m.get('Avg Exposure', 0):.1%}",
    }


def trades_to_list(trades_df, include_layer=False):
    """Convert trades DataFrame to JSON-safe list."""
    result = []
    if len(trades_df) == 0:
        return result
    for _, t in trades_df.iterrows():
        entry = {
            "ticker": t["ticker"],
            "direction": t["direction"],
            "entry_date": str(t["entry_date"].date()),
            "exit_date": str(t["exit_date"].date()),
            "entry_price": round(float(t["entry_price"]), 2),
            "exit_price": round(float(t["exit_price"]), 2),
            "pnl": round(float(t["pnl"]), 2),
            "return_pct": round(float(t["return_pct"]) * 100, 2),
            "bars_held": int(t["bars_held"]),
            "exit_reason": t["exit_reason"],
            "regime_score": round(float(t["regime_score"]), 3),
        }
        if include_layer and "layer" in t:
            entry["layer"] = int(t["layer"])
        result.append(entry)
    return result


def generate_html(r):
    """Generate the V3 comparison dashboard HTML."""
    v3_cfg = r["v3_cfg"]
    tickers = list(r["v3_signals"].keys())

    # Per-asset signal data (V3 has RSI, trend_strength, regime_scale)
    asset_data = {}
    for t in tickers:
        sdf = r["v3_signals"][t]
        asset_data[t] = {
            "dates": [d.strftime("%Y-%m-%d") for d in sdf.index],
            "close": to_js(sdf["close"]),
            "zscore": to_js(sdf["zscore"]),
            "regime_score": to_js(sdf["regime_score"]),
            "regime_scale": to_js(sdf["regime_scale"]),
            "rsi": to_js(sdf["rsi"]),
            "trend_strength": to_js(sdf["trend_strength"]),
            "halflife": to_js(sdf["halflife"]),
            "atr": to_js(sdf["atr"]),
        }

    # Equity curves
    def eq_data(eq):
        return {
            "dates": [d.strftime("%Y-%m-%d") for d in eq.index],
            "equity": to_js(eq["equity"]),
            "exposure": to_js(eq["exposure"]),
            "drawdown": to_js(eq["drawdown"]),
            "n_positions": to_js(eq["n_positions"]),
        }

    # Layer analysis
    layer_data = []
    v3t = r["v3_trades"]
    if len(v3t) > 0 and "layer" in v3t.columns:
        for layer in sorted(v3t["layer"].unique()):
            lt = v3t[v3t["layer"] == layer]
            wins = lt[lt["pnl"] > 0]
            losses = lt[lt["pnl"] <= 0]
            layer_data.append({
                "layer": int(layer),
                "trades": len(lt),
                "pnl": round(float(lt["pnl"].sum()), 2),
                "wr": round(len(wins) / len(lt) * 100, 1) if len(lt) > 0 else 0,
                "avg_return": round(float(lt["return_pct"].mean() * 100), 2),
                "avg_bars": round(float(lt["bars_held"].mean()), 1),
                "avg_win": round(float(wins["return_pct"].mean() * 100), 2) if len(wins) > 0 else 0,
                "avg_loss": round(float(losses["return_pct"].mean() * 100), 2) if len(losses) > 0 else 0,
            })

    # Fold results
    fold_data = []
    for fr in r["fold_results"]:
        fold_data.append({
            "fold": fr["fold"], "sharpe": round(fr["sharpe"], 3),
            "ret": round(fr["return"] * 100, 2), "trades": fr["trades"],
            "z": fr.get("best_z", "?"), "w": fr.get("best_w", "?"),
        })

    bp_data = [{"fold": bp["fold"], "z": bp["z_entry"], "w": bp["window"],
                "train_sharpe": round(bp["train_sharpe"], 3)} for bp in r["best_params"]]

    # Entry level descriptions
    entry_levels = [{"z": z, "size": sf, "rsi": req} for z, sf, req in v3_cfg.entry_levels]

    js_data = json.dumps({
        "tickers": tickers,
        "assets": asset_data,
        "v2_eq": eq_data(r["v2_eq"]),
        "v3_eq": eq_data(r["v3_eq"]),
        "wf_eq": {"dates": [d.strftime("%Y-%m-%d") for d in r["wf_eq"].index], "equity": to_js(r["wf_eq"]["equity"])} if len(r["wf_eq"]) > 0 else {},
        "v2_metrics": fmt_m(r["v2_metrics"]),
        "v3_metrics": fmt_m(r["v3_metrics"]),
        "wf_metrics": fmt_m(r["wf_metrics"]),
        "v2_per_asset": {t: fmt_asset(d["metrics"]) for t, d in r["v2_per_asset"].items()},
        "v3_per_asset": {t: fmt_asset(d["metrics"]) for t, d in r["v3_per_asset"].items()},
        "v2_per_asset_eq": {t: {"dates": [d.strftime("%Y-%m-%d") for d in d["equity"].index], "equity": to_js(d["equity"]["equity"])} for t, d in r["v2_per_asset"].items()},
        "v3_per_asset_eq": {t: {"dates": [d.strftime("%Y-%m-%d") for d in d["equity"].index], "equity": to_js(d["equity"]["equity"])} for t, d in r["v3_per_asset"].items()},
        "v3_trades": trades_to_list(r["v3_trades"], include_layer=True),
        "v2_trades": trades_to_list(r["v2_trades"]),
        "layers": layer_data,
        "folds": fold_data,
        "best_params": bp_data,
        "entry_levels": entry_levels,
        "params": {
            "z_exit": v3_cfg.z_exit,
            "regime_threshold": v3_cfg.regime_score_threshold,
            "corr_threshold": v3_cfg.correlation_threshold,
            "cluster_cap": v3_cfg.max_cluster_exposure,
            "max_exposure": v3_cfg.max_total_exposure,
            "stop_atr": v3_cfg.stop_atr_multiple,
        },
    })

    html = _HTML_TEMPLATE.replace("__JS_DATA__", js_data)
    return html


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Regime MR v3 — Before/After Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#c9d1d9}
  .hdr{background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);padding:20px 32px;border-bottom:1px solid #30363d;display:flex;align-items:center;justify-content:space-between}
  .hdr h1{font-size:20px;color:#f0f6fc;font-weight:600}
  .hdr .sub{color:#8b949e;font-size:12px;margin-top:4px}
  .mbar{display:flex;gap:8px;padding:12px 32px;background:#161b22;border-bottom:1px solid #30363d;flex-wrap:wrap;overflow-x:auto}
  .mc{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px 14px;min-width:100px}
  .mc .l{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
  .mc .v{font-size:16px;font-weight:700;margin-top:2px;font-variant-numeric:tabular-nums}
  .mc .v.pos{color:#3fb950}.mc .v.neg{color:#f85149}.mc .v.neu{color:#d2a8ff}
  .mc .delta{font-size:10px;margin-top:1px}
  .tabs{display:flex;gap:0;padding:0 32px;background:#161b22;border-bottom:1px solid #30363d;overflow-x:auto}
  .tab{padding:10px 16px;cursor:pointer;color:#8b949e;font-size:12px;font-weight:500;border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap}
  .tab:hover{color:#c9d1d9}.tab.active{color:#f0f6fc;border-bottom-color:#58a6ff}
  .cc{padding:16px 32px}
  .ch{background:#161b22;border:1px solid #30363d;border-radius:8px;margin-bottom:12px;overflow:hidden}
  .ct{padding:12px 16px;font-size:13px;font-weight:600;color:#f0f6fc;border-bottom:1px solid #21262d}
  .two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  @media(max-width:900px){.two{grid-template-columns:1fr}}
  .tbl{width:100%;border-collapse:collapse;font-size:12px}
  .tbl th{text-align:left;padding:8px 12px;background:#161b22;color:#8b949e;font-weight:500;text-transform:uppercase;font-size:10px;letter-spacing:.5px;border-bottom:1px solid #30363d;position:sticky;top:0;z-index:1}
  .tbl td{padding:6px 12px;border-bottom:1px solid #21262d}
  .tbl tr:hover{background:#1c2128}
  .badge{display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:600}
  .badge.long{background:#0d419d;color:#79c0ff}.badge.short{background:#6e1b1b;color:#ffa198}
  .win{color:#3fb950}.loss{color:#f85149}
  .scrl{max-height:500px;overflow-y:auto}
  .acards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-bottom:12px}
  .acard{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px 16px}
  .acard .nm{font-size:13px;font-weight:600;color:#f0f6fc;margin-bottom:6px}
  .acard .st{font-size:11px;color:#8b949e;margin-top:2px}
  .acard .st span{color:#c9d1d9;font-weight:500}
  .v2tag{color:#f85149;font-weight:600;font-size:10px}.v3tag{color:#58a6ff;font-weight:600;font-size:10px}
  .layer-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:12px}
  .lcard{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:14px;text-align:center}
  .lcard .ln{font-size:16px;font-weight:700;color:#f0f6fc}
  .lcard .lm{font-size:11px;color:#8b949e;margin-top:4px}
  .lcard .lm span{font-weight:500;color:#c9d1d9}
</style>
</head>
<body>

<div class="hdr">
  <div>
    <h1>Regime-Aware Mean Reversion &mdash; V2 vs V3 Comparison</h1>
    <div class="sub" id="hSub"></div>
  </div>
  <div style="text-align:right" id="hParams"></div>
</div>
<div class="mbar" id="mbar"></div>
<div class="tabs" id="tabBar"></div>
<div class="cc" id="content"></div>

<script>
var D = __JS_DATA__;

var PL = {
  paper_bgcolor:'#161b22',plot_bgcolor:'#0d1117',
  font:{color:'#c9d1d9',family:'-apple-system,sans-serif',size:11},
  margin:{l:55,r:15,t:8,b:35},
  xaxis:{gridcolor:'#21262d',linecolor:'#30363d',showgrid:true,zeroline:false},
  yaxis:{gridcolor:'#21262d',linecolor:'#30363d',showgrid:true,zeroline:false},
  legend:{bgcolor:'transparent',font:{size:10}},
  hovermode:'x unified'
};
var PC = {responsive:true,displayModeBar:true,displaylogo:false};

function el(tag,cls,txt){var e=document.createElement(tag);if(cls)e.className=cls;if(txt)e.textContent=txt;return e}
function clear(id){var c=document.getElementById(id);while(c.firstChild)c.removeChild(c.firstChild);return c}

// Header
(function(){
  document.getElementById('hSub').textContent='Multi-Entry + Regime Scaling + Correlation-Aware \\u2014 '+D.tickers.join(', ');
  var hp=document.getElementById('hParams');
  var d1=el('div','','V3: Z=['+D.entry_levels.map(function(l){return l.z}).join(',')+'] | Stop='+D.params.stop_atr+'\\u00d7ATR');
  d1.style.cssText='font-size:11px;color:#8b949e';
  var d2=el('div','','Corr\\u2265'+D.params.corr_threshold+' | ClusterCap='+Math.round(D.params.cluster_cap*100)+'% | MaxExp='+Math.round(D.params.max_exposure*100)+'%');
  d2.style.cssText='font-size:11px;color:#8b949e';
  hp.appendChild(d1);hp.appendChild(d2);
})();

// Metrics bar: V3 with delta from V2
function renderMetrics(){
  var m3=D.v3_metrics,m2=D.v2_metrics;
  var items=[
    ['CAGR',m3.CAGR,parseFloat(m3.CAGR),parseFloat(m2.CAGR)],
    ['Sharpe',m3.Sharpe,parseFloat(m3.Sharpe),parseFloat(m2.Sharpe)],
    ['Max DD',m3.MaxDD,-1,0],
    ['Trades',m3.Trades,0,0],
    ['Win Rate',m3.WinRate,parseFloat(m3.WinRate),parseFloat(m2.WinRate)],
    ['PF',m3.PF,parseFloat(m3.PF)-1,0],
    ['Exposure',m3.Exposure,0,0],
    ['Return',m3.Return,parseFloat(m3.Return),parseFloat(m2.Return)],
  ];
  var bar=clear('mbar');
  items.forEach(function(it){
    var label=it[0],value=it[1],sign=it[2],v2v=it[3];
    var cls=sign>0.001?'pos':sign<-0.001?'neg':'neu';
    if(label==='Trades'||label==='Max DD'||label==='Exposure')cls='neu';
    var card=el('div','mc');
    card.appendChild(el('div','l',label));
    var val=el('div','v '+cls,value);
    card.appendChild(val);
    if(v2v!==0 && label!=='Max DD' && label!=='Trades' && label!=='Exposure'){
      var delta=parseFloat(it[1])-v2v;
      var dEl=el('div','delta',(delta>=0?'+':'')+delta.toFixed(2)+' vs V2');
      dEl.style.color=delta>=0?'#3fb950':'#f85149';
      card.appendChild(dEl);
    }
    bar.appendChild(card);
  });
}

// Tabs
(function(){
  var tabBar=document.getElementById('tabBar');
  var tabs=[
    {id:'compare',label:'V2 vs V3'},
    {id:'equity',label:'Equity & DD'},
    {id:'overview',label:'Asset Comparison'},
    {id:'layers',label:'V3 Layers'},
  ];
  D.tickers.forEach(function(t){tabs.push({id:'asset_'+t.replace('-',''),label:t})});
  tabs.push({id:'wf',label:'Walk-Forward'});
  tabs.push({id:'trades',label:'Trade Log'});
  tabs.forEach(function(t,i){
    var d=el('div','tab'+(i===0?' active':''),t.label);
    d.dataset.tab=t.id;
    tabBar.appendChild(d);
  });
})();

function switchTab(id){
  document.querySelectorAll('.tab').forEach(function(t){t.classList.toggle('active',t.dataset.tab===id)});
  if(id==='compare')renderCompare();
  else if(id==='equity')renderEquity();
  else if(id==='overview')renderOverview();
  else if(id==='layers')renderLayers();
  else if(id==='wf')renderWF();
  else if(id==='trades')renderTrades();
  else if(id.startsWith('asset_')){
    var ticker=D.tickers.find(function(t){return 'asset_'+t.replace('-','')===id});
    if(ticker)renderAsset(ticker);
  }
}
document.getElementById('tabBar').addEventListener('click',function(e){
  if(e.target.classList.contains('tab'))switchTab(e.target.dataset.tab);
});

// ── TAB: V2 vs V3 Comparison ──
function renderCompare(){
  var c=clear('content');
  // Comparison table
  var wrap=el('div','ch');
  wrap.appendChild(el('div','ct','Before (V2) vs After (V3) Metrics'));
  var scrl=el('div','scrl');
  var tbl=el('table','tbl');
  var thead=el('thead');
  var hr=el('tr');
  ['Metric','V2 (Before)','V3 (After)','Delta'].forEach(function(h){hr.appendChild(el('th','',h))});
  thead.appendChild(hr);tbl.appendChild(thead);

  var tbody=el('tbody');
  var rows=[
    ['CAGR',D.v2_metrics.CAGR,D.v3_metrics.CAGR],
    ['Sharpe',D.v2_metrics.Sharpe,D.v3_metrics.Sharpe],
    ['Sortino',D.v2_metrics.Sortino,D.v3_metrics.Sortino],
    ['Calmar',D.v2_metrics.Calmar,D.v3_metrics.Calmar],
    ['Max DD',D.v2_metrics.MaxDD,D.v3_metrics.MaxDD],
    ['Win Rate',D.v2_metrics.WinRate,D.v3_metrics.WinRate],
    ['Profit Factor',D.v2_metrics.PF,D.v3_metrics.PF],
    ['Total Trades',D.v2_metrics.Trades,D.v3_metrics.Trades],
    ['Avg Exposure',D.v2_metrics.Exposure,D.v3_metrics.Exposure],
    ['Exp-Adj Return',D.v2_metrics.ExpAdj,D.v3_metrics.ExpAdj],
    ['Total Return',D.v2_metrics.Return,D.v3_metrics.Return],
  ];
  rows.forEach(function(r){
    var tr=el('tr');
    tr.appendChild(el('td','',r[0]));
    tr.appendChild(el('td','',r[1]));
    tr.appendChild(el('td','',r[2]));
    var d=parseFloat(r[2])-parseFloat(r[1]);
    var dtd=el('td');
    var dsp=el('span',isNaN(d)?'':d>=0?'win':'loss',isNaN(d)?'—':(d>=0?'+':'')+d.toFixed(3));
    dtd.appendChild(dsp);tr.appendChild(dtd);
    tbody.appendChild(tr);
  });
  tbl.appendChild(tbody);scrl.appendChild(tbl);wrap.appendChild(scrl);c.appendChild(wrap);

  // V3 entry levels
  var lw=el('div','ch');
  lw.appendChild(el('div','ct','V3 Entry Level Configuration'));
  var lt=el('table','tbl');
  var lth=el('thead');var lhr=el('tr');
  ['Level','Z Threshold','Size Factor','RSI Required'].forEach(function(h){lhr.appendChild(el('th','',h))});
  lth.appendChild(lhr);lt.appendChild(lth);
  var ltb=el('tbody');
  D.entry_levels.forEach(function(lv,i){
    var tr=el('tr');
    tr.appendChild(el('td','','L'+i));
    tr.appendChild(el('td','',String(lv.z)));
    tr.appendChild(el('td','',Math.round(lv.size*100)+'%'));
    tr.appendChild(el('td','',lv.rsi?'Yes':'No'));
    ltb.appendChild(tr);
  });
  lt.appendChild(ltb);lw.appendChild(lt);c.appendChild(lw);
}

// ── TAB: Equity & Drawdown ──
function renderEquity(){
  var c=clear('content');

  // Equity comparison
  var ew=el('div','ch');ew.appendChild(el('div','ct','Equity Curves: V2 vs V3'));
  var ed=el('div');ed.id='eqC';ed.style.height='380px';ew.appendChild(ed);c.appendChild(ew);

  var traces=[
    {x:D.v2_eq.dates,y:D.v2_eq.equity,name:'V2 (Before)',line:{color:'#f85149',width:1.5}},
    {x:D.v3_eq.dates,y:D.v3_eq.equity,name:'V3 (After)',line:{color:'#58a6ff',width:2}},
  ];
  if(D.wf_eq.dates&&D.wf_eq.dates.length>0)
    traces.push({x:D.wf_eq.dates,y:D.wf_eq.equity,name:'V3 WF OOS',line:{color:'#d29922',width:1.2}});
  traces.push({x:D.v3_eq.dates,y:D.v3_eq.dates.map(function(){return 100000}),name:'$100K',line:{color:'#484f58',width:1,dash:'dot'}});
  Plotly.newPlot('eqC',traces,Object.assign({},PL,{yaxis:Object.assign({},PL.yaxis,{title:'Equity ($)',tickprefix:'$',tickformat:',.0f'})}),PC);

  // DD + Exposure row
  var tw=el('div','two');
  var ddw=el('div','ch');ddw.appendChild(el('div','ct','Drawdown Comparison'));
  var ddd=el('div');ddd.id='ddC';ddd.style.height='240px';ddw.appendChild(ddd);tw.appendChild(ddw);

  var exw=el('div','ch');exw.appendChild(el('div','ct','Exposure Over Time'));
  var exd=el('div');exd.id='exC';exd.style.height='240px';exw.appendChild(exd);tw.appendChild(exw);
  c.appendChild(tw);

  Plotly.newPlot('ddC',[
    {x:D.v2_eq.dates,y:D.v2_eq.drawdown.map(function(v){return v!==null?-v*100:null}),fill:'tozeroy',fillcolor:'rgba(248,81,73,0.12)',line:{color:'#f85149',width:1},name:'V2 DD'},
    {x:D.v3_eq.dates,y:D.v3_eq.drawdown.map(function(v){return v!==null?-v*100:null}),fill:'tozeroy',fillcolor:'rgba(88,166,255,0.15)',line:{color:'#58a6ff',width:1},name:'V3 DD'},
  ],Object.assign({},PL,{yaxis:Object.assign({},PL.yaxis,{title:'DD %',ticksuffix:'%'})}),PC);

  Plotly.newPlot('exC',[
    {x:D.v2_eq.dates,y:D.v2_eq.exposure.map(function(v){return v!==null?v*100:null}),fill:'tozeroy',fillcolor:'rgba(248,81,73,0.12)',line:{color:'#f85149',width:1},name:'V2'},
    {x:D.v3_eq.dates,y:D.v3_eq.exposure.map(function(v){return v!==null?v*100:null}),fill:'tozeroy',fillcolor:'rgba(88,166,255,0.15)',line:{color:'#58a6ff',width:1},name:'V3'},
  ],Object.assign({},PL,{yaxis:Object.assign({},PL.yaxis,{title:'Exposure %',ticksuffix:'%',range:[0,60]})}),PC);

  // Position count
  var pw=el('div','ch');pw.appendChild(el('div','ct','V3 Active Positions'));
  var pd=el('div');pd.id='posC';pd.style.height='200px';pw.appendChild(pd);c.appendChild(pw);
  Plotly.newPlot('posC',[
    {x:D.v3_eq.dates,y:D.v3_eq.n_positions,fill:'tozeroy',fillcolor:'rgba(210,168,255,0.2)',line:{color:'#d2a8ff',width:1},name:'# Positions'},
  ],Object.assign({},PL,{yaxis:Object.assign({},PL.yaxis,{title:'# Active Positions'})}),PC);
}

// ── TAB: Asset Comparison ──
function renderOverview(){
  var c=clear('content');
  var cards=el('div','acards');
  D.tickers.forEach(function(t){
    var m2=D.v2_per_asset[t]||{},m3=D.v3_per_asset[t]||{};
    var card=el('div','acard');
    card.appendChild(el('div','nm',t));
    [['Sharpe','Sharpe'],['Trades','Trades'],['Exposure','Exposure'],['Max DD','MaxDD'],['Win Rate','WinRate']].forEach(function(pair){
      var st=el('div','st');
      st.textContent=pair[0]+': ';
      var v2s=el('span','v2tag',(m2[pair[1]]||'—'));
      var sep=document.createTextNode(' \\u2192 ');
      var v3s=el('span','v3tag',(m3[pair[1]]||'—'));
      st.appendChild(v2s);st.appendChild(sep);st.appendChild(v3s);
      card.appendChild(st);
    });
    cards.appendChild(card);
  });
  c.appendChild(cards);

  // Per-asset equity comparison
  var ew=el('div','ch');ew.appendChild(el('div','ct','Per-Asset Equity (V3, Isolated)'));
  var ed=el('div');ed.id='oEq';ed.style.height='350px';ew.appendChild(ed);c.appendChild(ew);
  var colors=['#58a6ff','#3fb950','#d2a8ff','#d29922','#f85149'];
  var traces=D.tickers.map(function(t,i){
    var d=D.v3_per_asset_eq[t];
    return {x:d.dates,y:d.equity,name:t,line:{color:colors[i%5],width:1.5}};
  });
  traces.push({x:D.v3_eq.dates,y:D.v3_eq.dates.map(function(){return 100000}),name:'$100K',line:{color:'#484f58',width:1,dash:'dot'}});
  Plotly.newPlot('oEq',traces,Object.assign({},PL,{yaxis:Object.assign({},PL.yaxis,{title:'Equity ($)',tickprefix:'$',tickformat:',.0f'})}),PC);
}

// ── TAB: V3 Layers ──
function renderLayers(){
  var c=clear('content');
  if(!D.layers||D.layers.length===0){c.appendChild(el('div','','No layer data.'));return}

  // Layer cards
  var cards=el('div','layer-cards');
  D.layers.forEach(function(l){
    var card=el('div','lcard');
    var zInfo=D.entry_levels[l.layer]||{z:'?',size:'?'};
    card.appendChild(el('div','ln','Layer '+l.layer+' (Z='+zInfo.z+')'));
    var pnlCls=l.pnl>=0?'win':'loss';
    var pm=el('div','lm');pm.textContent='PnL: ';
    var ps=el('span',pnlCls,'$'+l.pnl.toLocaleString());pm.appendChild(ps);card.appendChild(pm);
    [['Trades',l.trades],['Win Rate',l.wr+'%'],['Avg Return',l.avg_return+'%'],['Avg Hold',l.avg_bars+'d'],
     ['Avg Win','+'+l.avg_win+'%'],['Avg Loss',l.avg_loss+'%'],['Size Factor',Math.round(zInfo.size*100)+'%']
    ].forEach(function(p){
      var m=el('div','lm');m.textContent=p[0]+': ';
      m.appendChild(el('span','',String(p[1])));card.appendChild(m);
    });
    cards.appendChild(card);
  });
  c.appendChild(cards);

  // Layer PnL bar chart
  var bw=el('div','ch');bw.appendChild(el('div','ct','Total PnL by Entry Layer'));
  var bd=el('div');bd.id='lBar';bd.style.height='300px';bw.appendChild(bd);c.appendChild(bw);
  var lColors=D.layers.map(function(l){return l.pnl>=0?'#3fb950':'#f85149'});
  Plotly.newPlot('lBar',[{
    x:D.layers.map(function(l){return 'L'+l.layer+' (Z='+D.entry_levels[l.layer].z+')'}),
    y:D.layers.map(function(l){return l.pnl}),
    type:'bar',marker:{color:lColors,opacity:0.85},
    text:D.layers.map(function(l){return l.trades+' trades, WR='+l.wr+'%'}),hoverinfo:'text+y'
  }],Object.assign({},PL,{xaxis:Object.assign({},PL.xaxis,{type:'category'}),yaxis:Object.assign({},PL.yaxis,{title:'Total PnL ($)',tickprefix:'$',tickformat:',.0f'})}),PC);

  // Trade return by layer scatter
  var sw=el('div','ch');sw.appendChild(el('div','ct','Trade Returns by Layer'));
  var sd=el('div');sd.id='lScatter';sd.style.height='300px';sw.appendChild(sd);c.appendChild(sw);
  var lcolors2=['#58a6ff','#3fb950','#d2a8ff','#d29922'];
  var straces=[];
  D.entry_levels.forEach(function(lv,i){
    var lt=D.v3_trades.filter(function(t){return t.layer===i});
    if(lt.length===0)return;
    straces.push({
      x:lt.map(function(t){return t.entry_date}),
      y:lt.map(function(t){return t.return_pct}),
      mode:'markers',name:'L'+i+' (Z='+lv.z+')',
      marker:{color:lcolors2[i%4],size:6,opacity:0.7},
      text:lt.map(function(t){return t.ticker+' '+t.return_pct+'% ('+t.bars_held+'d)'}),hoverinfo:'text'
    });
  });
  Plotly.newPlot('lScatter',straces,Object.assign({},PL,{yaxis:Object.assign({},PL.yaxis,{title:'Return %',ticksuffix:'%'})}),PC);
}

// ── TAB: Single Asset ──
function renderAsset(ticker){
  var a=D.assets[ticker];if(!a)return;
  var c=clear('content');

  // Price + trades
  var pw=el('div','ch');pw.appendChild(el('div','ct',ticker+' \\u2014 Price & V3 Trades'));
  var pd=el('div');pd.id='aPrice';pd.style.height='300px';pw.appendChild(pd);c.appendChild(pw);

  var v3t=D.v3_trades.filter(function(t){return t.ticker===ticker});
  var eX=[],eY=[],eT=[],wX=[],wY=[],wT=[],lX=[],lY=[],lT=[];
  v3t.forEach(function(t){
    eX.push(t.entry_date);eY.push(t.entry_price);eT.push('L'+t.layer+' Entry $'+t.entry_price);
    if(t.pnl>=0){wX.push(t.exit_date);wY.push(t.exit_price);wT.push('Win +$'+t.pnl+' ('+t.exit_reason+')');}
    else{lX.push(t.exit_date);lY.push(t.exit_price);lT.push('Loss $'+t.pnl+' ('+t.exit_reason+')');}
  });
  Plotly.newPlot('aPrice',[
    {x:a.dates,y:a.close,name:'Close',line:{color:'#c9d1d9',width:1.5}},
    {x:eX,y:eY,mode:'markers',name:'Entry',marker:{symbol:'triangle-up',size:10,color:'#58a6ff',line:{width:1,color:'#fff'}},text:eT,hoverinfo:'text'},
    {x:wX,y:wY,mode:'markers',name:'Win',marker:{symbol:'triangle-down',size:10,color:'#3fb950',line:{width:1,color:'#fff'}},text:wT,hoverinfo:'text'},
    {x:lX,y:lY,mode:'markers',name:'Loss',marker:{symbol:'triangle-down',size:10,color:'#f85149',line:{width:1,color:'#fff'}},text:lT,hoverinfo:'text'},
  ],Object.assign({},PL,{yaxis:Object.assign({},PL.yaxis,{title:'Price ($)',tickprefix:'$',tickformat:',.0f'})}),PC);

  // Z-score + Regime Score
  var tw=el('div','two');
  var zw=el('div','ch');zw.appendChild(el('div','ct','Z-Score (Multi-Level Thresholds)'));
  var zd=el('div');zd.id='aZ';zd.style.height='220px';zw.appendChild(zd);tw.appendChild(zw);
  var rw=el('div','ch');rw.appendChild(el('div','ct','Regime Score & Scale'));
  var rd=el('div');rd.id='aR';rd.style.height='220px';rw.appendChild(rd);tw.appendChild(rw);
  c.appendChild(tw);

  var zTraces=[{x:a.dates,y:a.zscore,name:'Z-Score',line:{color:'#58a6ff',width:1}}];
  D.entry_levels.forEach(function(lv){
    zTraces.push({x:a.dates,y:a.dates.map(function(){return -lv.z}),name:'-Z='+lv.z,line:{color:'#3fb950',width:0.5,dash:'dash'},showlegend:true});
  });
  zTraces.push({x:a.dates,y:a.dates.map(function(){return 0}),line:{color:'#484f58',width:1,dash:'dot'},showlegend:false});
  Plotly.newPlot('aZ',zTraces,Object.assign({},PL,{yaxis:Object.assign({},PL.yaxis,{title:'Z-Score',range:[-4,4]})}),PC);

  Plotly.newPlot('aR',[
    {x:a.dates,y:a.regime_score,fill:'tozeroy',fillcolor:'rgba(210,168,255,0.2)',name:'Regime Score',line:{color:'#d2a8ff',width:1}},
    {x:a.dates,y:a.regime_scale,name:'Regime Scale',line:{color:'#d29922',width:1}},
    {x:a.dates,y:a.dates.map(function(){return D.params.regime_threshold}),name:'Floor',line:{color:'#f85149',width:1,dash:'dot'}},
  ],Object.assign({},PL,{yaxis:Object.assign({},PL.yaxis,{title:'Score',range:[0,1.5]})}),PC);

  // RSI + Trend Strength
  var tw2=el('div','two');
  var rsiw=el('div','ch');rsiw.appendChild(el('div','ct','RSI'));
  var rsid=el('div');rsid.id='aRSI';rsid.style.height='200px';rsiw.appendChild(rsid);tw2.appendChild(rsiw);
  var tsw=el('div','ch');tsw.appendChild(el('div','ct','Trend Strength'));
  var tsd=el('div');tsd.id='aTS';tsd.style.height='200px';tsw.appendChild(tsd);tw2.appendChild(tsw);
  c.appendChild(tw2);

  Plotly.newPlot('aRSI',[
    {x:a.dates,y:a.rsi,name:'RSI',line:{color:'#d29922',width:1}},
    {x:a.dates,y:a.dates.map(function(){return 40}),name:'Oversold (40)',line:{color:'#3fb950',width:0.7,dash:'dot'}},
    {x:a.dates,y:a.dates.map(function(){return 60}),name:'Overbought (60)',line:{color:'#f85149',width:0.7,dash:'dot'}},
  ],Object.assign({},PL,{yaxis:Object.assign({},PL.yaxis,{title:'RSI',range:[0,100]})}),PC);

  Plotly.newPlot('aTS',[
    {x:a.dates,y:a.trend_strength,fill:'tozeroy',fillcolor:'rgba(88,166,255,0.15)',name:'Trend Strength',line:{color:'#58a6ff',width:1}},
    {x:a.dates,y:a.dates.map(function(){return 0.3}),name:'Sideways (<0.3)',line:{color:'#3fb950',width:0.7,dash:'dot'}},
    {x:a.dates,y:a.dates.map(function(){return 0.6}),name:'Trending (>0.6)',line:{color:'#f85149',width:0.7,dash:'dot'}},
  ],Object.assign({},PL,{yaxis:Object.assign({},PL.yaxis,{title:'Trend Strength',range:[0,1]})}),PC);
}

// ── TAB: Walk-Forward ──
function renderWF(){
  var c=clear('content');
  if(!D.folds||D.folds.length===0){c.appendChild(el('div','','No walk-forward results.'));return}

  var avg=D.folds.reduce(function(s,f){return s+f.sharpe},0)/D.folds.length;
  var pos=D.folds.filter(function(f){return f.sharpe>0}).length;

  var info=el('div','');
  info.style.cssText='font-size:12px;color:#8b949e;padding:0 0 10px';
  info.textContent='Avg OOS Sharpe: '+avg.toFixed(3)+' | Positive Folds: '+pos+'/'+D.folds.length+' | Full-Sample Sharpe: '+D.v3_metrics.Sharpe;
  c.appendChild(info);

  var bw=el('div','ch');bw.appendChild(el('div','ct','OOS Sharpe by Fold'));
  var bd=el('div');bd.id='wfBar';bd.style.height='300px';bw.appendChild(bd);c.appendChild(bw);

  var labels=D.folds.map(function(f){return 'Fold '+f.fold});
  var sharpes=D.folds.map(function(f){return f.sharpe});
  var fColors=sharpes.map(function(s){return s>0?'#3fb950':'#f85149'});
  Plotly.newPlot('wfBar',[
    {x:labels,y:sharpes,type:'bar',marker:{color:fColors,opacity:.85},text:D.folds.map(function(f){return 'Ret: '+f.ret+'%, Trades: '+f.trades}),hoverinfo:'text+y'},
    {x:labels,y:labels.map(function(){return avg}),mode:'lines',name:'Avg ('+avg.toFixed(3)+')',line:{color:'#58a6ff',width:2,dash:'dash'}},
    {x:labels,y:labels.map(function(){return 0}),mode:'lines',line:{color:'#484f58',width:1},showlegend:false},
  ],Object.assign({},PL,{xaxis:Object.assign({},PL.xaxis,{type:'category'}),yaxis:Object.assign({},PL.yaxis,{title:'OOS Sharpe'})}),PC);

  // Params table
  if(D.best_params&&D.best_params.length>0){
    var tw=el('div','ch');tw.appendChild(el('div','ct','Best Parameters per Fold'));
    var sc=el('div','scrl');var tbl=el('table','tbl');
    var th=el('thead');var hr=el('tr');
    ['Fold','Z Entry','Window','Train Sharpe','OOS Sharpe','OOS Return','Trades'].forEach(function(h){hr.appendChild(el('th','',h))});
    th.appendChild(hr);tbl.appendChild(th);
    var tb=el('tbody');
    D.best_params.forEach(function(bp,i){
      var fr=D.folds[i]||{};var tr=el('tr');
      [[bp.fold],[bp.z],[bp.w],[bp.train_sharpe,bp.train_sharpe>0?'win':'loss'],
       [fr.sharpe||0,(fr.sharpe||0)>0?'win':'loss'],[(fr.ret||0)+'%',(fr.ret||0)>0?'win':'loss'],[fr.trades||0]
      ].forEach(function(cell){
        var td=el('td');
        if(cell[1]){var sp=el('span',cell[1],String(cell[0]));td.appendChild(sp)}
        else td.textContent=String(cell[0]);
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);sc.appendChild(tbl);tw.appendChild(sc);c.appendChild(tw);
  }
}

// ── TAB: Trade Log ──
function renderTrades(){
  var c=clear('content');
  var wrap=el('div','ch');
  wrap.appendChild(el('div','ct','V3 Trades ('+D.v3_trades.length+' total)'));
  var sc=el('div','scrl');var tbl=el('table','tbl');

  var th=el('thead');var hr=el('tr');
  ['Asset','Dir','Layer','Entry','Exit','Entry $','Exit $','P&L','Return','Held','Reason','Regime'].forEach(function(h){hr.appendChild(el('th','',h))});
  th.appendChild(hr);tbl.appendChild(th);

  var tb=el('tbody');
  D.v3_trades.forEach(function(t){
    var tr=el('tr');
    var cells=[
      {t:t.ticker},{t:t.direction,c:'badge '+(t.direction==='LONG'?'long':'short')},
      {t:'L'+(t.layer!==undefined?t.layer:'?')},
      {t:t.entry_date},{t:t.exit_date},
      {t:'$'+t.entry_price.toLocaleString()},{t:'$'+t.exit_price.toLocaleString()},
      {t:(t.pnl>=0?'+':'')+' $'+t.pnl.toLocaleString(),c:t.pnl>=0?'win':'loss'},
      {t:(t.return_pct>=0?'+':'')+t.return_pct+'%',c:t.return_pct>=0?'win':'loss'},
      {t:t.bars_held+'d'},{t:t.exit_reason},{t:String(t.regime_score)}
    ];
    cells.forEach(function(cell){
      var td=el('td');
      if(cell.c){var sp=el('span',cell.c,cell.t);td.appendChild(sp)}
      else td.textContent=cell.t;
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  tbl.appendChild(tb);sc.appendChild(tbl);wrap.appendChild(sc);c.appendChild(wrap);
}

// Init
renderMetrics();
renderCompare();
</script>
</body>
</html>"""


def main():
    print("Running V2+V3 comparison for dashboard...")
    results = run_comparison_for_dashboard()
    print("Generating V3 comparison dashboard HTML...")
    html = generate_html(results)
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", "dashboard.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"Dashboard saved to: {os.path.abspath(path)}")
    return os.path.abspath(path)


if __name__ == "__main__":
    main()
