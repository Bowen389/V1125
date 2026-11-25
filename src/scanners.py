"""Scanners for all 6 S-tier redesigned strategies (Next-Gen)."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

from data import fetch_lhb_day, fetch_turnover_batch, limit_up_price


def _is_mainboard(df: pd.DataFrame) -> pd.Series:
    return ((df["mkt"] == "sh") & df["code"].str.startswith(("600", "601", "603", "605"))) | \
           ((df["mkt"] == "sz") & df["code"].str.startswith(("000", "001", "002", "003")))


def _is_st_or_new(name: pd.Series) -> pd.Series:
    return name.str.contains(r"ST|退|^[NC]", case=False, na=True)


def _add_zt_flag(mb: pd.DataFrame) -> pd.Series:
    lu = np.floor(mb["pc"] * 110.0 + 0.5) / 100.0
    return (mb["close"] - lu).abs() < 0.005


def _precompute_mainboard(panel: pd.DataFrame) -> pd.DataFrame:
    mb = panel[_is_mainboard(panel)].copy()
    mb = mb.sort_values(["key", "date"]).reset_index(drop=True)
    mb["is_st"] = _is_st_or_new(mb["name"])
    mb["lu"] = np.floor(mb["pc"] * 110.0 + 0.5) / 100.0
    mb["zt"] = (mb["close"] - mb["lu"]).abs() < 0.005
    mb["hit_zt"] = (mb["high"] - mb["lu"]).abs() < 0.005
    mb["broken_zt"] = mb["hit_zt"] & (~mb["zt"])
    mb["yizi"] = (mb["high"] == mb["low"]) & mb["zt"]
    mb["t_board"] = (mb["open"] >= mb["lu"] - 0.005) & mb["zt"]
    mb["ret_d"] = mb["close"] / mb["pc"] - 1.0
    mb["open_ret"] = mb["open"] / mb["pc"] - 1.0
    mb["amp"] = (mb["high"] - mb["low"]) / mb["pc"]
    span = (mb["high"] - mb["low"]).replace(0, np.nan)
    mb["upper_shadow"] = (mb["high"] - np.maximum(mb["open"], mb["close"])) / span
    mb["vwap"] = mb["amount"] / mb["volume"]
    mb["vwap_dev"] = mb["close"] / mb["vwap"] - 1.0

    g = mb.groupby("key", sort=False)
    mb["bar_n"] = g.cumcount() + 1
    mb["prev_zt"] = g["zt"].shift(1).fillna(False)
    mb["prev_red"] = g["close"].shift(1) < g["open"].shift(1)
    mb["prev_broken"] = g["broken_zt"].shift(1).fillna(False)

    amt_s1 = g["amount"].shift(1)
    vol_s1 = g["volume"].shift(1)
    adjc_s1 = g["adj_c"].shift(1)
    zt_s2 = g["zt"].shift(2)

    mb["amt20"] = amt_s1.groupby(mb["key"], sort=False).rolling(20).mean().reset_index(level=0, drop=True)
    mb["vol20"] = vol_s1.groupby(mb["key"], sort=False).rolling(20).mean().reset_index(level=0, drop=True)
    mb["max20_c"] = adjc_s1.groupby(mb["key"], sort=False).rolling(20).max().reset_index(level=0, drop=True)
    mb["max60_c"] = adjc_s1.groupby(mb["key"], sort=False).rolling(60).max().reset_index(level=0, drop=True)
    mb["med60_c"] = adjc_s1.groupby(mb["key"], sort=False).rolling(60).median().reset_index(level=0, drop=True)
    mb["pos60_med"] = adjc_s1 / mb["med60_c"]
    mb["mom5"] = mb["adj_c"] / g["adj_c"].shift(5) - 1.0
    mb["zt_cnt_2_10"] = zt_s2.groupby(mb["key"], sort=False).rolling(9).sum().reset_index(level=0, drop=True)
    return mb


def scan_all(day: str, prev_day: str, panel: pd.DataFrame, indices: dict[str, pd.Series]) -> dict:
    """Run all 6 S-tier post-close scanners for `day`."""
    mb = _precompute_mainboard(panel)
    today_mb = mb[mb["date"] == day].copy()
    today_clean = today_mb[~today_mb["is_st"]].copy()

    # Index gates
    sse = indices["sh000001"].loc[:day]
    csi300 = indices["sh000300"].loc[:day]
    csi1000 = indices["sh000852"].loc[:day]

    sse_c, sse_ma20 = float(sse.iloc[-1]), float(sse.tail(20).mean())
    c300_c, c300_ma20 = float(csi300.iloc[-1]), float(csi300.tail(20).mean())
    c1000_c, c1000_ma20 = float(csi1000.iloc[-1]), float(csi1000.tail(20).mean())

    # All-A breadth on `day`
    today_all = panel[panel["date"] == day]
    breadth_all = float((today_all["close"] > today_all["pc"]).mean())
    sse_up = sse_c >= sse_ma20
    breadth_hi = breadth_all >= 0.55
    quadrant = ("Q1" if breadth_hi else "Q2") if sse_up else ("Q3" if breadth_hi else "Q4")

    # Identify all limit-up symbols needing exact turnover from Tencent
    zt_syms = today_clean[today_clean["zt"]]["key"].tolist()
    turn_map = fetch_turnover_batch(zt_syms, day)
    today_clean["turnover"] = today_clean["key"].map(turn_map)

    results = {
        "date": day,
        "market": {
            "sse_close": round(sse_c, 2), "sse_ma20": round(sse_ma20, 2),
            "csi300_close": round(c300_c, 2), "csi300_ma20": round(c300_ma20, 2),
            "csi1000_close": round(c1000_c, 2), "csi1000_ma20": round(c1000_ma20, 2),
            "breadth_all": round(breadth_all, 4), "quadrant": quadrant,
        },
        "strategies": {},
    }

    # --- 1. 2j3-Next (Top 5% amt20, non-yizi first board, turnover >= 6%, gap [2%, 6%]) ---
    today_clean["amt20_pct"] = today_clean["amt20"].rank(pct=True)
    c_2j3 = today_clean[
        today_clean["zt"] & (~today_clean["prev_zt"]) & (~today_clean["yizi"]) &
        (today_clean["amt20_pct"] >= 0.95) & (today_clean["turnover"] >= 6.0)
    ].sort_values(["amt20", "code"], ascending=[False, True])
    results["strategies"]["2j3_next"] = {
        "name": "2j3-Next（大容量一进二中军）",
        "gate": "零门控（全天候）",
        "gap_rule": "次日高开 [2%, 6%]，按高开降序",
        "exit_rule": "T+1封板→T+2开盘卖；未封→T+2收盘卖（3仓×33%，日限2只）",
        "gap_low": 0.02, "gap_high": 0.06, "gap_sort": "gap_desc",
        "candidates": [
            {"code": r.code, "mkt": r.mkt, "key": r.key, "name": r.name, "close": float(r.close),
             "amt20_yi": round(float(r.amt20) / 1e8, 2), "turnover": round(float(r.turnover), 2),
             "note": f"20日均额 {r.amt20/1e8:.2f}亿 · 换手 {r.turnover:.1f}%"}
            for r in c_2j3.itertuples()
        ],
    }

    # --- 2. Core-Rebound (amt20 >= 5亿, T-10..T-2 exactly 1 ZT, T-1 red non-broken, non-yizi, open>=pc, turn<20%) ---
    c_reb = today_clean[
        today_clean["zt"] & (~today_clean["yizi"]) & (~today_clean["prev_zt"]) &
        (today_clean["bar_n"] >= 60) & (today_clean["close"] >= 3.0) &
        (today_clean["zt_cnt_2_10"] == 1) & today_clean["prev_red"] & (~today_clean["prev_broken"]) &
        (today_clean["open_ret"] >= 0.0) & (today_clean["amt20"] >= 5e8) & (today_clean["turnover"] < 20.0)
    ].sort_values(["amt20", "code"], ascending=[False, True])
    results["strategies"]["core_rebound"] = {
        "name": "Core-Rebound（大容量单板反包）",
        "gate": "零门控（全天候）",
        "gap_rule": "次日缺口 [-5%, +6%]，按20日均额降序",
        "exit_rule": "T+2起收盘涨停顺延(最多5次)，否则收盘卖（6仓×16.7%，日限2只）",
        "gap_low": -0.05, "gap_high": 0.06, "gap_sort": "rank",
        "candidates": [
            {"code": r.code, "mkt": r.mkt, "key": r.key, "name": r.name, "close": float(r.close),
             "amt20_yi": round(float(r.amt20) / 1e8, 2), "turnover": round(float(r.turnover), 2),
             "note": f"20日均额 {r.amt20/1e8:.2f}亿 · 换手 {r.turnover:.1f}% · 开盘 {r.open_ret:+.1%}"}
            for r in c_reb.itertuples()
        ],
    }

    # --- 3. LHB-Next (CSI300 > MA20, single-day LHB, first-day, non-consecutive board, inst net >= 30M) ---
    lhb_gate = c300_c > c300_ma20
    lhb_cands, lhb_err = [], None
    if lhb_gate:
        try:
            snap_t = fetch_lhb_day(day)
            snap_p = fetch_lhb_day(prev_day)
            prev_codes = {str(r.get("SECURITY_CODE", "")).zfill(6) for r in snap_p["overview"]}
            # Build previous-day ZT lookup across all SH/SZ equities
            p_prev = panel[panel["date"] == prev_day].copy()
            p_prev["lu_r"] = np.where(p_prev["code"].str.startswith(("30", "68")), 1.20, 1.10)
            p_prev["lu"] = np.floor(p_prev["pc"] * p_prev["lu_r"] * 100.0 + 0.5) / 100.0
            prev_zt_all = set(p_prev[(p_prev["close"] - p_prev["lu"]).abs() < 0.005]["code"])

            # Filter single-day overview rows (exclude 3-day listings)
            single_codes = {}
            for r in snap_t["overview"]:
                code = str(r.get("SECURITY_CODE", "")).zfill(6)
                if not code.startswith(("00", "60", "30", "68")):
                    continue
                name = str(r.get("SECURITY_NAME_ABBR", ""))
                if "ST" in name.upper() or "退" in name:
                    continue
                expl = str(r.get("EXPLANATION", ""))
                if "三个交易日" in expl or "3个交易日" in expl:
                    continue
                if code in prev_codes or code in prev_zt_all:
                    continue
                single_codes[code] = {"name": name, "close": float(r.get("CLOSE_PRICE") or 0),
                                      "gain": float(r.get("CHANGE_RATE") or 0)}

            # Sum institutional net buy across buy & sell seat tables (dedup by trade_id/dept)
            inst_buy = defaultdict(float)
            inst_sell = defaultdict(float)
            seen_b, seen_s = set(), set()
            for r in snap_t["buy"]:
                code = str(r.get("SECURITY_CODE", "")).zfill(6)
                if code not in single_codes:
                    continue
                if "机构专用" not in str(r.get("OPERATEDEPT_NAME", "")):
                    continue
                k = (code, r.get("OPERATEDEPT_CODE"), r.get("BUY"), r.get("SELL"), r.get("RANK"))
                if k in seen_b:
                    continue
                seen_b.add(k)
                inst_buy[code] += float(r.get("BUY") or 0)
                inst_sell[code] += float(r.get("SELL") or 0)
            for r in snap_t["sell"]:
                code = str(r.get("SECURITY_CODE", "")).zfill(6)
                if code not in single_codes:
                    continue
                if "机构专用" not in str(r.get("OPERATEDEPT_NAME", "")):
                    continue
                k = (code, r.get("OPERATEDEPT_CODE"), r.get("BUY"), r.get("SELL"), r.get("RANK"))
                if k in seen_s:
                    continue
                seen_s.add(k)
                inst_buy[code] += float(r.get("BUY") or 0)
                inst_sell[code] += float(r.get("SELL") or 0)

            for code, info in single_codes.items():
                net = inst_buy[code] - inst_sell[code]
                if net >= 30_000_000:
                    mkt = "sh" if code.startswith("6") else "sz"
                    lhb_cands.append({
                        "code": code, "mkt": mkt, "key": mkt + code, "name": info["name"],
                        "close": info["close"], "inst_net_wan": round(net / 1e4, 1),
                        "note": f"机构总净买 {net/1e4:.0f}万 · 涨幅 {info['gain']:+.1f}%"
                    })
            lhb_cands.sort(key=lambda x: (-x["inst_net_wan"], x["code"]))
        except Exception as exc:
            lhb_err = str(exc)
    results["strategies"]["lhb_next"] = {
        "name": "LHB-Next（机构净买龙虎榜）",
        "gate": f"沪深300 {c300_c:.2f} / MA20 {c300_ma20:.2f}：{'通过' if lhb_gate else '拦截'}",
        "gap_rule": "次日缺口 ≥ -5%，按机构净买降序",
        "exit_rule": "T+2收盘卖，收盘涨停顺延(最多3次)（4仓×25%，日限2只）",
        "gap_low": -0.05, "gap_high": 0.25, "gap_sort": "rank",
        "error": lhb_err,
        "candidates": lhb_cands,
    }

    # --- 4. FB-Next (CSI1000 < MA20, non-yizi/non-T first board, amt>=1亿, turn<=10%, vwap_dev>=1.5%, pos60<1.0) ---
    fb_gate = c1000_c < c1000_ma20
    fb_cands = []
    if fb_gate:
        c_fb = today_clean[
            today_clean["zt"] & (~today_clean["prev_zt"]) & (~today_clean["yizi"]) & (~today_clean["t_board"]) &
            (today_clean["bar_n"] >= 61) & (today_clean["amount"] >= 1e8) &
            (today_clean["turnover"] <= 10.0) & (today_clean["vwap_dev"] >= 0.015) &
            (today_clean["pos60_med"] < 1.0)
        ].sort_values(["pos60_med", "code"])
        for r in c_fb.itertuples():
            tier = "C2" if r.pos60_med < 0.70 else "C1"
            fb_cands.append({
                "code": r.code, "mkt": r.mkt, "key": r.key, "name": r.name, "close": float(r.close),
                "tier": tier, "pos60_med": round(float(r.pos60_med), 3),
                "gap_low": -0.03 if tier == "C2" else -0.05,
                "gap_high": 0.03 if tier == "C2" else -0.02,
                "note": f"{tier} · pos60={r.pos60_med:.3f} · 换手 {r.turnover:.1f}% · 乖离 {r.vwap_dev:+.2%}"
            })
    results["strategies"]["fb_next"] = {
        "name": "FB-Next（弱市高质量首板低开）",
        "gate": f"中证1000 {c1000_c:.2f} / MA20 {c1000_ma20:.2f}：{'通过' if fb_gate else '拦截'}",
        "gap_rule": "C2 [-3%, +3%] 优先，C1 [-5%, -2%] 其次，同档按缺口升序",
        "exit_rule": "T+2收盘卖，收盘涨停顺延（6仓×16.7%）",
        "gap_low": -0.05, "gap_high": 0.03, "gap_sort": "fb_dual",
        "candidates": fb_cands,
    }

    # --- 5. CFGH-Next (4-Quadrant: Q1/Q2 breakout, Q3 shrink, Q4 cash; amt20>=5000万; gap [-3%, 0%)) ---
    cfgh_cands = []
    if quadrant in ("Q1", "Q2"):
        pool = today_clean[
            (today_clean["bar_n"] >= 59) & (today_clean["close"] >= 3.0) &
            (today_clean["amt20"] >= 5e7) &
            (today_clean["adj_c"] >= today_clean["max20_c"]) &
            (today_clean["volume"] >= 1.5 * today_clean["vol20"])
        ].copy()
        if len(pool) >= 3:
            r1 = pool["upper_shadow"].rank(pct=True)
            r2 = (-pool["amp"]).rank(pct=True) if quadrant == "Q1" else (-pool["mom5"]).rank(pct=True)
            pool["score"] = 0.5 * (r1 + r2)
            pool["pct"] = pool["score"].rank(pct=True)
            sel = pool[pool["pct"] >= 0.80].sort_values(["score", "code"], ascending=[False, True])
            for r in sel.itertuples():
                cfgh_cands.append({
                    "code": r.code, "mkt": r.mkt, "key": r.key, "name": r.name, "close": float(r.close),
                    "score": round(float(r.score), 3),
                    "note": f"{quadrant}突破 · 分={r.score:.3f} · 振幅 {r.amp:.1%} · 上影 {r.upper_shadow:.2f}"
                })
    elif quadrant == "Q3":
        # Shrink pool: launch k in {1,2,3} bars ago
        mb_clean = mb[~mb["is_st"]].copy()
        g = mb_clean.groupby("key", sort=False)
        launch = (mb_clean["ret_d"] > 0.05) & (mb_clean["volume"] >= 2.0 * mb_clean["vol20"]) & \
                 (mb_clean["close"] > mb_clean["open"]) & (mb_clean["adj_c"] >= mb_clean["max20_c"])
        shrink = pd.Series(False, index=mb_clean.index)
        mid = (mb_clean["adj_c"] + mb_clean["adj_o"]) / 2.0
        for k in (1, 2, 3):
            fl = np.minimum(g["adj_c"].transform(lambda s: mid.loc[s.index].shift(k)),
                            g["max20_c"].shift(k))
            ok = g["ret_d"].transform(lambda s: launch.loc[s.index].shift(k, fill_value=False)).astype(bool)
            for j in range(k):
                ok &= (g["adj_c"].shift(j) >= fl)
            a_mean = g["amount"].rolling(k).mean().reset_index(level=0, drop=True)
            shrink |= ok & (a_mean <= 0.65 * g["amount"].shift(k))
        pool = mb_clean[(mb_clean["date"] == day) & shrink & (mb_clean["bar_n"] >= 59) &
                        (mb_clean["close"] >= 3.0) & (mb_clean["amt20"] >= 5e7)].copy()
        pool = pool.sort_values(["vwap_dev", "code"], ascending=[True, True])
        for r in pool.itertuples():
            cfgh_cands.append({
                "code": r.code, "mkt": r.mkt, "key": r.key, "name": r.name, "close": float(r.close),
                "vwap_dev": round(float(r.vwap_dev), 4),
                "note": f"Q3缩量守位 · VWAP乖离 {r.vwap_dev:+.2%}"
            })
    results["strategies"]["cfgh_next"] = {
        "name": "CFGH-Next（零拟合四象限轮动）",
        "gate": f"当前象限 {quadrant}（上证 {sse_c:.1f}/MA20 {sse_ma20:.1f}，全A宽度 {breadth_all:.1%}）" + (" → 空仓防守" if quadrant == "Q4" else ""),
        "gap_rule": "次日微低开 [-3%, 0%)，按评分降序/乖离升序",
        "exit_rule": ("持2日+涨停顺延" if quadrant == "Q1" else "持1日+涨停顺延") + "（单笔10%，日限2~3只）",
        "gap_low": -0.03, "gap_high": 0.0, "gap_sort": "rank",
        "candidates": cfgh_cands[:15],  # display top 15; opening checks all saved
    }

    # --- 6. V3-Next (CSI300 >= MA20, 60d high breakout, amt20>=1亿, amp<6%, sort vwap_dev asc, gap [-2%, -0.5%)) ---
    v3_gate = c300_c >= c300_ma20
    v3_cands = []
    if v3_gate:
        c_v3 = today_clean[
            (today_clean["bar_n"] >= 61) & (today_clean["close"] >= 3.0) &
            (today_clean["amt20"] >= 1e8) & (today_clean["adj_c"] >= today_clean["max60_c"]) &
            (today_clean["amp"] < 0.06)
        ].sort_values(["vwap_dev", "code"], ascending=[True, True])
        for r in c_v3.head(20).itertuples():
            v3_cands.append({
                "code": r.code, "mkt": r.mkt, "key": r.key, "name": r.name, "close": float(r.close),
                "vwap_dev": round(float(r.vwap_dev), 4), "amp": round(float(r.amp), 4),
                "note": f"VWAP乖离 {r.vwap_dev:+.2%} · 振幅 {r.amp:.1%} · 20日均额 {r.amt20/1e8:.2f}亿"
            })
    results["strategies"]["v3_next"] = {
        "name": "V3-Next（低波低乖离突破）",
        "gate": f"沪深300 {c300_c:.2f} / MA20 {c300_ma20:.2f}：{'通过' if v3_gate else '拦截'}",
        "gap_rule": "次日温和低开 [-2%, -0.5%)，按VWAP乖离升序",
        "exit_rule": "持有2日收盘卖（5仓×20%，日限2只）",
        "gap_low": -0.02, "gap_high": -0.005, "gap_sort": "rank",
        "candidates": v3_cands,
    }

    return results
