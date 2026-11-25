"""CLI entry point for daily post-close scan and 09:31 opening-gap check."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from data import fetch_index_daily, fetch_realtime_quotes, limit_up_price, load_panel
from scanners import scan_all

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
STATE = ROOT / "state"
CST = ZoneInfo("Asia/Shanghai")


def render_postclose_md(res: dict) -> str:
    m = res["market"]
    lines = [
        f"# S 级六策略盘后预选 · {res['date']}",
        "",
        f"- 上证指数 `{m['sse_close']}` / MA20 `{m['sse_ma20']}` · 全 A 上涨宽度 `{m['breadth_all']:.1%}` → **象限 `{m['quadrant']}`**",
        f"- 沪深 300 `{m['csi300_close']}` / MA20 `{m['csi300_ma20']}` · 中证 1000 `{m['csi1000_close']}` / MA20 `{m['csi1000_ma20']}`",
        "",
    ]
    for key in ("2j3_next", "core_rebound", "lhb_next", "fb_next", "cfgh_next", "v3_next"):
        st = res["strategies"][key]
        lines += [
            f"## {st['name']}",
            f"- 门控：{st['gate']}",
            f"- 次日买点：{st['gap_rule']}",
            f"- 退出与仓位：{st['exit_rule']}",
            "",
        ]
        if st.get("error"):
            lines += [f"> 数据待核验：`{st['error']}`", ""]
            continue
        cands = st["candidates"]
        if not cands:
            lines += ["无符合条件候选。", ""]
            continue
        lines += ["| 序 | 代码 | 名称 | 收盘 | 关键因子 |", "|---:|---|---|---:|---|"]
        for idx, c in enumerate(cands[:12], 1):
            lines.append(f"| {idx} | `{c['code']}` | {c['name']} | {c['close']:.2f} | {c['note']} |")
        if len(cands) > 12:
            lines.append(f"\n*（共 {len(cands)} 只，仅展示前 12 只，完整列表见 `state/{res['date']}.json`）*")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_opening_md(rep: dict) -> str:
    lines = [
        f"# S 级六策略开盘筛选 · {rep['date']}",
        "",
        f"- 对应盘后信号日：`{rep['signal_date']}`",
        "",
    ]
    for key in ("2j3_next", "core_rebound", "lhb_next", "fb_next", "cfgh_next", "v3_next"):
        st = rep["strategies"][key]
        lines += [f"## {st['name']}", f"- 条件：{st['gap_rule']}", ""]
        rows = st["passed"]
        if not rows:
            lines += ["今日开盘无命中标的。", ""]
            continue
        lines += ["| 序 | 代码 | 名称 | 昨收 | 开盘 | 缺口 | 盘后因子 |", "|---:|---|---|---:|---:|---:|---|"]
        for idx, r in enumerate(rows[:10], 1):
            lines.append(
                f"| {idx} | `{r['code']}` | {r['name']} | {r['pc']:.2f} | {r['open']:.2f} | {r['gap']:+.2%} | {r['note']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_postclose(day: str | None) -> None:
    indices = {s: fetch_index_daily(s, count=120) for s in ("sh000001", "sh000300", "sh000852")}
    all_days = sorted(indices["sh000001"].index)
    target = day or all_days[-1]
    if target not in all_days:
        print(f"{target} is not a completed trading session (latest: {all_days[-1]}), skipping.")
        return
    pos = all_days.index(target)
    if pos < 70:
        raise RuntimeError(f"Insufficient index history before {target}")
    window = all_days[pos - 70:pos + 1]
    prev_day = all_days[pos - 1]

    panel = load_panel(window)
    res = scan_all(target, prev_day, panel, indices)

    REPORTS.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / f"{target}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = render_postclose_md(res)
    (REPORTS / f"{target}.md").write_text(md, encoding="utf-8")
    (REPORTS / "latest.md").write_text(md, encoding="utf-8")
    print(f"Published post-close report for {target}")


def run_opening(day: str | None) -> None:
    today = day or dt.datetime.now(CST).date().isoformat()
    stamp = today.replace("-", "")
    idx_q = fetch_realtime_quotes(["sh000001"]).get("sh000001")
    if not idx_q or not idx_q["stamp"].startswith(stamp):
        print(f"No live opening quote for {today}, skipping.")
        return
    # Find latest state file strictly before today
    state_files = sorted(p for p in STATE.glob("????-??-??.json") if p.stem < today)
    if not state_files:
        raise RuntimeError("No previous post-close state file found")
    prev_state = json.loads(state_files[-1].read_text(encoding="utf-8"))

    all_syms = sorted({c["key"] for st in prev_state["strategies"].values() for c in st["candidates"]})
    quotes = fetch_realtime_quotes(all_syms)

    out = {"date": today, "signal_date": prev_state["date"], "strategies": {}}
    for key, st in prev_state["strategies"].items():
        passed = []
        for idx, c in enumerate(st["candidates"]):
            q = quotes.get(c["key"])
            if not q or not q["stamp"].startswith(stamp) or q["open"] <= 0 or q["pc"] <= 0:
                continue
            ratio = 1.20 if c["code"].startswith(("30", "68")) else 1.10
            lu = limit_up_price(q["pc"], ratio)
            if q["open"] >= lu - 0.005:
                continue  # skip open limit-up
            gap = q["open"] / q["pc"] - 1.0
            gl = c.get("gap_low", st["gap_low"])
            gh = c.get("gap_high", st["gap_high"])
            # Inclusive endpoints except strict upper 0.0 / -0.005
            ok = (gap >= gl) and (gap < gh if gh in (0.0, -0.005) else gap <= gh)
            if ok:
                passed.append({**c, "pc": q["pc"], "open": q["open"], "gap": gap, "_rank": idx})
        if st["gap_sort"] == "gap_desc":
            passed.sort(key=lambda x: (-x["gap"], x["code"]))
        elif st["gap_sort"] == "fb_dual":
            passed.sort(key=lambda x: (0 if x.get("tier") == "C2" else 1, x["gap"], x["code"]))
        else:
            passed.sort(key=lambda x: (x["_rank"], x["code"]))
        out["strategies"][key] = {"name": st["name"], "gap_rule": st["gap_rule"], "passed": passed}

    op_dir = REPORTS / "opening"
    st_dir = STATE / "opening"
    op_dir.mkdir(parents=True, exist_ok=True)
    st_dir.mkdir(parents=True, exist_ok=True)
    (st_dir / f"{today}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = render_opening_md(out)
    (op_dir / f"{today}.md").write_text(md, encoding="utf-8")
    (op_dir / "latest.md").write_text(md, encoding="utf-8")
    print(f"Published opening report for {today}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--opening", action="store_true", help="Run 09:31 opening-gap check")
    p.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD")
    args = p.parse_args()
    if args.opening:
        run_opening(args.date)
    else:
        run_postclose(args.date)


if __name__ == "__main__":
    main()
