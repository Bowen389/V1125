"""Public market data layer for GitHub Actions (TDX daily packages + Tencent quotes/indices + Eastmoney LHB)."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import io
import json
import math
import re
import struct
import time
import zipfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "tdx"
CST = ZoneInfo("Asia/Shanghai")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TDX_URL = "https://www.tdx.com.cn/products/data/data/g4day/{ymd}.zip"
EM_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def _http_get(url: str, encoding: str = "utf-8", timeout: int = 25, retries: int = 3) -> str:
    for i in range(retries):
        try:
            with urlopen(Request(url, headers=UA), timeout=timeout) as resp:
                return resp.read().decode(encoding, errors="replace")
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1 + i)
    raise RuntimeError(f"GET failed: {url}")


def fetch_index_daily(symbol: str, count: int = 140) -> pd.Series:
    """Fetch index daily closes from Tencent (e.g. sh000001, sh000300, sh000852)."""
    today_cst = dt.datetime.now(CST).date()
    end = today_cst.isoformat()
    start = (today_cst - dt.timedelta(days=int(count * 1.8) + 30)).isoformat()
    q = urlencode({"param": f"{symbol},day,{start},{end},{count},"})
    data = json.loads(_http_get(f"https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get?{q}"))
    bars = data["data"][symbol]["day"]
    s = pd.Series({b[0]: float(b[2]) for b in bars}).sort_index()
    return s


def fetch_tdx_day(day: str) -> Path:
    """Download and cache one trading day of TDX daily package as parquet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{day}.parquet"
    if dest.exists():
        return dest
    ymd = day.replace("-", "")
    url = TDX_URL.format(ymd=ymd)
    raw = None
    headers = {**UA, "Connection": "close"}
    for i in range(5):
        try:
            with urlopen(Request(url, headers=headers), timeout=45) as resp:
                raw = resp.read()
            break
        except Exception:
            if i == 4:
                raise
            time.sleep(1.5 * (i + 1))
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = set(z.namelist())
    rows = []
    for market in ("sh", "sz", "bj"):
        cod_n, md_n = f"{market}{ymd[2:]}.cod", f"{market}{ymd[2:]}.md1"
        if cod_n not in names or md_n not in names:
            if market == "bj":
                continue
            raise RuntimeError(f"{day}: missing {cod_n}/{md_n}")
        cod, md = z.read(cod_n), z.read(md_n)
        if len(cod) % 150 or len(md) % 512 or len(cod) // 150 != len(md) // 512:
            raise RuntimeError(f"{day} {market}: corrupt TDX lengths")
        for off in range(0, len(cod), 150):
            rec = cod[off:off + 150]
            code = rec[0:6].rstrip(b"\x00 ").decode("ascii", "replace")
            if not (len(code) == 6 and code.isdigit()):
                continue
            # Keep A-share equities (SH 60/68, SZ 00/30, BJ 4/8/92)
            if not ((market == "sh" and code.startswith(("60", "68")))
                    or (market == "sz" and code.startswith(("00", "30")))
                    or (market == "bj" and code.startswith(("4", "8", "92")))):
                continue
            seq = struct.unpack("<H", rec[32:34])[0]
            b = md[seq * 512:(seq + 1) * 512]
            prev_close = struct.unpack("<d", b[4:12])[0]
            o, h, l, c = struct.unpack("<4d", b[12:44])
            vol = struct.unpack("<Q", b[56:64])[0]
            amt = struct.unpack("<d", b[72:80])[0]
            if not all(math.isfinite(v) for v in (prev_close, o, h, l, c, amt)) or c <= 0 or vol <= 0:
                continue
            name = rec[40:72].split(b"\x00")[0].decode("gbk", "replace").strip()
            rows.append((day, market, code, name, round(prev_close, 4), round(o, 4),
                         round(h, 4), round(l, 4), round(c, 4), float(vol), round(amt, 2)))
    df = pd.DataFrame(rows, columns=["date", "mkt", "code", "name", "pc", "open",
                                     "high", "low", "close", "volume", "amount"])
    if len(df) < 3500:
        raise RuntimeError(f"{day}: too few equity rows ({len(df)})")
    tmp = dest.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(dest)
    return dest


def load_panel(days: list[str], workers: int = 4) -> pd.DataFrame:
    """Download missing TDX days in parallel and return sorted panel with causal adj factor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        paths = list(ex.map(fetch_tdx_day, days))
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["key"] = df["mkt"] + df["code"]
    df = df.sort_values(["key", "date"]).reset_index(drop=True)

    # Causal backward-adjust factor using official ex-rights pc vs previous bar close
    prev_c = df.groupby("key", sort=False)["close"].shift(1)
    ratio = (df["pc"] / prev_c).where(prev_c > 0, 1.0)
    # Only treat as ex-rights when discrepancy > 0.2% (avoid rounding noise)
    ratio = ratio.where((ratio - 1.0).abs() > 0.002, 1.0)
    # Forward cumprod of (1/ratio) gives continuous price series comparable across days
    df["adj"] = (1.0 / ratio).groupby(df["key"], sort=False).cumprod()
    df["adj_c"] = df["close"] * df["adj"]
    df["adj_o"] = df["open"] * df["adj"]
    return df


def limit_up_price(pc: float, ratio: float = 1.10) -> float:
    return float((Decimal(str(pc)) * Decimal(str(ratio))).quantize(Decimal(".01"), rounding=ROUND_HALF_UP))


def fetch_turnover_batch(symbols: list[str], day: str) -> dict[str, float]:
    """Fetch exact daily turnover (%) for a small list of symbols on `day` via Tencent kline."""
    if not symbols:
        return {}
    start = (dt.date.fromisoformat(day) - dt.timedelta(days=12)).isoformat()

    def one(sym: str):
        q = urlencode({"param": f"{sym},day,{start},{day},10,"})
        try:
            res = json.loads(_http_get(f"https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get?{q}", timeout=15))
            bars = res["data"][sym]["day"]
            for b in bars:
                if b[0] == day and len(b) >= 8 and b[7]:
                    return sym, float(b[7])
        except Exception:
            pass
        return sym, float("nan")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        return dict(ex.map(one, symbols))


def fetch_realtime_quotes(symbols: list[str]) -> dict[str, dict]:
    """Fetch real-time opening quotes from Tencent qt.gtimg.cn."""
    out = {}
    for start in range(0, len(symbols), 40):
        chunk = ",".join(symbols[start:start + 40])
        text = _http_get(f"https://qt.gtimg.cn/q={chunk}", encoding="gb18030", timeout=18)
        for sym, payload in re.findall(r'v_([a-z0-9]+)="([^"]*)"', text):
            parts = payload.split("~")
            if len(parts) < 35:
                continue
            try:
                pc = float(parts[4]) if parts[4] else 0.0
                op = float(parts[5]) if parts[5] else 0.0
                stamp = parts[30]
                out[sym] = {"pc": pc, "open": op, "stamp": stamp, "name": parts[1]}
            except ValueError:
                continue
    return out


def _em_paged(report: str, day: str, page_size: int = 500) -> list[dict]:
    rows, page = [], 1
    while True:
        params = {
            "reportName": report,
            "columns": "ALL",
            "filter": f"(TRADE_DATE='{day}')",
            "pageNumber": page,
            "pageSize": page_size,
            "source": "WEB",
            "client": "WEB",
        }
        r = requests.get(EM_URL, params=params, headers=UA, timeout=25)
        r.raise_for_status()
        payload = r.json()
        res = payload.get("result")
        if not res or not res.get("data"):
            break
        rows.extend(res["data"])
        if page >= int(res.get("pages", 1)):
            break
        page += 1
        time.sleep(0.15)
    return rows


def fetch_lhb_day(day: str) -> dict:
    """Fetch Eastmoney Dragon-Tiger List overview + buy/sell seats for `day`."""
    return {
        "overview": _em_paged("RPT_DAILYBILLBOARD_DETAILS", day),
        "buy": _em_paged("RPT_BILLBOARD_DAILYDETAILSBUY", day),
        "sell": _em_paged("RPT_BILLBOARD_DAILYDETAILSSELL", day),
    }
