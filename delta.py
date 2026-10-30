"""
거래량 델타(Volume Delta) 스캐너
- S&P 500 + 관심종목(watchlist.txt) + ETF(etfs.txt)
- 하루를 5분봉으로 나눠 "오른 5분봉 거래량 − 내린 5분봉 거래량" = 델타를 계산 (TradingView Volume Delta와 같은 근사 방식)
- 델타가 그날 거래량의 큰 비율을 차지하고(매수 쏠림), 최근 20일 평균 델타 크기보다 훨씬 큰 종목 = "큰 양봉 델타"를 디스코드로 보냅니다.
- 수동 실행 때 티커를 넣으면 그 종목의 최근 델타를 보여 줍니다.
"""
import datetime as dt
import io
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 (여기만 고치면 됩니다) ─────────────
MIN_RATIO = float(os.environ.get("MIN_RATIO", "30"))           # 델타 ÷ 그날 거래량 ≥ 이 % (매수 쏠림)
MIN_MULT = float(os.environ.get("MIN_MULT", "2"))              # 델타 ≥ 최근 20일 평균 델타 크기 × 이 배수 (이례적 크기)
BASE_DAYS = int(os.environ.get("BASE_DAYS", "20"))             # 평균 델타 크기 비교 기간(거래일)
# 누적 판정 (최근 N일 합계). 기간: (쏠림% 최소, 양봉 델타 최소 일수, 큰 양봉 최소 횟수)
WINDOWS = {5: (20, 4, 1), 10: (15, 6, 2)}
# 쏠림% = N일 델타 합계 ÷ N일 거래량 합계 · 큰 양봉 = 그날 델타 ≥ 평소 평균 델타 크기 × MIN_MULT
# 평소 = 누적 구간 바로 전 BASE_DAYS일 (최근 큰 막대가 평균을 부풀리지 않도록 구간과 겹치지 않게)
MIN_PRICE = float(os.environ.get("MIN_PRICE", "5"))
MIN_DOLLAR_VOL = float(os.environ.get("MIN_DOLLAR_VOL", "10000000"))       # 종목: 20일 평균 거래대금($)
MIN_DOLLAR_VOL_ETF = float(os.environ.get("MIN_DOLLAR_VOL_ETF", "5000000"))  # ETF: 20일 평균 거래대금($)
MAX_ROWS = int(os.environ.get("MAX_ROWS", "30"))               # 표마다 최대 줄 수 (델타 배수 큰 순)
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"
DIAGNOSE = [t.strip().upper() for t in os.environ.get("DIAGNOSE", "").replace(" ", ",").split(",") if t.strip()]
# ──────────────────────────────────────────────────────

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (volume-delta)"}


# ───────────── 종목 목록 ─────────────
def _sp500() -> pd.DataFrame:
    cache = STATE / "sp500.csv"
    if not cache.exists() or time.time() - cache.stat().st_mtime > 7 * 86400:
        try:
            html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30).text
            t = pd.read_html(io.StringIO(html))[0]
            df = pd.DataFrame({"ticker": t["Symbol"].astype(str).str.replace(".", "-", regex=False), "name": t["Security"]})
            if len(df) > 400:
                df.to_csv(cache, index=False)
        except Exception as e:
            print(f"[경고] S&P 500 목록 갱신 실패: {e}")
    return pd.read_csv(cache).fillna("") if cache.exists() else pd.DataFrame(columns=["ticker", "name"])


def _txt(name: str) -> pd.DataFrame:
    rows = []
    p = ROOT / name
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if ln:
                tk, _, nm = ln.partition(",")
                rows.append({"ticker": re.sub(r"\.([A-Z])$", r"-\1", tk.strip().upper()), "name": nm.strip()})
    return pd.DataFrame(rows, columns=["ticker", "name"]).drop_duplicates("ticker")


def load_all() -> dict[str, tuple[str, str]]:
    """티커 → (이름, 종류 'stock' / 'etf')"""
    out = {}
    for r in _txt("etfs.txt").itertuples():
        out[r.ticker] = (r.name, "etf")
    for df in (_sp500(), _txt("watchlist.txt")):
        for r in df.itertuples():
            if r.ticker not in out:
                out[r.ticker] = (r.name, "stock")
    return out


# ───────────── 시세 ─────────────
def _download(tickers: list[str], interval: str, period: str, chunk: int) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        for attempt in range(2):
            try:
                data = yf.download(part, period=period, interval=interval, group_by="ticker",
                                   auto_adjust=False, prepost=False, threads=True, progress=False)
                break
            except Exception as e:
                print(f"[경고] {interval} 다운로드 실패({part[0]}…): {e}")
                data = None
                time.sleep(5)
        if data is None or data.empty:
            continue
        if not isinstance(data.columns, pd.MultiIndex):
            data = pd.concat({part[0]: data}, axis=1)
        have = set(data.columns.get_level_values(0))
        for t in part:
            if t in have:
                df = data[t].dropna(subset=["Close"])
                if len(df):
                    out[t] = df
        time.sleep(1)
    return out


def daily_delta(m5: pd.DataFrame) -> pd.DataFrame:
    """5분봉 → 날짜별 매수·매도 거래량과 델타 (정규장만)"""
    df = m5.copy()
    idx = df.index.tz_convert(NY) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(NY)
    df.index = idx
    tod = idx.hour * 60 + idx.minute
    df = df[(tod >= 570) & (tod < 960)]                               # 09:30 ~ 16:00
    if df.empty:
        return pd.DataFrame()
    prev = df["Close"].shift(1)
    # 종가 > 시가 → 매수, 종가 < 시가 → 매도, 같으면 직전 5분봉 종가와 비교 (TradingView 방식)
    up = (df["Close"] > df["Open"]) | ((df["Close"] == df["Open"]) & (df["Close"] > prev))
    dn = (df["Close"] < df["Open"]) | ((df["Close"] == df["Open"]) & (df["Close"] < prev))
    v = df["Volume"].fillna(0)
    g = pd.DataFrame({"buy": v.where(up, 0), "sell": v.where(dn, 0), "vol": v, "bars": 1}, index=df.index)
    d = g.groupby(idx.date).sum()
    d["delta"] = d["buy"] - d["sell"]
    return d


def finished_days(d: pd.DataFrame) -> pd.DataFrame:
    """장중에 실행하면 오늘은 아직 미완성 → 빼고 마감된 날만"""
    now = dt.datetime.now(NY)
    if len(d) and d.index[-1] == now.date() and now.weekday() < 5 and now.time() < dt.time(16, 10):
        return d.iloc[:-1]
    return d


# ───────────── 판정 ─────────────
def analyze(t: str, m5: pd.DataFrame, day: pd.DataFrame, kind: str) -> tuple[dict | None, str]:
    d = finished_days(daily_delta(m5))
    if len(d) < BASE_DAYS // 2 + 1:
        return None, "5분봉 기간 부족"
    last_date = d.index[-1]
    day = day[day.index.date <= last_date]
    if len(day) < 60:
        return None, "일봉 기간 부족 (상장 3개월 미만)"
    c = day["Close"]
    price = float(c.iloc[-1])
    dollar = float((c * day["Volume"]).tail(20).mean())
    if price < MIN_PRICE or dollar < (MIN_DOLLAR_VOL_ETF if kind == "etf" else MIN_DOLLAR_VOL):
        return None, f"주가 또는 거래대금 미달 (${price:.2f}, 20일 평균 ${dollar/1e6:.0f}M)"
    today = d.iloc[-1]
    base = d.iloc[-(BASE_DAYS + 1):-1]
    avg_abs = float(base["delta"].abs().mean())
    if today["vol"] <= 0 or avg_abs <= 0:
        return None, "거래량 없음"
    ratio = today["delta"] / today["vol"] * 100
    mult = today["delta"] / avg_abs
    last3 = d["delta"].iloc[-3:]
    ma50 = float(c.rolling(50).mean().iloc[-1])
    vavg = float(base["vol"].mean())
    r = {
        "t": t, "date": str(last_date), "kind": kind, "delta": float(today["delta"]), "delta_usd": float(today["delta"]) * price,
        "ratio": float(ratio), "mult": float(mult), "rvol": float(today["vol"] / vavg) if vavg else float("nan"),
        "chg": float((c.iloc[-1] / c.iloc[-2] - 1) * 100), "streak": "".join("+" if x > 0 else "-" for x in last3),
        "ratio3": float(last3.sum() / d["vol"].iloc[-3:].sum() * 100), "above50": price > ma50, "price": price,
        "partial": int(today["bars"]) < 70,                               # 5분봉 일부 누락(조기 폐장 등)
    }
    # 누적 (5일·10일)
    r["win"] = {}
    for N, (min_ratio, min_up, min_big) in WINDOWS.items():
        if len(d) < N + 5:
            continue
        w = d.iloc[-N:]
        b = d.iloc[-(N + BASE_DAYS):-N]
        b_abs = float(b["delta"].abs().mean())
        if b_abs <= 0 or w["vol"].sum() <= 0:
            continue
        wr = float(w["delta"].sum() / w["vol"].sum() * 100)
        up = int((w["delta"] > 0).sum())
        big = int((w["delta"] >= b_abs * MIN_MULT).sum())
        x = {"delta_usd": float(w["delta"].sum()) * price, "ratio": wr, "up": up, "big": big,
             "mult": float(w["delta"].sum() / b_abs), "rvol": float(w["vol"].mean() / b["vol"].mean()),
             "chg": float((c.iloc[-1] / c.iloc[-1 - N] - 1) * 100) if len(c) > N else float("nan")}
        if x["delta_usd"] <= 0:
            x["why"] = f"{N}일 누적 델타 마이너스 ({wr:+.0f}%)"
        elif wr < min_ratio:
            x["why"] = f"{N}일 쏠림 부족 ({wr:+.0f}% < {min_ratio}%)"
        elif up < min_up:
            x["why"] = f"{N}일 중 양봉 델타 {up}일 < {min_up}일"
        elif big < min_big:
            x["why"] = f"{N}일 중 큰 양봉 {big}회 < {min_big}회"
        else:
            x["why"] = ""
        r["win"][N] = x
    if r["delta"] <= 0:
        return r, f"델타 마이너스 (매도 우위 {ratio:+.0f}%)"
    if ratio < MIN_RATIO:
        return r, f"매수 쏠림 부족 (델타 {ratio:+.0f}% < {MIN_RATIO:g}%)"
    if mult < MIN_MULT:
        return r, f"평소 대비 크기 부족 (평균의 {mult:.1f}배 < {MIN_MULT:g}배)"
    return r, ""


# ───────────── 디스코드 ─────────────
def post(content: str):
    if not WEBHOOK:
        print(content)
        return
    for _ in range(5):
        res = requests.post(WEBHOOK, json={"content": content}, timeout=30)
        if res.status_code == 429:
            time.sleep(float(res.json().get("retry_after", 2)) + 0.5)
            continue
        res.raise_for_status()
        time.sleep(0.6)
        return


def width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def fit(s: str, w: int, right: bool = False) -> str:
    out = ""
    for ch in s:
        if width(out + ch) > w:
            break
        out += ch
    pad = " " * (w - width(out))
    return pad + out if right else out + pad


def money(x: float) -> str:
    a = abs(x)
    s = f"{a/1e9:.1f}B" if a >= 1e9 else f"{a/1e6:.0f}M" if a >= 1e6 else f"{a/1e3:.0f}K"
    return ("+" if x >= 0 else "-") + s


def send_table(title: str, rows: list[dict]):
    head = (fit("TICKER", 7) + fit("델타$", 8, True) + fit("쏠림%", 7, True) + fit("평균x", 6, True)
            + fit("RVOL", 6, True) + fit("등락%", 7, True) + fit("3일", 5, True) + fit("50일", 5, True) + "  이름")
    lines = [fit(r["t"], 7) + f"{money(r['delta_usd']):>8}{r['ratio']:>+7.0f}{r['mult']:>6.1f}{r['rvol']:>6.1f}"
             f"{r['chg']:>+7.1f}{r['streak']:>5}" + fit('위' if r['above50'] else '아래', 5, True) + "  " + r["name"][:18]
             for r in rows]
    first, part = True, 1
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1700:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first else f"**{title} ({part})**\n"
        post(f"{header}```\n{head}\n" + "\n".join(block) + "\n```")
        first, part = False, part + 1


def send_win_table(title: str, rows: list[dict], N: int):
    head = (fit("TICKER", 7) + fit("누적델타$", 9, True) + fit("쏠림%", 7, True) + fit("양봉", 6, True)
            + fit("큰양봉", 7, True) + fit("RVOL", 6, True) + fit(f"{N}일%", 7, True) + fit("50일", 5, True) + "  이름")
    lines = []
    for r in rows:
        x = r["win"][N]
        lines.append(fit(r["t"], 7) + f"{money(x['delta_usd']):>9}{x['ratio']:>+7.0f}" + fit(f"{x['up']}/{N}", 6, True)
                     + f"{x['big']:>7}{x['rvol']:>6.1f}{x['chg']:>+7.1f}" + fit('위' if r['above50'] else '아래', 5, True)
                     + "  " + r["name"][:18])
    first, part = True, 1
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1700:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first else f"**{title} ({part})**\n"
        post(f"{header}```\n{head}\n" + "\n".join(block) + "\n```")
        first, part = False, part + 1


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(NY)
    if now.weekday() >= 5 and not FORCE and not DIAGNOSE:
        print("주말. 종료.")
        return
    universe = load_all()
    tickers = DIAGNOSE or list(universe)
    print(f"대상 {len(tickers)}종목")
    m5 = _download(tickers, "5m", "60d", 50)
    day = _download(list(m5), "1d", "1y", 200)
    results, why = [], {}
    for t in tickers:
        if t not in m5 or t not in day:
            why[t] = "시세를 받지 못함 (티커 확인 필요)"
            continue
        name, kind = universe.get(t, ("", "stock"))
        try:
            r, w = analyze(t, m5[t], day[t], kind)
        except Exception as e:
            r, w = None, f"계산 오류: {e}"
        if r:
            r["name"] = name
            r["hit"] = not w
            results.append(r)
        why[t] = w

    if DIAGNOSE:
        lines = []
        for t in DIAGNOSE:
            r = next((x for x in results if x["t"] == t), None)
            if r is None:
                lines.append(f"**{t}** ❌ {why.get(t, '알 수 없음')}")
                continue
            mark = "✅ 하루 큰 양봉 델타" if r["hit"] else f"❌ 하루: {why[t]}"
            lines.append(f"**{t}** {mark}\n　{r['date']} 델타 {money(r['delta_usd'])} ({r['delta']/1e3:+,.0f}K주) · 쏠림 {r['ratio']:+.0f}% · "
                         f"평균의 {r['mult']:.1f}배 · RVOL {r['rvol']:.1f} · 등락 {r['chg']:+.1f}% · 최근 3일 {r['streak']}")
            for N, x in r.get("win", {}).items():
                lines.append(f"　{N}일 {'✅' if not x['why'] else '❌ ' + x['why']} · 누적 {money(x['delta_usd'])} · 쏠림 {x['ratio']:+.0f}% · "
                             f"양봉 {x['up']}/{N}일 · 큰 양봉 {x['big']}회 · RVOL {x['rvol']:.1f} · {N}일 등락 {x['chg']:+.1f}%")
        post("**거래량 델타 진단**\n" + "\n".join(lines))
        return

    if not results:
        post("⚠️ 거래량 델타: 시세를 받지 못했습니다. 일시적 오류일 수 있습니다.")
        return
    session = max(r["date"] for r in results)
    cur = [r for r in results if r["date"] == session]
    day_hits = sorted([r for r in cur if r["hit"]], key=lambda r: -r["mult"])
    win_hits = {N: sorted([r for r in cur if N in r.get("win", {}) and not r["win"][N]["why"]],
                          key=lambda r, N=N: -r["win"][N]["mult"]) for N in WINDOWS}

    def cnt(rows, kind):
        return sum(r["kind"] == kind for r in rows)
    summary = " / ".join([f"하루 {cnt(day_hits, 'stock')}·{cnt(day_hits, 'etf')}"]
                         + [f"{N}일 {cnt(v, 'stock')}·{cnt(v, 'etf')}" for N, v in win_hits.items()])
    post(f"**🟩 거래량 델타 · 매수 우위 | {session} 마감**\n"
         f"종목·ETF 수: {summary} (확인 {len(results)}개)")

    def cap(title, rows):
        return (title + f" · 상위 {MAX_ROWS}개 (전체 {len(rows)})", rows[:MAX_ROWS]) if len(rows) > MAX_ROWS else (title, rows)
    for kind, label in (("stock", "📈 종목"), ("etf", "🧺 ETF")):
        rows = [r for r in day_hits if r["kind"] == kind]
        if rows:
            send_table(*cap(f"{label} · 하루 큰 양봉 (쏠림 {MIN_RATIO:g}%↑ + 평균의 {MIN_MULT:g}배↑)", rows))
        for N, (mr, mu, mb) in WINDOWS.items():
            rows = [r for r in win_hits[N] if r["kind"] == kind]
            if rows:
                send_win_table(*cap(f"{label} · {N}일 누적 매수 우위 (쏠림 {mr}%↑ · 양봉 {mu}일↑ · 큰 양봉 {mb}회↑)", rows), N)
    if day_hits or any(win_hits.values()):
        post("델타$ = (오른 5분봉 거래량 − 내린 5분봉 거래량) × 종가 · 쏠림% = 델타 ÷ 거래량 · 평균x = 델타 ÷ 평소 평균 델타 크기\n"
             "양봉 = 델타 플러스인 날 수 · 큰양봉 = 평소의 2배 넘는 양봉 델타 횟수 · RVOL = 기간 평균 거래량 ÷ 평소 · "
             "누적 표는 누적델타 ÷ 평소 크기가 큰 순")
    if len(results) < len(tickers) * 0.5:
        post(f"⚠️ 거래량 델타: 시세를 {len(results)}/{len(tickers)}종목만 받았습니다. 일시적 오류일 수 있습니다.")
    print(f"완료: 확인 {len(results)}개, 하루 {len(day_hits)}개, 누적 " + ", ".join(f"{N}일 {len(v)}개" for N, v in win_hits.items()))


if __name__ == "__main__":
    sys.exit(main())
