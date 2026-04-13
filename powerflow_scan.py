
import math
import os
import json
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date as date_type
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf


# =========================
# ✅ 텔레그램 설정
# =========================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")


def send_telegram(message: str) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("[WARN] TELEGRAM_TOKEN 또는 CHAT_ID 환경변수가 없습니다.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
        print("[Telegram] 전송 완료")
    except Exception as e:
        print(f"[Telegram ERROR] {e}")


def format_alert(row: dict, rank: int) -> str:
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "🔹")
    mode = row.get("price_mode", "REVERSAL")
    return (
        f"{medal} <b>#{rank} {row['ticker']}</b>\n"
        f"⏰ {row['signal_bar_et']}\n"
        f"💰 Close: <b>${row['close']:.2f}</b>\n"
        f"💪 Strength: <b>{row['strength']:.2f}</b>\n"
        f"🧭 Mode: <b>{mode}</b> | Activity Hits: <b>{row['activity_hits']}</b>\n"
        f"📊 MF Z: {row['mf_z']:.2f} | RVOL(slot): {row['rvol_slot']:.2f}\n"
        f"📈 Post CLV: {row['post_clv']:.3f} | Pre CLV: {row['pre_clv']:.3f}\n"
        f"📏 Range/ATR: {row['range_atr_ratio']:.2f} | Trend: {row['trend_score']:.2f}\n"
        f"{'✅ Above EMA' if row['close_above_ema'] else '⚠️ Below EMA'}"
    )


# =========================
# ✅ 중복 알림 방지
# =========================

SENT_LOG_FILE = "/tmp/powerflow_sent.json"


def load_sent_log() -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if os.path.exists(SENT_LOG_FILE):
        try:
            with open(SENT_LOG_FILE, "r") as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
        except Exception:
            pass
    return {"date": today, "sent": []}


def save_sent_log(log: dict) -> None:
    with open(SENT_LOG_FILE, "w") as f:
        json.dump(log, f)


def is_already_sent(log: dict, ticker: str, signal_bar_et: str) -> bool:
    return f"{ticker}|{signal_bar_et}" in log["sent"]


def mark_as_sent(log: dict, ticker: str, signal_bar_et: str) -> bool:
    key = f"{ticker}|{signal_bar_et}"
    if key not in log["sent"]:
        log["sent"].append(key)
        return True
    return False


def send_scan_results(df: pd.DataFrame) -> None:
    now_et = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M ET")
    log = load_sent_log()

    if df.empty:
        print("[INFO] 신호 없음 — 알림 생략")
        return

    new_signals = []
    for _, row in df.iterrows():
        if not is_already_sent(log, row["ticker"], row["signal_bar_et"]):
            new_signals.append(row.to_dict())

    if not new_signals:
        print("[INFO] 새 시그널 없음 (이미 보낸 것들) — 알림 생략")
        return

    header = (
        f"🚨 <b>PowerFlow 60m Scan Results v7 (balanced)</b>\n"
        f"⏰ {now_et}\n"
        f"📋 새 시그널: {len(new_signals)}개 종목\n"
        f"{'─' * 28}"
    )
    send_telegram(header)

    for rank, row in enumerate(new_signals, start=1):
        msg = format_alert(row, rank)
        send_telegram(msg)
        mark_as_sent(log, row["ticker"], row["signal_bar_et"])

    save_sent_log(log)


# =========================
# Params
# =========================

@dataclass(frozen=True)
class PowerFlow60mParams:
    interval: str = "60m"
    lookback_days: int = 30
    min_bars: int = 40

    pre_window: int = 3
    post_window: int = 2

    # 완화된 reversal 조건
    pre_clv_max: float = 0.10
    post_clv_min: float = 0.15

    # continuation 허용: 이미 강한 종목도 통과 가능
    use_continuation_branch: bool = True
    continuation_post_clv_min: float = 0.45
    continuation_clv_min: float = 0.35

    z_window: int = 20
    mf_z_th: float = 1.5

    vol_window: int = 20
    vol_mult: float = 1.6

    use_ema_filter: bool = True
    ema_span: int = 20
    require_close_above_ema: bool = True

    use_trend_filter: bool = True
    trend_ema_fast: int = 20
    trend_ema_slow: int = 50
    min_trend_score: float = -0.20

    use_slot_rvol: bool = True
    slot_rvol_window: int = 10
    slot_rvol_th: float = 1.35

    # 활동성 3개(MFZ / VOL / RVOL) 중 최소 몇 개 충족할지
    min_activity_hits: int = 2

    use_range_quality_filter: bool = True
    atr_window: int = 14
    min_range_atr_ratio: float = 0.60

    last_k_bars_to_check: int = 8
    today_only: bool = True

    normalize_volume_by_bar_minutes: bool = True
    completed_bars_only: bool = True

    session_tz: str = "America/New_York"
    session_close_time: str = "16:00"
    naive_index_tz_default: str = "America/New_York"
    auto_guess_naive_index_tz: bool = True

    batch_size: int = 20


# =========================
# Utilities
# =========================

def _parse_interval_minutes(interval: str) -> int:
    s = str(interval).strip().lower()
    if s.endswith("m"):
        return int(s[:-1])
    if s.endswith("h"):
        return int(s[:-1]) * 60
    raise ValueError(f"Unsupported interval format: {interval}")


def _guess_naive_index_tz(idx: pd.DatetimeIndex) -> str:
    if len(idx) == 0:
        return "America/New_York"
    s = pd.to_datetime(pd.Series(idx), errors="coerce").dropna()
    if s.empty:
        return "America/New_York"

    hrs = s.dt.hour
    mins = s.dt.minute
    score_ny = ((mins == 30) & (hrs.isin([9, 10, 11, 12, 13, 14, 15]))).mean()
    score_utc = ((mins == 30) & (hrs.isin([13, 14, 15, 16, 17, 18, 19, 20]))).mean()
    return "UTC" if score_utc > score_ny else "America/New_York"


def _to_et(idx: pd.DatetimeIndex, p: PowerFlow60mParams) -> pd.DatetimeIndex:
    if idx.tz is None:
        tz = _guess_naive_index_tz(idx) if p.auto_guess_naive_index_tz else p.naive_index_tz_default
        idx = idx.tz_localize(tz)
    return idx.tz_convert(p.session_tz)


def _compute_bar_minutes(
    idx_et: pd.DatetimeIndex,
    interval_minutes: int,
    session_close_time: str,
) -> pd.Series:
    hh, mm = session_close_time.split(":")
    close_offset = pd.to_timedelta(int(hh), unit="h") + pd.to_timedelta(int(mm), unit="m")
    close_dt = idx_et.normalize() + close_offset
    mins_to_close = ((close_dt - idx_et) / pd.to_timedelta(1, unit="m")).astype(float)

    bar_min = np.where(
        mins_to_close <= 0,
        float(interval_minutes),
        np.minimum(float(interval_minutes), np.maximum(1.0, mins_to_close)),
    )
    return pd.Series(bar_min, index=idx_et)


def _filter_completed_bars(df: pd.DataFrame, p: PowerFlow60mParams) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    interval_minutes = _parse_interval_minutes(p.interval)
    idx_et = _to_et(df.index, p)
    bar_min = _compute_bar_minutes(idx_et, interval_minutes, p.session_close_time)
    now_et = pd.Timestamp.now(tz=p.session_tz)
    bar_end = idx_et + pd.to_timedelta(bar_min.values, unit="m")
    return df.loc[bar_end <= now_et].copy()


def _get_today_mask(df: pd.DataFrame, p: PowerFlow60mParams) -> np.ndarray:
    idx_et = _to_et(df.index, p)
    today_et: date_type = pd.Timestamp.now(tz=p.session_tz).date()
    return np.array([ts.date() == today_et for ts in idx_et], dtype=bool)


def _select_scan_bars(feat: pd.DataFrame, p: PowerFlow60mParams) -> pd.DataFrame:
    if p.today_only:
        mask = _get_today_mask(feat, p)
        sub = feat.loc[mask].copy()
        if sub.empty:
            sub = feat.iloc[-p.last_k_bars_to_check:].copy()
    else:
        sub = feat.iloc[-p.last_k_bars_to_check:].copy()
    return sub


def _chunked(seq: List[str], size: int) -> List[List[str]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


# =========================
# yfinance
# =========================

def _normalize_single_ticker_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(df.columns):
        return pd.DataFrame()

    out = df[required].copy()
    out = out.dropna(subset=["Close"])
    return out


def fetch_60m_yf_batch(tickers: List[str], p: PowerFlow60mParams) -> Dict[str, pd.DataFrame]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=p.lookback_days)

    result: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return result

    for chunk in _chunked(tickers, p.batch_size):
        try:
            raw = yf.download(
                tickers=chunk,
                start=start,
                end=end,
                interval=p.interval,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
                prepost=False,
            )
        except Exception as e:
            print(f"[WARN] batch download failed for chunk={chunk}: {type(e).__name__}: {e}")
            continue

        if raw is None or raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            level0 = list(raw.columns.get_level_values(0))

            # case A: group_by="ticker" => (ticker, field)
            if any(t in level0 for t in chunk):
                for t in chunk:
                    if t in raw.columns.get_level_values(0):
                        sub = raw[t].copy()
                        sub = _normalize_single_ticker_ohlcv(sub)
                        if not sub.empty:
                            if p.completed_bars_only:
                                sub = _filter_completed_bars(sub, p)
                            result[t] = sub

            # case B: group_by="column" => (field, ticker)
            else:
                for t in chunk:
                    data = {}
                    for f in ["Open", "High", "Low", "Close", "Volume"]:
                        if (f, t) in raw.columns:
                            data[f] = raw[(f, t)]
                    if data:
                        sub = pd.DataFrame(data, index=raw.index)
                        sub = _normalize_single_ticker_ohlcv(sub)
                        if not sub.empty:
                            if p.completed_bars_only:
                                sub = _filter_completed_bars(sub, p)
                            result[t] = sub
        else:
            # single ticker fallback
            if len(chunk) == 1:
                sub = _normalize_single_ticker_ohlcv(raw)
                if not sub.empty:
                    if p.completed_bars_only:
                        sub = _filter_completed_bars(sub, p)
                    result[chunk[0]] = sub

    return result


# =========================
# Feature engineering
# =========================

def compute_clv(df: pd.DataFrame) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    rng = (high - low).replace(0, np.nan)
    return (((close - low) - (high - close)) / rng).fillna(0.0).clip(-1, 1)


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window, min_periods=max(5, window // 2)).mean()


def add_features(df: pd.DataFrame, p: PowerFlow60mParams) -> pd.DataFrame:
    out = df.copy()
    out["CLV"] = compute_clv(out)

    vol = out["Volume"].astype(float)

    if p.normalize_volume_by_bar_minutes:
        interval_minutes = _parse_interval_minutes(p.interval)
        idx_et = _to_et(out.index, p)
        bar_min = _compute_bar_minutes(idx_et, interval_minutes, p.session_close_time)
        out["BAR_MIN"] = bar_min.values
        vol_base = vol / out["BAR_MIN"].replace(0, np.nan)
    else:
        out["BAR_MIN"] = np.nan
        vol_base = vol

    out["VOL_BASE"] = vol_base
    out["MoneyFlow"] = out["CLV"] * out["VOL_BASE"]

    mf = out["MoneyFlow"]
    mf_mean = mf.rolling(p.z_window, min_periods=max(5, p.z_window // 2)).mean()
    mf_std = mf.rolling(p.z_window, min_periods=max(5, p.z_window // 2)).std(ddof=0)
    out["MF_Z"] = (mf - mf_mean) / mf_std.replace(0, np.nan)

    v_mean = out["VOL_BASE"].rolling(p.vol_window, min_periods=max(5, p.vol_window // 2)).mean()
    out["VOL_SPIKE"] = out["VOL_BASE"] / v_mean.replace(0, np.nan)

    idx_et = _to_et(out.index, p)
    out["SLOT_KEY"] = idx_et.strftime("%H:%M")

    if p.use_slot_rvol:
        slot_avg = (
            out.groupby("SLOT_KEY")["VOL_BASE"]
            .transform(lambda s: s.shift(1).rolling(p.slot_rvol_window, min_periods=3).mean())
        )
        out["RVOL_SLOT"] = out["VOL_BASE"] / slot_avg.replace(0, np.nan)
    else:
        out["RVOL_SLOT"] = np.nan

    out["EMA"] = out["Close"].ewm(span=p.ema_span, adjust=False).mean()
    out["EMA_FAST"] = out["Close"].ewm(span=p.trend_ema_fast, adjust=False).mean()
    out["EMA_SLOW"] = out["Close"].ewm(span=p.trend_ema_slow, adjust=False).mean()

    out["ATR"] = compute_atr(out, p.atr_window)
    out["BAR_RANGE"] = (out["High"] - out["Low"]).astype(float)
    out["RANGE_ATR_RATIO"] = out["BAR_RANGE"] / out["ATR"].replace(0, np.nan)

    out["TREND_SCORE"] = (
        ((out["EMA_FAST"] - out["EMA_SLOW"]) / out["EMA_SLOW"].replace(0, np.nan)).fillna(0.0) * 100.0
    )

    return out


def detect_powerflow(df: pd.DataFrame, p: PowerFlow60mParams) -> pd.DataFrame:
    out = add_features(df, p)
    clv = out["CLV"]

    out["PRE_CLV"] = (
        clv.shift(p.post_window)
        .rolling(p.pre_window, min_periods=max(2, p.pre_window))
        .mean()
    )
    out["POST_CLV"] = clv.rolling(p.post_window, min_periods=max(1, p.post_window)).mean()

    # 1) reversal + 2) continuation 둘 다 허용
    cond_reversal = (out["PRE_CLV"] <= p.pre_clv_max) & (out["POST_CLV"] >= p.post_clv_min)

    if p.use_continuation_branch:
        cond_continuation = (
            (out["POST_CLV"] >= p.continuation_post_clv_min)
            & (out["CLV"] >= p.continuation_clv_min)
        )
    else:
        cond_continuation = pd.Series(False, index=out.index)

    cond_price = cond_reversal | cond_continuation

    # 활동성 3종 중 2개 이상 충족
    cond_mfz = out["MF_Z"] >= p.mf_z_th
    cond_vol = out["VOL_SPIKE"] >= p.vol_mult

    if p.use_slot_rvol:
        cond_rvol_slot = out["RVOL_SLOT"].fillna(0) >= p.slot_rvol_th
    else:
        cond_rvol_slot = pd.Series(True, index=out.index)

    activity_components = [cond_mfz, cond_vol]
    if p.use_slot_rvol:
        activity_components.append(cond_rvol_slot)

    activity_hits = pd.Series(0, index=out.index, dtype=int)
    for comp in activity_components:
        activity_hits = activity_hits.add(comp.astype(int), fill_value=0).astype(int)

    required_hits = min(p.min_activity_hits, len(activity_components))
    cond_activity = activity_hits >= required_hits

    if p.use_ema_filter and p.require_close_above_ema:
        cond_ema = out["Close"] >= out["EMA"]
    else:
        cond_ema = pd.Series(True, index=out.index)

    if p.use_trend_filter:
        cond_trend = out["TREND_SCORE"].fillna(-999) >= p.min_trend_score
    else:
        cond_trend = pd.Series(True, index=out.index)

    if p.use_range_quality_filter:
        cond_range = out["RANGE_ATR_RATIO"].fillna(0) >= p.min_range_atr_ratio
    else:
        cond_range = pd.Series(True, index=out.index)

    out["POWERFLOW"] = cond_price & cond_activity & cond_ema & cond_trend & cond_range

    out["ACTIVITY_HITS"] = activity_hits
    out["PRICE_MODE"] = np.select(
        [
            cond_reversal & cond_continuation,
            cond_continuation,
            cond_reversal,
        ],
        [
            "BOTH",
            "CONTINUATION",
            "REVERSAL",
        ],
        default="NONE",
    )

    out["STRENGTH"] = (
        (out["POST_CLV"].clip(lower=0).fillna(0) * 2.0)
        + out["MF_Z"].fillna(0).clip(lower=0) * 1.1
        + np.log1p(out["VOL_SPIKE"].fillna(0).clip(lower=0)) * 1.0
        + np.log1p(out["RVOL_SLOT"].fillna(0).clip(lower=0)) * 0.7
        + out["RANGE_ATR_RATIO"].fillna(0).clip(lower=0, upper=3) * 0.5
        + out["TREND_SCORE"].fillna(0).clip(lower=0, upper=3) * 0.25
        + out["ACTIVITY_HITS"].fillna(0).clip(lower=0, upper=3) * 0.40
        + (cond_continuation.astype(int) * 0.20)
    )

    out["COND_REVERSAL"] = cond_reversal
    out["COND_CONTINUATION"] = cond_continuation
    out["COND_PRICE"] = cond_price
    out["COND_MFZ"] = cond_mfz
    out["COND_VOL"] = cond_vol
    out["COND_RVOL_SLOT"] = cond_rvol_slot
    out["COND_ACTIVITY"] = cond_activity
    out["COND_EMA"] = cond_ema
    out["COND_TREND"] = cond_trend
    out["COND_RANGE"] = cond_range

    return out


def load_tickers_from_file(file_path: str) -> List[str]:
    path = os.path.abspath(file_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ticker file not found: {path}")

    tickers: List[str] = []
    seen = set()

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            normalized = line.replace(",", " ")
            for token in normalized.split():
                ticker = token.strip().upper()
                if ticker and ticker not in seen:
                    seen.add(ticker)
                    tickers.append(ticker)

    if not tickers:
        raise ValueError(f"ticker file is empty: {path}")

    return tickers


# =========================
# Scanner
# =========================

def scan_one_ticker_after_close_from_df(
    ticker: str,
    df: pd.DataFrame,
    p: PowerFlow60mParams,
) -> Optional[dict]:
    if df.empty or "Close" not in df.columns or len(df) < p.min_bars:
        return None

    feat = detect_powerflow(df, p)
    sub = _select_scan_bars(feat, p)
    if sub.empty:
        return None

    hits = sub[sub["POWERFLOW"] == True]
    if hits.empty:
        return None

    ranked = hits.sort_values(
        ["STRENGTH", "ACTIVITY_HITS", "MF_Z", "RVOL_SLOT", "VOL_SPIKE"],
        ascending=False,
    )
    best = ranked.iloc[0]
    best_ts = ranked.index[0]

    best_ts_et = _to_et(pd.DatetimeIndex([best_ts]), p)[0]
    if getattr(best_ts_et, "tzinfo", None) is not None:
        signal_bar_utc = str(pd.Timestamp(best_ts_et).tz_convert("UTC"))
    else:
        signal_bar_utc = str(pd.Timestamp(best_ts))

    return {
        "ticker": ticker,
        "signal_bar_et": str(best_ts_et),
        "signal_bar_utc": signal_bar_utc,
        "close": float(best["Close"]),
        "strength": float(best["STRENGTH"]),
        "pre_clv": float(best["PRE_CLV"]) if pd.notna(best["PRE_CLV"]) else 0.0,
        "post_clv": float(best["POST_CLV"]) if pd.notna(best["POST_CLV"]) else 0.0,
        "mf_z": float(best["MF_Z"]) if pd.notna(best["MF_Z"]) else 0.0,
        "vol_spike": float(best["VOL_SPIKE"]) if pd.notna(best["VOL_SPIKE"]) else 0.0,
        "rvol_slot": float(best["RVOL_SLOT"]) if pd.notna(best["RVOL_SLOT"]) else 0.0,
        "range_atr_ratio": float(best["RANGE_ATR_RATIO"]) if pd.notna(best["RANGE_ATR_RATIO"]) else 0.0,
        "trend_score": float(best["TREND_SCORE"]) if pd.notna(best["TREND_SCORE"]) else 0.0,
        "close_above_ema": bool(best["Close"] >= best["EMA"]) if "EMA" in best and pd.notna(best["EMA"]) else True,
        "activity_hits": int(best["ACTIVITY_HITS"]) if pd.notna(best["ACTIVITY_HITS"]) else 0,
        "price_mode": str(best["PRICE_MODE"]),
    }


def scan_universe_after_close(
    tickers: List[str],
    p: PowerFlow60mParams,
    top_n: int = 20,
) -> pd.DataFrame:
    tickers = list(dict.fromkeys([str(t).strip().upper() for t in tickers if str(t).strip()]))

    data_map = fetch_60m_yf_batch(tickers, p)
    if not data_map:
        print("[WARN] 다운로드된 데이터가 없습니다.")
        return pd.DataFrame()

    results: List[dict] = []
    errors: Dict[str, str] = {}

    for t in tickers:
        try:
            df = data_map.get(t, pd.DataFrame())
            r = scan_one_ticker_after_close_from_df(t, df, p)
            if r is not None:
                results.append(r)
        except Exception as e:
            errors[t] = f"{type(e).__name__}: {e}"

    if errors:
        print(f"[WARN] {len(errors)} tickers had errors:")
        for t, msg in list(errors.items())[:10]:
            print(f"  - {t}: {msg}")

    df = pd.DataFrame(results)
    if df.empty:
        print("[INFO] 조건을 만족한 신호가 없습니다.")
        return df

    df = df.sort_values(
        ["strength", "activity_hits", "mf_z", "rvol_slot", "vol_spike", "range_atr_ratio"],
        ascending=False,
    ).head(top_n)

    return df.reset_index(drop=True)


# =========================
# RUN
# =========================

if __name__ == "__main__":
    ticker_file = os.environ.get("TICKER_FILE", "ticker.txt")
    tickers = load_tickers_from_file(ticker_file)

    p = PowerFlow60mParams(
        lookback_days=30,
        pre_window=3,
        post_window=2,
        last_k_bars_to_check=8,
        pre_clv_max=0.10,
        post_clv_min=0.15,
        use_continuation_branch=True,
        continuation_post_clv_min=0.45,
        continuation_clv_min=0.35,
        mf_z_th=1.5,
        vol_mult=1.6,
        today_only=True,
        use_ema_filter=True,
        ema_span=20,
        require_close_above_ema=True,
        use_trend_filter=True,
        trend_ema_fast=20,
        trend_ema_slow=50,
        min_trend_score=-0.20,
        use_slot_rvol=True,
        slot_rvol_window=10,
        slot_rvol_th=1.35,
        min_activity_hits=2,
        use_range_quality_filter=True,
        atr_window=14,
        min_range_atr_ratio=0.60,
        normalize_volume_by_bar_minutes=True,
        completed_bars_only=True,
        batch_size=20,
    )

    print("===== PowerFlow 60m Scan v7 (balanced) 시작 =====")
    print(f"[INFO] ticker file: {os.path.abspath(ticker_file)}")
    print(f"[INFO] loaded tickers: {len(tickers)}")

    df_result = scan_universe_after_close(tickers, p, top_n=20)

    print("\n===== PowerFlow 60m Scan Results (v7 balanced) =====")
    if not df_result.empty:
        print(df_result.to_string(index=False))
    else:
        print("신호 없음")

    send_scan_results(df_result)
