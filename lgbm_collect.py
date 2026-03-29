"""
LightGBM 학습용 데이터 수집
- 미너비니 백테스트 시그널 기반
- 라벨: 손절(-7%) 안 걸리고 10일 내 +8% 도달 → 1
- 피처: 가격/거래량 시계열 + RS + 기술지표 + 섹터/시총 OHE
"""
import os, time, warnings, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

MASSIVE = os.environ.get("MASSIVE_TOKEN", "")
TOK     = os.environ.get("TELEGRAM_TOKEN", "")
CID     = os.environ.get("TELEGRAM_CHAT_ID", "")

WINDOW      = 150
SKIP        = 2
VOL_MA      = 20
MIN_PRICE   = 5.0      # 페니스탁 필터
MIN_VOL     = 100000   # 일평균 거래량 최소
STOP_PCT    = 0.93     # 손절 -7%
TARGET_PCT  = 1.08     # 목표 +8%
HOLD_DAYS   = 10       # 보유 기간


def send(text):
    print(text)
    if TOK:
        try:
            requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",
                         data={"chat_id": CID, "text": text}, timeout=10)
        except: pass


def get_ohlcv(ticker, start, end):
    url    = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": MASSIVE}
    for _ in range(2):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429: time.sleep(12); continue
            if resp.status_code != 200: return None
            results = resp.json().get("results", [])
            if not results: return None
            df = pd.DataFrame(results)
            df["date"] = (pd.to_datetime(df["t"], unit="ms")
                         .dt.tz_localize("UTC").dt.tz_convert("America/New_York")
                         .dt.normalize().dt.tz_localize(None))
            df = df.set_index("date").sort_index()
            df = df.rename(columns={"c": "Close", "h": "High", "l": "Low", "v": "Volume"})
            return df[["Close", "High", "Low", "Volume"]].astype(float).dropna()
        except: time.sleep(3)
    return None


def calc_label(df, sig_idx, entry_price):
    """손절 안 걸리고 10일 내 +8% 도달 여부"""
    stop   = entry_price * STOP_PCT
    target = entry_price * TARGET_PCT
    future = df.iloc[sig_idx + 1: sig_idx + 1 + HOLD_DAYS]
    if len(future) == 0:
        return None
    for _, row in future.iterrows():
        low  = row["Low"]
        high = row["High"]
        # 손절 먼저 체크
        if low <= stop:
            return 0
        # 목표가 도달
        if high >= target:
            return 1
    return 0


def calc_rsi(closes, period=14):
    """RSI 계산"""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 4)


def calc_features(df, spy_df, d_idx, ticker_info):
    """피처 계산"""
    start_idx = d_idx - SKIP - WINDOW
    end_idx   = d_idx - SKIP
    if start_idx < max(VOL_MA, 252): return None

    w = df.iloc[start_idx:end_idx]
    if len(w) < WINDOW: return None

    close  = w["Close"].values
    high   = w["High"].values
    low    = w["Low"].values
    volume = w["Volume"].values

    # 페니스탁 필터
    if close[-1] < MIN_PRICE: return None
    if np.mean(volume[-20:]) < MIN_VOL: return None

    # 종가 정규화
    c_min, c_max = close.min(), close.max()
    if c_max == c_min: return None
    close_norm = (close - c_min) / (c_max - c_min)

    # 등락률
    ret = np.zeros(WINDOW)
    ret[1:] = (close[1:] / close[:-1] - 1) * 100

    # 거래량 비율
    vol_ratio = np.zeros(WINDOW)
    for k in range(WINDOW):
        abs_idx  = start_idx + k
        past_vol = df["Volume"].iloc[max(0, abs_idx - VOL_MA): abs_idx]
        ma = past_vol.mean()
        vol_ratio[k] = volume[k] / ma if ma > 0 else 1.0

    feat = {}
    for k in range(WINDOW):
        feat[f"ret_{k+1}"]        = round(ret[k], 4)
        feat[f"close_norm_{k+1}"] = round(float(close_norm[k]), 4)
        feat[f"vol_ratio_{k+1}"]  = round(float(vol_ratio[k]), 4)

    # ── RS 피처 (미너비니 방식) ────────────────────────
    ref_idx = d_idx - SKIP
    td = df.index[:ref_idx + 1]
    sa = spy_df.reindex(td).ffill()
    tc = df["Close"].iloc[:ref_idx + 1].values
    sc = sa["Close"].values if not sa.isnull().all().any() else None

    if sc is not None and len(tc) >= 253:
        def pr(arr, n): return float(arr[-1] / arr[-n] - 1) if len(arr) >= n else 0.0
        w_ = [0.4, 0.2, 0.2, 0.2]; p_ = [63, 126, 189, 252]
        t_rs = sum(w_[i]*pr(tc,p_[i]) for i in range(4))
        s_rs = sum(w_[i]*pr(sc,p_[i]) for i in range(4))
        rs_at  = round((t_rs - s_rs) * 100, 4)
        rs_20  = round((pr(tc,20)  - pr(sc,20))  * 100, 4)
        rs_50  = round((pr(tc,50)  - pr(sc,50))  * 100, 4)
        rs_150 = round((pr(tc,150) - pr(sc,150)) * 100, 4)
    else:
        rs_at = rs_20 = rs_50 = rs_150 = 0.0
    feat["rs_at_d2"] = rs_at
    feat["rs_20"]    = rs_20
    feat["rs_50"]    = rs_50
    feat["rs_150"]   = rs_150
    feat["rs_trend"] = round(rs_20 - rs_50, 4)

    # ── 기술적 지표 ────────────────────────────────────
    # RSI
    feat["rsi_14"] = calc_rsi(close, 14)

    # 볼린저밴드 위치 (0~1)
    ma20  = np.mean(close[-20:])
    std20 = np.std(close[-20:])
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    bb_pos = (close[-1] - lower) / (upper - lower) if upper != lower else 0.5
    feat["bb_pos"] = round(float(np.clip(bb_pos, 0, 1)), 4)

    # 52주 고저점 위치
    year_close = df["Close"].iloc[max(0, d_idx-252):d_idx].values
    if len(year_close) > 0:
        y_high = year_close.max()
        y_low  = year_close.min()
        feat["pos_52w_high"] = round(float(close[-1] / y_high), 4)
        feat["pos_52w_low"]  = round(float(close[-1] / y_low),  4)
    else:
        feat["pos_52w_high"] = 1.0
        feat["pos_52w_low"]  = 1.0

    # ATR 비율 (14일 ATR / 현재가)
    tr_list = []
    h_arr = high[-15:]; l_arr = low[-15:]; c_arr = close[-15:]
    for k in range(1, len(h_arr)):
        tr = max(h_arr[k]-l_arr[k], abs(h_arr[k]-c_arr[k-1]), abs(l_arr[k]-c_arr[k-1]))
        tr_list.append(tr)
    atr = np.mean(tr_list) if tr_list else 0
    feat["atr_ratio"] = round(float(atr / close[-1]) if close[-1] > 0 else 0, 4)

    # 이격도 (20일, 50일)
    ma20_pos = close[-1] / np.mean(close[-20:]) if len(close) >= 20 else 1.0
    ma50_pos = close[-1] / np.mean(close[-50:]) if len(close) >= 50 else 1.0
    feat["ma20_pos"] = round(float(ma20_pos), 4)
    feat["ma50_pos"] = round(float(ma50_pos), 4)

    # ── 섹터 OHE ──────────────────────────────────────
    SECTORS = ["Technology","Healthcare","Financial","Consumer","Energy",
               "Industrial","Materials","Utilities","Real Estate","Communication","Other"]
    sector = str(ticker_info.get("sector", "Other"))
    for s in SECTORS:
        feat[f"sec_{s}"] = 1 if sector == s else 0

    # ── 시총 OHE ──────────────────────────────────────
    CAPS = ["MegaCap","LargeCap","MidCap","SmallCap"]
    cap = str(ticker_info.get("cap", "SmallCap"))
    for c in CAPS:
        feat[f"cap_{c}"] = 1 if cap == c else 0

    return feat


if __name__ == "__main__":
    if not MASSIVE:
        send("MASSIVE_TOKEN 없음!")
        exit(1)

    # 백테스트 로드
    try:
        bt = pd.read_csv("backtest_us_raw.csv", encoding="utf-8-sig")
        print(f"백테스트 시그널: {len(bt)}건")
    except Exception as e:
        send(f"backtest_us_raw.csv 로드 실패: {e}")
        exit(1)

    # 티커 정보 로드
    try:
        tdf = pd.read_csv("tickers_us.csv", encoding="utf-8-sig")
        ticker_info = {str(r["ticker"]).strip(): {
            "sector": str(r.get("sector", "Other")),
            "cap":    str(r.get("cap", "SmallCap"))
        } for _, r in tdf.iterrows()}
    except:
        ticker_info = {}

    # SPY 수집
    min_date = pd.to_datetime(bt["date"].min())
    max_date = pd.to_datetime(bt["date"].max())
    spy_start = (min_date - timedelta(days=600)).strftime("%Y-%m-%d")
    spy_end   = (max_date + timedelta(days=20)).strftime("%Y-%m-%d")
    print("SPY 수집 중...")
    spy_df = get_ohlcv("SPY", spy_start, spy_end)
    if spy_df is None:
        send("SPY 수집 실패!")
        exit(1)
    print(f"SPY: {len(spy_df)}일치")

    send(f"🌲 LightGBM 데이터 수집 시작\n{len(bt)}개 시그널\n필터: ${MIN_PRICE}+, 거래량{MIN_VOL//1000}K+")

    samples = []
    for i, row in bt.iterrows():
        if i % 100 == 0:
            print(f"[{i}/{len(bt)}] {len(samples)}건 수집됨")

        ticker   = str(row["ticker"])
        sig_date = pd.Timestamp(row["date"])
        entry    = float(row.get("entry") or row.get("pivot") or 0)
        if entry <= 0: continue

        # 데이터 수집 범위
        fetch_start = (sig_date - timedelta(days=600)).strftime("%Y-%m-%d")
        fetch_end   = (sig_date + timedelta(days=20)).strftime("%Y-%m-%d")

        df = get_ohlcv(ticker, fetch_start, fetch_end)
        if df is None or len(df) < WINDOW + SKIP + HOLD_DAYS + 5:
            time.sleep(0.05)
            continue

        # 시그널 날짜 인덱스
        idx_list = df.index.tolist()
        matches  = [x for x in idx_list if x.date() == sig_date.date()]
        if not matches: continue
        d_idx = idx_list.index(matches[0])
        if d_idx < WINDOW + SKIP: continue

        # 라벨 계산
        label = calc_label(df, d_idx, entry)
        if label is None: continue

        # 피처 계산
        info = ticker_info.get(ticker, {})
        feat = calc_features(df, spy_df, d_idx, info)
        if feat is None: continue

        feat["ticker"] = ticker
        feat["date"]   = sig_date.strftime("%Y-%m-%d")
        feat["label"]  = label
        feat["entry"]  = round(entry, 2)
        samples.append(feat)
        time.sleep(0.05)

    print(f"\n수집 완료: {len(samples)}건")
    if not samples:
        send("수집 결과 없음")
        exit(0)

    df_out = pd.DataFrame(samples)
    pos = (df_out["label"]==1).sum()
    neg = (df_out["label"]==0).sum()
    print(f"양성:{pos} 음성:{neg} 비율:{pos/(pos+neg)*100:.1f}%")
    df_out.to_csv("lgbm_raw.csv", index=False, encoding="utf-8-sig")
    send(f"🌲 LightGBM 데이터 수집 완료\n{len(df_out)}건 (양성:{pos} 음성:{neg})")
