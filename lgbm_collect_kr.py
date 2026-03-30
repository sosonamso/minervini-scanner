"""
국장 LightGBM 데이터 수집
- backtest_raw.csv (미너비니 시그널) 기반
- KRX API로 각 종목 피처 계산
- 라벨: 손절(-7%) 안 걸리고 10일 내 +8% 도달 → 1
"""
import os, time, warnings, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

TOK    = os.environ.get("TELEGRAM_TOKEN", "")
CID    = os.environ.get("TELEGRAM_CHAT_ID", "")
KRX    = os.environ.get("KRX_TOKEN", "")

WINDOW      = 150
SKIP        = 2
VOL_MA      = 20
MIN_PRICE   = 1000     # 최소 주가 1000원
MIN_TRDVAL  = 5        # 일평균 거래대금 최소 5억
STOP_PCT    = 0.93     # 손절 -7%
TARGET_PCT  = 1.08     # 목표 +8%
HOLD_DAYS   = 10

_row_meta = {}


def send(text):
    print(text)
    if TOK:
        try:
            requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",
                         data={"chat_id": CID, "text": text}, timeout=10)
        except: pass


def get_krx_data(date_str, market="KOSPI"):
    url = ("https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd"
           if market == "KOSPI"
           else "https://data-dbg.krx.co.kr/svc/apis/sto/ksq_bydd_trd")
    headers = {"AUTH_KEY": KRX.strip(),
               "Content-Type": "application/json",
               "Accept": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers,
                                json={"basDd": date_str}, timeout=30)
            if resp.status_code != 200: return {}
            block = resp.json().get("OutBlock_1", [])
            if not block: return {}
            result = {}
            for row in block:
                try:
                    ticker = str(row.get("ISU_CD", "")).strip()
                    if not ticker: continue
                    _row_meta[ticker] = {
                        "name":   str(row.get("ISU_NM", "")).strip(),
                        "sector": str(row.get("SECT_TP_NM", "기타")).strip() or "기타"
                    }
                    result[ticker] = {
                        "Close":  float(str(row.get("TDD_CLSPRC", "0")).replace(",", "")),
                        "High":   float(str(row.get("TDD_HGPRC", "0")).replace(",", "")),
                        "Low":    float(str(row.get("TDD_LWPRC", "0")).replace(",", "")),
                        "Volume": float(str(row.get("ACC_TRDVOL", "0")).replace(",", "")),
                        "TrdVal": float(str(row.get("ACC_TRDVAL", "0")).replace(",", "")),
                    }
                except: pass
            return result
        except Exception as e:
            print(f"KRX 오류(시도{attempt+1}): {e}")
            time.sleep(2 * (attempt + 1))
    return {}


def get_trading_dates(start_date, end_date):
    dates = []
    d = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while d <= end:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return dates


def build_ticker_ohlcv(ticker, market, start_date, end_date):
    """특정 종목의 OHLCV 구축"""
    trading_dates = get_trading_dates(start_date, end_date)
    rows = []
    for date_str in trading_dates:
        day_data = get_krx_data(date_str, market)
        if ticker in day_data:
            rows.append({"date": pd.Timestamp(date_str), **day_data[ticker]})
        time.sleep(0.3)

    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("date").sort_index()
    df = df[["Close", "High", "Low", "Volume", "TrdVal"]].astype(float)
    df = df[df["Close"] > 0].dropna()
    return df if len(df) >= WINDOW + SKIP + HOLD_DAYS else None


def calc_label(df, sig_idx, entry_price):
    """손절 안 걸리고 10일 내 +8% 도달 여부"""
    stop   = entry_price * STOP_PCT
    target = entry_price * TARGET_PCT
    future = df.iloc[sig_idx + 1: sig_idx + 1 + HOLD_DAYS]
    if len(future) == 0: return None
    for _, row in future.iterrows():
        if row["Low"] <= stop:   return 0
        if row["High"] >= target: return 1
    return 0


def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period-1) + gains[i]) / period
        avg_loss = (avg_loss * (period-1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 4)


def calc_features(df, mkt_df, d_idx, ticker_info):
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
    trdval = w["TrdVal"].values

    # 필터
    if close[-1] < MIN_PRICE: return None
    trdval_avg = np.mean(trdval[-20:]) / 1e8
    if trdval_avg < MIN_TRDVAL: return None

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

    # RS (미너비니 방식)
    ref_idx = d_idx - SKIP
    td = df.index[:ref_idx + 1]
    sa = mkt_df.reindex(td).ffill()
    tc = df["Close"].iloc[:ref_idx + 1].values
    sc = sa["Close"].values if not sa.isnull().all().any() else None

    if sc is not None and len(tc) >= 253:
        def pr(arr, n): return float(arr[-1] / arr[-n] - 1) if len(arr) >= n else 0.0
        w_ = [0.4, 0.2, 0.2, 0.2]; p_ = [63, 126, 189, 252]
        t_rs = sum(w_[i]*pr(tc, p_[i]) for i in range(4))
        s_rs = sum(w_[i]*pr(sc, p_[i]) for i in range(4))
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

    # 기술적 지표
    feat["rsi_14"] = calc_rsi(close, 14)

    ma20  = np.mean(close[-20:])
    std20 = np.std(close[-20:])
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    bb_pos = (close[-1] - lower) / (upper - lower) if upper != lower else 0.5
    feat["bb_pos"] = round(float(np.clip(bb_pos, 0, 1)), 4)

    year_close = df["Close"].iloc[max(0, d_idx-252):d_idx].values
    if len(year_close) > 0:
        feat["pos_52w_high"] = round(float(close[-1] / year_close.max()), 4)
        feat["pos_52w_low"]  = round(float(close[-1] / year_close.min()), 4)
    else:
        feat["pos_52w_high"] = 1.0
        feat["pos_52w_low"]  = 1.0

    tr_list = []
    h_arr = high[-15:]; l_arr = low[-15:]; c_arr = close[-15:]
    for k in range(1, len(h_arr)):
        tr = max(h_arr[k]-l_arr[k], abs(h_arr[k]-c_arr[k-1]), abs(l_arr[k]-c_arr[k-1]))
        tr_list.append(tr)
    atr = np.mean(tr_list) if tr_list else 0
    feat["atr_ratio"] = round(float(atr / close[-1]) if close[-1] > 0 else 0, 4)

    feat["ma20_pos"] = round(float(close[-1] / np.mean(close[-20:])), 4)
    feat["ma50_pos"] = round(float(close[-1] / np.mean(close[-50:])), 4)

    # 거래대금 평균 (피처로)
    feat["trdval_20"] = round(float(np.mean(trdval[-20:]) / 1e8), 2)

    # 섹터 OHE (국장)
    KR_SECTORS = ["음식료품", "섬유의복", "종이목재", "화학", "의약품",
                  "비금속광물", "철강금속", "기계", "전기전자", "의료정밀",
                  "운수장비", "유통업", "전기가스업", "건설업", "운수창고업",
                  "통신업", "금융업", "증권", "보험", "서비스업", "기타"]
    sector = str(ticker_info.get("sector", "기타"))
    for s in KR_SECTORS:
        feat[f"sec_{s}"] = 1 if sector == s else 0

    # 시장 OHE
    market = str(ticker_info.get("market", "KOSPI"))
    feat["mkt_KOSPI"]  = 1 if market == "KOSPI"  else 0
    feat["mkt_KOSDAQ"] = 1 if market == "KOSDAQ" else 0

    return feat


if __name__ == "__main__":
    if not KRX:
        send("KRX_TOKEN 없음!")
        exit(1)

    # 백테스트 로드
    try:
        bt = pd.read_csv("backtest_raw.csv", encoding="utf-8-sig")
        bt["date"] = pd.to_datetime(bt["date"])
        bt = bt.sort_values("date").reset_index(drop=True)
        print(f"백테스트 시그널: {len(bt)}건")
    except Exception as e:
        send(f"backtest_raw.csv 로드 실패: {e}")
        exit(1)

    # 지수 (KODEX200) 수집 - 전체 기간
    min_date = bt["date"].min()
    max_date = bt["date"].max()
    data_start = (min_date - timedelta(days=600)).strftime("%Y%m%d")
    data_end   = (max_date + timedelta(days=20)).strftime("%Y%m%d")

    send(f"🌲 국장 LightGBM 데이터 수집 시작\n{len(bt)}개 시그널\n필터: {MIN_PRICE}원+, 거래대금{MIN_TRDVAL}억+")

    # KODEX200 수집 (지수 대용)
    print("KODEX200 수집 중...")
    mkt_ticker = None
    mkt_df     = None

    # 전체 기간 날짜별 순회해서 KODEX200 찾기
    trading_dates = get_trading_dates(data_start, data_end)
    mkt_rows = []
    for date_str in trading_dates[:20]:  # 처음 몇 날만 확인해서 KODEX200 ticker 찾기
        day_data = get_krx_data(date_str, "KOSPI")
        for t, meta in _row_meta.items():
            if "KODEX" in meta["name"] and "200" in meta["name"] \
                    and "레버리지" not in meta["name"] and "인버스" not in meta["name"]:
                mkt_ticker = t
                print(f"KODEX200 발견: {meta['name']}({t})")
                break
        if mkt_ticker: break
        time.sleep(0.3)

    if mkt_ticker:
        print(f"KODEX200({mkt_ticker}) 전체 기간 수집 중...")
        mkt_df = build_ticker_ohlcv(mkt_ticker, "KOSPI", data_start, data_end)
        if mkt_df is not None:
            mkt_df = mkt_df[["Close"]]
            print(f"KODEX200: {len(mkt_df)}일치")

    if mkt_df is None:
        print("KODEX200 없음 → RS 피처 0으로 처리")
        mkt_df = pd.DataFrame(columns=["Close"])

    samples = []
    ticker_cache = {}  # 종목별 OHLCV 캐시

    for i, row in bt.iterrows():
        if i % 50 == 0:
            print(f"[{i}/{len(bt)}] {len(samples)}건 수집됨")

        ticker   = str(row["ticker"]).zfill(6)
        sig_date = pd.Timestamp(row["date"])
        market   = str(row["market"])
        entry    = float(row["entry"])
        if entry <= 0: continue

        # 종목 OHLCV (캐시 활용)
        if ticker not in ticker_cache:
            fetch_start = (sig_date - timedelta(days=600)).strftime("%Y%m%d")
            fetch_end   = (max_date + timedelta(days=20)).strftime("%Y%m%d")
            df = build_ticker_ohlcv(ticker, market, fetch_start, fetch_end)
            ticker_cache[ticker] = df
        else:
            df = ticker_cache[ticker]

        if df is None: continue

        # 시그널 날짜 인덱스
        idx_list = df.index.tolist()
        matches  = [x for x in idx_list if x.date() == sig_date.date()]
        if not matches: continue
        d_idx = idx_list.index(matches[0])
        if d_idx < WINDOW + SKIP: continue

        # 라벨 (r5/r10 컬럼 있으면 활용, 없으면 계산)
        r10 = row.get("r10")
        r5  = row.get("r5")
        if pd.notna(r10) and pd.notna(r5):
            # 간접 라벨: r10 >= 8% 이고 도중에 -7% 이하 없었는지
            # 정확한 라벨은 High/Low로 재계산
            label = calc_label(df, d_idx, entry)
        else:
            label = calc_label(df, d_idx, entry)

        if label is None: continue

        # 피처 계산
        info = {"sector": str(row.get("sector", "기타")),
                "market": market}
        feat = calc_features(df, mkt_df, d_idx, info)
        if feat is None: continue

        feat["ticker"] = ticker
        feat["date"]   = sig_date.strftime("%Y-%m-%d")
        feat["label"]  = label
        feat["entry"]  = round(entry, 0)
        feat["r5"]     = r5
        feat["r10"]    = r10
        samples.append(feat)

    print(f"\n수집 완료: {len(samples)}건")
    if not samples:
        send("수집 결과 없음")
        exit(0)

    df_out = pd.DataFrame(samples)
    pos = (df_out["label"] == 1).sum()
    neg = (df_out["label"] == 0).sum()
    print(f"양성:{pos} 음성:{neg} 비율:{pos/(pos+neg)*100:.1f}%")
    df_out.to_csv("lgbm_raw_kr.csv", index=False, encoding="utf-8-sig")
    send(f"🌲 국장 LightGBM 데이터 수집 완료\n{len(df_out)}건 (양성:{pos} 음성:{neg})")
