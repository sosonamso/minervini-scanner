"""
국장 LightGBM 데이터 수집 (올바른 구조)
- 날짜별로 전체 종목 한번에 수집 (backtest.py 방식)
- r5/r10으로 라벨 간접 계산
- API 호출 최소화
"""
import os, time, warnings, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

TOK    = os.environ.get("TELEGRAM_TOKEN", "")
CID    = os.environ.get("TELEGRAM_CHAT_ID", "")
KRX    = os.environ.get("KRX_TOKEN", "")

WINDOW     = 150
SKIP       = 2
VOL_MA     = 20
MIN_PRICE  = 1000
MIN_TRDVAL = 5

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


def get_trading_dates(start_str, end_str):
    dates = []
    d   = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    while d <= end:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return dates


def build_all_ohlcv(trading_dates):
    """날짜별로 전체 종목 수집 → 종목별 DataFrame"""
    ticker_data = {}
    total = len(trading_dates)
    for i, date_str in enumerate(trading_dates):
        if i % 50 == 0:
            print(f"  데이터 수집 [{i}/{total}] {date_str}")
        for mkt in ["KOSPI", "KOSDAQ"]:
            day = get_krx_data(date_str, mkt)
            for ticker, ohlcv in day.items():
                if ticker not in ticker_data:
                    ticker_data[ticker] = {"market": mkt, "rows": []}
                ticker_data[ticker]["rows"].append({
                    "date": pd.Timestamp(date_str), **ohlcv
                })
        time.sleep(0.3)

    result = {}
    for ticker, info in ticker_data.items():
        if len(info["rows"]) < 100: continue
        df = pd.DataFrame(info["rows"]).set_index("date").sort_index()
        df = df[["Close", "High", "Low", "Volume", "TrdVal"]].astype(float)
        df = df[df["Close"] > 0].dropna()
        if len(df) >= 100:
            meta = _row_meta.get(ticker, {})
            result[ticker] = {
                "market": info["market"],
                "df": df,
                "name": meta.get("name", ticker),
                "sector": meta.get("sector", "기타")
            }
    return result


def calc_label_indirect(r5, r10):
    if pd.isna(r5) or pd.isna(r10): return None
    if r5 < -7.0: return 0
    if r10 >= 8.0: return 1
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
    start_idx = d_idx - SKIP - WINDOW
    end_idx   = d_idx - SKIP
    if start_idx < VOL_MA: return None

    w = df.iloc[start_idx:end_idx]
    if len(w) < WINDOW: return None

    close  = w["Close"].values
    high   = w["High"].values
    low    = w["Low"].values
    volume = w["Volume"].values
    trdval = w["TrdVal"].values

    if close[-1] < MIN_PRICE: return None
    if np.mean(trdval[-20:]) / 1e8 < MIN_TRDVAL: return None

    c_min, c_max = close.min(), close.max()
    if c_max == c_min: return None
    close_norm = (close - c_min) / (c_max - c_min)

    ret = np.zeros(WINDOW)
    ret[1:] = (close[1:] / close[:-1] - 1) * 100

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

    # RS
    ref_idx = d_idx - SKIP
    td = df.index[:ref_idx + 1]
    tc = df["Close"].iloc[:ref_idx + 1].values
    rs_at = rs_20 = rs_50 = rs_150 = 0.0

    if len(mkt_df) > 0 and len(tc) >= 253:
        sa = mkt_df.reindex(td).ffill()
        if not sa.isnull().all().any():
            sc = sa["Close"].values
            def pr(arr, n): return float(arr[-1]/arr[-n]-1) if len(arr)>=n else 0.0
            w_ = [0.4,0.2,0.2,0.2]; p_ = [63,126,189,252]
            t_rs = sum(w_[i]*pr(tc,p_[i]) for i in range(4))
            s_rs = sum(w_[i]*pr(sc,p_[i]) for i in range(4))
            rs_at  = round((t_rs-s_rs)*100, 4)
            rs_20  = round((pr(tc,20) -pr(sc,20)) *100, 4)
            rs_50  = round((pr(tc,50) -pr(sc,50)) *100, 4)
            rs_150 = round((pr(tc,150)-pr(sc,150))*100, 4)

    feat["rs_at_d2"] = rs_at
    feat["rs_20"]    = rs_20
    feat["rs_50"]    = rs_50
    feat["rs_150"]   = rs_150
    feat["rs_trend"] = round(rs_20 - rs_50, 4)

    feat["rsi_14"] = calc_rsi(close, 14)

    ma20  = np.mean(close[-20:])
    std20 = np.std(close[-20:])
    upper = ma20 + 2*std20; lower = ma20 - 2*std20
    feat["bb_pos"] = round(float(np.clip(
        (close[-1]-lower)/(upper-lower) if upper!=lower else 0.5, 0, 1)), 4)

    yc = df["Close"].iloc[max(0,d_idx-252):d_idx].values
    feat["pos_52w_high"] = round(float(close[-1]/yc.max()), 4) if len(yc)>0 else 1.0
    feat["pos_52w_low"]  = round(float(close[-1]/yc.min()), 4) if len(yc)>0 else 1.0

    tr_list = []
    for k in range(1, len(high[-15:])):
        h=high[-15:]; l=low[-15:]; c=close[-15:]
        tr_list.append(max(h[k]-l[k], abs(h[k]-c[k-1]), abs(l[k]-c[k-1])))
    atr = np.mean(tr_list) if tr_list else 0
    feat["atr_ratio"] = round(float(atr/close[-1]) if close[-1]>0 else 0, 4)
    feat["ma20_pos"]  = round(float(close[-1]/np.mean(close[-20:])), 4)
    feat["ma50_pos"]  = round(float(close[-1]/np.mean(close[-50:])) if len(close)>=50 else 1.0, 4)
    feat["trdval_20"] = round(float(np.mean(trdval[-20:])/1e8), 2)

    KR_SECTORS = ["음식료품","섬유의복","종이목재","화학","의약품",
                  "비금속광물","철강금속","기계","전기전자","의료정밀",
                  "운수장비","유통업","전기가스업","건설업","운수창고업",
                  "통신업","금융업","증권","보험","서비스업","기타"]
    sector = str(ticker_info.get("sector","기타"))
    for s in KR_SECTORS:
        feat[f"sec_{s}"] = 1 if sector==s else 0

    market = str(ticker_info.get("market","KOSPI"))
    feat["mkt_KOSPI"]  = 1 if market=="KOSPI"  else 0
    feat["mkt_KOSDAQ"] = 1 if market=="KOSDAQ" else 0

    return feat


if __name__ == "__main__":
    if not KRX:
        send("KRX_TOKEN 없음!"); exit(1)

    # 백테스트 로드 + 라벨 계산
    bt = pd.read_csv("backtest_raw.csv", encoding="utf-8-sig")
    bt["date"] = pd.to_datetime(bt["date"])
    bt["label"] = bt.apply(
        lambda r: calc_label_indirect(r.get("r5"), r.get("r10")), axis=1)
    bt = bt.dropna(subset=["label"]).copy()
    bt["label"] = bt["label"].astype(int)
    bt = bt.sort_values("date").reset_index(drop=True)

    pos = (bt.label==1).sum(); neg = (bt.label==0).sum()
    print(f"시그널: {len(bt)}건 (양성:{pos} 음성:{neg})")

    # 수집 기간
    data_start = (bt["date"].min() - timedelta(days=600)).strftime("%Y%m%d")
    data_end   = bt["date"].max().strftime("%Y%m%d")
    trading_dates = get_trading_dates(data_start, data_end)
    print(f"수집 기간: {data_start} ~ {data_end} ({len(trading_dates)}거래일)")

    send(f"🌲 국장 LightGBM 데이터 수집 시작\n"
         f"{len(bt)}건 / 수집기간: {len(trading_dates)}거래일\n"
         f"라벨: r5>=-7% AND r10>=8%")

    # 전체 OHLCV 구축 (날짜별)
    print("전체 OHLCV 구축 중...")
    all_ohlcv = build_all_ohlcv(trading_dates)
    print(f"종목 구축 완료: {len(all_ohlcv)}개")

    # 지수 (KODEX200)
    mkt_df = pd.DataFrame(columns=["Close"])
    for ticker, info in all_ohlcv.items():
        name = info.get("name","")
        if "KODEX" in name and "200" in name \
                and "레버리지" not in name and "인버스" not in name:
            mkt_df = info["df"][["Close"]].copy()
            print(f"지수 ETF: {name}({ticker}) {len(mkt_df)}일치")
            break

    send(f"OHLCV 구축 완료: {len(all_ohlcv)}개 종목\n피처 계산 시작...")

    # 피처 계산
    samples = []
    target_tickers = set(bt["ticker"].astype(str).str.zfill(6))

    for i, (ticker_raw, group) in enumerate(bt.groupby("ticker")):
        ticker = str(ticker_raw).zfill(6)
        if i % 20 == 0:
            print(f"[{i}/{len(target_tickers)}] {ticker} | 수집:{len(samples)}건")

        if ticker not in all_ohlcv:
            continue

        info = all_ohlcv[ticker]
        df   = info["df"]
        idx_list = df.index.tolist()

        for _, row in group.iterrows():
            sig_date = pd.Timestamp(row["date"])
            matches  = [x for x in idx_list if x.date() == sig_date.date()]
            if not matches: continue
            d_idx = idx_list.index(matches[0])
            if d_idx < WINDOW + SKIP: continue

            feat = calc_features(df, mkt_df, d_idx, {
                "sector": str(row.get("sector","기타")),
                "market": info["market"]
            })
            if feat is None: continue

            feat["ticker"] = ticker
            feat["date"]   = sig_date.strftime("%Y-%m-%d")
            feat["label"]  = int(row["label"])
            feat["entry"]  = float(row["entry"])
            feat["r5"]     = row.get("r5")
            feat["r10"]    = row.get("r10")
            samples.append(feat)

    print(f"\n수집 완료: {len(samples)}건")
    if not samples:
        send("수집 결과 없음"); exit(0)

    df_out = pd.DataFrame(samples)
    pos2 = (df_out.label==1).sum(); neg2 = (df_out.label==0).sum()
    df_out.to_csv("lgbm_raw_kr.csv", index=False, encoding="utf-8-sig")
    send(f"🌲 국장 LightGBM 데이터 수집 완료\n"
         f"{len(df_out)}건 (양성:{pos2} 음성:{neg2})")
