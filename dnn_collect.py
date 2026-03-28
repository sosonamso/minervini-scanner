"""
DNN 학습용 데이터 수집 스크립트
- 미너비니 로직 없음, 순수 가격/거래량 + RS 데이터
- 양성: r5 최대값 >= 8%, 1년 이상 간격으로 최대 3개
- 음성: 양성 수 × 3개 (급등 구간 미겹치는 구간)
- RS 피처: rs_at_d2 / rs_20 / rs_50 / rs_150 / rs_trend
- 기간: 10년치 (3600일)
- 출력: dnn_raw_r5.csv / dnn_raw_r10.csv
"""
import os, time, warnings, random
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

MASSIVE      = os.environ.get("MASSIVE_TOKEN", "")
TOK          = os.environ.get("TELEGRAM_TOKEN", "")
CID          = os.environ.get("TELEGRAM_CHAT_ID", "")

HISTORY_DAYS = 3600   # 10년치
WINDOW       = 150    # 입력 윈도우
SKIP         = 2      # D-day 기준 D-2부터
VOL_MA       = 20     # 거래량 정규화 기준
MIN_GAP_DAYS = 252    # 양성 샘플 간 최소 간격 (1년 = 252 거래일)
MAX_POS      = 3      # 종목당 최대 양성 샘플 수
NEG_RATIO    = 3      # 양성 1개당 음성 샘플 수

R5_THRESH    = 0.08
R10_THRESH   = 0.14
NEG_R5       = 0.03
NEG_R10      = 0.05


def send(text):
    print(text)
    if TOK:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TOK}/sendMessage",
                data={"chat_id": CID, "text": text}, timeout=10
            )
        except:
            pass


def send_file(filepath, caption=""):
    if TOK:
        try:
            with open(filepath, "rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{TOK}/sendDocument",
                    data={"chat_id": CID, "caption": caption},
                    files={"document": f}, timeout=30
                )
        except Exception as e:
            print(f"파일 전송 실패: {e}")


def get_ohlcv(ticker, start, end):
    url    = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": MASSIVE}
    for _ in range(2):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(12)
                continue
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            if not results or len(results) < WINDOW + SKIP + 20:
                return None
            df = pd.DataFrame(results)
            df["date"] = (
                pd.to_datetime(df["t"], unit="ms")
                .dt.tz_localize("UTC")
                .dt.tz_convert("America/New_York")
                .dt.normalize()
                .dt.tz_localize(None)
            )
            df = df.set_index("date").sort_index()
            df = df.rename(columns={"c": "Close", "v": "Volume"})
            df = df[["Close", "Volume"]].astype(float).dropna()
            return df
        except:
            time.sleep(3)
    return None


def calc_rs_features(df, spy_df, d_idx):
    """D-2일 기준 RS 피처 5개 계산"""
    ref_idx = d_idx - SKIP

    if ref_idx < 150:
        return None

    ticker_dates = df.index[:ref_idx + 1]
    spy_aligned  = spy_df.reindex(ticker_dates).ffill()

    if spy_aligned.isnull().all().any():
        return None

    ticker_close = df["Close"].iloc[:ref_idx + 1].values
    spy_close    = spy_aligned["Close"].values

    if len(ticker_close) < 152:
        return None

    t_ret    = np.zeros(len(ticker_close))
    s_ret    = np.zeros(len(spy_close))
    t_ret[1:] = ticker_close[1:] / ticker_close[:-1] - 1
    s_ret[1:] = spy_close[1:]    / spy_close[:-1]    - 1
    rs_daily  = (t_ret - s_ret) * 100

    rs_at_d2 = round(float(rs_daily[-1]),          4)
    rs_20    = round(float(np.mean(rs_daily[-20:])),  4)
    rs_50    = round(float(np.mean(rs_daily[-50:])),  4)
    rs_150   = round(float(np.mean(rs_daily[-150:])), 4)
    rs_trend = round(float(rs_20 - rs_50),           4)

    return {
        "rs_at_d2": rs_at_d2,
        "rs_20":    rs_20,
        "rs_50":    rs_50,
        "rs_150":   rs_150,
        "rs_trend": rs_trend,
    }


def make_features(df, spy_df, d_idx):
    """d_idx 기준 피처 생성 (가격/거래량 + RS)"""
    start_idx = d_idx - SKIP - WINDOW
    end_idx   = d_idx - SKIP

    if start_idx < VOL_MA:
        return None

    window_df = df.iloc[start_idx:end_idx].copy()
    if len(window_df) < WINDOW:
        return None

    close  = window_df["Close"].values
    volume = window_df["Volume"].values

    c_min, c_max = close.min(), close.max()
    if c_max == c_min:
        return None
    close_norm = (close - c_min) / (c_max - c_min)

    ret = np.zeros(WINDOW)
    ret[1:] = (close[1:] / close[:-1] - 1) * 100

    vol_ratio = np.zeros(WINDOW)
    for k in range(WINDOW):
        abs_idx  = start_idx + k
        past_vol = df["Volume"].iloc[max(0, abs_idx - VOL_MA): abs_idx]
        ma       = past_vol.mean()
        vol_ratio[k] = volume[k] / ma if ma > 0 else 1.0

    feat = {}
    for k in range(WINDOW):
        feat[f"ret_{k+1}"]        = round(ret[k], 4)
        feat[f"close_norm_{k+1}"] = round(float(close_norm[k]), 4)
        feat[f"vol_ratio_{k+1}"]  = round(float(vol_ratio[k]), 4)

    rs_feat = calc_rs_features(df, spy_df, d_idx)
    if rs_feat is None:
        return None
    feat.update(rs_feat)

    return feat


def pick_pos_candidates(r_arr, r_thresh, n, min_gap, max_pos):
    """
    1년 이상 간격으로 최대 max_pos개 양성 후보 선택
    r_arr 값이 높은 순으로 뽑되 간격 조건 충족
    """
    valid = np.where(
        (~np.isnan(r_arr)) & (r_arr >= r_thresh) &
        (np.arange(n) >= WINDOW + SKIP)
    )[0]

    if len(valid) == 0:
        return []

    # r값 높은 순 정렬
    sorted_idx = valid[np.argsort(r_arr[valid])[::-1]]

    chosen = []
    for idx in sorted_idx:
        # 이미 선택된 것들과 간격 확인
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(idx)
        if len(chosen) >= max_pos:
            break

    return chosen


def extract_samples(ticker, df, spy_df, r5_thresh, r10_thresh, neg_r5, neg_r10):
    """종목 하나에서 양성/음성 샘플 추출"""
    n      = len(df)
    closes = df["Close"].values
    samples = []

    r5  = np.full(n, np.nan)
    r10 = np.full(n, np.nan)
    for i in range(n):
        if i + 5  < n: r5[i]  = closes[i + 5]  / closes[i] - 1
        if i + 10 < n: r10[i] = closes[i + 10] / closes[i] - 1

    # ── 양성 샘플 (r5) ───────────────────────────────────
    pos_idxs_r5 = pick_pos_candidates(r5, r5_thresh, n, MIN_GAP_DAYS, MAX_POS)

    for pos_idx in pos_idxs_r5:
        feat = make_features(df, spy_df, pos_idx)
        if feat:
            row = {
                "ticker": ticker,
                "date":   df.index[pos_idx].strftime("%Y-%m-%d"),
                "label":  1,
                "r5":     round(float(r5[pos_idx]) * 100, 2),
                "r10":    round(float(r10[pos_idx]) * 100, 2) if not np.isnan(r10[pos_idx]) else None,
            }
            row.update(feat)
            samples.append(("r5", row))

    # ── 양성 샘플 (r10) ──────────────────────────────────
    pos_idxs_r10 = pick_pos_candidates(r10, r10_thresh, n, MIN_GAP_DAYS, MAX_POS)

    for pos_idx in pos_idxs_r10:
        feat = make_features(df, spy_df, pos_idx)
        if feat:
            row = {
                "ticker": ticker,
                "date":   df.index[pos_idx].strftime("%Y-%m-%d"),
                "label":  1,
                "r5":     round(float(r5[pos_idx]) * 100, 2) if not np.isnan(r5[pos_idx]) else None,
                "r10":    round(float(r10[pos_idx]) * 100, 2),
            }
            row.update(feat)
            samples.append(("r10", row))

    # ── 음성 샘플 ────────────────────────────────────────
    # 양성 구간 전체 제외
    exclude = set()
    for pos_idx in pos_idxs_r5 + pos_idxs_r10:
        for k in range(pos_idx - WINDOW - SKIP - 10, pos_idx + 10):
            exclude.add(k)

    neg_candidates_r5 = [
        i for i in range(WINDOW + SKIP, n - 10)
        if i not in exclude
        and not np.isnan(r5[i]) and r5[i] <= neg_r5
    ]
    neg_candidates_r10 = [
        i for i in range(WINDOW + SKIP, n - 10)
        if i not in exclude
        and not np.isnan(r10[i]) and r10[i] <= neg_r10
    ]

    for model, neg_candidates, pos_idxs in [
        ("r5",  neg_candidates_r5,  pos_idxs_r5),
        ("r10", neg_candidates_r10, pos_idxs_r10),
    ]:
        if not neg_candidates:
            continue

        # 양성 수 × NEG_RATIO 만큼 음성 뽑기
        n_neg  = min(len(pos_idxs) * NEG_RATIO, len(neg_candidates))
        chunk  = max(1, len(neg_candidates) // max(n_neg, 1))
        chosen = []
        for k in range(n_neg):
            sub = neg_candidates[k*chunk:(k+1)*chunk]
            if sub:
                chosen.append(random.choice(sub))

        for neg_idx in chosen:
            feat = make_features(df, spy_df, neg_idx)
            if feat:
                row = {
                    "ticker": ticker,
                    "date":   df.index[neg_idx].strftime("%Y-%m-%d"),
                    "label":  0,
                    "r5":     round(float(r5[neg_idx]) * 100, 2) if not np.isnan(r5[neg_idx]) else None,
                    "r10":    round(float(r10[neg_idx]) * 100, 2) if not np.isnan(r10[neg_idx]) else None,
                }
                row.update(feat)
                samples.append((model, row))

    return samples


if __name__ == "__main__":
    if not MASSIVE:
        send("MASSIVE_TOKEN 없음!")
        exit(1)

    random.seed(42)

    end     = datetime.today()
    start   = (end - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    # 티커 로드
    try:
        tdf         = pd.read_csv("tickers_us.csv", encoding="utf-8-sig")
        ticker_list = [str(r["ticker"]).strip() for _, r in tdf.iterrows()
                       if str(r["ticker"]).strip()]
    except Exception as e:
        send(f"tickers_us.csv 로드 실패: {e}")
        exit(1)

    # SPY 먼저 수집
    send(f"🧠 DNN 데이터 수집 시작\n총 {len(ticker_list)}개 종목\nSPY 수집 중...")
    spy_df = get_ohlcv("SPY", start, end_str)
    if spy_df is None:
        send("SPY 데이터 수집 실패!")
        exit(1)
    print(f"SPY 수집 완료: {len(spy_df)}일치")

    send(
        f"SPY 수집 완료\n"
        f"기간: 10년({HISTORY_DAYS}일) | 윈도우: {WINDOW}일\n"
        f"양성: 1년 간격 최대 {MAX_POS}개 | 음성: 양성×{NEG_RATIO}\n"
        f"r5>={R5_THRESH*100:.0f}% / r10>={R10_THRESH*100:.0f}%\n"
        f"RS 피처: rs_at_d2/rs_20/rs_50/rs_150/rs_trend"
    )

    samples_r5  = []
    samples_r10 = []

    for i, ticker in enumerate(ticker_list):
        if i % 200 == 0:
            print(f"[{i}/{len(ticker_list)}] r5:{len(samples_r5)}건 r10:{len(samples_r10)}건")

        df = get_ohlcv(ticker, start, end_str)
        if df is None:
            continue

        try:
            results = extract_samples(
                ticker, df, spy_df,
                R5_THRESH, R10_THRESH,
                NEG_R5, NEG_R10
            )
            for model, row in results:
                if model == "r5":
                    samples_r5.append(row)
                else:
                    samples_r10.append(row)
        except Exception as e:
            print(f"  {ticker} 오류: {e}")

        time.sleep(0.05)

    print(f"\n수집 완료: r5={len(samples_r5)}건 / r10={len(samples_r10)}건")

    for fname, samples in [("dnn_raw_r5.csv", samples_r5), ("dnn_raw_r10.csv", samples_r10)]:
        if samples:
            df_out = pd.DataFrame(samples)
            pos = (df_out["label"] == 1).sum()
            neg = (df_out["label"] == 0).sum()
            df_out.to_csv(fname, index=False, encoding="utf-8-sig")
            send_file(fname, f"🧠 {fname} ({len(df_out)}건 / 양성:{pos} 음성:{neg})")
            print(f"{fname} 저장: {len(df_out)}건 (양성:{pos} 음성:{neg})")

    send(
        f"🧠 DNN 데이터 수집 완료\n"
        f"r5: {len(samples_r5)}건\n"
        f"r10: {len(samples_r10)}건"
    )
