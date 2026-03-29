"""
LSTM 급등 예측 스캐너
- 전체 종목 LSTM 점수 계산 → Top10 추출
- scanner_lstm_top10.csv 누적 관리 (10일치)
- 과거 종목 수익률 자동 업데이트
"""
import os, time, warnings, requests, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

TOK    = os.environ.get("TELEGRAM_TOKEN", "")
CID    = os.environ.get("TELEGRAM_CHAT_ID", "")
MASSIVE= os.environ.get("MASSIVE_TOKEN", "")

WINDOW   = 150
SKIP     = 2
VOL_MA   = 20
TOP_N    = 10
KEEP_DAYS= 10  # 최근 N일치 보관

# ── LSTM 모델 ─────────────────────────────────────────
class SurgeLSTM(nn.Module):
    def __init__(self, n_feat=3, n_rs=5, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.rs_fc = nn.Sequential(nn.Linear(n_rs, 32), nn.ReLU())
        self.head  = nn.Sequential(
            nn.Linear(hidden + 32, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x_seq, x_rs):
        out, _   = self.lstm(x_seq)
        lstm_out = out[:, -1, :]
        return self.head(torch.cat([lstm_out, self.rs_fc(x_rs)], dim=1)).squeeze()


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
            if not results or len(results) < 160: return None
            df = pd.DataFrame(results)
            df["date"] = (pd.to_datetime(df["t"], unit="ms")
                         .dt.tz_localize("UTC").dt.tz_convert("America/New_York")
                         .dt.normalize().dt.tz_localize(None))
            df = df.set_index("date").sort_index()
            df = df.rename(columns={"c": "Close", "v": "Volume"})
            return df[["Close", "Volume"]].astype(float).dropna()
        except: time.sleep(3)
    return None


def make_features(df, spy_df, d_idx):
    start_idx = d_idx - SKIP - WINDOW
    end_idx   = d_idx - SKIP
    if start_idx < VOL_MA: return None, None

    w = df.iloc[start_idx:end_idx]
    if len(w) < WINDOW: return None, None

    close  = w["Close"].values
    volume = w["Volume"].values

    c_min, c_max = close.min(), close.max()
    if c_max == c_min: return None, None
    close_norm = (close - c_min) / (c_max - c_min)

    ret = np.zeros(WINDOW)
    ret[1:] = (close[1:] / close[:-1] - 1) * 100

    vol_ratio = np.zeros(WINDOW)
    for k in range(WINDOW):
        abs_idx  = start_idx + k
        past_vol = df["Volume"].iloc[max(0, abs_idx - VOL_MA): abs_idx]
        ma = past_vol.mean()
        vol_ratio[k] = volume[k] / ma if ma > 0 else 1.0

    seq = np.stack([ret, close_norm, vol_ratio], axis=1).astype(np.float32)

    # RS 피처 (미너비니 방식)
    ref_idx = d_idx - SKIP
    if ref_idx < 252: return seq, np.zeros(5, dtype=np.float32)

    td = df.index[:ref_idx + 1]
    sa = spy_df.reindex(td).ffill()
    if sa.isnull().all().any(): return seq, np.zeros(5, dtype=np.float32)

    tc = df["Close"].iloc[:ref_idx + 1].values
    sc = sa["Close"].values
    if len(tc) < 253: return seq, np.zeros(5, dtype=np.float32)

    def pr(arr, n): return float(arr[-1] / arr[-n] - 1) if len(arr) >= n else 0.0
    w_ = [0.4, 0.2, 0.2, 0.2]; p_ = [63, 126, 189, 252]
    rs_at = (sum(w_[i]*pr(tc,p_[i]) for i in range(4)) -
             sum(w_[i]*pr(sc,p_[i]) for i in range(4))) * 100
    rs_20  = (pr(tc,20)  - pr(sc,20))  * 100
    rs_50  = (pr(tc,50)  - pr(sc,50))  * 100
    rs_150 = (pr(tc,150) - pr(sc,150)) * 100
    rs = np.array([rs_at, rs_20, rs_50, rs_150, rs_20-rs_50], dtype=np.float32)
    return seq, rs


def predict_score(df, spy_df, model, seq_sc, rs_sc, device):
    """종목 df의 마지막 날 기준 LSTM 점수"""
    d_idx = len(df) - 1
    seq, rs = make_features(df, spy_df, d_idx)
    if seq is None: return None

    seq_scaled = seq_sc.transform(seq.reshape(1, -1)).reshape(1, 150, 3).astype(np.float32)
    rs_scaled  = rs_sc.transform(rs.reshape(1, -1)).astype(np.float32)

    with torch.no_grad():
        xseq = torch.FloatTensor(seq_scaled).to(device)
        xrs  = torch.FloatTensor(rs_scaled).to(device)
        score = model(xseq, xrs).cpu().item()
    return round(score, 4)


def update_returns(df_hist, spy_df, today):
    """과거 종목들 수익률 업데이트"""
    updated = df_hist.copy()
    for idx, row in updated.iterrows():
        sig_date = pd.Timestamp(row["date"])
        ticker   = row["ticker"]

        # r5 업데이트
        if pd.isna(row.get("r5")) or row.get("r5") == "":
            target = sig_date + timedelta(days=7)
            if today >= target:
                df = get_ohlcv(ticker,
                               sig_date.strftime("%Y-%m-%d"),
                               (sig_date + timedelta(days=10)).strftime("%Y-%m-%d"))
                if df is not None and len(df) >= 6:
                    entry = float(df["Close"].iloc[0])
                    r5_val = round((float(df["Close"].iloc[5]) / entry - 1) * 100, 2)
                    updated.at[idx, "r5"] = r5_val
                time.sleep(0.05)

        # r10 업데이트
        if pd.isna(row.get("r10")) or row.get("r10") == "":
            target = sig_date + timedelta(days=14)
            if today >= target:
                df = get_ohlcv(ticker,
                               sig_date.strftime("%Y-%m-%d"),
                               (sig_date + timedelta(days=15)).strftime("%Y-%m-%d"))
                if df is not None and len(df) >= 11:
                    entry = float(df["Close"].iloc[0])
                    r10_val = round((float(df["Close"].iloc[10]) / entry - 1) * 100, 2)
                    updated.at[idx, "r10"] = r10_val
                time.sleep(0.05)

        # r20 업데이트
        if pd.isna(row.get("r20")) or row.get("r20") == "":
            target = sig_date + timedelta(days=28)
            if today >= target:
                df = get_ohlcv(ticker,
                               sig_date.strftime("%Y-%m-%d"),
                               (sig_date + timedelta(days=30)).strftime("%Y-%m-%d"))
                if df is not None and len(df) >= 21:
                    entry = float(df["Close"].iloc[0])
                    r20_val = round((float(df["Close"].iloc[20]) / entry - 1) * 100, 2)
                    updated.at[idx, "r20"] = r20_val
                time.sleep(0.05)

    return updated


if __name__ == "__main__":
    if not MASSIVE:
        send("MASSIVE_TOKEN 없음!")
        exit(1)

    device = torch.device("cpu")  # Actions에서는 CPU 사용
    today  = datetime.today()
    today_str = today.strftime("%Y-%m-%d")

    # 모델 & 스케일러 로드
    try:
        model_r5  = SurgeLSTM().to(device)
        model_r10 = SurgeLSTM().to(device)
        model_r5.load_state_dict(torch.load("model_lstm_r5.pth",  map_location=device))
        model_r10.load_state_dict(torch.load("model_lstm_r10.pth", map_location=device))
        model_r5.eval(); model_r10.eval()

        with open("seq_scaler_r5.pkl",  "rb") as f: seq_sc_r5  = pickle.load(f)
        with open("seq_scaler_r10.pkl", "rb") as f: seq_sc_r10 = pickle.load(f)
        with open("rs_scaler_r5.pkl",   "rb") as f: rs_sc_r5   = pickle.load(f)
        with open("rs_scaler_r10.pkl",  "rb") as f: rs_sc_r10  = pickle.load(f)
        print("모델 로드 완료")
    except Exception as e:
        send(f"모델 로드 실패: {e}")
        exit(1)

    # 티커 로드
    try:
        tdf = pd.read_csv("tickers_us.csv", encoding="utf-8-sig")
        ticker_list = [str(r["ticker"]).strip() for _, r in tdf.iterrows()
                      if str(r["ticker"]).strip()]
        ticker_info = {str(r["ticker"]).strip(): {
            "name": str(r.get("name", "")),
            "cap":  str(r.get("cap", "")),
            "sector": str(r.get("sector", ""))
        } for _, r in tdf.iterrows()}
    except Exception as e:
        send(f"tickers_us.csv 로드 실패: {e}")
        exit(1)

    # SPY 수집
    start = (today - timedelta(days=600)).strftime("%Y-%m-%d")
    end   = today_str
    spy_df = get_ohlcv("SPY", start, end)
    if spy_df is None:
        send("SPY 수집 실패!")
        exit(1)
    print(f"SPY: {len(spy_df)}일치")

    send(f"🧠 LSTM 스캐너 시작\n{len(ticker_list)}개 종목 점수 계산 중...")

    # 전체 종목 점수 계산
    scores = []
    for i, ticker in enumerate(ticker_list):
        if i % 500 == 0:
            print(f"[{i}/{len(ticker_list)}] 처리 중...")

        df = get_ohlcv(ticker, start, end)
        if df is None:
            time.sleep(0.05)
            continue

        s_r5  = predict_score(df, spy_df, model_r5,  seq_sc_r5,  rs_sc_r5,  device)
        s_r10 = predict_score(df, spy_df, model_r10, seq_sc_r10, rs_sc_r10, device)

        if s_r5 is not None and s_r10 is not None:
            info = ticker_info.get(ticker, {})
            scores.append({
                "ticker":   ticker,
                "name":     info.get("name", ""),
                "cap":      info.get("cap", ""),
                "sector":   info.get("sector", ""),
                "lstm_r5":  s_r5,
                "lstm_r10": s_r10,
                "lstm_avg": round((s_r5 + s_r10) / 2, 4),
            })
        time.sleep(0.05)

    print(f"점수 계산 완료: {len(scores)}건")

    if not scores:
        send("점수 계산 결과 없음")
        exit(0)

    # Top10 추출 (평균 점수 기준)
    scores_df = pd.DataFrame(scores).sort_values("lstm_avg", ascending=False)
    top10 = scores_df.head(TOP_N).copy()
    top10["date"]       = today_str
    top10["minervini"]  = ""  # scanner_us_raw.csv와 나중에 조인 가능
    top10["r5"]         = ""
    top10["r10"]        = ""
    top10["r20"]        = ""

    # 미너비니 시그널 여부 확인
    try:
        scanner_raw = pd.read_csv("scanner_us_raw.csv", encoding="utf-8-sig")
        minervini_tickers = set(scanner_raw["ticker"].astype(str))
        top10["minervini"] = top10["ticker"].apply(
            lambda t: "✅" if t in minervini_tickers else "❌"
        )
    except:
        top10["minervini"] = "-"

    # 기존 누적 데이터 로드
    CSV_FILE = "scanner_lstm_top10.csv"
    try:
        df_hist = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
        df_hist["date"] = pd.to_datetime(df_hist["date"]).dt.strftime("%Y-%m-%d")
    except:
        df_hist = pd.DataFrame()

    # 수익률 업데이트
    if len(df_hist) > 0:
        print("과거 수익률 업데이트 중...")
        df_hist = update_returns(df_hist, spy_df, today)

    # 오늘 데이터 추가
    cols = ["date", "ticker", "name", "cap", "sector",
            "lstm_r5", "lstm_r10", "lstm_avg", "minervini", "r5", "r10", "r20"]
    top10 = top10[cols]

    if len(df_hist) > 0:
        df_out = pd.concat([df_hist, top10], ignore_index=True)
    else:
        df_out = top10

    # 10일치만 보관
    df_out["date"] = pd.to_datetime(df_out["date"]).dt.strftime("%Y-%m-%d")
    cutoff = (today - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
    df_out = df_out[df_out["date"] >= cutoff].reset_index(drop=True)

    df_out.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {CSV_FILE} ({len(df_out)}건)")

    # 텔레그램 전송
    lines = [f"🧠 LSTM Top{TOP_N} ({today_str})", "─" * 28]
    for _, row in top10.iterrows():
        lines.append(
            f"{row['ticker']} {row['minervini']}\n"
            f"  r5:{row['lstm_r5']:.3f} r10:{row['lstm_r10']:.3f} avg:{row['lstm_avg']:.3f}\n"
            f"  {row['cap']} | {row['sector']}"
        )
    send("\n".join(lines))
