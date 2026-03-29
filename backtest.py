"""
미너비니 컵&핸들 백테스트 - KRX API 버전
- 조건 완화: 트렌드 + 컵핸들만 (거래량/RS 제거) → LGBM 학습 데이터용
- r5/r10 추가 저장
"""
import os, time, warnings, requests, json, statistics
import numpy as np, pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

TOK          = os.environ.get("TELEGRAM_TOKEN", "")
CID          = os.environ.get("TELEGRAM_CHAT_ID", "")
KRX          = os.environ.get("KRX_TOKEN", "")

LOOKBACK_DAYS = 1500
HISTORY_DAYS  = 2100
MAX_HOLD      = 90

_row_meta = {}


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
                        "Volume": float(str(row.get("ACC_TRDVOL", "0")).replace(",", "")),
                        "TrdVal": float(str(row.get("ACC_TRDVAL", "0")).replace(",", "")),
                    }
                except: pass
            return result
        except Exception as e:
            print(f"KRX 오류(시도{attempt+1}): {e}")
            time.sleep(2 * (attempt + 1))
    return {}


def get_trading_dates(days):
    dates = []
    d = datetime.today()
    while len(dates) < days:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))
        d -= timedelta(days=1)
        if (datetime.today() - d).days > days * 2: break
    return list(reversed(dates))


def build_ohlcv(trading_dates):
    ticker_data = {}
    total = len(trading_dates)
    for i, date_str in enumerate(trading_dates):
        if i % 100 == 0:
            print(f"데이터 수집 [{i}/{total}] {date_str}")
        for mkt in ["KOSPI", "KOSDAQ"]:
            day_data = get_krx_data(date_str, mkt)
            for ticker, ohlcv in day_data.items():
                if ticker not in ticker_data:
                    ticker_data[ticker] = {"market": mkt, "rows": []}
                ticker_data[ticker]["rows"].append({
                    "date": pd.Timestamp(date_str), **ohlcv
                })
        time.sleep(0.3)

    result = {}
    for ticker, info in ticker_data.items():
        rows = info["rows"]
        if len(rows) < 200: continue
        df = pd.DataFrame(rows).set_index("date").sort_index()
        df = df[["Close", "Volume", "TrdVal"]].astype(float)
        df = df[df["Close"] > 0].dropna()
        if len(df) >= 200:
            meta = _row_meta.get(ticker, {})
            result[ticker] = {
                "market": info["market"], "df": df,
                "name":   meta.get("name", ticker),
                "sector": meta.get("sector", "기타")
            }
    return result


def build_index(all_ohlcv):
    kospi_df = None; kosdaq_df = None
    for ticker, info in all_ohlcv.items():
        name = info.get("name", "")
        if kospi_df is None and "KODEX" in name and "200" in name \
                and "레버리지" not in name and "인버스" not in name:
            kospi_df = info["df"][["Close"]].copy()
            print(f"코스피 ETF: {name}({ticker}) {len(kospi_df)}일치")
        if kosdaq_df is None and "KODEX" in name and "코스닥" in name \
                and "레버리지" not in name and "인버스" not in name:
            kosdaq_df = info["df"][["Close"]].copy()
            print(f"코스닥 ETF: {name}({ticker}) {len(kosdaq_df)}일치")
        if kospi_df is not None and kosdaq_df is not None: break

    if kospi_df is None:
        tickers = [t for t, v in all_ohlcv.items() if v["market"] == "KOSPI"][:50]
        closes  = pd.concat([all_ohlcv[t]["df"]["Close"].rename(t) for t in tickers], axis=1)
        kospi_df = closes.mean(axis=1).to_frame("Close")
    if kosdaq_df is None:
        tickers = [t for t, v in all_ohlcv.items() if v["market"] == "KOSDAQ"][:50]
        closes  = pd.concat([all_ohlcv[t]["df"]["Close"].rename(t) for t in tickers], axis=1)
        kosdaq_df = closes.mean(axis=1).to_frame("Close")

    return kospi_df, kosdaq_df


def check_trend(df):
    if len(df) < 200: return False
    c = df["Close"]
    m50 = c.rolling(50).mean(); m150 = c.rolling(150).mean(); m200 = c.rolling(200).mean()
    cur = float(c.iloc[-1]); a = float(m50.iloc[-1]); b = float(m150.iloc[-1])
    m20v = m200.dropna()
    if len(m20v) < 1: return False
    d = float(m20v.iloc[-1]); d1 = float(m20v.iloc[-21]) if len(m20v) >= 21 else d
    if any(pd.isna([a, b, d])): return False
    lk = c.iloc[-252:] if len(c) >= 252 else c
    return all([cur > b and cur > d, b > d, d > d1, a > b and a > d, cur > a,
                cur >= lk.min() * 1.25, cur >= lk.max() * 0.70])


def detect(df):
    cl = df["Close"].values.astype(float)
    vl = df["Volume"].values.astype(float)
    n  = len(cl); idx = df.index
    if n < 60: return False, {}
    c = cl[-min(200, n):]; v = vl[-min(200, n):]; w = len(c)
    li = int(np.argmax(c[:w//2])); lh = c[li]
    cup = c[li:]
    if len(cup) < 20: return False, {}
    bi = li + int(np.argmin(cup)); bot = c[bi]; cd = (lh - bot) / lh
    if not (0.15 <= cd <= 0.50) or (bi - li) < 35: return False, {}
    rc = c[bi:]
    if len(rc) < 10: return False, {}
    ri = bi + int(np.argmax(rc)); rh = c[ri]
    if rh < lh * 0.90: return False, {}
    if rh > lh * 1.15: return False, {}
    hnd = c[ri:]; hl = len(hnd)
    if not (5 <= hl <= 20): return False, {}
    hlow = float(np.min(hnd)); hd = (rh - hlow) / rh
    if not (0.05 <= hd <= 0.15): return False, {}
    if (hlow - bot) / (lh - bot) < 0.60: return False, {}
    cur = cl[-1]
    if not (rh * 0.97 <= cur <= rh * 1.05): return False, {}
    vr = float(np.mean(v[-5:])) / float(np.mean(v[-40:-5])) if len(v) >= 40 else 1.0
    try:
        start_pos  = n - len(c)
        cup_start  = idx[start_pos + li].strftime("%y.%m.%d")
        cup_end    = idx[start_pos + ri].strftime("%y.%m.%d")
    except:
        cup_start = ""; cup_end = ""
    return True, {
        "cd": round(cd*100, 1), "hd": round(hd*100, 1),
        "cdays": ri - li, "hdays": hl,
        "pivot": round(float(rh), 0), "cur": round(float(cur), 0),
        "vr": round(vr, 2), "vs": vr >= 1.40,
        "cup_start": cup_start, "cup_end": cup_end
    }


def calc_rs(df, mkt_df):
    def p(d, n): return float(d["Close"].iloc[-1] / d["Close"].iloc[-n] - 1) if len(d) >= n else 0.0
    s = sum([0.4,0.2,0.2,0.2][i] * p(df,  [63,126,189,252][i]) for i in range(4))
    m = sum([0.4,0.2,0.2,0.2][i] * p(mkt_df, [63,126,189,252][i]) for i in range(4))
    return round((s - m) * 100, 1)


def calc_score(rs, vr, cd, hd):
    s_rs = 100 if rs>=25 else 80 if rs>=15 else 60 if rs>=10 else 40 if rs>=5 else 20
    s_vr = 100 if vr>=3.0 else 85 if vr>=2.5 else 70 if vr>=2.0 else 55 if vr>=1.7 else 40
    s_cd = 100 if 20<=cd<=35 else 75 if (15<=cd<20 or 35<cd<=40) else 50 if 40<cd<=50 else 30
    s_hd = 100 if 5<=hd<=10 else 75 if 10<hd<=12 else 50 if hd>12 else 60
    return round(s_rs*0.40 + s_vr*0.35 + s_cd*0.15 + s_hd*0.10)


if __name__ == "__main__":
    if not KRX:
        send("KRX_TOKEN 없음!")
        exit(1)

    end          = datetime.today()
    signal_start = end - timedelta(days=LOOKBACK_DAYS)
    signal_dates = set(pd.bdate_range(signal_start, end).map(pd.Timestamp))

    send(
        f"백테스트 시작 (KRX API)\n"
        f"기간: 최근 {LOOKBACK_DAYS}일\n"
        f"데이터 {HISTORY_DAYS}일치 수집 중...\n"
        f"조건: 트렌드 + 컵핸들 (거래량/RS 제거)\n"
        f"(약 40~50분 소요)"
    )

    trading_dates = get_trading_dates(HISTORY_DAYS)
    print(f"수집 대상: {len(trading_dates)}거래일")
    all_ohlcv = build_ohlcv(trading_dates)
    print(f"종목 구축 완료: {len(all_ohlcv)}개")

    send("지수 데이터 수집 중...")
    kospi_df, kosdaq_df = build_index(all_ohlcv)
    send(f"데이터 수집 완료: {len(all_ohlcv)}개 종목\n패턴 분석 시작...")

    all_signals = []

    for i, (ticker, info) in enumerate(all_ohlcv.items()):
        if i % 200 == 0:
            print(f"[{i}/{len(all_ohlcv)}] 시그널:{len(all_signals)}건")
        df  = info["df"]
        mkt = info["market"]
        idx = df.index.tolist()
        idx_df = kospi_df if mkt == "KOSPI" else kosdaq_df

        for j, sig_ts in enumerate(idx):
            if sig_ts not in signal_dates: continue
            sl = df.iloc[:j + 1]

            # ── 조건 완화: 트렌드 + 컵핸들만 ──────────
            if not check_trend(sl): continue
            ok, pat = detect(sl)
            if not ok: continue
            # ─────────────────────────────────────────

            # RS 계산 (조건 아님, 피처로만 사용)
            rs = 0.0
            if idx_df is not None and len(idx_df) > 10:
                try: rs = calc_rs(sl, idx_df.loc[:sig_ts])
                except: rs = 0.0

            entry = float(df["Close"].iloc[j])
            score = calc_score(rs, pat["vr"], pat["cd"], pat["hd"])

            # 거래대금 (20일 평균)
            trdval_20 = 0.0
            if "TrdVal" in sl.columns:
                trdval_20 = round(float(sl["TrdVal"].tail(20).mean()) / 1e8, 1)

            # r5 ~ r90 수익률
            daily_r = {}
            for hold in range(1, MAX_HOLD + 1):
                fi = j + hold
                if fi < len(idx):
                    daily_r[hold] = round((float(df["Close"].iloc[fi]) / entry - 1) * 100, 2)
                else:
                    daily_r[hold] = None

            all_signals.append({
                "date":         sig_ts.strftime("%Y-%m-%d"),
                "ticker":       ticker,
                "name":         info["name"],
                "market":       mkt,
                "sector":       info["sector"],
                "entry":        entry,
                "pivot":        pat["pivot"],
                "cup_depth":    pat["cd"],
                "handle_depth": pat["hd"],
                "cup_days":     pat["cdays"],
                "handle_days":  pat["hdays"],
                "cup_start":    pat.get("cup_start", ""),
                "cup_end":      pat.get("cup_end", ""),
                "vol_ratio":    pat["vr"],
                "vs":           pat["vs"],
                "rs":           rs,
                "score":        score,
                "trdval_20":    trdval_20,
                "daily_returns": daily_r,
            })

    print(f"백테스트 완료: {len(all_signals)}건")

    # RAW 저장
    rows = []
    for s in all_signals:
        row = {k: v for k, v in s.items() if k != "daily_returns"}
        for hold in [1, 3, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90]:
            row[f"r{hold}"] = s["daily_returns"].get(hold)
        rows.append(row)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv("backtest_raw.csv", index=False, encoding="utf-8-sig")
    print("RAW 저장 완료")

    send_file("backtest_raw.csv",
              f"📊 국장 백테스트 RAW ({len(all_signals)}건)")

    if not all_signals:
        send("시그널 없음")
    else:
        df = pd.DataFrame(rows)
        lines = [f"백테스트 결과 (KRX API)",
                 f"총 시그널: {len(df)}건 (조건 완화)",
                 f"거래량 충족(vs): {(raw_df['vs']==True).sum()}건",
                 "─" * 28]
        for col, label in [("r5","5일"), ("r10","10일"), ("r20","20일"), ("r60","60일")]:
            vals = df[col].dropna()
            if len(vals) == 0: continue
            win = sum(1 for v in vals if v > 0)
            lines.append(f"[{label}] n={len(vals)}")
            lines.append(f"  평균:{vals.mean():+.1f}% 중앙:{statistics.median(vals):+.1f}%")
            lines.append(f"  승률:{round(win/len(vals)*100, 1)}%")
        send("\n".join(lines))
