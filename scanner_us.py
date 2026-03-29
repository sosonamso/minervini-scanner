"""
미너비니 컵&핸들 스캐너 - 미국 주식
- 최근 SCAN_DAYS 거래일 내 시그널 탐색
- LightGBM 점수 추가 (모델 있을 때만)
- 결과: 텔레그램 전송 + CSV 저장
"""
import time, warnings, pickle, os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from detect_us import (
    MASSIVE,
    send, send_file,
    get_massive_ohlcv, load_tickers,
    check_trend, detect, calc_rs, calc_score,
)

warnings.filterwarnings("ignore")

SCAN_DAYS    = 7
HISTORY_DAYS = 1800
WINDOW       = 150
SKIP         = 2
VOL_MA       = 20
MIN_PRICE    = 5.0
MIN_VOL      = 100000


def get_recent_dates(n=7):
    dates = []
    d = datetime.today()
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
        d -= timedelta(days=1)
    return dates[:n]


def check_market(mkt_df):
    if mkt_df is None or len(mkt_df) < 200:
        return True, "데이터부족"
    c    = mkt_df["Close"]
    ma   = c.rolling(200).mean()
    cur  = float(c.iloc[-1])
    ma_v = float(ma.iloc[-1])
    if pd.isna(ma_v):
        return True, "데이터부족"
    if cur > ma_v:
        return True, "상승장(S&P500>200MA)"
    return False, "하락장(S&P500<200MA)"


def score_grade(s):
    return "S" if s >= 90 else "A" if s >= 80 else "B" if s >= 70 else "C" if s >= 60 else "D"


def cap_label(cap):
    return {"MegaCap": "초대형", "LargeCap": "대형",
            "MidCap": "중형", "SmallCap": "소형"}.get(cap, cap)


# ── LightGBM 피처 계산 ──────────────────────────────
def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
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


def calc_lgbm_features(df, spy_df, d_idx, ticker_info):
    start_idx = d_idx - SKIP - WINDOW
    end_idx   = d_idx - SKIP
    if start_idx < max(VOL_MA, 252): return None

    w = df.iloc[start_idx:end_idx]
    if len(w) < WINDOW: return None

    close  = w["Close"].values
    high   = w["High"].values if "High" in w.columns else close
    low    = w["Low"].values  if "Low"  in w.columns else close
    volume = w["Volume"].values

    if close[-1] < MIN_PRICE: return None
    if np.mean(volume[-20:]) < MIN_VOL: return None

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
    sa = spy_df.reindex(td).ffill()
    tc = df["Close"].iloc[:ref_idx + 1].values
    sc = sa["Close"].values if not sa.isnull().all().any() else None

    if sc is not None and len(tc) >= 253:
        def pr(arr, n): return float(arr[-1] / arr[-n] - 1) if len(arr) >= n else 0.0
        w_ = [0.4, 0.2, 0.2, 0.2]; p_ = [63, 126, 189, 252]
        t_rs = sum(w_[i]*pr(tc,p_[i]) for i in range(4))
        s_rs = sum(w_[i]*pr(sc,p_[i]) for i in range(4))
        rs_at  = round((t_rs - s_rs) * 100, 4)
        rs_20  = round((pr(tc,20) - pr(sc,20)) * 100, 4)
        rs_50  = round((pr(tc,50) - pr(sc,50)) * 100, 4)
        rs_150 = round((pr(tc,150)- pr(sc,150))* 100, 4)
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

    feat["ma20_pos"] = round(float(close[-1] / np.mean(close[-20:])) if len(close) >= 20 else 1.0, 4)
    feat["ma50_pos"] = round(float(close[-1] / np.mean(close[-50:])) if len(close) >= 50 else 1.0, 4)

    # 섹터 OHE
    SECTORS = ["Technology","Healthcare","Financial","Consumer","Energy",
               "Industrial","Materials","Utilities","Real Estate","Communication","Other"]
    sector = str(ticker_info.get("sector", "Other"))
    for s in SECTORS:
        feat[f"sec_{s}"] = 1 if sector == s else 0

    # 시총 OHE
    CAPS = ["MegaCap","LargeCap","MidCap","SmallCap"]
    cap = str(ticker_info.get("cap", "SmallCap"))
    for c in CAPS:
        feat[f"cap_{c}"] = 1 if cap == c else 0

    return feat


def predict_lgbm(df, spy_df, sig_ts, ticker_info, lgbm_model, feat_cols):
    """LightGBM 점수 계산"""
    try:
        import lightgbm as lgb
        idx_list = df.index.tolist()
        matches  = [x for x in idx_list if x.date() == sig_ts.date()]
        if not matches: return None
        d_idx = idx_list.index(matches[0])
        feat  = calc_lgbm_features(df, spy_df, d_idx, ticker_info)
        if feat is None: return None
        X = np.array([[feat.get(c, 0) for c in feat_cols]], dtype=np.float32)
        return round(float(lgbm_model.predict(X)[0]), 4)
    except:
        return None


if __name__ == "__main__":
    if not MASSIVE:
        send("MASSIVE_TOKEN이 없어요! GitHub Secrets 확인해주세요.")
        exit(1)

    end      = datetime.today()
    start    = (end - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end_str  = end.strftime("%Y-%m-%d")

    sig_dates   = get_recent_dates(SCAN_DAYS)
    data_cutoff = pd.Timestamp(sig_dates[0]) - timedelta(days=7)
    print(f"탐색날짜: {sig_dates}")

    # LightGBM 모델 로드 (있을 때만)
    lgbm_model = None
    feat_cols  = None
    try:
        import lightgbm as lgb
        lgbm_model = lgb.Booster(model_file="model_lgbm.txt")
        with open("feat_cols_lgbm.pkl", "rb") as f:
            feat_cols = pickle.load(f)
        print(f"LightGBM 모델 로드 완료 (피처:{len(feat_cols)}개)")
    except Exception as e:
        print(f"LightGBM 모델 없음 (점수 미계산): {e}")

    # SPY
    mkt_df = get_massive_ohlcv("SPY", start, end_str)
    market_ok, market_str = check_market(mkt_df)

    # 티커
    all_tickers = load_tickers()
    if not all_tickers:
        send("tickers_us.csv 없음!")
        exit(1)

    send(
        f"🇺🇸 미국 스캐너 시작 (Massive)\n"
        f"최근 {SCAN_DAYS}거래일 | {market_str}\n"
        f"{len(all_tickers)}개 종목 수집 중..."
    )
    if not market_ok:
        send("⚠️ S&P500 200MA 하방 - 시그널 신뢰도 낮음!")

    # 데이터 수집
    valid_data = {}
    ticker_list = list(all_tickers.keys())

    for i, ticker in enumerate(ticker_list):
        if i % 200 == 0:
            print(f"[{i}/{len(ticker_list)}] 수신:{len(valid_data)}")
        df = get_massive_ohlcv(ticker, start, end_str)
        if df is None: continue
        if df.index[-1] < data_cutoff: continue
        valid_data[ticker] = df
        time.sleep(0.05)

    send(f"다운로드 완료\n수신: {len(valid_data)}/{len(ticker_list)}개\n패턴 분석 시작...")

    # 패턴 분석
    signals    = []
    all_scores = []
    trend_pass = 0

    for ticker in ticker_list:
        df = valid_data.get(ticker)
        if df is None: continue

        info    = all_tickers[ticker]
        matched = False

        for sig_str in sig_dates:
            sig_ts = pd.Timestamp(sig_str)
            if sig_ts not in df.index: continue

            pos = df.index.tolist().index(sig_ts)
            sl  = df.iloc[:pos + 1]
            cur = float(sl["Close"].iloc[-1])
            rs  = calc_rs(sl, mkt_df.loc[:sig_ts]) if mkt_df is not None else 0.0

            if not check_trend(sl):
                all_scores.append({
                    "ticker": ticker, "name": info["name"],
                    "cap": info["cap"], "sector": info["sector"],
                    "exchange": info["exchange"],
                    "cur": round(cur, 2), "rs": rs,
                    "trend_ok": False, "pattern_ok": False, "signal": False,
                    "score": 0, "grade": "D", "pivot": 0,
                    "cup_depth": 0, "handle_depth": 0, "vol_ratio": 0,
                    "cup_days": 0, "handle_days": 0,
                    "reason": "트렌드 미통과", "pct_from_pivot": 0, "safety": "",
                    "lgbm_score": None,
                })
                break

            trend_pass += 1
            ok, pat = detect(sl)

            if not ok:
                score = calc_score(rs, 1.0, 0, 0)
                all_scores.append({
                    "ticker": ticker, "name": info["name"],
                    "cap": info["cap"], "sector": info["sector"],
                    "exchange": info["exchange"],
                    "cur": round(cur, 2), "rs": rs,
                    "trend_ok": True, "pattern_ok": False, "signal": False,
                    "score": score, "grade": score_grade(score), "pivot": 0,
                    "cup_depth": 0, "handle_depth": 0, "vol_ratio": 0,
                    "cup_days": 0, "handle_days": 0,
                    "reason": "패턴 미감지", "pct_from_pivot": 0, "safety": "",
                    "lgbm_score": None,
                })
                break

            score  = calc_score(rs, pat["vr"], pat["cd"], pat["hd"])
            grade  = score_grade(score)
            pivot  = pat["pivot"]
            pct    = round((cur - pivot) / pivot * 100, 1) if pivot > 0 else 0
            safety = ("safe" if cur >= pivot * 0.93
                      else "caution" if cur >= pivot * 0.90
                      else "danger")
            signal = pat["vs"] and rs > 30 and pat["cd"] < 30

            # LightGBM 점수
            lgbm_score = None
            if lgbm_model and feat_cols and signal:
                lgbm_score = predict_lgbm(
                    df, mkt_df, sig_ts,
                    {"sector": info["sector"], "cap": info["cap"]},
                    lgbm_model, feat_cols
                )

            all_scores.append({
                "ticker": ticker, "name": info["name"],
                "cap": info["cap"], "sector": info["sector"],
                "exchange": info["exchange"],
                "cur": pat["cur"], "rs": rs,
                "trend_ok": True, "pattern_ok": True, "signal": signal,
                "score": score, "grade": grade, "pivot": pivot,
                "cup_depth": pat["cd"], "handle_depth": pat["hd"],
                "vol_ratio": pat["vr"], "cup_days": pat["cdays"],
                "handle_days": pat["hdays"],
                "reason": ("시그널" if signal
                           else "거래량 미충족" if not pat["vs"]
                           else "RS 미충족"),
                "pct_from_pivot": pct, "safety": safety,
                "lgbm_score": lgbm_score,
            })

            if not signal:
                break

            signals.append({
                "sig_date":   sig_str,
                "ticker":     ticker,
                "name":       info["name"],
                "cap":        info["cap"],
                "sector":     info["sector"],
                "exchange":   info["exchange"],
                "cur":        pat["cur"],
                "pivot":      pivot,
                "cd":         pat["cd"],
                "hd":         pat["hd"],
                "cdays":      pat["cdays"],
                "hdays":      pat["hdays"],
                "cup_start":  pat.get("cup_start", ""),
                "cup_end":    pat.get("cup_end", ""),
                "vr":         pat["vr"],
                "rs":         rs,
                "score":      score,
                "grade":      grade,
                "lgbm_score": lgbm_score,
            })
            matched = True
            break

    # 중복 제거
    seen   = set()
    deduped = []
    for r in sorted(signals, key=lambda x: (x["sig_date"], x.get("lgbm_score") or x["score"]), reverse=True):
        if r["ticker"] not in seen:
            seen.add(r["ticker"])
            deduped.append(r)
    signals = deduped

    print(f"완료: {len(signals)}개 발견 / 트렌드 통과: {trend_pass}개")
    send(f"스캔 완료\n트렌드 통과: {trend_pass}개\n패턴+거래량+RS: {len(signals)}개")

    # CSV 저장
    rows = []
    for r in signals:
        rows.append({
            "date":         r["sig_date"],
            "ticker":       r["ticker"],
            "name":         r["name"],
            "cap":          r["cap"],
            "sector":       r["sector"],
            "exchange":     r["exchange"],
            "entry":        r["cur"],
            "pivot":        r["pivot"],
            "cup_depth":    r["cd"],
            "handle_depth": r["hd"],
            "cup_days":     r["cdays"],
            "handle_days":  r["hdays"],
            "cup_start":    r.get("cup_start", ""),
            "cup_end":      r.get("cup_end", ""),
            "vol_ratio":    r["vr"],
            "rs":           r["rs"],
            "score":        r["score"],
            "grade":        r["grade"],
            "lgbm_score":   r.get("lgbm_score"),
        })

    pd.DataFrame(rows if rows else []).to_csv(
        "scanner_us_raw.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_scores if all_scores else []).to_csv(
        "scanner_us_all.csv", index=False, encoding="utf-8-sig"
    )

    if rows:
        send_file("scanner_us_raw.csv",
                  f"🇺🇸 미장 스캐너 RAW ({len(rows)}건) {datetime.today().strftime('%Y-%m-%d')}")

    # 텔레그램
    if not signals:
        send(f"🇺🇸 미국 스캐너\n최근 {SCAN_DAYS}거래일 | {market_str}\n조건 충족 종목 없음")
    else:
        hdr = (f"🇺🇸 미너비니 컵&핸들(미국)\n"
               f"최근 {SCAN_DAYS}거래일 | {market_str}\n"
               f"{len(signals)}개 발견\n" + "─" * 24 + "\n")
        msg = hdr
        grade_emoji = {"S": "🏆", "A": "🥇", "B": "🥈", "C": "🥉", "D": "📊"}

        for r in signals:
            up  = round((r["pivot"] / r["cur"] - 1) * 100, 1)
            cup_date = (f"({r.get('cup_start','')}~{r.get('cup_end','')})"
                        if r.get("cup_start") else "")
            lgbm_str = (f"  🌲LGBM: {r['lgbm_score']:.3f}\n"
                        if r.get("lgbm_score") is not None else "")
            blk = (
                f"[{r['sig_date']}] [{cap_label(r['cap'])}] {r['sector']}\n"
                f"◆ {r['ticker']} {r['name']}\n"
                f"  AI점수: {grade_emoji.get(r['grade'],'📊')}{r['score']}점({r['grade']}등급)\n"
                f"{lgbm_str}"
                f"  현재가: ${r['cur']:,.2f}\n"
                f"  피벗: ${r['pivot']:,.2f} ({up:+.1f}%)\n"
                f"  컵:{r['cd']}%/{r['cdays']}일{cup_date} 핸들:{r['hd']}%/{r['hdays']}일\n"
                f"  거래량:{r['vr']}x🔥 RS:{r['rs']:+.1f}%\n\n"
            )
            if len(msg) + len(blk) > 4000:
                send(msg)
                msg = "(이어서)\n\n" + blk
            else:
                msg += blk
        send(msg)
