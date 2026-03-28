"""
미너비니 컵&핸들 스캐너 - 미국 주식
- 최근 SCAN_DAYS 거래일 내 시그널 탐색
- 결과: 텔레그램 전송 + CSV 저장
"""
import time, warnings
import pandas as pd
from datetime import datetime, timedelta
from detect_us import (
    MASSIVE,
    send, send_file,
    get_massive_ohlcv, load_tickers,
    check_trend, detect, calc_rs, calc_score,
)

warnings.filterwarnings("ignore")

SCAN_DAYS    = 7     # 최근 N 거래일 탐색
HISTORY_DAYS = 1800  # 데이터 수집 기간


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

    # SPY 시장 상태
    mkt_df = get_massive_ohlcv("SPY", start, end_str)
    market_ok, market_str = check_market(mkt_df)

    # 티커 로드
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
        if df is None:
            continue
        if df.index[-1] < data_cutoff:
            continue
        valid_data[ticker] = df
        time.sleep(0.05)

    send(f"다운로드 완료\n수신: {len(valid_data)}/{len(ticker_list)}개\n패턴 분석 시작...")

    # 패턴 분석
    signals    = []
    all_scores = []
    trend_pass = 0

    for ticker in ticker_list:
        df = valid_data.get(ticker)
        if df is None:
            continue

        info = all_tickers[ticker]
        matched = False

        for sig_str in sig_dates:
            sig_ts = pd.Timestamp(sig_str)
            if sig_ts not in df.index:
                continue

            pos = df.index.tolist().index(sig_ts)
            sl  = df.iloc[:pos + 1]
            cur = float(sl["Close"].iloc[-1])
            rs  = calc_rs(sl, mkt_df.loc[:sig_ts]) if mkt_df is not None else 0.0

            # 트렌드 체크
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
            })

            if not signal:
                break

            signals.append({
                "sig_date":  sig_str,
                "ticker":    ticker,
                "name":      info["name"],
                "cap":       info["cap"],
                "sector":    info["sector"],
                "exchange":  info["exchange"],
                "cur":       pat["cur"],
                "pivot":     pivot,
                "cd":        pat["cd"],
                "hd":        pat["hd"],
                "cdays":     pat["cdays"],
                "hdays":     pat["hdays"],
                "cup_start": pat.get("cup_start", ""),
                "cup_end":   pat.get("cup_end", ""),
                "vr":        pat["vr"],
                "rs":        rs,
                "score":     score,
                "grade":     grade,
            })
            matched = True
            break

    # 중복 제거 (종목당 최신 1개)
    seen = set()
    deduped = []
    for r in sorted(signals, key=lambda x: (x["sig_date"], x["score"]), reverse=True):
        if r["ticker"] not in seen:
            seen.add(r["ticker"])
            deduped.append(r)
    signals = deduped

    print(f"완료: {len(signals)}개 발견 / 트렌드 통과: {trend_pass}개")
    send(
        f"스캔 완료\n"
        f"트렌드 통과: {trend_pass}개\n"
        f"패턴+거래량+RS: {len(signals)}개"
    )

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

    # 텔레그램 메시지
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
            blk = (
                f"[{r['sig_date']}] [{cap_label(r['cap'])}] {r['sector']}\n"
                f"◆ {r['ticker']} {r['name']}\n"
                f"  AI점수: {grade_emoji.get(r['grade'],'📊')}{r['score']}점({r['grade']}등급)\n"
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
