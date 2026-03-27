"""
미너비니 컵&핸들 백테스트 - 미국 주식
- 종목당 최근 컵핸들 1개 추출
- 출력: backtest_us_raw.csv (label_cup / label_win 컬럼 포함)
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

HISTORY_DAYS = 1800   # 데이터 수집 기간 (탐색 1250 + 수익률 계산 여유)
MAX_HOLD     = 90     # 최대 보유일 수익률 추적


if __name__ == "__main__":
    if not MASSIVE:
        send("MASSIVE_TOKEN 없음!")
        exit(1)

    end      = datetime.today()
    start    = (end - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end_str  = end.strftime("%Y-%m-%d")

    # 티커 로드
    all_tickers = load_tickers()
    if not all_tickers:
        send("tickers_us.csv 없음!")
        exit(1)

    send(
        f"🇺🇸 미국 백테스트 시작\n"
        f"종목당 최근 컵 1개 | {len(all_tickers)}개 종목\n"
        f"(약 1~2시간 소요)"
    )

    # SPY 기준지수
    spy_df = get_massive_ohlcv("SPY", start, end_str)
    print(f"SPY: {len(spy_df) if spy_df is not None else 0}일치")

    all_signals = []
    ticker_list = list(all_tickers.keys())

    for i, ticker in enumerate(ticker_list):
        if i % 100 == 0:
            print(f"[{i}/{len(ticker_list)}] 시그널:{len(all_signals)}건")

        df = get_massive_ohlcv(ticker, start, end_str)
        if df is None:
            continue

        idx = df.index.tolist()
        total = len(idx)

        # 역순 탐색: 최근 시그널부터
        # - 최소 200일 필요 (MA 계산)
        # - 최소 60일 미래 필요 (r60 계산)
        scan_end   = total - 61   # r60 확보
        scan_start = 200          # MA 계산 최소

        found = False
        for j in range(scan_end, scan_start, -1):
            sl = df.iloc[:j + 1]

            if not check_trend(sl):
                continue

            ok, pat = detect(sl)
            if not ok:
                continue

            # 시그널 확정
            sig_ts = idx[j]
            entry  = float(df["Close"].iloc[j])
            rs     = calc_rs(sl, spy_df.loc[:sig_ts]) if spy_df is not None else 0.0
            score  = calc_score(rs, pat["vr"], pat["cd"], pat["hd"])

            # 보유일별 수익률
            daily_r = {}
            for hold in range(1, MAX_HOLD + 1):
                fi = j + hold
                if fi < total:
                    daily_r[hold] = round(
                        (float(df["Close"].iloc[fi]) / entry - 1) * 100, 2
                    )
                else:
                    daily_r[hold] = None

            # SPY 알파 (5/20/60일)
            alpha = {}
            try:
                if spy_df is not None and sig_ts in spy_df.index:
                    spy_idx   = spy_df.index.tolist()
                    spy_j     = spy_idx.index(sig_ts)
                    spy_entry = float(spy_df["Close"].iloc[spy_j])
                    for hold in [5, 20, 60]:
                        spy_fi = spy_j + hold
                        if spy_fi < len(spy_idx):
                            spy_r = (float(spy_df["Close"].iloc[spy_fi]) / spy_entry - 1) * 100
                            sr = daily_r.get(hold)
                            if sr is not None:
                                alpha[hold] = round(sr - spy_r, 2)
            except:
                pass

            # label_win: r20 기준 자동 계산 (양수=1, 음수=0)
            r20       = daily_r.get(20)
            label_win = (1 if r20 > 0 else 0) if r20 is not None else ""

            all_signals.append({
                "date":         sig_ts.strftime("%Y-%m-%d"),
                "ticker":       ticker,
                "cap":          all_tickers[ticker]["cap"],
                "sector":       all_tickers[ticker]["sector"],
                "entry":        entry,
                "pivot":        pat["pivot"],
                "cup_depth":    pat["cd"],
                "handle_depth": pat["hd"],
                "cup_days":     pat["cdays"],
                "handle_days":  pat["hdays"],
                "cup_start":    pat.get("cup_start", ""),
                "cup_end":      pat.get("cup_end", ""),
                "vol_ratio":    pat["vr"],
                "rs":           rs,
                "score":        score,
                "r5":           daily_r.get(5),
                "r20":          r20,
                "r60":          daily_r.get(60),
                "alpha5":       alpha.get(5),
                "alpha20":      alpha.get(20),
                "alpha60":      alpha.get(60),
                "label_cup":    "",        # 수동 라벨링 (1=진짜컵, 0=노이즈)
                "label_win":    label_win, # r20 기준 자동 (1=수익, 0=손실)
            })
            found = True
            break  # 종목당 최근 1개

        time.sleep(0.05)

    print(f"백테스트 완료: {len(all_signals)}건")

    if not all_signals:
        send("시그널 없음")
    else:
        raw_df = pd.DataFrame(all_signals)
        raw_df.to_csv("backtest_us_raw.csv", index=False, encoding="utf-8-sig")
        send_file(
            "backtest_us_raw.csv",
            f"🇺🇸 백테스트 RAW ({len(all_signals)}건) {datetime.today().strftime('%Y-%m-%d')}"
        )
        send(f"🇺🇸 백테스트 완료\n총 {len(all_signals)}건\nCSV 저장 완료")
