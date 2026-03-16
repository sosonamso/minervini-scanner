"""
미너비니 컵&핸들 백테스트
- 대상: KOSPI200
- 기간: 최근 1년
- 출력: 날짜 / 섹터 / 종목 / 5일 / 20일 / 60일 수익률 테이블
- 결과: 텔레그램 전송
"""

import os, time, warnings, requests
import numpy as np
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

LOOKBACK_DAYS = 365
HISTORY_DAYS  = 800
HOLD_PERIODS  = [5, 20, 60]

# ══════════════════════════════════════
# 데이터 수집
# ══════════════════════════════════════
def get_kospi200():
    try:
        today = datetime.today().strftime("%Y%m%d")
        tickers = stock.get_index_portfolio_deposit_file(today, "1028")
        return list(tickers)
    except:
        today = datetime.today().strftime("%Y%m%d")
        return stock.get_market_ticker_list(today, market="KOSPI")[:200]

def get_ohlcv_full(ticker, start_str, end_str):
    for _ in range(2):
        try:
            df = stock.get_market_ohlcv(start_str, end_str, ticker)
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                "시가":"Open","고가":"High",
                "저가":"Low","종가":"Close","거래량":"Volume"
            })
            df = df[["Open","High","Low","Close","Volume"]].dropna()
            if len(df) >= 100: return df
        except: time.sleep(0.3)
    return None

# 섹터 캐시
_sector_cache = {}
def get_sector(ticker):
    if ticker not in _sector_cache:
        try:
            today = datetime.today().strftime("%Y%m%d")
            df = stock.get_market_sector_classifications(today, "KOSPI")
            for t, row in df.iterrows():
                _sector_cache[str(t)] = row.get("업종명", "기타")
        except: pass
    return _sector_cache.get(ticker, "기타")

_name_cache = {}
def get_name(ticker):
    if ticker not in _name_cache:
        try: _name_cache[ticker] = stock.get_market_ticker_name(ticker)
        except: _name_cache[ticker] = ticker
    return _name_cache[ticker]


# ══════════════════════════════════════
# 미너비니 조건
# ══════════════════════════════════════
def check_trend(df):
    if len(df) < 200: return False
    c = df["Close"]
    ma50=c.rolling(50).mean(); ma150=c.rolling(150).mean(); ma200=c.rolling(200).mean()
    cur=c.iloc[-1]; m50=ma50.iloc[-1]; m150=ma150.iloc[-1]; m200=ma200.iloc[-1]
    if pd.isna(m50) or pd.isna(m200): return False
    m200_1m = ma200.iloc[-21] if len(ma200.dropna()) >= 21 else m200
    lk = c.iloc[-252:] if len(c) >= 252 else c
    return all([
        cur>m150 and cur>m200, m150>m200, m200>m200_1m,
        m50>m150 and m50>m200, cur>m50,
        cur>=lk.min()*1.25, cur>=lk.max()*0.70,
    ])

def detect_cup_handle(df):
    close=df["Close"].values; volume=df["Volume"].values; n=len(close)
    if n<60: return False,{}
    c=close[-min(200,n):]; v=volume[-min(200,n):]; w=len(c)
    lh_idx=int(np.argmax(c[:w//2])); lh=c[lh_idx]
    cup=c[lh_idx:]
    if len(cup)<20: return False,{}
    bi=lh_idx+int(np.argmin(cup)); bot=c[bi]; cd=(lh-bot)/lh
    if not(0.15<=cd<=0.50) or (bi-lh_idx)<35: return False,{}
    rec=c[bi:]
    if len(rec)<10: return False,{}
    ri=bi+int(np.argmax(rec)); rh=c[ri]
    if rh<lh*0.90: return False,{}
    hnd=c[ri:]; hl=len(hnd)
    if not(5<=hl<=20): return False,{}
    hlow=float(np.min(hnd)); hd=(rh-hlow)/rh
    if not(0.05<=hd<=0.15): return False,{}
    if (hlow-bot)/(lh-bot)<0.60: return False,{}
    cur=close[-1]
    if not(rh*0.97<=cur<=rh*1.05): return False,{}
    vr=float(np.mean(v[-5:]))/float(np.mean(v[-40:-5])) if len(v)>=40 else 1.0
    return True,{
        "pivot":round(float(rh),0),"cur":round(float(cur),0),
        "cd":round(cd*100,1),"hd":round(hd*100,1),
        "vr":round(vr,2),"vs":vr>=1.40
    }


# ══════════════════════════════════════
# 백테스트
# ══════════════════════════════════════
def backtest_ticker(ticker, full_df, signal_dates):
    signals = []
    dates = full_df.index.tolist()
    for i, sig_date in enumerate(dates):
        if sig_date not in signal_dates: continue
        slice_df = full_df.iloc[:i+1]
        if len(slice_df) < 200: continue
        if not check_trend(slice_df): continue
        found, info = detect_cup_handle(slice_df)
        if not found: continue
        entry = info["cur"]
        returns = {}
        for hold in HOLD_PERIODS:
            fi = i + hold
            if fi < len(full_df):
                fp = full_df["Close"].iloc[fi]
                returns[f"r{hold}d"] = round((fp/entry-1)*100, 2)
            else:
                returns[f"r{hold}d"] = None
        signals.append({
            "시그널일"  : sig_date.strftime("%Y-%m-%d"),
            "종목코드"  : ticker,
            "종목명"    : get_name(ticker),
            "섹터"      : get_sector(ticker),
            "진입가"    : f"{entry:,.0f}",
            "피벗"      : f"{info['pivot']:,.0f}",
            "컵깊이"    : f"{info['cd']}%",
            "거래량"    : "급증" if info["vs"] else "보통",
            "5일수익"   : f"{returns['r5d']:+.1f}%" if returns['r5d'] is not None else "-",
            "20일수익"  : f"{returns['r20d']:+.1f}%" if returns['r20d'] is not None else "-",
            "60일수익"  : f"{returns['r60d']:+.1f}%" if returns['r60d'] is not None else "-",
            "_r20"      : returns["r20d"],
        })
    return signals

# ══════════════════════════════════════
# 요약 통계
# ══════════════════════════════════════
def summary_stats(all_signals):
    if not all_signals: return "⚠️ 시그널 없음"
    df = pd.DataFrame(all_signals)
    total = len(df)
    lines = [
        f"📊 미너비니 백테스트 요약",
        f"📅 최근 1년 | KOSPI200",
        f"총 시그널: {total}건",
        f"{'─'*30}",
    ]
    for hold in HOLD_PERIODS:
        col = f"r{hold}d"
        vals = [s[col] for s in all_signals if s.get(col) is not None]
        if not vals: continue
        win = sum(1 for v in vals if v > 0)
        lines += [
            f"[{hold}일 후]",
            f"  승률  : {round(win/len(vals)*100,1)}% ({win}/{len(vals)})",
            f"  평균  : {round(sum(vals)/len(vals),2):+.2f}%",
            f"  최고  : {max(vals):+.2f}%",
            f"  최저  : {min(vals):+.2f}%",
        ]

    # 섹터별 승률
    df2 = pd.DataFrame(all_signals)
    df2["win20"] = df2["_r20"].apply(lambda x: 1 if x and x>0 else 0)
    sector_g = df2.groupby("섹터").agg(
        건수=("_r20","count"),
        승률=("win20","mean"),
        평균수익=("_r20","mean")
    ).sort_values("건수", ascending=False).head(8)

    lines += [f"{'─'*30}", "📂 섹터별 성과 (20일 기준)"]
    for sec, row in sector_g.iterrows():
        lines.append(
            f"  {sec[:8]:<8} | {int(row['건수'])}건 | "
            f"승률{round(row['승률']*100)}% | "
            f"평균{row['평균수익']:+.1f}%"
        )
    return "\n".join(lines)


def format_table(all_signals):
    """시그널 테이블을 텔레그램용 텍스트로 변환"""
    if not all_signals: return []

    # 20일 수익률 내림차순 정렬
    sorted_s = sorted(all_signals, key=lambda x: x["_r20"] or -999, reverse=True)

    header = (
        f"{'시그널일':<12}"
        f"{'종목명':<10}"
        f"{'섹터':<10}"
        f"{'5일':>6}"
        f"{'20일':>7}"
        f"{'60일':>7}"
        f"{'거래량':>5}\n"
        f"{'─'*57}\n"
    )

    rows = ""
    for s in sorted_s:
        rows += (
            f"{s['시그널일']:<12}"
            f"{s['종목명'][:6]:<10}"
            f"{s['섹터'][:6]:<10}"
            f"{s['5일수익']:>6}"
            f"{s['20일수익']:>7}"
            f"{s['60일수익']:>7}"
            f"{'🔥' if s['거래량']=='급증' else '  ':>5}\n"
        )

    # 4000자 단위로 분할
    full = f"📋 시그널 상세 테이블\n(20일 수익률 순)\n\n" + header + rows
    chunks = []
    while len(full) > 4000:
        chunks.append(full[:4000])
        full = full[4000:]
    chunks.append(full)
    return chunks
# ══════════════════════════════════════
# 텔레그램
# ══════════════════════════════════════
def send(text):
    print(text)
    if TELEGRAM_TOKEN:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                timeout=10
            )
        except: pass


# ══════════════════════════════════════
# 메인
# ══════════════════════════════════════
if __name__ == "__main__":
    end_dt    = datetime.today()
    start_dt  = end_dt - timedelta(days=HISTORY_DAYS)
    end_str   = end_dt.strftime("%Y%m%d")
    start_str = start_dt.strftime("%Y%m%d")

    signal_start = end_dt - timedelta(days=LOOKBACK_DAYS)
    signal_dates = set(pd.bdate_range(signal_start, end_dt).map(pd.Timestamp))

    print("📡 KOSPI200 종목 수집...")
    tickers = get_kospi200()

    # 섹터 일괄 캐싱
    try:
        today = datetime.today().strftime("%Y%m%d")
        sec_df = stock.get_market_sector_classifications(today, "KOSPI")
        for t, row in sec_df.iterrows():
            _sector_cache[str(t)] = row.get("업종명", "기타")
        print(f"섹터 캐싱 완료: {len(_sector_cache)}개")
    except:
        print("섹터 캐싱 실패 - 기타로 표시됩니다")

    print(f"→ {len(tickers)}개 종목 백테스트 시작\n")

    all_signals = []
    for i, ticker in enumerate(tickers):
        if i % 20 == 0:
            print(f"[{i}/{len(tickers)}] 시그널: {len(all_signals)}건")
        full_df = get_ohlcv_full(ticker, start_str, end_str)
        if full_df is None: continue
        sigs = backtest_ticker(ticker, full_df, signal_dates)
        all_signals.extend(sigs)
        time.sleep(0.1)

    print(f"\n✅ 완료: {len(all_signals)}건 시그널")

    # ① 요약 통계 전송
    send(summary_stats(all_signals))
    time.sleep(1)

    # ② 상세 테이블 전송
    for chunk in format_table(all_signals):
        send(chunk)
        time.sleep(0.5)
