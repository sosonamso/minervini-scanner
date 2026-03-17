"""
미너비니 컵&핸들 스캐너
- 대상: KOSPI + KOSDAQ 시총 상위 30~50%
- 섹터: pykrx KRX 한국어 업종명
- 결과: 텔레그램 전송
"""

import os, time, warnings, requests
import numpy as np, pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
HISTORY_DAYS     = 420


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


def get_latest_trading_date():
    for i in range(10):
        d = (datetime.today() - timedelta(days=i)).strftime("%Y%m%d")
        try:
            t = stock.get_market_ticker_list(d, market="KOSPI")
            if t: return d
        except: pass
    return datetime.today().strftime("%Y%m%d")
 def get_universe(today_str):
    """KOSPI + KOSDAQ 시총 상위 30~50% 종목"""
    universe = []
    for mkt in ["KOSPI", "KOSDAQ"]:
        try:
            cap_df = stock.get_market_cap(today_str, today_str, market=mkt)
            if cap_df.empty: continue
            cap_df = cap_df[["시가총액"]].copy()
            cap_df = cap_df[cap_df["시가총액"] > 0].sort_values("시가총액", ascending=False)
            total  = len(cap_df)
            top30  = int(total * 0.30)
            top50  = int(total * 0.50)
            selected = cap_df.iloc[top30:top50].index.tolist()

            # 섹터 매핑
            try:
                sec_df  = stock.get_market_sector_classifications(today_str, market=mkt)
                sec_map = {str(t): row.get("업종명","기타") for t, row in sec_df.iterrows()}
            except:
                sec_map = {}

            for ticker in selected:
                t = str(ticker)
                universe.append({
                    "ticker": t,
                    "market": mkt,
                    "sector": sec_map.get(t, "기타"),
                })
            print(f"{mkt}: 전체 {total}개 → 상위 30~50% {len(selected)}개")
            time.sleep(0.5)
        except Exception as e:
            print(f"{mkt} 유니버스 실패: {e}")
    return universe


def get_ohlcv(ticker, start_str, end_str):
    for _ in range(2):
        try:
            df = stock.get_market_ohlcv(start_str, end_str, ticker)
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
            df = df[["Open","High","Low","Close","Volume"]].dropna()
            if len(df) >= 60: return df
        except: time.sleep(0.3)
    return None


def check_trend(df):
    if len(df) < 200: return False
    c = df["Close"]
    ma50=c.rolling(50).mean(); ma150=c.rolling(150).mean(); ma200=c.rolling(200).mean()
    cur=c.iloc[-1]; m50=ma50.iloc[-1]; m150=ma150.iloc[-1]; m200=ma200.iloc[-1]
    if any(pd.isna([m50,m150,m200])): return False
    m200_1m = ma200.iloc[-21] if len(ma200.dropna())>=21 else m200
    lk = c.iloc[-252:] if len(c)>=252 else c
    return all([
        cur>m150 and cur>m200, m150>m200, m200>m200_1m,
        m50>m150 and m50>m200, cur>m50,
        cur>=lk.min()*1.25, cur>=lk.max()*0.70,
    ])


def detect_cup_handle(df):
    close=df["Close"].values.astype(float); volume=df["Volume"].values.astype(float); n=len(close)
    if n<60: return False,{}
    c=close[-min(200,n):]; v=volume[-min(200,n):]; w=len(c)
    li=int(np.argmax(c[:w//2])); lh=c[li]
    cup=c[li:]
    if len(cup)<20: return False,{}
    bi=li+int(np.argmin(cup)); bot=c[bi]; cd=(lh-bot)/lh
    if not(0.15<=cd<=0.50) or (bi-li)<35: return False,{}
    rec=c[bi:]
    if len(rec)<10: return False,{}
    ri=bi+int(np.argmax(rec)); rh=c[ri]
    if rh<lh*0.90: return False,{}
    hnd=c[ri:]; hl=len(hnd)
    if not(5<=hl<=20): return False,{}
    hlow=float(np.min(hnd)); hd=(rh-hlow)/rh
    if not(0.05<=hd<=0.15): return False,{}
    if(hlow-bot)/(lh-bot)<0.60: return False,{}
    cur=close[-1]
    if not(rh*0.97<=cur<=rh*1.05): return False,{}
    vr=float(np.mean(v[-5:]))/float(np.mean(v[-40:-5])) if len(v)>=40 else 1.0
    return True,{
        "cup_depth":round(cd*100,1),"handle_depth":round(hd*100,1),
        "cup_days":bi-li,"handle_days":hl,
        "pivot":round(float(rh),0),"current":round(float(cur),0),
        "vol_ratio":round(vr,2),"vol_surge":vr>=1.40,
    }


def calc_rs(df, mkt_df):
    def p(d,n): return float(d["Close"].iloc[-1]/d["Close"].iloc[-n]-1) if len(d)>=n else 0.0
    s=sum([0.4,0.2,0.2,0.2][i]*p(df,[63,126,189,252][i]) for i in range(4))
    m=sum([0.4,0.2,0.2,0.2][i]*p(mkt_df,[63,126,189,252][i]) for i in range(4))
    return round((s-m)*100,1)


def format_results(results, today_str):
    date_fmt = f"{today_str[:4]}/{today_str[4:6]}/{today_str[6:]}"
    if not results:
        return [f"📊 미너비니 스캐너\n📅 {date_fmt} 기준\n{'─'*28}\n⚠️ 조건 충족 종목 없음"]

    header = f"📊 미너비니 컵&핸들\n📅 {date_fmt} 종가 기준\n✅ {len(results)}개 발견 (RS순)\n{'─'*28}\n"
    msgs=[]; cur=header

    for r in results:
        upside = round((r['pivot']/r['current']-1)*100,1) if r['current']>0 else 0
        mkt_lbl = "🔵코스피" if r['market']=="KOSPI" else "🟢코스닥"
        vol_lbl = "🔥급증" if r['vol_surge'] else "보통"
        block = (
            f"{mkt_lbl} [{r['sector']}]\n"
            f"🔹 {r['name']} ({r['ticker']})\n"
            f"   현재가: {r['current']:>10,.0f}원\n"
            f"   피벗  : {r['pivot']:>10,.0f}원 ({upside:+.1f}%)\n"
            f"   컵    : {r['cup_depth']}% / {r['cup_days']}일\n"
            f"   핸들  : {r['handle_depth']}% / {r['handle_days']}일\n"
            f"   거래량: {r['vol_ratio']}x {vol_lbl}\n"
            f"   RS    : {r['rs']:+.1f}%\n\n"
        )
        if len(cur)+len(block)>4000:
            msgs.append(cur); cur="📊 (이어서)\n\n"+block
        else:
            cur+=block
    msgs.append(cur)
    return msgs


if __name__ == "__main__":
    today_str = get_latest_trading_date()
    start_str = (datetime.strptime(today_str,"%Y%m%d")-timedelta(days=HISTORY_DAYS)).strftime("%Y%m%d")

    send(f"🚀 스캐너 시작\n📅 {today_str} 기준\n잠시만 기다려주세요...")

    try:
        mkt_df = stock.get_index_ohlcv(start_str, today_str, "1028")
        mkt_df.index = pd.to_datetime(mkt_df.index)
        mkt_df = mkt_df.rename(columns={"종가":"Close"})
    except:
        mkt_df = None

    universe = get_universe(today_str)
    send(f"📡 {len(universe)}개 종목 스캔 중...")

    name_cache = {}
    def get_name(ticker):
        if ticker not in name_cache:
            try: name_cache[ticker] = stock.get_market_ticker_name(ticker)
            except: name_cache[ticker] = ticker
        return name_cache[ticker]

    results=[]; passed=0

    for i, info in enumerate(universe):
        ticker = info["ticker"]
        if i%50==0: print(f"[{i}/{len(universe)}] 트렌드통과:{passed} 발견:{len(results)}")

        df = get_ohlcv(ticker, start_str, today_str)
        if df is None: continue
        if not check_trend(df): continue
        passed+=1

        found, pat = detect_cup_handle(df)
        if not found: continue

        rs = calc_rs(df, mkt_df) if mkt_df is not None else 0.0
        results.append({
            "ticker"      : ticker,
            "name"        : get_name(ticker),
            "market"      : info["market"],
            "sector"      : info["sector"],
            "current"     : pat["current"],
            "pivot"       : pat["pivot"],
            "cup_depth"   : pat["cup_depth"],
            "handle_depth": pat["handle_depth"],
            "cup_days"    : pat["cup_days"],
            "handle_days" : pat["handle_days"],
            "vol_ratio"   : pat["vol_ratio"],
            "vol_surge"   : pat["vol_surge"],
            "rs"          : rs,
        })
        time.sleep(0.05)

    results.sort(key=lambda x: x["rs"], reverse=True)
    print(f"✅ 완료: {len(results)}개 / 트렌드통과: {passed}개")

    for msg in format_results(results, today_str):
        send(msg)
        time.sleep(0.5)
