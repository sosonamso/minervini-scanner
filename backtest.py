import os,time,warnings,requests
import numpy as np
import pandas as pd
from pykrx import stock
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")

TELEGRAM_TOKEN=os.environ.get("TELEGRAM_TOKEN","")
TELEGRAM_CHAT_ID=os.environ.get("TELEGRAM_CHAT_ID","")
LOOKBACK_DAYS=365; HISTORY_DAYS=800

def get_tickers():
    today=datetime.today().strftime("%Y%m%d")
    t=stock.get_market_ticker_list(today,market="KOSPI")
    print(f"KOSPI: {len(t)}개")
    return list(t)

def get_name(t):
    try:return stock.get_market_ticker_name(t)
    except:return t

def get_ohlcv(ticker,s,e):
    for _ in range(2):
        try:
            df=stock.get_market_ohlcv(s,e,ticker)
            df.index=pd.to_datetime(df.index)
            df=df.rename(columns={"시가":"O","고가":"H","저가":"L","종가":"Close","거래량":"Volume"})
            df=df[["Close","Volume"]].dropna()
            if len(df)>=100:return df
        except:time.sleep(0.3)
    return None

def check_trend(df):
    if len(df)<200:return False
    c=df["Close"]
    m50=c.rolling(50).mean();m150=c.rolling(150).mean();m200=c.rolling(200).mean()
    cur=c.iloc[-1];a=m50.iloc[-1];b=m150.iloc[-1];d=m200.iloc[-1]
    if any(pd.isna([a,b,d])):return False
    d1m=m200.iloc[-21] if len(m200.dropna())>=21 else d
    lk=c.iloc[-252:]if len(c)>=252 else c
    return all([cur>b and cur>d,b>d,d>d1m,a>b and a>d,cur>a,cur>=lk.min()*1.25,cur>=lk.max()*0.70])

def detect(df):
    close=df["Close"].values;vol=df["Volume"].values;n=len(close)
    if n<60:return False,{}
    c=close[-min(200,n):];v=vol[-min(200,n):];w=len(c)
    li=int(np.argmax(c[:w//2]));lh=c[li]
    cup=c[li:]
    if len(cup)<20:return False,{}
    bi=li+int(np.argmin(cup));bot=c[bi];cd=(lh-bot)/lh
    if not(0.15<=cd<=0.50)or(bi-li)<35:return False,{}
    rec=c[bi:]
    if len(rec)<10:return False,{}
    ri=bi+int(np.argmax(rec));rh=c[ri]
    if rh<lh*0.90:return False,{}
    hnd=c[ri:];hl=len(hnd)
    if not(5<=hl<=20):return False,{}
    hlow=float(np.min(hnd));hd=(rh-hlow)/rh
    if not(0.05<=hd<=0.15):return False,{}
    if(hlow-bot)/(lh-bot)<0.60:return False,{}
    cur=close[-1]
    if not(rh*0.97<=cur<=rh*1.05):return False,{}
    vr=float(np.mean(v[-5:]))/float(np.mean(v[-40:-5]))if len(v)>=40 else 1.0
    return True,{"cd":round(cd*100,1),"hd":round(hd*100,1),"cdays":bi-li,"hdays":hl,
                 "pivot":round(float(rh),0),"cur":round(float(cur),0),"vr":round(vr,2),"vs":vr>=1.40}

def send(text):
    print(text)
    if TELEGRAM_TOKEN:
        try:requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                         data={"chat_id":TELEGRAM_CHAT_ID,"text":text},timeout=10)
        except:pass

if __name__=="__main__":
    end_dt=datetime.today()
    start_dt=end_dt-timedelta(days=HISTORY_DAYS)
    end_str=end_dt.strftime("%Y%m%d")
    start_str=start_dt.strftime("%Y%m%d")
    sig_start=end_dt-timedelta(days=LOOKBACK_DAYS)
    sig_dates=set(pd.bdate_range(sig_start,end_dt).map(pd.Timestamp))

    tickers=get_tickers()
    print(f"→ {len(tickers)}개 종목 시작")

    results=[]
    for i,ticker in enumerate(tickers):
        if i%50==0:print(f"[{i}/{len(tickers)}] 발견:{len(results)}")
        df=get_ohlcv(ticker,start_str,end_str)
        if df is None:continue
        dates=df.index.tolist()
        for j,sd in enumerate(dates):
            if sd not in sig_dates:continue
            sl=df.iloc[:j+1]
            if len(sl)<200:continue
            if not check_trend(sl):continue
            found,info=detect(sl)
            if not found:continue
            entry=info["cur"]
            r5=round((df["Close"].iloc[j+5]/entry-1)*100,2)if j+5<len(df)else None
            r20=round((df["Close"].iloc[j+20]/entry-1)*100,2)if j+20<len(df)else None
            r60=round((df["Close"].iloc[j+60]/entry-1)*100,2)if j+60<len(df)else None
            results.append({"날짜":sd.strftime("%Y-%m-%d"),"종목":get_name(ticker),
                           "코드":ticker,"컵깊이":f"{info['cd']}%",
                           "거래량":"급증"if info["vs"]else"보통",
                           "5일":f"{r5:+.1f}%"if r5 else"-",
                           "20일":f"{r20:+.1f}%"if r20 else"-",
                           "60일":f"{r60:+.1f}%"if r60 else"-",
                           "_r20":r20})

    results.sort(key=lambda x:x["_r20"]or-999,reverse=True)
    print(f"✅ {len(results)}건 완료")

    if not results:
        send("⚠️ 백테스트 시그널 없음");exit()

    # 요약
    vals=[r["_r20"]for r in results if r["_r20"]]
    win=sum(1 for v in vals if v>0)
    summary=(f"📊 미너비니 백테스트\n📅 최근1년 KOSPI\n총:{len(results)}건\n"
             f"[20일후] 승률{round(win/len(vals)*100,1)}% 평균{round(sum(vals)/len(vals),2):+.2f}%\n"
             f"최고:{max(vals):+.1f}% 최저:{min(vals):+.1f}%")
    send(summary)
    time.sleep(1)

    # 테이블
    header=f"📋 시그널 상세(20일수익순)\n{'날짜':<11}{'종목':<8}{'5일':>6}{'20일':>7}{'60일':>7}{'거래량'}\n{'─'*48}\n"
    body=""
    for r in results:
        body+=f"{r['날짜']:<11}{r['종목'][:6]:<8}{r['5일']:>6}{r['20일']:>7}{r['60일']:>7}{'🔥'if r['거래량']=='급증'else'  '}\n"
    full=header+body
    while len(full)>4000:
        send(full[:4000]);full=full[4000:]
        time.sleep(0.5)
    send(full)
