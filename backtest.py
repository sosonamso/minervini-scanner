import os,time,warnings,requests
import numpy as np,pandas as pd
from pykrx import stock
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")
TOK=os.environ.get("TELEGRAM_TOKEN","")
CID=os.environ.get("TELEGRAM_CHAT_ID","")

def send(t):
 print(t)
 if TOK:
  try:requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",data={"chat_id":CID,"text":t},timeout=10)
  except:pass

def get_tickers():
 try:
  headers={"User-Agent":"Mozilla/5.0","Referer":"http://data.krx.co.kr/"}
  r=requests.post("http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd",
   data={"bld":"dbms/MDC/STAT/standard/MDCSTAT01901","locale":"ko_KR","mktId":"STK","share":"1","money":"1","csvxls_isNo":"false"},
   headers=headers,timeout=10)
  items=r.json()["OutBlock_1"]
  tickers=[x["ISU_SRT_CD"] for x in items]
  send(f"KRX직접: {len(tickers)}개")
  return tickers
 except Exception as e:
  send(f"KRX직접 실패: {e}")
  return []

def get_name(t):
 try:return stock.get_market_ticker_name(t)
 except:return t

def ohlcv(ticker,s,e):
 for _ in range(2):
  try:
   df=stock.get_market_ohlcv(s,e,ticker)
   df.index=pd.to_datetime(df.index)
   df=df.rename(columns={"종가":"Close","거래량":"Volume"})
   df=df[["Close","Volume"]].dropna()
   if len(df)>=100:return df
  except:time.sleep(0.3)
 return None

def trend(df):
 if len(df)<200:return False
 c=df["Close"]
 m5=c.rolling(50).mean();m15=c.rolling(150).mean();m20=c.rolling(200).mean()
 cur=c.iloc[-1];a=m5.iloc[-1];b=m15.iloc[-1];d=m20.iloc[-1]
 if any(pd.isna([a,b,d])):return False
 d1=m20.iloc[-21] if len(m20.dropna())>=21 else d
 lk=c.iloc[-252:]if len(c)>=252 else c
 return all([cur>b and cur>d,b>d,d>d1,a>b and a>d,cur>a,cur>=lk.min()*1.25,cur>=lk.max()*0.70])

def cup(df):
 cl=df["Close"].values;vl=df["Volume"].values;n=len(cl)
 if n<60:return False,{}
 c=cl[-min(200,n):];v=vl[-min(200,n):];w=len(c)
 li=int(np.argmax(c[:w//2]));lh=c[li]
 cp=c[li:]
 if len(cp)<20:return False,{}
 bi=li+int(np.argmin(cp));bot=c[bi];cd=(lh-bot)/lh
 if not(0.15<=cd<=0.50)or(bi-li)<35:return False,{}
 rc=c[bi:]
 if len(rc)<10:return False,{}
 ri=bi+int(np.argmax(rc));rh=c[ri]
 if rh<lh*0.90:return False,{}
 hd=c[ri:];hl=len(hd)
 if not(5<=hl<=20):return False,{}
 hl2=float(np.min(hd));hdd=(rh-hl2)/rh
 if not(0.05<=hdd<=0.15):return False,{}
 if(hl2-bot)/(lh-bot)<0.60:return False,{}
 cur=cl[-1]
 if not(rh*0.97<=cur<=rh*1.05):return False,{}
 vr=float(np.mean(v[-5:]))/float(np.mean(v[-40:-5]))if len(v)>=40 else 1.0
 return True,{"cd":round(cd*100,1),"pivot":round(float(rh),0),"cur":round(float(cur),0),"vs":vr>=1.40}

if __name__=="__main__":
 end=datetime.today();s=(end-timedelta(days=800)).strftime("%Y%m%d");e=end.strftime("%Y%m%d")
 sd=set(pd.bdate_range(end-timedelta(days=365),end).map(pd.Timestamp))
 tickers=get_tickers()
 if not tickers:send("종목리스트 실패");exit()
 res=[]
 for i,t in enumerate(tickers):
  if i%100==0:print(f"[{i}/{len(tickers)}] 발견:{len(res)}")
  df=ohlcv(t,s,e)
  if df is None:continue
  dates=df.index.tolist()
  for j,dt in enumerate(dates):
   if dt not in sd:continue
   sl=df.iloc[:j+1]
   if not trend(sl):continue
   ok,info=cup(sl)
   if not ok:continue
   ep=info["cur"]
   r20=round((df["Close"].iloc[j+20]/ep-1)*100,2)if j+20<len(df)else None
   r60=round((df["Close"].iloc[j+60]/ep-1)*100,2)if j+60<len(df)else None
   res.append({"날짜":dt.strftime("%Y-%m-%d"),"종목":get_name(t),"5일":round((df["Close"].iloc[j+5]/ep-1)*100,2)if j+5<len(df)else None,"20일":r20,"60일":r60,"급등":info["vs"],"_r":r20})
 res.sort(key=lambda x:x["_r"]or-999,reverse=True)
 print(f"완료:{len(res)}건")
 if not res:send("⚠️ 시그널 없음");exit()
 vals=[r["_r"]for r in res if r["_r"]]
 win=sum(1 for v in vals if v>0)
 send(f"📊 백테스트결과\n총:{len(res)}건\n[20일]승률{round(win/len(vals)*100,1)}% 평균{round(sum(vals)/len(vals),2):+.2f}%\n최고:{max(vals):+.1f}% 최저:{min(vals):+.1f}%")
 time.sleep(1)
 hdr=f"📋 시그널(20일순)\n{'날짜':<11}{'종목':<8}{'5일':>6}{'20일':>7}{'60일':>7}\n{'─'*45}\n"
 body=""
 for r in res:
  body+=f"{r['날짜']:<11}{r['종목'][:6]:<8}{str(r['5일']or'-'):>6}{str(r['20일']or'-'):>7}{str(r['60일']or'-'):>7}{'🔥'if r['급등']else''}\n"
 full=hdr+body
 while len(full)>4000:send(full[:4000]);full=full[4000:];time.sleep(0.5)
 send(full)
