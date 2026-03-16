import os,time,warnings,requests,numpy as np,pandas as pd
from pykrx import stock
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")

TELEGRAM_TOKEN=os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID=os.environ["TELEGRAM_CHAT_ID"]
TARGET_MARKET="ALL"
MAX_RESULTS=25

end_dt=datetime.today()
start_dt=end_dt-timedelta(days=420)
end_str=end_dt.strftime("%Y%m%d")
start_str=start_dt.strftime("%Y%m%d")

def get_tickers():
 t=[]
 if TARGET_MARKET in("KOSPI","ALL"):t+=stock.get_market_ticker_list(end_str,market="KOSPI")
 if TARGET_MARKET in("KOSDAQ","ALL"):t+=stock.get_market_ticker_list(end_str,market="KOSDAQ")
 return t

def get_name(ticker):
 try:return stock.get_market_ticker_name(ticker)
 except:return ticker

def get_ohlcv(ticker):
 for _ in range(2):
  try:
   df=stock.get_market_ohlcv(start_str,end_str,ticker)
   df.index=pd.to_datetime(df.index)
   df=df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
   df=df[["Open","High","Low","Close","Volume"]].dropna()
   if len(df)>=60:return df
  except:time.sleep(0.3)
 return None

def get_index():
 try:
  df=stock.get_index_ohlcv(start_str,end_str,"1028")
  df.index=pd.to_datetime(df.index)
  return df.rename(columns={"종가":"Close"})
 except:return None

def check_trend(df):
 if len(df)<200:return False
 c=df["Close"]
 ma50=c.rolling(50).mean();ma150=c.rolling(150).mean();ma200=c.rolling(200).mean()
 cur=c.iloc[-1];m50=ma50.iloc[-1];m150=ma150.iloc[-1];m200=ma200.iloc[-1];m200_1m=ma200.iloc[-21]
 lk=c.iloc[-252:]if len(c)>=252 else c
 return all([cur>m150 and cur>m200,m150>m200,m200>m200_1m,m50>m150 and m50>m200,cur>m50,cur>=lk.min()*1.25,cur>=lk.max()*0.70])

def detect(df):
 close=df["Close"].values;volume=df["Volume"].values;n=len(close)
 if n<60:return False,{}
 c=close[-min(200,n):];v=volume[-min(200,n):];w=len(c)
 lh_idx=int(np.argmax(c[:w//2]));lh=c[lh_idx]
 cup=c[lh_idx:]
 if len(cup)<20:return False,{}
 bi=lh_idx+int(np.argmin(cup));bot=c[bi];cd=(lh-bot)/lh
 if not(0.15<=cd<=0.50)or(bi-lh_idx)<35:return False,{}
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
 return True,{"cd":round(cd*100,1),"hd":round(hd*100,1),"cdays":bi-lh_idx,"hdays":hl,"pivot":round(float(rh),0),"cur":round(float(cur),0),"vr":round(vr,2),"vs":vr>=1.40}

def rs(df,mkt):
 def p(d,n):return float(d["Close"].iloc[-1]/d["Close"].iloc[-n]-1)if len(d)>=n else 0.0
 s=sum([0.4,0.2,0.2,0.2][i]*p(df,[63,126,189,252][i])for i in range(4))
 m=sum([0.4,0.2,0.2,0.2][i]*p(mkt,[63,126,189,252][i])for i in range(4))
 return round((s-m)*100,1)

def send(text):
 try:requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",data={"chat_id":TELEGRAM_CHAT_ID,"text":text},timeout=10)
 except:pass

def build(results):
 today=datetime.today().strftime("%Y/%m/%d")
 hdr=f"📊 미너비니 컵&핸들\n📅 {today} 기준\n{'─'*26}\n"
 if not results:return[hdr+"⚠️ 조건 충족 종목 없음"]
 hdr+=f"✅ {len(results)}개 발견\n\n"
 msgs=[];cur=hdr
 for r in results[:MAX_RESULTS]:
  up=round((r['pivot']/r['cur']-1)*100,1)if r['cur']>0 else 0
  b=(f"🔹 {r['name']} ({r['ticker']})\n"
     f"   현재가 : {r['cur']:>10,.0f}원\n"
     f"   피벗   : {r['pivot']:>10,.0f}원 ({up:+.1f}%)\n"
     f"   컵/핸들: {r['cd']}%({r['cdays']}일)/{r['hd']}%({r['hdays']}일)\n"
     f"   거래량 : {r['vr']}x {'🔥급증'if r['vs']else'보통'}\n"
     f"   RS     : {r['rsv']:+.1f}%\n\n")
  if len(cur)+len(b)>4000:msgs.append(cur);cur="📊(이어서)\n\n"+b
  else:cur+=b
 msgs.append(cur)
 return msgs

if __name__=="__main__":
 print("스캔 시작")
 tickers=get_tickers();mkt=get_index()
 print(f"{len(tickers)}개 종목")
 results=[];passed=0
 for i,ticker in enumerate(tickers):
  if i%200==0:print(f"[{i}/{len(tickers)}] 통과:{passed} 발견:{len(results)}")
  df=get_ohlcv(ticker)
  if df is None:continue
  if not check_trend(df):continue
  passed+=1
  found,info=detect(df)
  if not found:continue
  r=rs(df,mkt)if mkt is not None else 0.0
  results.append({"ticker":ticker,"name":get_name(ticker),**info,"rsv":r})
 results.sort(key=lambda x:x["rsv"],reverse=True)
 print(f"완료: {len(results)}개")
 for msg in build(results):send(msg);time.sleep(0.5)
 print("텔레그램 전송 완료")
