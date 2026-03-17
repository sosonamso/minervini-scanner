import os,time,warnings,requests
import numpy as np,pandas as pd
from pykrx import stock
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")

TOK=os.environ.get("TELEGRAM_TOKEN","")
CID=os.environ.get("TELEGRAM_CHAT_ID","")
SCAN_DAYS=7

def send(text):
 print(text)
 if TOK:
  try:requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",data={"chat_id":CID,"text":text},timeout=10)
  except:pass

def get_recent_dates(n=7):
 dates=[]
 d=datetime.today()
 while len(dates)<n:
  ds=d.strftime("%Y%m%d")
  try:
   t=stock.get_market_ticker_list(ds,market="KOSPI")
   if t:dates.append(ds)
  except:pass
  d-=timedelta(days=1)
  if(datetime.today()-d).days>30:break
 return dates

def get_universe(today):
 uni=[]
 for mkt in ["KOSPI","KOSDAQ"]:
  try:
   cap=stock.get_market_cap(today,today,market=mkt)
   if cap.empty:continue
   cap=cap[["시가총액"]].copy()
   cap=cap[cap["시가총액"]>0].sort_values("시가총액",ascending=False)
   n=len(cap)
   sel=cap.iloc[int(n*0.30):int(n*0.50)].index.tolist()
   try:
    sec=stock.get_market_sector_classifications(today,market=mkt)
    sm={str(t):r.get("업종명","기타") for t,r in sec.iterrows()}
   except:
    sm={}
   for t in sel:
    uni.append({"ticker":str(t),"market":mkt,"sector":sm.get(str(t),"기타")})
   print(f"{mkt}: {n}개 → {len(sel)}개")
   time.sleep(0.5)
  except Exception as e:
   print(f"{mkt} 실패: {e}")
 return uni

def get_ohlcv(ticker,s,e):
 result=None
 for _ in range(2):
  try:
   df=stock.get_market_ohlcv(s,e,ticker)
   df.index=pd.to_datetime(df.index)
   df=df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
   df=df[["Open","High","Low","Close","Volume"]].dropna()
   if len(df)>=60:
    result=df
    break
  except:
   time.sleep(0.3)
 return result

def check_trend(df):
 if len(df)<200:return False
 c=df["Close"]
 m50=c.rolling(50).mean();m150=c.rolling(150).mean();m200=c.rolling(200).mean()
 cur=c.iloc[-1];a=m50.iloc[-1];b=m150.iloc[-1];d=m200.iloc[-1]
 if any(pd.isna([a,b,d])):return False
 d1=m200.iloc[-21]if len(m200.dropna())>=21 else d
 lk=c.iloc[-252:]if len(c)>=252 else c
 return all([cur>b and cur>d,b>d,d>d1,a>b and a>d,cur>a,cur>=lk.min()*1.25,cur>=lk.max()*0.70])

def detect(df):
 cl=df["Close"].values.astype(float);vl=df["Volume"].values.astype(float);n=len(cl)
 if n<60:return False,{}
 c=cl[-min(200,n):];v=vl[-min(200,n):];w=len(c)
 li=int(np.argmax(c[:w//2]));lh=c[li]
 cup=c[li:]
 if len(cup)<20:return False,{}
 bi=li+int(np.argmin(cup));bot=c[bi];cd=(lh-bot)/lh
 if not(0.15<=cd<=0.50)or(bi-li)<35:return False,{}
 rc=c[bi:]
 if len(rc)<10:return False,{}
 ri=bi+int(np.argmax(rc));rh=c[ri]
 if rh<lh*0.90:return False,{}
 hnd=c[ri:];hl=len(hnd)
 if not(5<=hl<=20):return False,{}
 hlow=float(np.min(hnd));hd=(rh-hlow)/rh
 if not(0.05<=hd<=0.15):return False,{}
 if(hlow-bot)/(lh-bot)<0.60:return False,{}
 cur=cl[-1]
 if not(rh*0.97<=cur<=rh*1.05):return False,{}
 vr=float(np.mean(v[-5:]))/float(np.mean(v[-40:-5]))if len(v)>=40 else 1.0
 return True,{"cd":round(cd*100,1),"hd":round(hd*100,1),"cdays":bi-li,"hdays":hl,
              "pivot":round(float(rh),0),"cur":round(float(cur),0),"vr":round(vr,2),"vs":vr>=1.40}

def calc_rs(df,mkt):
 def p(d,n):return float(d["Close"].iloc[-1]/d["Close"].iloc[-n]-1)if len(d)>=n else 0.0
 s=sum([0.4,0.2,0.2,0.2][i]*p(df,[63,126,189,252][i])for i in range(4))
 m=sum([0.4,0.2,0.2,0.2][i]*p(mkt,[63,126,189,252][i])for i in range(4))
 return round((s-m)*100,1)

if __name__=="__main__":
 today=datetime.today().strftime("%Y%m%d")
 start=(datetime.today()-timedelta(days=420)).strftime("%Y%m%d")
 sig_dates=get_recent_dates(SCAN_DAYS)
 print(f"탐색 날짜: {sig_dates}")
 send(f"🚀 스캐너 시작\n📅 최근 {SCAN_DAYS}거래일\n({sig_dates[-1]}~{sig_dates[0]})\n잠시만 기다려주세요...")
 try:
  mkt_df=stock.get_index_ohlcv(start,today,"1028")
  mkt_df.index=pd.to_datetime(mkt_df.index)
  mkt_df=mkt_df.rename(columns={"종가":"Close"})
 except:mkt_df=None
 uni=get_universe(sig_dates[0])
 send(f"📡 {len(uni)}개 종목 스캔 중...")
 nc={}
 def nm(t):
  if t not in nc:
   try:nc[t]=stock.get_market_ticker_name(t)
   except:nc[t]=t
  return nc[t]
 res=[]
 for i,info in enumerate(uni):
  t=info["ticker"]
  if i%50==0:print(f"[{i}/{len(uni)}] 발견:{len(res)}")
  df=get_ohlcv(t,start,today)
  if df is None:continue
  for sig_str in sig_dates:
   sig_ts=pd.Timestamp(sig_str)
   if sig_ts not in df.index:continue
   pos=df.index.tolist().index(sig_ts)
   sl=df.iloc[:pos+1]
   if not check_trend(sl):continue
   ok,pat=detect(sl)
   if not ok:continue
   rs=calc_rs(sl,mkt_df.loc[:sig_ts])if mkt_df is not None else 0.0
   res.append({"sig_date":sig_str,"ticker":t,"name":nm(t),
               "market":info["market"],"sector":info["sector"],
               "cur":pat["cur"],"pivot":pat["pivot"],
               "cd":pat["cd"],"hd":pat["hd"],"cdays":pat["cdays"],"hdays":pat["hdays"],
               "vr":pat["vr"],"vs":pat["vs"],"rs":rs})
  time.sleep(0.05)
 res.sort(key=lambda x:(x["sig_date"],x["rs"]),reverse=True)
 seen=set();deduped=[]
 for r in res:
  if r["ticker"] not in seen:
   seen.add(r["ticker"]);deduped.append(r)
 res=deduped
 print(f"✅ {len(res)}개 발견")
 if not res:
  send(f"📊 미너비니 스캐너\n📅 최근 {SCAN_DAYS}거래일\n⚠️ 조건 충족 종목 없음")
 else:
  hdr=f"📊 미너비니 컵&핸들\n📅 최근 {SCAN_DAYS}거래일\n✅ {len(res)}개 발견\n{'─'*24}\n"
  msg=hdr
  for r in res:
   up=round((r['pivot']/r['cur']-1)*100,1)
   mkt="🔵코스피"if r['market']=="KOSPI"else"🟢코스닥"
   vol="🔥"if r['vs']else"  "
   blk=(f"📅{r['sig_date'][:4]}/{r['sig_date'][4:6]}/{r['sig_date'][6:]}\n"
        f"{mkt}[{r['sector']}]\n"
        f"🔹{r['name']}({r['ticker']})\n"
        f"  현재가:{r['cur']:,.0f}원\n"
        f"  피벗:{r['pivot']:,.0f}원({up:+.1f}%)\n"
        f"  컵:{r['cd']}%/{r['cdays']}일 핸들:{r['hd']}%/{r['hdays']}일\n"
        f"  거래량:{r['vr']}x{vol} RS:{r['rs']:+.1f}%\n\n")
   if len(msg)+len(blk)>4000:
    send(msg);msg="📊(이어서)\n\n"+blk
   else:msg+=blk
  send(msg)
