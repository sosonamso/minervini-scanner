import os,time,warnings,requests
import numpy as np,pandas as pd
import yfinance as yf
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")
TOK=os.environ.get("TELEGRAM_TOKEN","")
CID=os.environ.get("TELEGRAM_CHAT_ID","")
def send(t):
 print(t)
 if TOK:
  try:requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",data={"chat_id":CID,"text":t},timeout=10)
  except:pass

TICKERS=[
 "005930.KS","000660.KS","207940.KS","005380.KS","000270.KS",
 "068270.KS","105560.KS","012330.KS","055550.KS","066570.KS",
 "028260.KS","035420.KS","000810.KS","051910.KS","003550.KS",
 "034730.KS","096770.KS","003490.KS","017670.KS","032830.KS",
 "011200.KS","018260.KS","009150.KS","010130.KS","086790.KS",
 "033780.KS","316140.KS","003670.KS","009830.KS","010950.KS",
 "011070.KS","047050.KS","034020.KS","010140.KS","021240.KS",
 "000100.KS","161390.KS","002790.KS","008770.KS","004020.KS",
 "139480.KS","011170.KS","006400.KS","009540.KS","010620.KS",
 "000720.KS","004170.KS","035250.KS","001800.KS","002380.KS",
 "047810.KS","030200.KS","097950.KS","023530.KS","018880.KS",
 "004990.KS","007070.KS","003410.KS","016360.KS","008930.KS",
 "000880.KS","005490.KS","011780.KS","000150.KS","001040.KS",
 "002160.KS","004800.KS","006800.KS","007310.KS","009240.KS",
 "010060.KS","011500.KS","012750.KS","014820.KS","015760.KS",
 "017800.KS","019170.KS","020150.KS","021820.KS","023150.KS",
 "024110.KS","025540.KS","026960.KS","028050.KS","029780.KS",
 "032640.KS","033530.KS","034230.KS","035000.KS","036570.KS",
 "042660.KS","044880.KS","051600.KS","055490.KS","057050.KS",
 "064350.KS","069960.KS","078930.KS","086280.KS","267250.KS"
]

def get_ohlcv(ticker,start,end):
 for _ in range(2):
  try:
   raw=yf.download(ticker,start=start,end=end,progress=False,auto_adjust=True)
   if raw.empty or len(raw)<100:return None
   # MultiIndex 처리
   close=pd.to_numeric(raw["Close"][ticker],errors="coerce")
   volume=pd.to_numeric(raw["Volume"][ticker],errors="coerce")
   df=pd.DataFrame({"Close":close,"Volume":volume}).dropna()
   if len(df)<100:return None
   return df
  except:time.sleep(1)
 return None

def trend(df):
 if len(df)<150:return False
 c=df["Close"]
 m5=c.rolling(50).mean()
 m15=c.rolling(150).mean()
 m20=c.rolling(200).mean()
 cur=float(c.iloc[-1])
 a=float(m5.iloc[-1]);b=float(m15.iloc[-1])
 m20v=m20.dropna()
 if len(m20v)<1:return False
 d=float(m20v.iloc[-1])
 d1=float(m20v.iloc[-21])if len(m20v)>=21 else d
 if any(pd.isna([a,b,d])):return False
 return all([cur>b and cur>d,b>d,d>d1,a>b,cur>a])

def cup(df):
 cl=df["Close"].values.astype(float)
 vl=df["Volume"].values.astype(float)
 n=len(cl)
 if n<60:return False,{}
 c=cl[-min(200,n):];v=vl[-min(200,n):];w=len(c)
 li=int(np.argmax(c[:w//2]));lh=c[li]
 cp=c[li:]
 if len(cp)<20:return False,{}
 bi=li+int(np.argmin(cp));bot=c[bi];cd=(lh-bot)/lh
 if not(0.12<=cd<=0.55)or(bi-li)<25:return False,{}
 rc=c[bi:]
 if len(rc)<10:return False,{}
 ri=bi+int(np.argmax(rc));rh=c[ri]
 if rh<lh*0.85:return False,{}
 hd=c[ri:];hl=len(hd)
 if not(3<=hl<=25):return False,{}
 hl2=float(np.min(hd));hdd=(rh-hl2)/rh
 if not(0.03<=hdd<=0.20):return False,{}
 if(hl2-bot)/(lh-bot)<0.50:return False,{}
 cur=cl[-1]
 if not(rh*0.95<=cur<=rh*1.08):return False,{}
 vr=float(np.mean(v[-5:]))/float(np.mean(v[-40:-5]))if len(v)>=40 else 1.0
 return True,{"cd":round(cd*100,1),"cur":round(float(cur),0),"vs":vr>=1.30}

if __name__=="__main__":
 end=datetime.today()
 s=(end-timedelta(days=800)).strftime("%Y-%m-%d")
 e=end.strftime("%Y-%m-%d")
 sd=set(pd.bdate_range(end-timedelta(days=365),end).map(pd.Timestamp))
 send(f"스캔시작: {len(TICKERS)}개")
 res=[];tp=0
 for i,t in enumerate(TICKERS):
  if i%20==0:print(f"[{i}/{len(TICKERS)}] 트렌드:{tp} 발견:{len(res)}")
  df=get_ohlcv(t,s,e)
  if df is None:continue
  dates=df.index.tolist()
  for j,dt in enumerate(dates):
   if pd.Timestamp(dt).tz_localize(None) not in sd:continue
   sl=df.iloc[:j+1]
   if not trend(sl):continue
   tp+=1
   ok,info=cup(sl)
   if not ok:continue
   ep=info["cur"]
   if ep==0:continue
   r5=round((float(df["Close"].iloc[j+5])/ep-1)*100,2)if j+5<len(df)else None
   r20=round((float(df["Close"].iloc[j+20])/ep-1)*100,2)if j+20<len(df)else None
   r60=round((float(df["Close"].iloc[j+60])/ep-1)*100,2)if j+60<len(df)else None
   res.append({"날짜":pd.Timestamp(dt).strftime("%Y-%m-%d"),"종목":t.replace(".KS",""),
    "5일":r5,"20일":r20,"60일":r60,"급등":info["vs"],"_r":r20})
 res.sort(key=lambda x:x["_r"]or-999,reverse=True)
 send(f"완료:{len(res)}건 (트렌드통과:{tp})")
 if not res:send("⚠️ 시그널 없음");exit()
 vals=[r["_r"]for r in res if r["_r"]]
 win=sum(1 for v in vals if v>0)
 send(f"📊 백테스트\n총:{len(res)}건\n승률{round(win/len(vals)*100,1)}% 평균{round(sum(vals)/len(vals),2):+.2f}%\n최고:{max(vals):+.1f}% 최저:{min(vals):+.1f}%")
 time.sleep(1)
 hdr=f"📋 시그널(20일순)\n{'날짜':<11}{'종목':<8}{'5일':>6}{'20일':>7}{'60일':>7}\n{'─'*45}\n"
 body=""
 for r in res:
  body+=f"{r['날짜']:<11}{r['종목']:<8}{str(r['5일']or'-'):>6}{str(r['20일']or'-'):>7}{str(r['60일']or'-'):>7}{'🔥'if r['급등']else''}\n"
 full=hdr+body
 while len(full)>4000:send(full[:4000]);full=full[4000:];time.sleep(0.5)
 send(full)
