import os,requests,pandas as pd
import yfinance as yf
from datetime import datetime,timedelta
import warnings
warnings.filterwarnings("ignore")
TOK=os.environ.get("TELEGRAM_TOKEN","")
CID=os.environ.get("TELEGRAM_CHAT_ID","")
def send(t):
 print(t)
 if TOK:
  try:requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",data={"chat_id":CID,"text":t},timeout=10)
  except:pass

end=datetime.today()
s=(end-timedelta(days=800)).strftime("%Y-%m-%d")
e=end.strftime("%Y-%m-%d")

# 삼성전자 하나만 테스트
df=yf.download("005930.KS",start=s,end=e,progress=False,auto_adjust=True)
send(f"shape: {df.shape}")
send(f"columns: {list(df.columns)}")
send(f"column type: {type(df.columns)}")
send(f"head:\n{df.tail(3).to_string()}")

# Close 추출 시도
try:
 if isinstance(df.columns,pd.MultiIndex):
  close=df["Close"]["005930.KS"]
 else:
  close=df["Close"]
 close=pd.to_numeric(close,errors="coerce").dropna()
 send(f"Close 길이:{len(close)}, 마지막값:{close.iloc[-1]:.0f}")
except Exception as ex:
 send(f"Close 추출 실패: {ex}")
