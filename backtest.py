import requests,time
from pykrx import stock
from datetime import datetime,timedelta
import os

TOK=os.environ.get("TELEGRAM_TOKEN","")
CID=os.environ.get("TELEGRAM_CHAT_ID","")
def send(t):
 print(t)
 if TOK:
  try:requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",data={"chat_id":CID,"text":t},timeout=10)
  except:pass

send("진단 시작")

# 날짜 5일치 시도
for d in range(5):
 td=(datetime.today()-timedelta(days=d)).strftime("%Y%m%d")
 try:
  t=stock.get_market_ticker_list(td,market="KOSPI")
  send(f"날짜:{td} → {len(t)}개")
  if t:break
 except Exception as e:
  send(f"날짜:{td} → 에러: {e}")
 time.sleep(1)

# KRX 직접 연결 테스트
try:
 r=requests.get("http://data.krx.co.kr",timeout=5)
 send(f"KRX 연결: {r.status_code}")
except Exception as e:
 send(f"KRX 연결 실패: {e}")
