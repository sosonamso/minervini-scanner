"""
미너비니 컵&핸들 스캐너 - 미국 주식
- 대상: S&P1500 (S&P500 + MidCap400 + SmallCap600)
- 데이터: yfinance (미국은 GitHub Actions에서 정상 동작)
- 결과: 텔레그램 전송 (한국어)
"""
import os,time,warnings,requests
import numpy as np,pandas as pd
import yfinance as yf
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")

TOK=os.environ.get("TELEGRAM_TOKEN","")
CID=os.environ.get("TELEGRAM_CHAT_ID","")
SCAN_DAYS=7
BATCH_SIZE=50

# ─────────────────────────────────────
# S&P1500 종목 리스트 수집 (Wikipedia)
# ─────────────────────────────────────
def get_sp1500():
    tickers={}
    sources=[
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies","S&P500",0),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies","MidCap400",0),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies","SmallCap600",0),
    ]
    for url,cap,tbl_idx in sources:
        try:
            tables=pd.read_html(url)
            df=tables[tbl_idx]
            # 티커 컬럼 찾기
            tick_col=None
            for col in df.columns:
                if "ticker" in str(col).lower() or "symbol" in str(col).lower():
                    tick_col=col
                    break
            if tick_col is None:
                tick_col=df.columns[0]
            # 섹터 컬럼 찾기
            sec_col=None
            for col in df.columns:
                if "sector" in str(col).lower() or "gics" in str(col).lower():
                    sec_col=col
                    break
            for _,row in df.iterrows():
                t=str(row[tick_col]).strip().replace(".","-")
                if not t or t=="nan" or len(t)>6:continue
                sec=str(row[sec_col]).strip() if sec_col else "기타"
                if sec=="nan":sec="기타"
                tickers[t]={"cap":cap,"sector":sec}
            print(f"{cap}: {len([k for k,v in tickers.items() if v['cap']==cap])}개")
            time.sleep(1)
        except Exception as e:
            print(f"{cap} 수집 실패: {e}")
    return tickers

# 섹터 한글 매핑
SECTOR_KO={
    "Information Technology":"IT/기술",
    "Health Care":"헬스케어",
    "Financials":"금융",
    "Consumer Discretionary":"임의소비재",
    "Communication Services":"커뮤니케이션",
    "Industrials":"산업재",
    "Consumer Staples":"필수소비재",
    "Energy":"에너지",
    "Utilities":"유틸리티",
    "Real Estate":"리츠/부동산",
    "Materials":"소재",
    "기타":"기타",
}

def sec_ko(s):
    return SECTOR_KO.get(s,s)

# ─────────────────────────────────────
# 유틸
# ─────────────────────────────────────
def send(text):
    print(text)
    if TOK:
        try:requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",
                          data={"chat_id":CID,"text":text},timeout=10)
        except:pass

def get_recent_dates(n=7):
    dates=[]
    d=datetime.today()
    while len(dates)<n:
        if d.weekday()<5:  # 미국 주말만 제외 (공휴일은 데이터 없으면 자동 스킵)
            dates.append(d.strftime("%Y-%m-%d"))
        d-=timedelta(days=1)
        if len(dates)>=n*3:break
    return dates[:n]

def batch_download(tickers,start,end):
    result={}
    ticker_list=list(tickers)
    for i in range(0,len(ticker_list),BATCH_SIZE):
        batch=ticker_list[i:i+BATCH_SIZE]
        print(f"배치 {i+1}~{min(i+BATCH_SIZE,len(ticker_list))}/{len(ticker_list)}")
        for attempt in range(3):
            try:
                raw=yf.download(" ".join(batch),start=start,end=end,
                                progress=False,auto_adjust=True,group_by="ticker")
                if raw.empty:break
                if isinstance(raw.columns,pd.MultiIndex):
                    for t in batch:
                        try:
                            df=pd.DataFrame({
                                "Close":pd.to_numeric(raw[t]["Close"],errors="coerce"),
                                "Volume":pd.to_numeric(raw[t]["Volume"],errors="coerce")
                            }).dropna()
                            if len(df)>=100:result[t]=df
                        except:pass
                else:
                    if len(batch)==1:
                        t=batch[0]
                        try:
                            df=pd.DataFrame({
                                "Close":pd.to_numeric(raw["Close"],errors="coerce"),
                                "Volume":pd.to_numeric(raw["Volume"],errors="coerce")
                            }).dropna()
                            if len(df)>=100:result[t]=df
                        except:pass
                break
            except Exception as e:
                print(f"배치 오류(시도{attempt+1}): {e}")
                time.sleep(5*(attempt+1))
        time.sleep(1)
    return result

# ─────────────────────────────────────
# 미너비니 로직
# ─────────────────────────────────────
def check_market(mkt_df):
    """S&P500 200MA 위에 있을 때만 True"""
    if mkt_df is None or len(mkt_df)<200:return True
    c=mkt_df["Close"]
    ma200=c.rolling(200).mean()
    cur=float(c.iloc[-1]);ma=float(ma200.iloc[-1])
    if pd.isna(ma):return True
    return cur>ma

def check_trend(df):
    if len(df)<200:return False
    c=df["Close"]
    m50=c.rolling(50).mean();m150=c.rolling(150).mean();m200=c.rolling(200).mean()
    cur=float(c.iloc[-1]);a=float(m50.iloc[-1]);b=float(m150.iloc[-1])
    m20v=m200.dropna()
    if len(m20v)<1:return False
    d=float(m20v.iloc[-1]);d1=float(m20v.iloc[-21])if len(m20v)>=21 else d
    if any(pd.isna([a,b,d])):return False
    lk=c.iloc[-252:]if len(c)>=252 else c
    return all([cur>b and cur>d,b>d,d>d1,a>b and a>d,cur>a,
                cur>=lk.min()*1.25,cur>=lk.max()*0.70])

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
                 "pivot":round(float(rh),2),"cur":round(float(cur),2),
                 "vr":round(vr,2),"vs":vr>=1.40}

def calc_rs(df,mkt):
    def p(d,n):return float(d["Close"].iloc[-1]/d["Close"].iloc[-n]-1)if len(d)>=n else 0.0
    s=sum([0.4,0.2,0.2,0.2][i]*p(df,[63,126,189,252][i])for i in range(4))
    m=sum([0.4,0.2,0.2,0.2][i]*p(mkt,[63,126,189,252][i])for i in range(4))
    return round((s-m)*100,1)

def get_past_signals(df,exclude_ts):
    results=[]
    idx=df.index.tolist()
    check_start=max(0,len(idx)-180)
    for i in range(check_start,len(idx)-5):
        ts=idx[i]
        if ts>=exclude_ts:continue
        sl=df.iloc[:i+1]
        if not check_trend(sl):continue
        ok,pat=detect(sl)
        if not ok:continue
        close_on=float(df["Close"].iloc[i])
        r5=None;r20=None
        if i+5<len(idx):r5=round((float(df["Close"].iloc[i+5])/close_on-1)*100,1)
        if i+20<len(idx):r20=round((float(df["Close"].iloc[i+20])/close_on-1)*100,1)
        results.append({"date":ts.strftime("%y-%m-%d"),"r5":r5,"r20":r20,"vs":pat["vs"]})
    return results

def format_past(history):
    if not history:return ""
    lines=[]
    wins=sum(1 for h in history if h["r20"] is not None and h["r20"]>0)
    total=sum(1 for h in history if h["r20"] is not None)
    for h in history[-3:]:
        r5_s=f"{h['r5']:+.1f}%"if h["r5"] is not None else"-"
        r20_s=f"{h['r20']:+.1f}%"if h["r20"] is not None else"미완"
        ok="OK"if(h["r20"] is not None and h["r20"]>0)else("..."if h["r20"] is None else"XX")
        vol="VOL"if h["vs"]else""
        lines.append(f"  {h['date']}: 5일{r5_s}/20일{r20_s} {ok}{vol}")
    result="[과거이력]\n"+"\n".join(lines)
    if total>0:
        wr=round(wins/total*100)
        result+=f"\n  -> 과거{total}회 승률{wr}%"
    return result

# ─────────────────────────────────────
# 메인
# ─────────────────────────────────────
if __name__=="__main__":
    end=datetime.today()
    start=(end-timedelta(days=420)).strftime("%Y-%m-%d")
    end_str=end.strftime("%Y-%m-%d")
    sig_dates=get_recent_dates(SCAN_DAYS)
    data_cutoff=pd.Timestamp(sig_dates[0])-timedelta(days=7)
    print(f"탐색날짜: {sig_dates}")

    # S&P500 지수 (기준)
    try:
        mkt_raw=yf.download("SPY",start=start,end=end_str,progress=False,auto_adjust=True)
        if isinstance(mkt_raw.columns,pd.MultiIndex):mkt_raw.columns=mkt_raw.columns.get_level_values(0)
        mkt_df=pd.DataFrame({"Close":pd.to_numeric(mkt_raw["Close"],errors="coerce")}).dropna()
        print(f"S&P500(SPY) {len(mkt_df)}일치 수신")
    except:mkt_df=None

    market_ok=check_market(mkt_df)
    market_str="상승장(S&P500>200MA)"if market_ok else"하락장(S&P500<200MA)"

    # S&P1500 종목 수집
    send(f"🇺🇸 미국 스캐너 시작\n최근 {SCAN_DAYS}거래일 | {market_str}\nS&P1500 종목 수집 중...")
    if not market_ok:
        send("S&P500 200MA 하방 - 시그널 신뢰도 낮음, 주의!")

    sp1500=get_sp1500()
    send(f"S&P1500 {len(sp1500)}개 종목 배치 다운로드 중...")

    # 배치 다운로드
    all_data=batch_download(list(sp1500.keys()),start,end_str)

    # 데이터 유효성 체크
    data_ok=0;data_old=0;last_dates=[]
    valid_data={}
    for ticker,df in all_data.items():
        last_date=df.index[-1]
        if last_date<data_cutoff:
            data_old+=1
            continue
        valid_data[ticker]=df
        data_ok+=1
        last_dates.append(last_date)

    if last_dates:
        last_dates.sort()
        median_date=last_dates[len(last_dates)//2].strftime("%Y-%m-%d")
        date_stat=f"데이터 기준일: {median_date}(중앙)"
    else:
        date_stat="데이터 기준일: 없음"

    send(f"다운로드 완료\n수신: {data_ok}/{len(sp1500)}개\n{date_stat}\n패턴 분석 시작...")

    # 패턴 분석
    res=[];trend_pass=0
    for i,(ticker,info) in enumerate(sp1500.items()):
        if i%200==0:print(f"[{i}/{len(sp1500)}] 트렌드:{trend_pass} 발견:{len(res)}")
        df=valid_data.get(ticker)
        if df is None:continue
        for sig_str in sig_dates:
            sig_ts=pd.Timestamp(sig_str)
            if sig_ts not in df.index:continue
            pos=df.index.tolist().index(sig_ts)
            sl=df.iloc[:pos+1]
            if not check_trend(sl):continue
            trend_pass+=1
            ok,pat=detect(sl)
            if not ok:continue
            if not pat["vs"]:continue  # 거래량 급증 필수
            rs=calc_rs(sl,mkt_df.loc[:sig_ts])if mkt_df is not None else 0.0
            if rs<=0:continue  # RS 양수 필터
            history=get_past_signals(df,sig_ts)
            res.append({
                "sig_date":sig_str,
                "ticker":ticker,
                "cap":info["cap"],
                "sector":sec_ko(info["sector"]),
                "cur":pat["cur"],"pivot":pat["pivot"],
                "cd":pat["cd"],"hd":pat["hd"],
                "cdays":pat["cdays"],"hdays":pat["hdays"],
                "vr":pat["vr"],"vs":pat["vs"],
                "rs":rs,"history":history,
            })
            break

    res.sort(key=lambda x:(x["sig_date"],x["rs"]),reverse=True)
    seen=set();deduped=[]
    for r in res:
        if r["ticker"] not in seen:
            seen.add(r["ticker"]);deduped.append(r)
    res=deduped
    print(f"완료: {len(res)}개 발견")

    send(f"스캔 완료\n데이터 수신: {data_ok}/{len(sp1500)}개\n{date_stat}\n트렌드 통과: {trend_pass}개\n패턴+거래량+RS: {len(res)}개")

    if not res:
        send(f"🇺🇸 미국 스캐너\n최근 {SCAN_DAYS}거래일 | {market_str}\n조건 충족 종목 없음\n(거래량급증+RS양수 기준)")
    else:
        hdr=f"🇺🇸 미너비니 컵&핸들 (미국)\n최근 {SCAN_DAYS}거래일 | {market_str}\n{len(res)}개 발견(RS순)\n"+"─"*24+"\n"
        msg=hdr
        for r in res:
            up=round((r["pivot"]/r["cur"]-1)*100,1)
            past=format_past(r["history"])
            blk=(f"[{r['sig_date']}] [{r['cap']}] {r['sector']}\n"
                 f"{r['ticker']}\n"
                 f"현재가: ${r['cur']:,.2f}\n"
                 f"피벗  : ${r['pivot']:,.2f} ({up:+.1f}%)\n"
                 f"컵:{r['cd']}%/{r['cdays']}일 핸들:{r['hd']}%/{r['hdays']}일\n"
                 f"거래량:{r['vr']}x VOL RS:{r['rs']:+.1f}%\n"
                 +(past+"\n" if past else "")+"\n")
            if len(msg)+len(blk)>4000:
                send(msg);msg="(이어서)\n\n"+blk
            else:msg+=blk
        send(msg)
