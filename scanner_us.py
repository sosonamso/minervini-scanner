"""
미너비니 컵&핸들 스캐너 - 미국 주식 통합버전
- 대상: tickers_us.csv (2800개+)
- 데이터: yfinance
- 결과: 텔레그램 전송 + CSV 저장
"""
import os,time,warnings,requests
import numpy as np,pandas as pd
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")

TOK=os.environ.get("TELEGRAM_TOKEN","")
CID=os.environ.get("TELEGRAM_CHAT_ID","")
SCAN_DAYS=7
HISTORY_DAYS=420
BATCH_SIZE=50

def send(text):
    print(text)
    if TOK:
        try:requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",
                          data={"chat_id":CID,"text":text},timeout=10)
        except:pass

def send_file(filepath,caption=""):
    if TOK:
        try:
            with open(filepath,"rb") as f:
                requests.post(f"https://api.telegram.org/bot{TOK}/sendDocument",
                    data={"chat_id":CID,"caption":caption},
                    files={"document":f},timeout=30)
        except:pass

def get_recent_dates(n=7):
    dates=[];d=datetime.today()
    while len(dates)<n:
        if d.weekday()<5:dates.append(d.strftime("%Y-%m-%d"))
        d-=timedelta(days=1)
        if len(dates)>=n*3:break
    return dates[:n]

def load_tickers():
    """tickers_us.csv 로드. 없으면 Wikipedia S&P1500 fallback"""
    try:
        df=pd.read_csv("tickers_us.csv",encoding="utf-8-sig")
        tickers={}
        for _,row in df.iterrows():
            t=str(row["ticker"]).strip().replace(".","-")
            if not t or t=="nan" or len(t)>6:continue
            tickers[t]={
                "cap":str(row.get("cap","SmallCap")),
                "sector":str(row.get("sector","기타")),
                "name":str(row.get("name",t)),
                "exchange":str(row.get("exchange","NYSE")),
            }
        print(f"tickers_us.csv 로드: {len(tickers)}개")
        return tickers
    except Exception as e:
        print(f"tickers_us.csv 로드 실패({e}) → Wikipedia S&P1500 사용")
        return get_sp1500_fallback()

def get_sp1500_fallback():
    tickers={}
    sources=[
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies","LargeCap",0),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies","MidCap",0),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies","SmallCap",0),
    ]
    for url,cap,tbl_idx in sources:
        try:
            tables=pd.read_html(url)
            df=tables[tbl_idx]
            tick_col=next((c for c in df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()),df.columns[0])
            sec_col=next((c for c in df.columns if "sector" in str(c).lower() or "gics" in str(c).lower()),None)
            for _,row in df.iterrows():
                t=str(row[tick_col]).strip().replace(".","-")
                if not t or t=="nan" or len(t)>6:continue
                sec=str(row[sec_col]).strip() if sec_col else "기타"
                tickers[t]={"cap":cap,"sector":sec,"name":t,"exchange":"NYSE"}
            print(f"{cap}: {len([k for k,v in tickers.items() if v['cap']==cap])}개")
            time.sleep(1)
        except Exception as e:
            print(f"{cap} 실패: {e}")
    return tickers

def batch_download(ticker_list,start,end):
    result={}
    for i in range(0,len(ticker_list),BATCH_SIZE):
        batch=ticker_list[i:i+BATCH_SIZE]
        if i%500==0:print(f"  다운로드 [{i}/{len(ticker_list)}]")
        for attempt in range(3):
            try:
                import yfinance as yf
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
        time.sleep(0.5)
    return result

def check_market(mkt_df):
    if mkt_df is None or len(mkt_df)<200:return True,"데이터부족"
    c=mkt_df["Close"]
    ma200=c.rolling(200).mean()
    cur=float(c.iloc[-1]);ma=float(ma200.iloc[-1])
    if pd.isna(ma):return True,"데이터부족"
    if cur>ma:return True,"상승장(S&P500>200MA)"
    return False,"하락장(S&P500<200MA)"

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

def calc_rs(df,mkt_df):
    def p(d,n):return float(d["Close"].iloc[-1]/d["Close"].iloc[-n]-1)if len(d)>=n else 0.0
    s=sum([0.4,0.2,0.2,0.2][i]*p(df,[63,126,189,252][i])for i in range(4))
    m=sum([0.4,0.2,0.2,0.2][i]*p(mkt_df,[63,126,189,252][i])for i in range(4))
    return round((s-m)*100,1)

def calc_score(rs,vr,cd,hd):
    s=50
    s+=min(30,max(-10,rs*0.3))
    if vr>=3:s+=15
    elif vr>=2:s+=10
    elif vr>=1.4:s+=5
    if 15<=cd<=25:s+=10
    elif 25<=cd<=35:s+=5
    if hd<=7:s+=10
    elif hd<=10:s+=5
    return min(100,max(0,int(s)))

def score_grade(s):
    return "S" if s>=90 else "A" if s>=80 else "B" if s>=70 else "C" if s>=60 else "D"

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
    wins=sum(1 for h in history if h["r20"] is not None and h["r20"]>0)
    total=sum(1 for h in history if h["r20"] is not None)
    lines=[]
    for h in history[-4:]:
        r5s=f"+{h['r5']}%" if h['r5'] and h['r5']>0 else f"{h['r5']}%"
        r20s=f"+{h['r20']}%" if h['r20'] and h['r20']>0 else f"{h['r20']}%"
        tag="OK" if h['r20'] and h['r20']>0 else "NG"
        if h['vs']:tag+="VOL"
        lines.append(f"  {h['date']}: 5일{r5s}/20일{r20s} {tag}")
    if total>0:lines.append(f"  -> 과거{len(history)}회 승률{round(wins/total*100)}%")
    return "\n".join(lines)

def cap_label(cap):
    m={"MegaCap":"초대형","LargeCap":"대형","MidCap":"중형","SmallCap":"소형"}
    return m.get(cap,cap)

def main():
    import yfinance as yf

    end=datetime.today()
    start=(end-timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end_str=end.strftime("%Y-%m-%d")
    sig_dates=get_recent_dates(SCAN_DAYS)
    data_cutoff=pd.Timestamp(sig_dates[0])-timedelta(days=7)
    print(f"탐색날짜: {sig_dates}")

    # SPY 지수
    try:
        mkt_raw=yf.download("SPY",start=start,end=end_str,progress=False,auto_adjust=True)
        if isinstance(mkt_raw.columns,pd.MultiIndex):mkt_raw.columns=mkt_raw.columns.get_level_values(0)
        mkt_df=pd.DataFrame({"Close":pd.to_numeric(mkt_raw["Close"],errors="coerce")}).dropna()
        print(f"SPY {len(mkt_df)}일치 수신")
    except:mkt_df=None

    market_ok,market_str=check_market(mkt_df)

    # 티커 로드
    all_tickers=load_tickers()
    send(f"🇺🇸 미국 스캐너 시작\n최근 {SCAN_DAYS}거래일 | {market_str}\n{len(all_tickers)}개 종목 다운로드 중...\n(약 2~3시간 소요)")
    if not market_ok:
        send("⚠️ S&P500 200MA 하방 - 시그널 신뢰도 낮음!")

    # 배치 다운로드
    all_data=batch_download(list(all_tickers.keys()),start,end_str)

    # 유효 데이터 필터
    valid_data={}
    for ticker,df in all_data.items():
        if df.index[-1]>=data_cutoff:
            valid_data[ticker]=df
    send(f"다운로드 완료: {len(valid_data)}/{len(all_tickers)}개 유효\n패턴 분석 시작...")

    # 패턴 분석
    res=[];trend_pass=0
    for i,(ticker,info) in enumerate(all_tickers.items()):
        if i%500==0:print(f"[{i}/{len(all_tickers)}] 트렌드:{trend_pass} 발견:{len(res)}")
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
            if not pat["vs"]:continue
            rs=calc_rs(sl,mkt_df.loc[:sig_ts])if mkt_df is not None else 0.0
            if rs<=0:continue
            score=calc_score(rs,pat["vr"],pat["cd"],pat["hd"])
            grade=score_grade(score)
            history=get_past_signals(df,sig_ts)
            res.append({
                "sig_date":sig_str,"ticker":ticker,
                "cap":info["cap"],"sector":info["sector"],"name":info.get("name",ticker),
                "cur":pat["cur"],"pivot":pat["pivot"],
                "cd":pat["cd"],"hd":pat["hd"],
                "cdays":pat["cdays"],"hdays":pat["hdays"],
                "vr":pat["vr"],"vs":pat["vs"],
                "rs":rs,"score":score,"grade":grade,"history":history,
            })
            break

    res.sort(key=lambda x:(x["sig_date"],x["rs"]),reverse=True)
    seen=set();deduped=[]
    for r in res:
        if r["ticker"] not in seen:
            seen.add(r["ticker"]);deduped.append(r)
    res=deduped
    print(f"완료: {len(res)}개 발견")
    send(f"스캔 완료\n유효 데이터: {len(valid_data)}/{len(all_tickers)}개\n트렌드 통과: {trend_pass}개\n패턴+거래량+RS: {len(res)}개")

    if not res:
        send(f"🇺🇸 미국 스캐너\n최근 {SCAN_DAYS}거래일 | {market_str}\n조건 충족 종목 없음")
    else:
        hdr=f"🇺🇸 미너비니 컵&핸들 (미국)\n최근 {SCAN_DAYS}거래일 | {market_str}\n{len(res)}개 발견(RS순)\n"+"─"*24+"\n"
        msg=hdr
        for r in res:
            up=round((r["pivot"]/r["cur"]-1)*100,1)
            past=format_past(r["history"])
            h_cnt=len(r["history"])
            h_win=sum(1 for h in r["history"] if h["r20"] and h["r20"]>0)
            hist_str=f"\n[과거이력]\n{past}" if past else ""
            blk=(f"[{r['sig_date']}] [{cap_label(r['cap'])}] {r['sector']}\n"
                 f"◆{r['ticker']}\n"
                 f"  AI점수: {r['score']}점({r['grade']}등급)\n"
                 f"  현재가: ${r['cur']:,.2f}\n"
                 f"  피벗: ${r['pivot']:,.2f} ({up:+.1f}%)\n"
                 f"  컵:{r['cd']}%/{r['cdays']}일 핸들:{r['hd']}%/{r['hdays']}일\n"
                 f"  거래량:{r['vr']}x RS:{r['rs']:+.1f}%"
                 +hist_str+"\n\n")
            if len(msg)+len(blk)>4000:
                send(msg);msg="(이어서)\n\n"+blk
            else:msg+=blk
        send(msg)

    # CSV 저장
    rows=[]
    for r in res:
        h=r["history"]
        h_cnt=len(h);h_win=sum(1 for x in h if x["r20"] and x["r20"]>0)
        h_rate=round(h_win/h_cnt*100) if h_cnt else 0
        h_detail="|".join(f"{x['date']}:{x['r5']}/{x['r20']}" for x in h)
        rows.append({
            "date":r["sig_date"],"ticker":r["ticker"],"cap":r["cap"],
            "sector":r["sector"],"entry":r["cur"],"pivot":r["pivot"],
            "cup_depth":r["cd"],"handle_depth":r["hd"],
            "cup_days":r["cdays"],"handle_days":r["hdays"],
            "vol_ratio":r["vr"],"rs":r["rs"],
            "score":r["score"],"grade":r["grade"],
            "hist_count":h_cnt,"hist_winrate":h_rate,"hist_detail":h_detail,
        })
    if rows:
        pd.DataFrame(rows).to_csv("scanner_us_raw.csv",index=False,encoding="utf-8-sig")
        send_file("scanner_us_raw.csv",f"🇺🇸 미국 스캐너 RAW ({len(rows)}건) {datetime.today().strftime('%Y-%m-%d')}")
    else:
        pd.DataFrame().to_csv("scanner_us_raw.csv",index=False,encoding="utf-8-sig")

if __name__=="__main__":
    main()
