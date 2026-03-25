"""
미너비니 컵&핸들 스캐너 - 미국 주식 (Massive API + tickers_us.csv)
- 대상: tickers_us.csv (2800개+)
- 데이터: Massive.com (구 Polygon.io) REST API
- 결과: 텔레그램 전송 + CSV 저장
"""
import os,time,warnings,requests
import numpy as np,pandas as pd
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")

TOK=os.environ.get("TELEGRAM_TOKEN","")
CID=os.environ.get("TELEGRAM_CHAT_ID","")
MASSIVE=os.environ.get("MASSIVE_TOKEN","")
SCAN_DAYS=7
HISTORY_DAYS=420

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
        except Exception as e:
            print(f"파일 전송 실패: {e}")

def get_recent_dates(n=7):
    dates=[];d=datetime.today()
    while len(dates)<n:
        if d.weekday()<5:dates.append(d.strftime("%Y-%m-%d"))
        d-=timedelta(days=1)
        if len(dates)>=n*3:break
    return dates[:n]

def load_tickers():
    """tickers_us.csv에서 티커 로드"""
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
            }
        print(f"tickers_us.csv 로드: {len(tickers)}개")
        return tickers
    except Exception as e:
        print(f"tickers_us.csv 로드 실패({e})")
        return {}

def get_massive_ohlcv(ticker,start,end):
    url=f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params={"adjusted":"true","sort":"asc","limit":50000,"apiKey":MASSIVE}
    for attempt in range(2):
        try:
            resp=requests.get(url,params=params,timeout=30)
            if resp.status_code==429:
                time.sleep(12);continue
            if resp.status_code!=200:return None
            data=resp.json()
            results=data.get("results",[])
            if not results or len(results)<100:return None
            df=pd.DataFrame(results)
            df["date"]=pd.to_datetime(df["t"],unit="ms").dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
            df=df.set_index("date").sort_index()
            df=df.rename(columns={"c":"Close","v":"Volume"})
            df=df[["Close","Volume"]].astype(float).dropna()
            if len(df)>=100:return df
        except Exception as e:
            time.sleep(3)
    return None

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

    # ① 기간 확장: 500일
    c=cl[-min(500,n):];v=vl[-min(500,n):];w=len(c)

    # ② 급등 필터: 최근 20일 +40% 이상이면 제외 (이벤트성 급등)
    if len(c)>=20:
        recent_gain=(c[-1]-c[-20])/c[-20]
        if recent_gain>0.40:return False,{}

    # ③ 로컬 최고점 후보 탐색
    peak_candidates=[]
    for i in range(10, w-55):
        lo=max(0,i-10);hi=min(w,i+10)
        if c[i]==np.max(c[lo:hi]) and c[i]==np.max(c[max(0,i-5):min(w,i+5)]):
            peak_candidates.append(i)

    if not peak_candidates:
        peak_candidates=[int(np.argmax(c[:w//2]))]

    # ④ 모든 후보 검사 → 가장 긴 컵 패턴 선택
    best=None
    best_cup_days=0

    for li in peak_candidates:
        lh=c[li]
        cup=c[li:]
        if len(cup)<55:continue

        bi=li+int(np.argmin(cup))
        bot=c[bi];cd=(lh-bot)/lh
        cup_days=bi-li

        if not(0.20<=cd<=0.50):continue
        if cup_days<35:continue

        rc=c[bi:]
        if len(rc)<10:continue
        ri=bi+int(np.argmax(rc));rh=c[ri]

        if rh<lh*0.90:continue

        hnd=c[ri:];hl=len(hnd)
        if not(5<=hl<=20):continue

        hlow=float(np.min(hnd));hd=(rh-hlow)/rh
        if not(0.05<=hd<=0.15):continue

        if(hlow-bot)/(lh-bot)<0.60:continue

        cur=cl[-1]
        if not(rh*0.97<=cur<=rh*1.05):continue

        vr=float(np.mean(v[-5:]))/float(np.mean(v[-40:-5]))if len(v)>=40 else 1.0

        # 가장 긴 컵 선택
        if cup_days>best_cup_days:
            best_cup_days=cup_days
            best={"cd":round(cd*100,1),"hd":round(hd*100,1),"cdays":cup_days,"hdays":hl,
                  "pivot":round(float(rh),2),"cur":round(float(cur),2),
                  "vr":round(vr,2),"vs":vr>=1.40}

    if best is None:return False,{}
    return True,best

def calc_rs(df,mkt):
    def p(d,n):return float(d["Close"].iloc[-1]/d["Close"].iloc[-n]-1)if len(d)>=n else 0.0
    s=sum([0.4,0.2,0.2,0.2][i]*p(df,[63,126,189,252][i])for i in range(4))
    m=sum([0.4,0.2,0.2,0.2][i]*p(mkt,[63,126,189,252][i])for i in range(4))
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
        r5s=f"{h['r5']:+.1f}%"if h["r5"] is not None else"-"
        r20s=f"{h['r20']:+.1f}%"if h["r20"] is not None else"미완"
        tag="OK"if(h["r20"] is not None and h["r20"]>0)else("..."if h["r20"] is None else"NG")
        if h["vs"]:tag+="VOL"
        lines.append(f"  {h['date']}: 5일{r5s}/20일{r20s} {tag}")
    if total>0:lines.append(f"  -> 과거{len(history)}회 승률{round(wins/total*100)}%")
    return "\n".join(lines)

def cap_label(cap):
    m={"MegaCap":"초대형","LargeCap":"대형","MidCap":"중형","SmallCap":"소형"}
    return m.get(cap,cap)

if __name__=="__main__":
    if not MASSIVE:
        send("MASSIVE_TOKEN이 없어요! GitHub Secrets 확인해주세요.")
        exit(1)

    end=datetime.today()
    start=(end-timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end_str=end.strftime("%Y-%m-%d")
    sig_dates=get_recent_dates(SCAN_DAYS)
    data_cutoff=pd.Timestamp(sig_dates[0])-timedelta(days=7)
    print(f"탐색날짜: {sig_dates}")

    # SPY 시장 상태
    mkt_df=get_massive_ohlcv("SPY",start,end_str)
    market_ok,market_str=check_market(mkt_df)

    # 티커 로드
    all_tickers=load_tickers()
    if not all_tickers:
        send("tickers_us.csv 없음! 레포에 파일 확인해주세요.")
        exit(1)

    send(f"🇺🇸 미국 스캐너 시작 (Massive)\n최근 {SCAN_DAYS}거래일 | {market_str}\n{len(all_tickers)}개 종목 수집 중...")
    if not market_ok:
        send("⚠️ S&P500 200MA 하방 - 시그널 신뢰도 낮음!")

    # 데이터 수집
    valid_data={};data_ok=0;last_dates=[]
    ticker_list=list(all_tickers.keys())
    for i,ticker in enumerate(ticker_list):
        if i%200==0:print(f"[{i}/{len(ticker_list)}] 수신:{data_ok}")
        df=get_massive_ohlcv(ticker,start,end_str)
        if df is None:continue
        if df.index[-1]<data_cutoff:continue
        valid_data[ticker]=df
        data_ok+=1
        last_dates.append(df.index[-1])
        time.sleep(0.05)

    if last_dates:
        last_dates.sort()
        median_date=last_dates[len(last_dates)//2].strftime("%Y-%m-%d")
        date_stat=f"데이터 기준일: {median_date}(중앙)"
    else:
        date_stat="데이터 기준일: 없음"

    send(f"다운로드 완료\n수신: {data_ok}/{len(ticker_list)}개\n{date_stat}\n패턴 분석 시작...")

    # 패턴 분석
    res=[];trend_pass=0;all_scores=[]
    for i,ticker in enumerate(ticker_list):
        if i%500==0:print(f"[{i}/{len(ticker_list)}] 트렌드:{trend_pass} 발견:{len(res)}")
        df=valid_data.get(ticker)
        if df is None:continue
        info=all_tickers[ticker]
        for sig_str in sig_dates:
            sig_ts=pd.Timestamp(sig_str)
            if sig_ts not in df.index:continue
            pos=df.index.tolist().index(sig_ts)
            sl=df.iloc[:pos+1]
            trend_ok=check_trend(sl)
            cur=float(sl["Close"].iloc[-1])
            rs=calc_rs(sl,mkt_df.loc[:sig_ts])if mkt_df is not None else 0.0

            if not trend_ok:
                all_scores.append({
                    "ticker":ticker,"name":info.get("name",ticker),
                    "cap":info["cap"],"sector":info["sector"],
                    "exchange":info.get("exchange","NYSE"),
                    "cur":round(cur,2),"rs":rs,
                    "trend_ok":False,"pattern_ok":False,"signal":False,
                    "score":0,"grade":"D","pivot":0,
                    "cup_depth":0,"handle_depth":0,"vol_ratio":0,
                    "cup_days":0,"handle_days":0,
                    "reason":"트렌드 미통과","pct_from_pivot":0,"safety":""
                })
                break

            trend_pass+=1
            ok,pat=detect(sl)

            if not ok:
                score=calc_score(rs,1.0,0,0)
                all_scores.append({
                    "ticker":ticker,"name":info.get("name",ticker),
                    "cap":info["cap"],"sector":info["sector"],
                    "exchange":info.get("exchange","NYSE"),
                    "cur":round(cur,2),"rs":rs,
                    "trend_ok":True,"pattern_ok":False,"signal":False,
                    "score":score,"grade":score_grade(score),"pivot":0,
                    "cup_depth":0,"handle_depth":0,"vol_ratio":0,
                    "cup_days":0,"handle_days":0,
                    "reason":"패턴 미감지","pct_from_pivot":0,"safety":""
                })
                break

            score=calc_score(rs,pat["vr"],pat["cd"],pat["hd"])
            grade=score_grade(score)
            pivot=pat["pivot"]
            pct=round((cur-pivot)/pivot*100,1) if pivot>0 else 0
            if cur>=pivot*0.93:safety="safe"
            elif cur>=pivot*0.90:safety="caution"
            else:safety="danger"
            signal=pat["vs"] and rs>0

            all_scores.append({
                "ticker":ticker,"name":info.get("name",ticker),
                "cap":info["cap"],"sector":info["sector"],
                "exchange":info.get("exchange","NYSE"),
                "cur":pat["cur"],"rs":rs,
                "trend_ok":True,"pattern_ok":True,"signal":signal,
                "score":score,"grade":grade,"pivot":pivot,
                "cup_depth":pat["cd"],"handle_depth":pat["hd"],
                "vol_ratio":pat["vr"],"cup_days":pat["cdays"],
                "handle_days":pat["hdays"],
                "reason":"시그널" if signal else ("거래량 미충족" if not pat["vs"] else "RS 미충족"),
                "pct_from_pivot":pct,"safety":safety
            })

            if not signal:break

            history=get_past_signals(df,sig_ts)
            res.append({
                "sig_date":sig_str,"ticker":ticker,
                "cap":info["cap"],"sector":info["sector"],
                "exchange":info.get("exchange","NYSE"),
                "cur":pat["cur"],"pivot":pivot,
                "cd":pat["cd"],"hd":pat["hd"],
                "cdays":pat["cdays"],"hdays":pat["hdays"],
                "vr":pat["vr"],"vs":pat["vs"],
                "rs":rs,"score":score,"grade":grade,"history":history,
            })
            break

    res.sort(key=lambda x:(x["sig_date"],x["score"],x["rs"]),reverse=True)
    seen=set();deduped=[]
    for r in res:
        if r["ticker"] not in seen:
            seen.add(r["ticker"]);deduped.append(r)
    res=deduped
    print(f"완료: {len(res)}개 발견")
    send(f"스캔 완료\n수신: {data_ok}/{len(ticker_list)}개\n{date_stat}\n트렌드 통과: {trend_pass}개\n패턴+거래량+RS: {len(res)}개")

    # CSV 저장
    rows=[]
    for r in res:
        h=r["history"]
        h_cnt=len(h);h_win=sum(1 for x in h if x["r20"] and x["r20"]>0)
        h_rate=round(h_win/h_cnt*100)if h_cnt else 0
        h_detail="|".join(f"{x['date']}:{x['r5']}/{x['r20']}"for x in h)
        rows.append({
            "date":r["sig_date"],"ticker":r["ticker"],
            "cap":r["cap"],"sector":r["sector"],
            "exchange":r.get("exchange","NYSE"),
            "entry":r["cur"],"pivot":r["pivot"],
            "cup_depth":r["cd"],"handle_depth":r["hd"],
            "cup_days":r["cdays"],"handle_days":r["hdays"],
            "vol_ratio":r["vr"],"rs":r["rs"],
            "score":r["score"],"grade":r["grade"],
            "hist_count":h_cnt,"hist_winrate":h_rate,"hist_detail":h_detail,
        })

    pd.DataFrame(rows if rows else []).to_csv("scanner_us_raw.csv",index=False,encoding="utf-8-sig")
    if rows:
        send_file("scanner_us_raw.csv",f"🇺🇸 미장 스캐너 RAW ({len(rows)}건) {datetime.today().strftime('%Y-%m-%d')}")

    # 전체 종목 점수 저장 (종목 검색용)
    pd.DataFrame(all_scores if all_scores else []).to_csv("scanner_us_all.csv",index=False,encoding="utf-8-sig")
    print(f"전체 종목 점수 저장: scanner_us_all.csv ({len(all_scores)}건)")

    if not res:
        send(f"🇺🇸 미국 스캐너\n최근 {SCAN_DAYS}거래일 | {market_str}\n조건 충족 종목 없음")
    else:
        hdr=f"🇺🇸 미너비니 컵&핸들(미국)\n최근 {SCAN_DAYS}거래일 | {market_str}\n{len(res)}개 발견(점수순)\n"+"─"*24+"\n"
        msg=hdr
        for r in res:
            up=round((r["pivot"]/r["cur"]-1)*100,1)
            grade_emoji={"S":"🏆","A":"🥇","B":"🥈","C":"🥉","D":"📊"}.get(r["grade"],"📊")
            past=format_past(r["history"])
            blk=(f"[{r['sig_date']}] [{cap_label(r['cap'])}] {r['sector']}\n"
                 f"◆{r['ticker']}\n"
                 f"  AI점수: {grade_emoji}{r['score']}점({r['grade']}등급)\n"
                 f"  현재가: ${r['cur']:,.2f}\n"
                 f"  피벗: ${r['pivot']:,.2f} ({up:+.1f}%)\n"
                 f"  컵:{r['cd']}%/{r['cdays']}일 핸들:{r['hd']}%/{r['hdays']}일\n"
                 f"  거래량:{r['vr']}x🔥 RS:{r['rs']:+.1f}%\n"
                 +(past+"\n"if past else"")+"\n")
            if len(msg)+len(blk)>4000:
                send(msg);msg="(이어서)\n\n"+blk
            else:msg+=blk
        send(msg)
