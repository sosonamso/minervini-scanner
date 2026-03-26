"""
미너비니 컵&핸들 백테스트 - KRX API 버전
- 데이터: KRX Open API (안정적, IP차단 없음)
- 출력: RAW CSV + 일별수익률 JSON + 알파분석 + 피크분석
"""
import os,time,warnings,requests,json,statistics
import numpy as np,pandas as pd
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")

TOK=os.environ.get("TELEGRAM_TOKEN","")
CID=os.environ.get("TELEGRAM_CHAT_ID","")
KRX=os.environ.get("KRX_TOKEN","")

LOOKBACK_DAYS=1500  # 시그널 탐색 기간
HISTORY_DAYS=2100   # 데이터 수집 기간
MAX_HOLD=90         # 최대 보유일
_row_meta={}        # ticker -> {name, sector}

def send_file(filepath, caption=""):
    if TOK:
        try:
            with open(filepath,"rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{TOK}/sendDocument",
                    data={"chat_id":CID,"caption":caption},
                    files={"document":f},
                    timeout=30
                )
        except Exception as e:
            print(f"파일 전송 실패: {e}")

def send(text):
    print(text)
    if TOK:
        try:requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",
                          data={"chat_id":CID,"text":text},timeout=10)
        except:pass

# ─────────────────────────────────────
# KRX API
# ─────────────────────────────────────
def get_krx_data(date_str,market="KOSPI"):
    if market=="KOSPI":
        url="https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd"
    else:
        url="https://data-dbg.krx.co.kr/svc/apis/sto/ksq_bydd_trd"
    headers={"AUTH_KEY":KRX.strip(),"Content-Type":"application/json","Accept":"application/json"}
    for attempt in range(3):
        try:
            resp=requests.post(url,headers=headers,json={"basDd":date_str},timeout=30)
            if resp.status_code!=200:return {}
            block=resp.json().get("OutBlock_1",[])
            if not block:return {}
            result={}
            for row in block:
                try:
                    ticker=str(row.get("ISU_CD","")).strip()
                    if not ticker:continue
                    _row_meta[ticker]={
                        "name":str(row.get("ISU_NM","")).strip(),
                        "sector":str(row.get("SECT_TP_NM","기타")).strip() or "기타"
                    }
                    result[ticker]={
                        "Close":float(str(row.get("TDD_CLSPRC","0")).replace(",","")),
                        "Volume":float(str(row.get("ACC_TRDVOL","0")).replace(",","")),
                    }
                except:pass
            return result
        except Exception as e:
            print(f"KRX 오류(시도{attempt+1}): {e}")
            time.sleep(2*(attempt+1))
    return {}

def get_krx_index(date_str,index_name="코스피"):
    url="https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd"
    headers={"AUTH_KEY":KRX.strip(),"Content-Type":"application/json"}
    try:
        resp=requests.post(url,headers=headers,json={"basDd":date_str},timeout=30)
        if resp.status_code!=200:return None,None
        block=resp.json().get("OutBlock_1",[])
        for row in block:
            if str(row.get("IDX_NM",""))==index_name:
                return float(str(row.get("CLSPRC_IDX","0")).replace(",","")),index_name
    except:pass
    return None,None

def get_trading_dates(days):
    dates=[]
    d=datetime.today()
    while len(dates)<days:
        if d.weekday()<5:
            dates.append(d.strftime("%Y%m%d"))
        d-=timedelta(days=1)
        if (datetime.today()-d).days>days*2:break
    return list(reversed(dates))

# ─────────────────────────────────────
# OHLCV 구축
# ─────────────────────────────────────
def build_ohlcv(trading_dates):
    ticker_data={}
    total=len(trading_dates)
    for i,date_str in enumerate(trading_dates):
        if i%100==0:print(f"데이터 수집 [{i}/{total}] {date_str}")
        for mkt in ["KOSPI","KOSDAQ"]:
            day_data=get_krx_data(date_str,mkt)
            for ticker,ohlcv in day_data.items():
                if ticker not in ticker_data:
                    ticker_data[ticker]={"market":mkt,"rows":[]}
                ticker_data[ticker]["rows"].append({
                    "date":pd.Timestamp(date_str),
                    **ohlcv
                })
            time.sleep(0.3)
    result={}
    for ticker,info in ticker_data.items():
        rows=info["rows"]
        if len(rows)<200:continue
        df=pd.DataFrame(rows).set_index("date").sort_index()
        df=df[["Close","Volume"]].astype(float)
        df=df[df["Close"]>0].dropna()
        if len(df)>=200:
            meta=_row_meta.get(ticker,{})
            result[ticker]={"market":info["market"],"df":df,
                           "name":meta.get("name",ticker),
                           "sector":meta.get("sector","기타")}
    return result

def build_index(all_ohlcv):
    """ETF 데이터로 지수 대체 (KODEX200, KODEX코스닥150)"""
    kospi_df=None;kosdaq_df=None
    for ticker,info in all_ohlcv.items():
        name=info.get("name","")
        if kospi_df is None and "KODEX" in name and "200" in name and "레버리지" not in name and "인버스" not in name:
            kospi_df=info["df"][["Close"]].copy()
            print(f"코스피 지수 ETF: {name}({ticker}) {len(kospi_df)}일치")
        if kosdaq_df is None and "KODEX" in name and "코스닥" in name and "레버리지" not in name and "인버스" not in name:
            kosdaq_df=info["df"][["Close"]].copy()
            print(f"코스닥 지수 ETF: {name}({ticker}) {len(kosdaq_df)}일치")
        if kospi_df is not None and kosdaq_df is not None:
            break
    # ETF 없으면 시장 평균으로 대체
    if kospi_df is None:
        kospi_tickers=[t for t,v in all_ohlcv.items() if v["market"]=="KOSPI"][:50]
        closes=pd.concat([all_ohlcv[t]["df"]["Close"].rename(t) for t in kospi_tickers],axis=1)
        kospi_df=closes.mean(axis=1).to_frame("Close")
        print(f"코스피 지수(평균): {len(kospi_df)}일치")
    if kosdaq_df is None:
        kosdaq_tickers=[t for t,v in all_ohlcv.items() if v["market"]=="KOSDAQ"][:50]
        closes=pd.concat([all_ohlcv[t]["df"]["Close"].rename(t) for t in kosdaq_tickers],axis=1)
        kosdaq_df=closes.mean(axis=1).to_frame("Close")
        print(f"코스닥 지수(평균): {len(kosdaq_df)}일치")
    return kospi_df,kosdaq_df

# ─────────────────────────────────────
# 미너비니 로직
# ─────────────────────────────────────
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
    idx=df.index
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
    if rh>lh*1.15:return False,{}
    hnd=c[ri:];hl=len(hnd)
    if not(5<=hl<=20):return False,{}
    hlow=float(np.min(hnd));hd=(rh-hlow)/rh
    if not(0.05<=hd<=0.15):return False,{}
    if(hlow-bot)/(lh-bot)<0.60:return False,{}
    cur=cl[-1]
    if not(rh*0.97<=cur<=rh*1.05):return False,{}
    vr=float(np.mean(v[-5:]))/float(np.mean(v[-40:-5]))if len(v)>=40 else 1.0
    try:
        start_pos=n-len(c)
        cup_start=idx[start_pos+li].strftime("%y.%m.%d")
        cup_end=idx[start_pos+ri].strftime("%y.%m.%d")
    except:
        cup_start="";cup_end=""
    return True,{"cd":round(cd*100,1),"hd":round(hd*100,1),"cdays":ri-li,"hdays":hl,
                 "pivot":round(float(rh),0),"cur":round(float(cur),0),
                 "vr":round(vr,2),"vs":vr>=1.40,
                 "cup_start":cup_start,"cup_end":cup_end}

def calc_rs(df,mkt_df):
    def p(d,n):return float(d["Close"].iloc[-1]/d["Close"].iloc[-n]-1)if len(d)>=n else 0.0
    s=sum([0.4,0.2,0.2,0.2][i]*p(df,[63,126,189,252][i])for i in range(4))
    m=sum([0.4,0.2,0.2,0.2][i]*p(mkt_df,[63,126,189,252][i])for i in range(4))
    return round((s-m)*100,1)

def calc_score(rs,vr,cd,hd):
    if rs>=25:s_rs=100
    elif rs>=15:s_rs=80
    elif rs>=10:s_rs=60
    elif rs>=5:s_rs=40
    else:s_rs=20
    if vr>=3.0:s_vr=100
    elif vr>=2.5:s_vr=85
    elif vr>=2.0:s_vr=70
    elif vr>=1.7:s_vr=55
    else:s_vr=40
    if 20<=cd<=35:s_cd=100
    elif 15<=cd<20 or 35<cd<=40:s_cd=75
    elif 40<cd<=50:s_cd=50
    else:s_cd=30
    if 5<=hd<=10:s_hd=100
    elif 10<hd<=12:s_hd=75
    elif hd>12:s_hd=50
    else:s_hd=60
    return round(s_rs*0.40+s_vr*0.35+s_cd*0.15+s_hd*0.10)

# ─────────────────────────────────────
# 메인
# ─────────────────────────────────────
if __name__=="__main__":
    if not KRX:
        send("KRX_TOKEN 없음!")
        exit(1)

    end=datetime.today()
    signal_start=end-timedelta(days=LOOKBACK_DAYS)
    signal_dates=set(pd.bdate_range(signal_start,end).map(pd.Timestamp))

    send(f"백테스트 시작 (KRX API)\n기간: 최근 {LOOKBACK_DAYS}일\n데이터 {HISTORY_DAYS}일치 수집 중...\n(약 40~50분 소요)")

    # 데이터 수집
    trading_dates=get_trading_dates(HISTORY_DAYS)
    print(f"수집 대상: {len(trading_dates)}거래일")
    all_ohlcv=build_ohlcv(trading_dates)
    print(f"종목 구축 완료: {len(all_ohlcv)}개")

    # 지수 수집
    send("지수 데이터 수집 중...")
    kospi_df,kosdaq_df=build_index(all_ohlcv)

    send(f"데이터 수집 완료: {len(all_ohlcv)}개 종목\n패턴 분석 시작...")

    # 백테스트
    all_signals=[]
    for i,(ticker,info) in enumerate(all_ohlcv.items()):
        if i%200==0:print(f"[{i}/{len(all_ohlcv)}] 시그널:{len(all_signals)}건")
        df=info["df"]
        mkt=info["market"]
        mkt_df=kospi_df  # 코스닥도 일단 코스피로 비교
        idx=df.index.tolist()

        for j,sig_ts in enumerate(idx):
            if sig_ts not in signal_dates:continue
            sl=df.iloc[:j+1]
            if not check_trend(sl):continue
            ok,pat=detect(sl)
            if not ok or not pat["vs"]:continue
            if mkt_df is None:continue
            # 시장별 지수 선택 (KOSPI→kospi_df, KOSDAQ→kosdaq_df)
            idx_df=kospi_df if mkt=="KOSPI" else kosdaq_df
            if idx_df is not None and len(idx_df)>10:
                try:
                    rs=calc_rs(sl,idx_df.loc[:sig_ts])
                except:rs=0.0
                if rs<=0:continue  # RS 필터 (지수 있을 때만)
            else:
                rs=0.0  # 지수 없으면 RS 필터 스킵

            entry=float(df["Close"].iloc[j])
            score=calc_score(rs,pat["vr"],pat["cd"],pat["hd"])

            # 1~90일 수익률
            daily_r={}
            for hold in range(1,MAX_HOLD+1):
                fi=j+hold
                if fi<len(idx):
                    daily_r[hold]=round((float(df["Close"].iloc[fi])/entry-1)*100,2)
                else:
                    daily_r[hold]=None

            # 알파 계산 (5/20/60일)
            alpha={}
            try:
                idx_df2=kospi_df if mkt=="KOSPI" else kosdaq_df
                if idx_df2 is not None and sig_ts in idx_df2.index:
                    mkt_idx=idx_df2.index.tolist()
                    mkt_j=mkt_idx.index(sig_ts)
                    mkt_entry=float(idx_df2["Close"].iloc[mkt_j])
                    for hold in [5,20,60]:
                        mkt_fi=mkt_j+hold
                        if mkt_fi<len(mkt_idx):
                            mkt_r=(float(idx_df2["Close"].iloc[mkt_fi])/mkt_entry-1)*100
                            sr=daily_r.get(hold)
                            if sr is not None:
                                alpha[hold]=round(sr-mkt_r,2)
            except:pass

            all_signals.append({
                "date":sig_ts.strftime("%Y-%m-%d"),
                "ticker":ticker,
                "name":info["name"],
                "market":mkt,
                "sector":info["sector"],
                "entry":entry,
                "pivot":pat["pivot"],
                "cup_depth":pat["cd"],"handle_depth":pat["hd"],
                "cup_days":pat["cdays"],"handle_days":pat["hdays"],
                "cup_start":pat.get("cup_start",""),"cup_end":pat.get("cup_end",""),
                "vol_ratio":pat["vr"],
                "rs":rs,"score":score,
                "r5":daily_r.get(5),"r20":daily_r.get(20),"r60":daily_r.get(60),
                "alpha5":alpha.get(5),"alpha20":alpha.get(20),"alpha60":alpha.get(60),
                "daily_returns":daily_r,
            })

    print(f"백테스트 완료: {len(all_signals)}건")

    # RAW 저장
    rows=[]
    for s in all_signals:
        row={k:v for k,v in s.items() if k!="daily_returns"}
        for hold in [1,3,5,10,15,20,30,40,50,60,75,90]:
            row[f"r{hold}"]=s["daily_returns"].get(hold)
        rows.append(row)

    raw_df=pd.DataFrame(rows)
    raw_df.to_csv("backtest_raw.csv",index=False,encoding="utf-8-sig")

    daily_json=[{"date":s["date"],"ticker":s["ticker"],"name":s["name"],
                 "score":s["score"],"daily_returns":s["daily_returns"]}
                for s in all_signals]
    with open("backtest_daily.json","w",encoding="utf-8")as f:
        json.dump(daily_json,f,ensure_ascii=False,indent=2)

    print("RAW 저장 완료")
    send_file("backtest_raw.csv", f"📊 국장 백테스트 RAW ({len(all_signals)}건)")
    send_file("backtest_daily.json", f"📋 국장 백테스트 일별수익률 ({len(all_signals)}건)")

    if not all_signals:
        send("시그널 없음 — 조건 충족 종목이 없어요.")
    else:
        df=pd.DataFrame(rows)

        # 기본 요약
        lines=["백테스트 결과 (KRX API)",f"총 시그널: {len(df)}건","─"*28]
        for col,label in [("r5","5일"),("r20","20일"),("r60","60일")]:
            vals=df[col].dropna()
            if len(vals)==0:continue
            win=sum(1 for v in vals if v>0)
            lines.append(f"[{label}] n={len(vals)}")
            lines.append(f"  평균:{vals.mean():+.1f}% 중앙:{statistics.median(vals):+.1f}%")
            lines.append(f"  승률:{round(win/len(vals)*100,1)}%")
        send("\n".join(lines))

        # 알파 요약
        lines2=["vs KOSPI 알파 분석","─"*28]
        for col,label in [("alpha5","5일"),("alpha20","20일"),("alpha60","60일")]:
            vals=df[col].dropna()
            if len(vals)==0:continue
            beat=sum(1 for v in vals if v>0)
            lines2.append(f"[{label}] n={len(vals)}")
            lines2.append(f"  평균알파:{vals.mean():+.1f}% 중앙:{statistics.median(vals):+.1f}%")
            lines2.append(f"  지수초과확률:{round(beat/len(vals)*100,1)}%")
        send("\n".join(lines2))

        # 점수별 수익률
        lines3=["점수등급별 20일 수익률","─"*28]
        for grade,label in [("S","S(90+)"),("A","A(80-89)"),("B","B(70-79)"),("C","C(60-69)")]:
            grade_map={"S":(90,100),"A":(80,89),"B":(70,79),"C":(60,69)}
            lo,hi=grade_map[grade]
            sub=df[(df["score"]>=lo)&(df["score"]<=hi)]["r20"].dropna()
            if len(sub)==0:continue
            lines3.append(f"[{label}] n={len(sub)}")
            lines3.append(f"  평균:{sub.mean():+.1f}% 중앙:{statistics.median(sub):+.1f}%")
        send("\n".join(lines3))

        # 피크 분석
        all_daily_data=[]
        for s in all_signals:
            dr=s["daily_returns"]
            vals=[(h,r)for h,r in dr.items()if r is not None]
            if vals:
                peak=max(vals,key=lambda x:x[1])
                all_daily_data.append(peak[0])
        if all_daily_data:
            lines4=["최고수익 시점 분석","─"*28,
                    f"평균 피크: {round(sum(all_daily_data)/len(all_daily_data),1)}거래일",
                    f"중앙값 피크: {statistics.median(all_daily_data):.0f}거래일"]
            for lo,hi in [(1,5),(6,10),(11,20),(21,30),(31,45),(46,60),(61,90)]:
                cnt=sum(1 for d in all_daily_data if lo<=d<=hi)
                pct=round(cnt/len(all_daily_data)*100)
                lines4.append(f"  {lo:2d}~{hi:2d}일: {cnt}건({pct}%)")
            send("\n".join(lines4))

        # 일별 수익률 커브
        curve=["일별 수익률 커브","─"*28]
        for hold in [1,3,5,7,10,15,20,25,30,40,50,60,75,90]:
            col=f"r{hold}"
            if col in df.columns:
                vals=df[col].dropna()
                if len(vals)>5:
                    curve.append(f"  {hold:2d}일: 평균{vals.mean():+.1f}% 중앙{statistics.median(vals):+.1f}%")
        send("\n".join(curve))
