"""
미너비니 컵&핸들 스캐너 - 미국 소형주 (Massive API)
- 대상: SmallCap600 + Russell 2000 (IWM 구성종목)
- 시장 필터: SPY 정배열 (50MA > 150MA > 200MA, 200MA 상승)
- 데이터: Massive.com REST API
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

# S&P SmallCap600
SP600 = [
    "ACAD","ACLS","ADMA","ADUS","AEHR","AHCO","AIRC","ALCO","ALGT","ALRM","AMBC","AMEH","AMKR","AMMO","AMSC","ANF","ANGO","AORT","AOSL","APAM","APPN","APLE","ARKO","ARLO","ARRY","ARWR","ASLE","ASND","ASPS","ASRT","ASTE","ATEN","ATNI","ATSG","AUBN","AUPH","AVNW","AWI","AXNX","AXSM","BAND","BANF","BANR","BCEL","BCPC","BFS","BGFV","BKD","BLDR","BLFS","BLNK","BMTC","BNL","BOOT","BPOP","BRKL","BRSP","BSIG","BSRR","BSVN","BURL","BUSE","BZH","CACC","CAKE","CALM","CARA","CARE","CARS","CASH","CASY","CATO","CBAN","CBRL","CBSH","CC","CCNE","CDMO","CDNA","CDRE","CEIX","CENT","CENX","CEVA","CFFI","CFFN","CHCO","CHDN","CHEF","CHUY","CIVB","CLAR","CLB","CLBK","CLDT","CLNE","CLPR","CMAX","CMCO","COHU","COLM","COOP","COUR","CPRI","CPRX","CRAI","CRDX","CRK","CRSP","CRVL","CSGS","CSII","CSTR","CTBI","CTLP","CTMX","CTRE","CTRN","CUBI","CULP","CUTR","CW","CWCO","DAKT","DAVA","DCOM","DFIN","DGII","DH","DHIL","DINO","DIOD","DK","DKL","DLX","DNOW","DORM","DRH","DRVN","DSP","DXC","DXPE","DXYN","EAF","EARN","EBC","EBMT","EFC","EGP","EIG","ELAN","ELY","EPC","EPRT","ESS","ESTE","EVTC","EXLS","EXPI","EXTN","EYE","EZPW","FARO","FBNC","FBRT","FCFS","FELE","FFBC","FFIN","FN","FORM","FOUR","FRAF","FRME","FULT","GBX","GCI","GFF","GHC","GKOS","GMS","GOLF","GOOD","GPX","GRPN","GSBC","GTLS","GWRE","HAFC","HAIN","HASI","HBI","HBT","HCSG","HFWA","HIBB","HIMS","HLNE","HMN","HNI","HOFT","HOMB","HQY","HRMY","HSII","HTH","HTLD","HTLF","HUBG","HWC","HZO","IART","IBCP","IBOC","IBTX","IDCC","IDEX","IESC","IIIN","IMKTA","IMXI","INDB","INFU","INGN","INMD","INSP","INSW","IRBT","IRWD","ISBA","ITRI","JACK","JBLU","JBSS","JELD","JOUT","JWN","KAI","KALU","KFY","KMPR","KNSA","KREF","KRNT","KSS","KTOS","KW","LADR","LAUR","LCNB","LGIH","LGND","LKFN","LMB","LNTH","LOPE","LSTR","LWAY","LXP","MAIN","MATV","MBIN","MBUU","MCBS","MCF","MCRI","MDGL","MED","MEDP","MEI","MERC","MFIN","MGY","MKSI","MLAB","MLKN","MMI","MMSI","MNRO","MOD","MOFG","MRC","MRCY","MSEX","MSTR","MTG","MTSI","MTRN","MTRX","MVBF","MYR","MYRG","NATH","NBTB","NCOM","NCNO","NEO","NEOG","NFBK","NHC","NMIH","NNBR","NPO","NRC","NRIM","NRP","NTST","NVT","NVTS","NWE","NXST","OBK","OCFC","OCSL","OFG","OGE","OII","OIS","ONTO","OPBK","OPCH","OSPN","OTTR","OUT","OXM","PAHC","PATK","PBF","PBPB","PDCO","PEGA","PENN","PFBC","PFIS","PFSI","PGNY","PHR","PKST","PLBC","PLMR","PLNT","PNM","POOL","POWL","PPBI","PRAA","PRDO","PRGO","PRIM","PRK","PRKS","PSN","PTCT","PTEN","PTLO","PTVE","PUMP","QCRH","QDEL","QGEN","QTWO","RBC","RCKT","RCKY","RCM","RDNT","RES","REVG","RGEN","RGP","RICK","RILY","RMBS","RMNI","RMR","RPRX","RRR","RUSHA","RWT","RYAM","SAFE","SANM","SASR","SBCF","SBSI","SCVL","SEIC","SF","SFST","SHAK","SHBI","SHO","SIG","SIGI","SIT","SITM","SKX","SLG","SLGN","SLVM","SM","SMBC","SNCY","SNDR","SNEX","SOFI","SPFI","SPOK","SRC","SRI","SSBK","SSRM","STBA","STLD","STRA","SUPN","SWX","SYBT","SYNA","TALO","TBNK","TCBK","TCMD","TEX","TFSL","TGI","TILE","TNDM","TOWN","TPIC","TREX","TRNO","TRMK","TROW","TRUP","TTGT","TTMI","TWI","TXRH","UBCP","UCBI","UFPI","ULCC","UNFI","UNVR","UPST","USPH","UVV","VBTX","VCEL","VCYT","VECO","VICR","VIRT","VLGEA","VRTS","VSAT","VSEC","WAFD","WASH","WD","WDFC","WERN","WFRD","WINA","WMS","WOOF","WOR","WPC","WSFS","WTS","XRX"
]

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

def get_recent_dates(n=7):
    dates=[]
    d=datetime.today()
    while len(dates)<n:
        if d.weekday()<5:
            dates.append(d.strftime("%Y-%m-%d"))
        d-=timedelta(days=1)
        if len(dates)>=n*3:break
    return dates[:n]

# ─────────────────────────────────────
# Massive API
# ─────────────────────────────────────
def get_massive_ohlcv(ticker,start,end):
    url=f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params={"adjusted":"true","sort":"asc","limit":50000,"apiKey":MASSIVE}
    for attempt in range(2):
        try:
            resp=requests.get(url,params=params,timeout=30)
            if resp.status_code==429:time.sleep(12);continue
            if resp.status_code!=200:return None
            results=resp.json().get("results",[])
            if not results or len(results)<100:return None
            df=pd.DataFrame(results)
            df["date"]=pd.to_datetime(df["t"],unit="ms").dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
            df=df.set_index("date").sort_index()
            df=df.rename(columns={"c":"Close","v":"Volume"})
            df=df[["Close","Volume"]].astype(float).dropna()
            if len(df)>=100:return df
        except:time.sleep(3)
    return None

def get_russell2000_tickers():
    """Massive API에서 Russell 2000 구성종목 가져오기"""
    tickers=set()
    url="https://api.polygon.io/v3/reference/tickers"
    params={
        "market":"stocks",
        "exchange":"XNAS,XNYS,XASE",  # 나스닥+뉴욕+아멕스
        "active":"true",
        "limit":1000,
        "apiKey":MASSIVE
    }
    # IWM ETF holdings 방식으로 Russell 2000 근사치 가져오기
    # Massive API의 indices 엔드포인트 활용
    try:
        url2=f"https://api.polygon.io/v3/snapshot/indices?ticker=I:R2000&apiKey={MASSIVE}"
        resp=requests.get(url2,timeout=30)
        print(f"Russell 2000 지수 확인: {resp.status_code}")
    except:pass

    # 대안: Massive에서 소형주 필터링 (시총 기준)
    cursor=None
    page=0
    while page<20:  # 최대 20,000개
        p={**params,"limit":1000}
        if cursor:p["cursor"]=cursor
        try:
            resp=requests.get(url,params=p,timeout=30)
            if resp.status_code!=200:break
            data=resp.json()
            for t in data.get("results",[]):
                ticker=t.get("ticker","")
                if ticker and len(ticker)<=5 and ticker.isalpha():
                    tickers.add(ticker)
            cursor=data.get("next_url","")
            if not cursor:break
            # cursor에서 실제 cursor 값 추출
            if "cursor=" in cursor:
                cursor=cursor.split("cursor=")[-1].split("&")[0]
            else:
                break
            page+=1
            time.sleep(0.1)
        except:break
    return list(tickers)

# ─────────────────────────────────────
# 시장 필터 — 정배열 (미너비니 정석)
# ─────────────────────────────────────
def check_market_strict(spy_df):
    """
    SPY 정배열 확인:
    현재가 > 50MA > 150MA > 200MA
    AND 200MA가 21일 전보다 상승
    """
    if spy_df is None or len(spy_df)<200:return True,"데이터부족"
    c=spy_df["Close"]
    m50=float(c.rolling(50).mean().iloc[-1])
    m150=float(c.rolling(150).mean().iloc[-1])
    m200=float(c.rolling(200).mean().iloc[-1])
    cur=float(c.iloc[-1])
    m200v=c.rolling(200).mean().dropna()
    m200_21=float(m200v.iloc[-21]) if len(m200v)>=21 else m200

    cond1=cur>m50      # 현재가 > 50MA
    cond2=m50>m150     # 50MA > 150MA
    cond3=m150>m200    # 150MA > 200MA
    cond4=m200>m200_21 # 200MA 상승 중

    ok=all([cond1,cond2,cond3,cond4])
    if ok:
        status="완전 상승장(정배열)"
    elif cond3 and cond4:
        status="부분 상승장(200MA 위)"
    else:
        status="하락장/조정장"
    return ok,status

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

def score_grade(score):
    if score>=90:return "S"
    elif score>=80:return "A"
    elif score>=70:return "B"
    elif score>=60:return "C"
    else:return "D"

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
    if not MASSIVE:
        send("MASSIVE_TOKEN 없음!");exit(1)

    end=datetime.today()
    start=(end-timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end_str=end.strftime("%Y-%m-%d")
    sig_dates=get_recent_dates(SCAN_DAYS)
    data_cutoff=pd.Timestamp(sig_dates[0])-timedelta(days=7)
    print(f"탐색날짜: {sig_dates}")

    # SPY 데이터 + 정배열 확인
    spy_df=get_massive_ohlcv("SPY",start,end_str)
    market_ok,market_str=check_market_strict(spy_df)
    print(f"시장 상태: {market_str}")

    # 유니버스 구성: SP600 + IWM(Russell2000) 근사
    # IWM ETF 구성종목을 Massive API로 가져오기
    send(f"🇺🇸 미국 소형주 스캐너 시작\n최근 {SCAN_DAYS}거래일\n시장: {market_str}\nRussell 2000 구성종목 수집 중...")

    # IWM holdings 가져오기
    iwm_tickers=set()
    try:
        url=f"https://api.polygon.io/v3/reference/tickers?underlying_asset_id=IWM&apiKey={MASSIVE}"
        # 대안: IWM ETF의 holdings 직접 조회
        url2=f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/IWM?apiKey={MASSIVE}"
        resp=requests.get(url2,timeout=30)
        print(f"IWM 확인: {resp.status_code}")
    except:pass

    # Russell 2000 근사 - Massive ticker 필터링 (시총 $300M 이하 소형주)
    print("Russell 2000 티커 수집 중...")
    r2000_tickers=[]
    cursor=None
    collected=0
    for page in range(40):  # 최대 40,000개 조회
        params={
            "market":"stocks",
            "active":"true",
            "limit":1000,
            "sort":"market_cap",
            "order":"desc",
            "apiKey":MASSIVE
        }
        if cursor:params["cursor"]=cursor
        try:
            resp=requests.get("https://api.polygon.io/v3/reference/tickers",
                             params=params,timeout=30)
            if resp.status_code!=200:break
            data=resp.json()
            results=data.get("results",[])
            if not results:break
            for t in results:
                ticker=t.get("ticker","")
                market_cap=t.get("market_cap",0) or 0
                # 소형주 기준: 시총 $3억~$20억
                if (ticker and len(ticker)<=5 and ticker.replace("-","").isalpha()
                    and 300_000_000<=market_cap<=2_000_000_000):
                    r2000_tickers.append(ticker)
                    collected+=1
            next_url=data.get("next_url","")
            if not next_url or collected>=2000:break
            if "cursor=" in next_url:
                cursor=next_url.split("cursor=")[-1].split("&")[0]
            else:break
            time.sleep(0.2)
        except Exception as e:
            print(f"티커 수집 오류: {e}");break

    print(f"Russell 2000 근사: {len(r2000_tickers)}개")

    # 전체 유니버스: SP600 + Russell 2000 합산
    all_tickers=list(dict.fromkeys(SP600+r2000_tickers))
    print(f"전체 유니버스: {len(all_tickers)}개")

    send(f"유니버스 구성 완료\nSP600: {len(SP600)}개\nRussell2000 근사: {len(r2000_tickers)}개\n총 {len(all_tickers)}개 데이터 수집 시작...")

    if not market_ok:
        send(f"⚠️ {market_str}\n정배열 미충족 — 시그널 신뢰도 낮음\n(50MA>150MA>200MA 조건 미달)")

    # 데이터 수집
    valid_data={};data_ok=0;last_dates=[]
    for i,ticker in enumerate(all_tickers):
        if i%200==0:print(f"[{i}/{len(all_tickers)}] 수신:{data_ok}")
        df=get_massive_ohlcv(ticker,start,end_str)
        if df is None:continue
        last_date=df.index[-1]
        if last_date<data_cutoff:continue
        valid_data[ticker]=df
        data_ok+=1
        last_dates.append(last_date)
        time.sleep(0.05)

    if last_dates:
        last_dates.sort()
        median_date=last_dates[len(last_dates)//2].strftime("%Y-%m-%d")
        date_stat=f"데이터 기준일: {median_date}(중앙)"
    else:
        date_stat="데이터 기준일: 없음"

    send(f"다운로드 완료\n수신: {data_ok}/{len(all_tickers)}개\n{date_stat}\n패턴 분석 시작...")

    # 패턴 분석
    res=[];trend_pass=0
    for i,ticker in enumerate(all_tickers):
        if i%300==0:print(f"[{i}/{len(all_tickers)}] 트렌드:{trend_pass} 발견:{len(res)}")
        df=valid_data.get(ticker)
        if df is None:continue
        cap="SmallCap600" if ticker in SP600 else "Russell2000"
        for sig_str in sig_dates:
            sig_ts=pd.Timestamp(sig_str)
            if sig_ts not in df.index:continue
            pos=df.index.tolist().index(sig_ts)
            sl=df.iloc[:pos+1]
            if not check_trend(sl):continue
            trend_pass+=1
            ok,pat=detect(sl)
            if not ok or not pat["vs"]:continue
            rs=calc_rs(sl,spy_df.loc[:sig_ts])if spy_df is not None else 0.0
            if rs<=0:continue
            score=calc_score(rs,pat["vr"],pat["cd"],pat["hd"])
            grade=score_grade(score)
            history=get_past_signals(df,sig_ts)
            res.append({
                "sig_date":sig_str,"ticker":ticker,"cap":cap,
                "cur":pat["cur"],"pivot":pat["pivot"],
                "cd":pat["cd"],"hd":pat["hd"],
                "cdays":pat["cdays"],"hdays":pat["hdays"],
                "vr":pat["vr"],"vs":pat["vs"],
                "rs":rs,"score":score,"grade":grade,
                "history":history,
            })
            break

    res.sort(key=lambda x:(x["sig_date"],x["score"],x["rs"]),reverse=True)
    seen=set();deduped=[]
    for r in res:
        if r["ticker"] not in seen:
            seen.add(r["ticker"]);deduped.append(r)
    res=deduped
    print(f"완료: {len(res)}개 발견")

    # RAW 저장
    if res:
        rows=[]
        for r in res:
            rows.append({
                "date":r["sig_date"],"ticker":r["ticker"],
                "cap":r["cap"],"cur":r["cur"],"pivot":r["pivot"],
                "cup_depth":r["cd"],"handle_depth":r["hd"],
                "cup_days":r["cdays"],"handle_days":r["hdays"],
                "vol_ratio":r["vr"],"rs":r["rs"],
                "score":r["score"],"grade":r["grade"],
            })
        pd.DataFrame(rows).to_csv("scanner_us_small_raw.csv",index=False,encoding="utf-8-sig")
        print(f"RAW 저장: scanner_us_small_raw.csv ({len(rows)}건)")
        send_file("scanner_us_small_raw.csv",f"🇺🇸 미장 소형주 RAW ({len(rows)}건) {datetime.today().strftime('%Y-%m-%d')}")

    send(f"스캔 완료\n데이터 수신: {data_ok}/{len(all_tickers)}개\n{date_stat}\n트렌드 통과: {trend_pass}개\n패턴+거래량+RS: {len(res)}개")

    if not res:
        send(f"🇺🇸 미국 소형주 스캐너\n최근 {SCAN_DAYS}거래일 | {market_str}\n조건 충족 종목 없음")
    else:
        hdr=f"🇺🇸 미너비니 소형주(미국)\n최근 {SCAN_DAYS}거래일 | {market_str}\n{len(res)}개 발견(점수순)\n"+"─"*24+"\n"
        msg=hdr
        for r in res:
            up=round((r["pivot"]/r["cur"]-1)*100,1)
            grade_emoji={"S":"🏆","A":"🥇","B":"🥈","C":"🥉","D":"📊"}.get(r["grade"],"📊")
            past=format_past(r["history"])
            blk=(f"[{r['sig_date']}] [{r['cap']}]\n"
                 f"{r['ticker']}\n"
                 f"  AI점수: {grade_emoji}{r['score']}점({r['grade']}등급)\n"
                 f"  현재가: ${r['cur']:,.2f}\n"
                 f"  피벗  : ${r['pivot']:,.2f} ({up:+.1f}%)\n"
                 f"  컵:{r['cd']}%/{r['cdays']}일 핸들:{r['hd']}%/{r['hdays']}일\n"
                 f"  거래량:{r['vr']}x🔥 RS:{r['rs']:+.1f}%\n"
                 +(past+"\n" if past else "")+"\n")
            if len(msg)+len(blk)>4000:
                send(msg);msg="(이어서)\n\n"+blk
            else:msg+=blk
        send(msg)
