"""
미너비니 컵&핸들 스캐너 - 미국 주식 (Tiingo API 버전)
- 대상: S&P1500 하드코딩
- 데이터: Tiingo API (하루 50,000건 무료)
- 결과: 텔레그램 전송 (한국어)
"""
import os,time,warnings,requests
import numpy as np,pandas as pd
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")

TOK=os.environ.get("TELEGRAM_TOKEN","")
CID=os.environ.get("TELEGRAM_CHAT_ID","")
TIINGO=os.environ.get("TIINGO_TOKEN","")
SCAN_DAYS=7
HISTORY_DAYS=420

# ─────────────────────────────────────
# S&P1500 종목 리스트 (하드코딩)
# ─────────────────────────────────────
SP500 = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BK","BBWI","BAX","BDX","WRB","BBY","TECH","BIIB","BLK","BX","BA","BSX","BMY","AVGO","BR","BRO","BLDR","BG","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNX","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP","COST","CTRA","CRWD","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY","DE","DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LLY","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR","PANW","PARA","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WHR","WMB","WTW","GWW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS"
]
SP400 = [
    "AAN","ACC","ACHC","ACM","ACIW","ADNT","AEO","AGCO","AIT","AKR","AL","ALG","ALEX","ALGT","ALKS","ALRM","AM","AMG","AMKR","AMPH","AMWD","AN","ANF","AOUT","APAM","APOG","APPF","ARCB","AROC","ARW","ASB","ASGN","ASH","ASTE","ATMU","ATR","AUB","AVAV","AVT","AWI","AX","AYI","AZTA","B","BC","BCO","BCPC","BDC","BKH","BKU","BL","BLKB","BMI","BOOT","BOX","BRC","BRX","BURL","BWXT","BXC","CABO","CACI","CALM","CALX","CARS","CASH","CATY","CBT","CBZ","CC","CCOI","CDP","CECO","CENX","CHE","CHEF","CHH","CHS","CLB","CLF","CMC","CMCO","CNDT","CNO","COHU","COLB","COLM","CORT","CPRI","CR","CRUS","CRVL","CSGS","CSWI","CUZ","CVBF","CW","CWT","DAR","DBD","DCO","DEN","DFIN","DINO","DKS","DLX","DNOW","DORM","DRH","DSP","DV","DXC","DXPE","EEFT","EFC","EGP","EHC","ELAN","ELF","EME","EPC","ESE","ESNT","ESTE","EVTC","EXLS","EXPO","EXTN","EYE","FAF","FBIN","FBP","FCFS","FELE","FHB","FHN","FLO","FLR","FMC","FN","FORM","FR","FRME","FRPT","FSS","FUL","G","GATX","GFF","GGG","GHC","GKOS","GMS","GNRC","GOLF","GRBK","GRC","GTES","GTLS","HAFC","HBI","HCSG","HHH","HIBB","HNI","HOMB","HOPE","HP","HRB","HSII","HTH","HTLD","HUBG","HWC","IBP","ICFI","IDCC","IDA","INGR","ITRI","JACK","JBT","JELD","JJSF","JLL","KAI","KALU","KBH","KBR","KFY","KMPR","KNX","KRC","KRG","KSS","KTOS","KW","LAUR","LBRT","LEA","LECO","LII","LM","LNC","LNW","LOPE","LPX","LSTR","LTC","LXP","LZB","MANT","MATX","MCY","MD","MDU","MED","MEDP","MGA","MGY","MIDD","MMS","MP","MRC","MRCY","MSA","MSGS","MTG","MTSI","MUR","NAVI","NBR","NEO","NHC","NMRK","NOVA","NRC","NSA","NVT","NVTS","NX","NXST","OFG","OGE","OGS","OHI","OII","OIS","OLN","ONTO","OSCR","OUT","OZK","PAAS","PB","PCAR","PCH","PENN","PFSI","PJT","POR","POWL","PRG","PRGO","PRIM","PRK","PRKS","PSN","PTCT","PTEN","PTVE","PVH","QDEL","QGEN","QTWO","RCM","RDN","RHP","RNG","RNR","ROG","ROAD","RPM","RRX","RS","RUSHA","RXO","RYAM","SAFE","SANM","SBCF","SCI","SEIC","SF","SFM","SHAK","SHO","SIG","SITE","SIX","SKX","SLGN","SM","SMAR","SMPL","SNV","SPSC","SRC","SRI","SSB","SSNC","STAA","STC","STER","STL","STNE","SUPN","SWX","SYBT","SYNA","TALO","TBI","TBBK","TDOC","TDS","TEX","TGI","THO","TILE","TNET","TNL","TOL","TOWN","TPX","TREX","TRNO","TRMK","TRUP","TTGT","TTMI","TWI","TXRH","UE","UFPI","UHAL","UNFI","UNVR","UVV","VAC","VCEL","VCYT","VRTS","VSAT","VVV","WAFD","WASH","WD","WDFC","WEX","WINA","WMS","WOR","WPC","WPM","WSFS","WTS","WU","XPO"
]
SP600 = [
    "ACAD","ACLS","ADMA","ADUS","AEHR","AHCO","AIRC","ALCO","ALGT","ALRM","AMBC","AMEH","AMKR","AMMO","AMSC","ANF","ANGO","AORT","AOSL","APAM","APPN","APLE","ARKO","ARLO","ARRY","ARWR","ASLE","ASND","ASPS","ASRT","ASTE","ATEN","ATNI","ATSG","AUBN","AUPH","AVNW","AWI","AXNX","AXSM","BAND","BANF","BANR","BCEL","BCPC","BFS","BGFV","BKD","BLDR","BLFS","BLNK","BMTC","BNL","BOOT","BPOP","BRKL","BRSP","BSIG","BSRR","BSVN","BURL","BUSE","BZH","CACC","CAKE","CALM","CARA","CARE","CARS","CASH","CASY","CATO","CBAN","CBRL","CBSH","CC","CCNE","CDMO","CDNA","CDRE","CEIX","CENT","CENX","CEVA","CFFI","CFFN","CHCO","CHDN","CHEF","CHUY","CIVB","CLAR","CLB","CLBK","CLDT","CLNE","CLPR","CMAX","CMCO","COHU","COLM","COOP","COUR","CPRI","CPRX","CRAI","CRDX","CRK","CRSP","CRVL","CSGS","CSII","CSTR","CTBI","CTLP","CTMX","CTRE","CTRN","CUBI","CULP","CUTR","CW","CWCO","DAKT","DAVA","DCOM","DFIN","DGII","DH","DHIL","DINO","DIOD","DK","DKL","DLX","DNOW","DORM","DRH","DRVN","DSP","DXC","DXPE","DXYN","EAF","EARN","EBC","EBMT","EFC","EGP","EIG","ELAN","ELY","EPC","EPRT","ESS","ESTE","EVTC","EXLS","EXPI","EXTN","EYE","EZPW","FARO","FBNC","FBRT","FCFS","FELE","FFBC","FFIN","FN","FORM","FOUR","FRAF","FRME","FULT","GBX","GCI","GFF","GHC","GKOS","GMS","GOLF","GOOD","GPX","GRPN","GSBC","GTLS","GWRE","HAFC","HAIN","HASI","HBI","HBT","HCSG","HFWA","HIBB","HIMS","HLNE","HMN","HNI","HOFT","HOMB","HQY","HRMY","HSII","HTH","HTLD","HTLF","HUBG","HWC","HZO","IART","IBCP","IBOC","IBTX","IDCC","IDEX","IESC","IIIN","IMKTA","IMXI","INDB","INFU","INGN","INMD","INSP","INSW","IRBT","IRWD","ISBA","ITRI","JACK","JBLU","JBSS","JELD","JOUT","JWN","KAI","KALU","KFY","KMPR","KNSA","KREF","KRNT","KSS","KTOS","KW","LADR","LAUR","LCNB","LGIH","LGND","LKFN","LMB","LNTH","LOPE","LSTR","LWAY","LXP","MAIN","MATV","MBIN","MBUU","MCBS","MCF","MCRI","MDGL","MED","MEDP","MEI","MERC","MFIN","MGY","MKSI","MLAB","MLKN","MMI","MMSI","MNRO","MOD","MOFG","MRC","MRCY","MSEX","MSTR","MTG","MTSI","MTRN","MTRX","MVBF","MYR","MYRG","NATH","NBTB","NCOM","NCNO","NEO","NEOG","NFBK","NHC","NMIH","NNBR","NPO","NRC","NRIM","NRP","NTST","NVT","NVTS","NWE","NXST","OBK","OCFC","OCSL","OFG","OGE","OII","OIS","ONTO","OPBK","OPCH","OSPN","OTTR","OUT","OXM","PAHC","PATK","PBF","PBPB","PDCO","PEGA","PENN","PFBC","PFIS","PFSI","PGNY","PHR","PKST","PLBC","PLMR","PLNT","PNM","POOL","POWL","PPBI","PRAA","PRDO","PRGO","PRIM","PRK","PRKS","PSN","PTCT","PTEN","PTLO","PTVE","PUMP","QCRH","QDEL","QGEN","QTWO","RBC","RCKT","RCKY","RCM","RDNT","RES","REVG","RGEN","RGP","RICK","RILY","RMBS","RMNI","RMR","RPRX","RRR","RUSHA","RWT","RYAM","SAFE","SANM","SASR","SBCF","SBSI","SCVL","SEIC","SF","SFST","SHAK","SHBI","SHO","SIG","SIGI","SIT","SITM","SKX","SLG","SLGN","SLVM","SM","SMBC","SNCY","SNDR","SNEX","SOFI","SPFI","SPOK","SRC","SRI","SSBK","SSRM","STBA","STLD","STRA","SUPN","SWX","SYBT","SYNA","TALO","TBNK","TCBK","TCMD","TEX","TFSL","TGI","TILE","TNDM","TOWN","TPIC","TREX","TRNO","TRMK","TROW","TRUP","TTGT","TTMI","TWI","TXRH","UBCP","UCBI","UFPI","ULCC","UNFI","UNVR","UPST","USPH","UVV","VBTX","VCEL","VCYT","VECO","VICR","VIRT","VLGEA","VRTS","VSAT","VSEC","WAFD","WASH","WD","WDFC","WERN","WFRD","WINA","WMS","WOOF","WOR","WPC","WSFS","WTS","XRX"
]

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
# Tiingo API 호출
# ─────────────────────────────────────
def get_tiingo_ohlcv(ticker, start, end):
    url=f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    headers={"Content-Type":"application/json","Authorization":f"Token {TIINGO}"}
    params={"startDate":start,"endDate":end,"resampleFreq":"daily"}
    for attempt in range(2):
        try:
            resp=requests.get(url,headers=headers,params=params,timeout=30)
            if resp.status_code!=200:return None
            data=resp.json()
            if not data or len(data)<100:return None
            df=pd.DataFrame(data)
            df["date"]=pd.to_datetime(df["date"]).dt.tz_localize(None)
            df=df.set_index("date").sort_index()
            df=df.rename(columns={"adjClose":"Close","adjVolume":"Volume"})
            if "Close" not in df.columns:
                df=df.rename(columns={"close":"Close","volume":"Volume"})
            df=df[["Close","Volume"]].astype(float).dropna()
            if len(df)>=100:return df
        except Exception as e:
            time.sleep(2)
    return None

# ─────────────────────────────────────
# 미너비니 로직
# ─────────────────────────────────────
def check_market(mkt_df):
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
    if not TIINGO:
        send("TIINGO_TOKEN이 없어요! GitHub Secrets 확인해주세요.")
        exit(1)

    end=datetime.today()
    start=(end-timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end_str=end.strftime("%Y-%m-%d")
    sig_dates=get_recent_dates(SCAN_DAYS)
    data_cutoff=pd.Timestamp(sig_dates[0])-timedelta(days=7)
    print(f"탐색날짜: {sig_dates}")

    # SPY로 시장 상태 체크
    mkt_df=get_tiingo_ohlcv("SPY",start,end_str)
    market_ok=check_market(mkt_df)
    market_str="상승장(S&P500>200MA)"if market_ok else"하락장(S&P500<200MA)"

    all_tickers=list(dict.fromkeys(SP500+SP400+SP600))
    cap_map={t:"S&P500" for t in SP500}
    cap_map.update({t:"MidCap400" for t in SP400 if t not in cap_map})
    cap_map.update({t:"SmallCap600" for t in SP600 if t not in cap_map})

    send(f"🇺🇸 미국 스캐너 시작 (Tiingo)\n최근 {SCAN_DAYS}거래일 | {market_str}\nS&P1500 {len(all_tickers)}개 종목 수집 중...\n(약 30~40분 소요)")
    if not market_ok:
        send("S&P500 200MA 하방 - 시그널 신뢰도 낮음, 주의!")

    # 개별 종목 데이터 수집
    valid_data={};data_ok=0;data_old=0;last_dates=[]
    for i,ticker in enumerate(all_tickers):
        if i%100==0:print(f"[{i}/{len(all_tickers)}] 수신:{data_ok} 발견예정")
        df=get_tiingo_ohlcv(ticker,start,end_str)
        if df is None:continue
        last_date=df.index[-1]
        if last_date<data_cutoff:
            data_old+=1
            continue
        valid_data[ticker]=df
        data_ok+=1
        last_dates.append(last_date)
        time.sleep(1.5)  # Tiingo 무료 시간당 50개 제한 대응

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
        if i%200==0:print(f"[{i}/{len(all_tickers)}] 트렌드:{trend_pass} 발견:{len(res)}")
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
            history=get_past_signals(df,sig_ts)
            res.append({
                "sig_date":sig_str,"ticker":ticker,
                "cap":cap_map.get(ticker,"기타"),
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

    send(f"스캔 완료\n데이터 수신: {data_ok}/{len(all_tickers)}개\n{date_stat}\n트렌드 통과: {trend_pass}개\n패턴+거래량+RS: {len(res)}개")

    if not res:
        send(f"🇺🇸 미국 스캐너\n최근 {SCAN_DAYS}거래일 | {market_str}\n조건 충족 종목 없음\n(거래량급증+RS양수 기준)")
    else:
        hdr=f"🇺🇸 미너비니 컵&핸들(미국)\n최근 {SCAN_DAYS}거래일 | {market_str}\n{len(res)}개 발견(RS순)\n"+"─"*24+"\n"
        msg=hdr
        for r in res:
            up=round((r["pivot"]/r["cur"]-1)*100,1)
            past=format_past(r["history"])
            blk=(f"[{r['sig_date']}] [{r['cap']}]\n"
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
