"""
미너비니 컵&핸들 백테스트 - 미국 주식 (Massive API 버전)
- 대상: S&P1500
- 데이터: Massive.com (구 Polygon.io)
- 출력: RAW CSV + 일별수익률 JSON + 알파분석 + 피크분석
"""
import os,time,warnings,requests,json,statistics
import numpy as np,pandas as pd
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")

TOK=os.environ.get("TELEGRAM_TOKEN","")
CID=os.environ.get("TELEGRAM_CHAT_ID","")
MASSIVE=os.environ.get("MASSIVE_TOKEN","")

LOOKBACK_DAYS=1500
HISTORY_DAYS=2100
MAX_HOLD=90

SP500 = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BK","BBWI","BAX","BDX","WRB","BBY","TECH","BIIB","BLK","BX","BA","BSX","BMY","AVGO","BR","BRO","BLDR","BG","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNX","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP","COST","CTRA","CRWD","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY","DE","DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LLY","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR","PANW","PARA","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WHR","WMB","WTW","GWW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS"
]
SP400 = [
    "AAN","ACC","ACHC","ACM","ACIW","ADNT","AEO","AGCO","AIT","AKR","AL","ALG","ALEX","ALGT","ALKS","ALRM","AM","AMG","AMKR","AMPH","AMWD","AN","ANF","AOUT","APAM","APOG","APPF","ARCB","AROC","ARW","ASB","ASGN","ASH","ASTE","ATMU","ATR","AUB","AVAV","AVT","AWI","AX","AYI","AZTA","B","BC","BCO","BCPC","BDC","BKH","BKU","BL","BLKB","BMI","BOOT","BOX","BRC","BRX","BURL","BWXT","BXC","CABO","CACI","CALM","CALX","CARS","CASH","CATY","CBT","CBZ","CC","CCOI","CDP","CECO","CENX","CHE","CHEF","CHH","CHS","CLB","CLF","CMC","CMCO","CNDT","CNO","COHU","COLB","COLM","CORT","CPRI","CR","CRUS","CRVL","CSGS","CSWI","CUZ","CVBF","CW","CWT","DAR","DBD","DCO","DEN","DFIN","DINO","DKS","DLX","DNOW","DORM","DRH","DSP","DV","DXC","DXPE","EEFT","EFC","EGP","EHC","ELAN","ELF","EME","EPC","ESE","ESNT","ESTE","EVTC","EXLS","EXPO","EXTN","EYE","FAF","FBIN","FBP","FCFS","FELE","FHB","FHN","FLO","FLR","FMC","FN","FORM","FR","FRME","FRPT","FSS","FUL","G","GATX","GFF","GGG","GHC","GKOS","GMS","GNRC","GOLF","GRBK","GRC","GTES","GTLS","HAFC","HBI","HCSG","HHH","HIBB","HNI","HOMB","HOPE","HP","HRB","HSII","HTH","HTLD","HUBG","HWC","IBP","ICFI","IDCC","IDA","INGR","ITRI","JACK","JBT","JELD","JJSF","JLL","KAI","KALU","KBH","KBR","KFY","KMPR","KNX","KRC","KRG","KSS","KTOS","KW","LAUR","LBRT","LEA","LECO","LII","LM","LNC","LNW","LOPE","LPX","LSTR","LTC","LXP","LZB","MANT","MATX","MCY","MD","MDU","MED","MEDP","MGA","MGY","MIDD","MMS","MP","MRC","MRCY","MSA","MSGS","MTG","MTSI","MUR","NAVI","NBR","NEO","NHC","NMRK","NOVA","NRC","NSA","NVT","NVTS","NX","NXST","OFG","OGE","OGS","OHI","OII","OIS","OLN","ONTO","OSCR","OUT","OZK","PAAS","PB","PCAR","PCH","PENN","PFSI","PJT","POR","POWL","PRG","PRGO","PRIM","PRK","PRKS","PSN","PTCT","PTEN","PTVE","PVH","QDEL","QGEN","QTWO","RCM","RDN","RHP","RNG","RNR","ROG","ROAD","RPM","RRX","RS","RUSHA","RXO","RYAM","SAFE","SANM","SBCF","SCI","SEIC","SF","SFM","SHAK","SHO","SIG","SITE","SIX","SKX","SLGN","SM","SMAR","SMPL","SNV","SPSC","SRC","SRI","SSB","SSNC","STAA","STC","STER","STL","STNE","SUPN","SWX","SYBT","SYNA","TALO","TBI","TBBK","TDOC","TDS","TEX","TGI","THO","TILE","TNET","TNL","TOL","TOWN","TPX","TREX","TRNO","TRMK","TRUP","TTGT","TTMI","TWI","TXRH","UE","UFPI","UHAL","UNFI","UNVR","UVV","VAC","VCEL","VCYT","VRTS","VSAT","VVV","WAFD","WASH","WD","WDFC","WEX","WINA","WMS","WOR","WPC","WPM","WSFS","WTS","WU","XPO"
]
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

def get_massive_ohlcv(ticker,start,end):
    url=f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params={"adjusted":"true","sort":"asc","limit":50000,"apiKey":MASSIVE}
    for attempt in range(2):
        try:
            resp=requests.get(url,params=params,timeout=30)
            if resp.status_code==429:time.sleep(12);continue
            if resp.status_code!=200:return None
            results=resp.json().get("results",[])
            if not results or len(results)<200:return None
            df=pd.DataFrame(results)
            df["date"]=pd.to_datetime(df["t"],unit="ms").dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
            df=df.set_index("date").sort_index()
            df=df.rename(columns={"c":"Close","v":"Volume"})
            df=df[["Close","Volume"]].astype(float).dropna()
            if len(df)>=200:return df
        except:time.sleep(3)
    return None

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

    # 500일 + 급등 필터
    c=cl[-min(500,n):];v=vl[-min(500,n):];w=len(c)
    if len(c)>=20:
        recent_gain=(c[-1]-c[-20])/c[-20]
        if recent_gain>0.40:return False,{}

    # 로컬 최고점 후보 탐색
    peak_candidates=[]
    for i in range(10, w-55):
        lo=max(0,i-10);hi=min(w,i+10)
        if c[i]==np.max(c[lo:hi]) and c[i]==np.max(c[max(0,i-5):min(w,i+5)]):
            peak_candidates.append(i)
    if not peak_candidates:
        peak_candidates=[int(np.argmax(c[:w//2]))]

    # 가장 긴 컵 선택
    best=None;best_cup_days=0
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
        if rh>lh*1.05:continue
        hnd=c[ri:];hl=len(hnd)
        if not(5<=hl<=20):continue
        hlow=float(np.min(hnd));hd=(rh-hlow)/rh
        if not(0.05<=hd<=0.15):continue
        if(hlow-bot)/(lh-bot)<0.60:continue
        cur=cl[-1]
        if not(rh*0.97<=cur<=rh*1.05):continue
        vr=float(np.mean(v[-5:]))/float(np.mean(v[-40:-5]))if len(v)>=40 else 1.0
        full_cup_days=ri-li
        if full_cup_days>best_cup_days:
            best_cup_days=full_cup_days
            best={"cd":round(cd*100,1),"hd":round(hd*100,1),"cdays":full_cup_days,"hdays":hl,
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

if __name__=="__main__":
    if not MASSIVE:
        send("MASSIVE_TOKEN 없음!");exit(1)

    end=datetime.today()
    start=(end-timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end_str=end.strftime("%Y-%m-%d")
    signal_start=end-timedelta(days=LOOKBACK_DAYS)
    signal_dates=set(pd.bdate_range(signal_start,end).map(pd.Timestamp))

    all_tickers=list(dict.fromkeys(SP500+SP400+SP600))
    cap_map={t:"S&P500" for t in SP500}
    cap_map.update({t:"MidCap400" for t in SP400 if t not in cap_map})
    cap_map.update({t:"SmallCap600" for t in SP600 if t not in cap_map})

    send(f"🇺🇸 미국 백테스트 시작 (Massive)\n기간: 최근 {LOOKBACK_DAYS}일\nS&P1500 {len(all_tickers)}개 데이터 수집 중...\n(약 1~2시간 소요)")

    # SPY 기준지수
    spy_df=get_massive_ohlcv("SPY",start,end_str)
    print(f"SPY: {len(spy_df) if spy_df is not None else 0}일치")

    # 백테스트
    all_signals=[]
    for i,ticker in enumerate(all_tickers):
        if i%100==0:print(f"[{i}/{len(all_tickers)}] 시그널:{len(all_signals)}건")
        df=get_massive_ohlcv(ticker,start,end_str)
        if df is None:continue
        idx=df.index.tolist()
        for j,sig_ts in enumerate(idx):
            if sig_ts not in signal_dates:continue
            sl=df.iloc[:j+1]
            if not check_trend(sl):continue
            ok,pat=detect(sl)
            if not ok or not pat["vs"]:continue
            if spy_df is None:continue
            try:rs=calc_rs(sl,spy_df.loc[:sig_ts])
            except:rs=0.0
            if rs<=0:continue
            entry=float(df["Close"].iloc[j])
            score=calc_score(rs,pat["vr"],pat["cd"],pat["hd"])
            # 1~90일 수익률
            daily_r={}
            for hold in range(1,MAX_HOLD+1):
                fi=j+hold
                if fi<len(idx):daily_r[hold]=round((float(df["Close"].iloc[fi])/entry-1)*100,2)
                else:daily_r[hold]=None
            # SPY 알파
            alpha={}
            try:
                if spy_df is not None and sig_ts in spy_df.index:
                    spy_idx=spy_df.index.tolist()
                    spy_j=spy_idx.index(sig_ts)
                    spy_entry=float(spy_df["Close"].iloc[spy_j])
                    for hold in [5,20,60]:
                        spy_fi=spy_j+hold
                        if spy_fi<len(spy_idx):
                            spy_r=(float(spy_df["Close"].iloc[spy_fi])/spy_entry-1)*100
                            sr=daily_r.get(hold)
                            if sr is not None:alpha[hold]=round(sr-spy_r,2)
            except:pass
            all_signals.append({
                "date":sig_ts.strftime("%Y-%m-%d"),
                "ticker":ticker,
                "cap":cap_map.get(ticker,"기타"),
                "entry":entry,"pivot":pat["pivot"],
                "cup_depth":pat["cd"],"handle_depth":pat["hd"],
                "cup_days":pat["cdays"],"handle_days":pat["hdays"],
                "cup_start":pat.get("cup_start",""),"cup_end":pat.get("cup_end",""),
                "vol_ratio":pat["vr"],"rs":rs,"score":score,
                "r5":daily_r.get(5),"r20":daily_r.get(20),"r60":daily_r.get(60),
                "alpha5":alpha.get(5),"alpha20":alpha.get(20),"alpha60":alpha.get(60),
                "daily_returns":daily_r,
            })
        time.sleep(0.05)

    print(f"백테스트 완료: {len(all_signals)}건")

    # RAW 저장
    rows=[]
    for s in all_signals:
        row={k:v for k,v in s.items() if k!="daily_returns"}
        for hold in [1,3,5,10,15,20,30,40,50,60,75,90]:
            row[f"r{hold}"]=s["daily_returns"].get(hold)
        rows.append(row)
    raw_df=pd.DataFrame(rows)
    raw_df.to_csv("backtest_us_raw.csv",index=False,encoding="utf-8-sig")
    daily_json=[{"date":s["date"],"ticker":s["ticker"],"cap":s["cap"],
                 "score":s["score"],"daily_returns":s["daily_returns"]}for s in all_signals]
    with open("backtest_us_daily.json","w",encoding="utf-8")as f:
        json.dump(daily_json,f,ensure_ascii=False,indent=2)
    print("RAW 저장 완료")

    if not all_signals:
        send("시그널 없음")
    else:
        df=pd.DataFrame(rows)
        # 기본 요약
        lines=["🇺🇸 미국 백테스트 결과 (Massive)",f"총 시그널: {len(df)}건","─"*28]
        for col,label in [("r5","5일"),("r20","20일"),("r60","60일")]:
            vals=df[col].dropna()
            if len(vals)==0:continue
            win=sum(1 for v in vals if v>0)
            lines+=[f"[{label}] n={len(vals)}",
                    f"  평균:{vals.mean():+.1f}% 중앙:{statistics.median(vals):+.1f}%",
                    f"  승률:{round(win/len(vals)*100,1)}%"]
        send("\n".join(lines))

        # 알파 요약
        lines2=["vs SPY 알파 분석","─"*28]
        for col,label in [("alpha5","5일"),("alpha20","20일"),("alpha60","60일")]:
            vals=df[col].dropna()
            if len(vals)==0:continue
            beat=sum(1 for v in vals if v>0)
            lines2+=[f"[{label}] n={len(vals)}",
                     f"  평균알파:{vals.mean():+.1f}% 중앙:{statistics.median(vals):+.1f}%",
                     f"  SPY초과확률:{round(beat/len(vals)*100,1)}%"]
        send("\n".join(lines2))

        # 점수등급별
        lines3=["점수등급별 20일 수익률","─"*28]
        for grade,lo,hi in [("S(90+)",90,100),("A(80-89)",80,89),("B(70-79)",70,79),("C(60-69)",60,69)]:
            sub=df[(df["score"]>=lo)&(df["score"]<=hi)]["r20"].dropna()
            if len(sub)==0:continue
            lines3+=[f"[{grade}] n={len(sub)}",
                     f"  평균:{sub.mean():+.1f}% 중앙:{statistics.median(sub):+.1f}%"]
        send("\n".join(lines3))

        # 피크 분석
        peaks=[max([(h,r)for h,r in s["daily_returns"].items()if r is not None],key=lambda x:x[1])[0]
               for s in all_signals if any(r is not None for r in s["daily_returns"].values())]
        if peaks:
            lines4=["최고수익 시점 분석","─"*28,
                    f"평균 피크: {round(sum(peaks)/len(peaks),1)}거래일",
                    f"중앙값 피크: {statistics.median(peaks):.0f}거래일"]
            for lo,hi in [(1,5),(6,10),(11,20),(21,30),(31,45),(46,60),(61,90)]:
                cnt=sum(1 for d in peaks if lo<=d<=hi)
                lines4.append(f"  {lo:2d}~{hi:2d}일: {cnt}건({round(cnt/len(peaks)*100)}%)")
            send("\n".join(lines4))

        # 일별 커브
        curve=["일별 수익률 커브","─"*28]
        for hold in [1,3,5,7,10,15,20,25,30,40,50,60,75,90]:
            col=f"r{hold}"
            if col in df.columns:
                vals=df[col].dropna()
                if len(vals)>5:
                    curve.append(f"  {hold:2d}일: 평균{vals.mean():+.1f}% 중앙{statistics.median(vals):+.1f}%")
        send("\n".join(curve))

        # 상위 시그널 종목
        top=df.nlargest(10,"r20")[["date","ticker","cap","score","r5","r20","r60"]]
        lines5=["20일 수익률 TOP10","─"*28]
        for _,row in top.iterrows():
            lines5.append(f"  {row['ticker']} [{row['cap']}] score:{row['score']}점")
            lines5.append(f"    {row['date']} 5일:{row['r5']:+.1f}% 20일:{row['r20']:+.1f}% 60일:{row['r60'] if row['r60'] is not None else '-'}")
        send("\n".join(lines5))
