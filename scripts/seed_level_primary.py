"""Round-5 primary analysis: seed-level paired t-tests (95%/90% t-intervals, TOST at +/-0.01 and +/-0.02,
minimum detectable effect at 80% power) for every attention effect at the best and final checkpoints, pooled
seed-level estimates (16 main, 12 without MobileNetV2, 8 robustness, collapsed-run sensitivity) and
DerSimonian-Laird random-effects pooling. Input: results/round3/per_seed_all_runs.csv. Output: results/round5/."""
import itertools, numpy as np, pandas as pd
from scipy import stats
import os; ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(ROOT,'results','round5')+'/'; os.makedirs(OUT,exist_ok=True)
d=pd.read_csv(os.path.join(ROOT,'results','round3','per_seed_all_runs.csv')); d['variant']=d.variant.fillna('')
AL={'resnet50':'ResNet50','densenet121':'DenseNet121','mobilenetv2':'MobileNetV2','customcnn':'Custom CNN'}; TL={'se':'SE','cbam':'CBAM'}
def f(a,t,c,v='',ck='f1_best'): return d[(d.arch==a)&(d.attention==t)&(d.condition==c)&(d.variant==v)].set_index('seed')[ck]
def tci(x,lvl=.95):
    n=len(x); se=x.std(ddof=1)/np.sqrt(n); q=stats.t.ppf(1-(1-lvl)/2,n-1); return x.mean()-q*se, x.mean()+q*se
def signflip(x):
    x=np.asarray(x); obs=abs(x.mean()); return np.mean([abs((x*np.array(s)).mean())>=obs-1e-12 for s in itertools.product([1,-1],repeat=len(x))])
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for k,i in enumerate(o): run=max(run,(m-k)*p[i]); adj[i]=min(1,run)
    return adj
MDEK=stats.t.ppf(.975,4)+stats.t.ppf(.80,4)
def contrasts(ck):
    rows=[]; diffs={}
    for c in ['imbalanced','balanced']:
        for a in AL:
            for t in TL:
                x=(f(a,t,c,ck=ck)-f(a,'none',c,ck=ck)).dropna(); diffs[('main',a,t,c)]=x
    for a,v,lab,bv in [('resnet50','_lungcrop-True','lung crop','_lungcrop-True'),('resnet50','_val-grouped','grouped validation','_val-grouped'),('resnet50','_placement-multi','multi-depth',''),('densenet121','_pretrain-xrv','CXR pretraining','_pretrain-xrv')]:
        for t in TL:
            diffs[('rob',a,t,lab)]=(f(a,t,'imbalanced',v,ck)-f(a,'none','imbalanced',bv,ck)).dropna()
    for k,x in diffs.items():
        lo,hi=tci(x); lo9,hi9=tci(x,.90); se=x.std(ddof=1)/np.sqrt(len(x))
        rows.append(dict(family=k[0],arch=k[1],att=k[2],cond=k[3],effect=f'{TL[k[2]]}, {AL[k[1]]}, {k[3]}',n=len(x),delta=x.mean(),se=se,
          t_lo=lo,t_hi=hi,t90_lo=lo9,t90_hi=hi9,p_t=stats.ttest_1samp(x,0).pvalue,p_perm=signflip(x.values),mde80=MDEK*se,
          eq01=bool(lo9>-0.01 and hi9<0.01),eq02=bool(lo9>-0.02 and hi9<0.02)))
    R=pd.DataFrame(rows)
    for fam in ['main','rob']:
        m=R.family==fam; R.loc[m,'p_t_holm']=holm(R.loc[m,'p_t']); R.loc[m,'p_perm_holm']=holm(R.loc[m,'p_perm'])
    return R,diffs
def pooled(diffs,keys,label):
    P=pd.concat([diffs[k] for k in keys],axis=1).dropna().mean(axis=1); lo,hi=tci(P); lo9,hi9=tci(P,.9); n=len(P); se=P.std(ddof=1)/np.sqrt(n)
    tost=max(stats.t.sf((P.mean()+0.01)/se,n-1), stats.t.cdf((P.mean()-0.01)/se,n-1)); tost2=max(stats.t.sf((P.mean()+0.02)/se,n-1), stats.t.cdf((P.mean()-0.02)/se,n-1))
    return dict(pool=label,k=len(keys),n_seeds=n,delta=P.mean(),t_lo=lo,t_hi=hi,t90_lo=lo9,t90_hi=hi9,p_t=stats.ttest_1samp(P,0).pvalue,p_tost01=tost,p_tost02=tost2,eq01=bool(lo9>-.01 and hi9<.01),eq02=bool(lo9>-.02 and hi9<.02))
def re_pool(R):
    y=R.delta.values; v=R.se.values**2; w=1/v; ybar=(w*y).sum()/w.sum(); Q=(w*(y-ybar)**2).sum(); k=len(y)
    tau2=max(0,(Q-(k-1))/(w.sum()-(w**2).sum()/w.sum())); ws=1/(v+tau2); mu=(ws*y).sum()/ws.sum(); se=np.sqrt(1/ws.sum())
    I2=max(0,(Q-(k-1))/Q) if Q>0 else 0
    return dict(k=k,mu_re=mu,ci_lo=mu-1.96*se,ci_hi=mu+1.96*se,tau2=tau2,tau=np.sqrt(tau2),Q=Q,Q_p=stats.chi2.sf(Q,k-1),I2=I2,mu_fixed=ybar)
res={}
for ck in ['f1_best','f1_final']:
    R,diffs=contrasts(ck); R['ckpt']=ck; res[ck]=R
    main=[k for k in diffs if k[0]=='main']; clean=[k for k in main if k[1]!='mobilenetv2']; rob=[k for k in diffs if k[0]=='rob']
    P=[pooled(diffs,main,'16 main'),pooled(diffs,clean,'12 main (MobileNetV2 excluded)'),pooled(diffs,rob,'8 robustness')]
    # collapsed run: custom CNN se imbalanced
    x=diffs[('main','customcnn','se','imbalanced')]; x4=x.drop(1)
    P.append(dict(pool='Custom CNN SE imbalanced, all 5',k=1,n_seeds=5,delta=x.mean(),t_lo=tci(x)[0],t_hi=tci(x)[1],p_t=stats.ttest_1samp(x,0).pvalue))
    P.append(dict(pool='Custom CNN SE imbalanced, without seed 1',k=1,n_seeds=4,delta=x4.mean(),t_lo=tci(x4)[0],t_hi=tci(x4)[1],p_t=stats.ttest_1samp(x4,0).pvalue))
    PP=pd.DataFrame(P); PP['ckpt']=ck; res['pool_'+ck]=PP
    REm=re_pool(R[R.family=='main']); REc=re_pool(R[(R.family=='main')&(R.arch!='mobilenetv2')])
    res['re_'+ck]=pd.DataFrame([dict(pool='16 main',**REm),dict(pool='12 main (no MobileNetV2)',**REc)]).assign(ckpt=ck)
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
for ck in ['f1_best','f1_final']:
    print('=====',ck); R=res[ck]; m=R[R.family=='main']
    print(R[['family','effect','delta','t_lo','t_hi','p_t','p_perm','mde80','eq01','eq02']].round(4).to_string())
    print('main: hi<0.05',(m.t_hi<0.05).sum(),'hi<0.03',(m.t_hi<0.03).sum(),'hi<0.02',(m.t_hi<0.02).sum(),'eq02',m.eq02.sum(),'eq01',m.eq01.sum(),'CI excl0',((m.t_lo>0)|(m.t_hi<0)).sum(),'min p',m.p_t.min().round(3),'median MDE',m.mde80.median().round(4),'median halfwidth',((m.t_hi-m.t_lo)/2).median().round(4))
    rr=R[R.family=='rob']; print('rob CI excl0',((rr.t_lo>0)|(rr.t_hi<0)).sum(),'min p',rr.p_t.min().round(3))
    print(res['pool_'+ck].round(4).to_string()); print(res['re_'+ck].round(4).to_string())
for k,v in res.items(): v.to_csv(OUT+f'{k}.csv',index=False)
# collapsed seed values per ckpt
print(d[d.run_id.str.startswith('customcnn_se_imbalanced_s')][['run_id','f1_best','f1_final']])
