"""
70_actual_vs_detected.py

WHAT THIS DOES
    Puts the person's own answer (actual stress) next to what the model
    detected, one row per survey answer, for each of the four ways of
    splitting the data, with the model run without context ("signals only")
    and with it ("signals + context").

    Reads only files that already exist in the bundle:
        runs/C0_*/answer_level.csv       signals only
        runs/C7_*, runs/C7b_*/answer_level.csv   signals + context
        data/processed/labels/responses_scored.csv   the STAI-6 answers

    Writes comparisons/ :
        Actual_vs_Detected_Stress.xlsx   every table, one sheet each
        csv/                             the same tables as CSV
        figures/                         confusion matrices, score against
                                         STAI-6, per-person class mix

HOW TO RUN (from the bundle root)
    pip install pandas numpy scipy scikit-learn matplotlib openpyxl
    python3 btp/70_actual_vs_detected.py

    Nothing is trained here. The predictions were made when the runs were
    trained (btp/34_train_users.py); this only lays them out.
"""
import pandas as pd, numpy as np, json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from scipy.stats import spearmanr
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

SRC=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
OUT=os.path.join(SRC,'comparisons')
os.makedirs(OUT+'/csv',exist_ok=True); os.makedirs(OUT+'/figures',exist_ok=True)
CL=['Low','Medium','High']
# split -> (title, run with signals only, run with context)
SPL=[('random','Random split of windows (recall)','C0_random','C7_random'),
     ('response','Answers kept whole','C0_response','C7b_response'),
     ('user','A person never seen','C0_user','C7b_user'),
     ('temporal','Earlier train, later test','C0_temporal','C7_temporal')]
resp=pd.read_csv(f'{SRC}/data/processed/labels/responses_scored.csv')
resp['response']=resp.Subject_ID+'|'+resp.response_time.astype(str)
resp=resp.set_index('response')

def load(run):
    d=pd.read_csv(f'{SRC}/runs/{run}/answer_level.csv'); m=json.load(open(f'{SRC}/runs/{run}/metrics.json'))
    return d,m
data={}
for key,title,r0,r1 in SPL:
    d0,m0=load(r0); d1,m1=load(r1)
    a=d0[['response','Subject_ID','true','pred','score','STAI6','P0','P1','P2']].rename(columns={'pred':'pred0','score':'score0','P0':'L0','P1':'M0','P2':'H0'})
    b=d1[['response','pred','score','P0','P1','P2']].rename(columns={'pred':'pred1','score':'score1','P0':'L1','P1':'M1','P2':'H1'})
    t=a.merge(b,on='response',how='inner')
    assert len(t)==len(d0)==len(d1), (key,len(t),len(d0),len(d1))
    t=t.join(resp[['STAI6_RAW','SURVEY_ACTIVITY','SURVEY_SITUATION','SURVEY_ENVIRONMENT','SURVEY_TIREDNESS','SURVEY_STRESS_SOURCE']],on='response')
    t['time']=t.response.str.split('|').str[1].str[:16]
    t=t.sort_values(['Subject_ID','time']).reset_index(drop=True)
    data[key]=(title,t,m0,m1)

# ---------- tables
rows=[]; 
for key,(title,t,m0,m1) in data.items():
    for tag,m in (('signals only',m0),('signals + context',m1)):
        al=m['answer_level']; p=al['point']; ci=al['ci95']
        rows.append({'Split':title,'Model':tag,'Answers':al['n_answers'],
          'Exact match (accuracy)':round(p['accuracy'],3),
          'Balanced accuracy':round(p['balanced_accuracy'],3),
          'Balanced acc. 95% CI':f"{ci['balanced_accuracy'][0]:.2f} to {ci['balanced_accuracy'][1]:.2f}",
          'Within one class':round(p['within_one_class'],3),
          'Rank corr. score vs STAI-6':round(p['rank_corr_score_vs_stai6'],3),
          'Rank corr. 95% CI':f"{ci['rank_corr_score_vs_stai6'][0]:.2f} to {ci['rank_corr_score_vs_stai6'][1]:.2f}",
          'Always-commonest accuracy':round(al['always_commonest_accuracy'],3),
          'Chance (balanced acc.)':0.333,
          'Window-level balanced acc.':round(m['mean_balanced_accuracy'],3)})
summary=pd.DataFrame(rows)

conf_rows=[]
for key,(title,t,m0,m1) in data.items():
    for tag,col in (('signals only','pred0'),('signals + context','pred1')):
        cm=confusion_matrix(t.true,t[col],labels=[0,1,2])
        for i,a in enumerate(CL):
            conf_rows.append({'Split':title,'Model':tag,'Actual (STAI-6 class)':a,'Detected Low':cm[i,0],'Detected Medium':cm[i,1],'Detected High':cm[i,2],'Answers':cm[i].sum(),
                              'Correct':f"{cm[i,i]}/{cm[i].sum()}"})
conf=pd.DataFrame(conf_rows)

pp_rows=[]
for key,(title,t,m0,m1) in data.items():
    for u,g in t.groupby('Subject_ID'):
        r={'Split':title,'Person':u,'Answers':len(g),
           'Actual Low':(g.true==0).sum(),'Actual Medium':(g.true==1).sum(),'Actual High':(g.true==2).sum(),
           'Mean STAI-6':round(g.STAI6.mean(),1),'Mean score (signals only)':round(g.score0.mean(),2),'Mean score (+context)':round(g.score1.mean(),2)}
        for tag,col in (('signals only','pred0'),('+context','pred1')):
            r[f'Detected Low ({tag})']=(g[col]==0).sum(); r[f'Detected Medium ({tag})']=(g[col]==1).sum(); r[f'Detected High ({tag})']=(g[col]==2).sum()
            r[f'Correct ({tag})']=f"{(g[col]==g.true).sum()}/{len(g)}"
        pp_rows.append(r)
perp=pd.DataFrame(pp_rows)

def answers_table(t):
    o=pd.DataFrame({'Person':t.Subject_ID,'Answered at':t.time,'STAI-6 score (20-80)':t.STAI6.round(1),
      'Actual stress':t.true.map(dict(enumerate(CL))),
      'Detected (signals only)':t.pred0.map(dict(enumerate(CL))),'Score 0-2 (signals only)':t.score0.round(2),
      'Detected (+context)':t.pred1.map(dict(enumerate(CL))),'Score 0-2 (+context)':t.score1.round(2),
      'P(Low) +ctx':t.L1.round(2),'P(Medium) +ctx':t.M1.round(2),'P(High) +ctx':t.H1.round(2),
      'Match (signals only)':np.where(t.pred0==t.true,'yes','no'),'Match (+context)':np.where(t.pred1==t.true,'yes','no'),
      'Activity (survey)':t.SURVEY_ACTIVITY,'Situation (survey)':t.SURVEY_SITUATION,'Tiredness 1-5 (survey)':t.SURVEY_TIREDNESS,'Stress source (survey)':t.SURVEY_STRESS_SOURCE})
    return o
ans={k:answers_table(v[1]) for k,v in data.items()}

# ---------- csv
summary.to_csv(f'{OUT}/csv/summary_by_split.csv',index=False)
conf.to_csv(f'{OUT}/csv/confusion_by_split.csv',index=False)
perp.to_csv(f'{OUT}/csv/per_person.csv',index=False)
for k,v in ans.items(): v.to_csv(f'{OUT}/csv/answers_{k}.csv',index=False)

# ---------- xlsx
wb=Workbook(); thin=Side(style='thin',color='BBBBBB'); bd=Border(left=thin,right=thin,top=thin,bottom=thin)
HF=PatternFill('solid',fgColor='1F3A5F'); GREEN=PatternFill('solid',fgColor='D9EAD3'); RED=PatternFill('solid',fgColor='F4CCCC')
CLS={'Low':PatternFill('solid',fgColor='DCE9F7'),'Medium':PatternFill('solid',fgColor='FFF2CC'),'High':PatternFill('solid',fgColor='F8D7D3')}
def put(ws,df,r0=1,widths=None):
    for j,c in enumerate(df.columns,1):
        x=ws.cell(r0,j,c); x.font=Font(bold=True,color='FFFFFF'); x.fill=HF; x.alignment=Alignment(wrap_text=True,vertical='center',horizontal='center'); x.border=bd
    for i,row in enumerate(df.itertuples(index=False),r0+1):
        for j,v in enumerate(row,1):
            if isinstance(v,(np.integer,)): v=int(v)
            elif isinstance(v,(np.floating,)): v=float(v)
            x=ws.cell(i,j,v); x.border=bd; x.alignment=Alignment(vertical='center',horizontal='center' if j>1 else 'left',wrap_text=True)
    ws.row_dimensions[r0].height=44
    for j,c in enumerate(df.columns,1):
        w=max(len(str(c))*0.62,*(len(str(v)) for v in df.iloc[:,j-1]))+2 if len(df) else 12
        ws.column_dimensions[get_column_letter(j)].width=min(max(w,10),34)
    ws.freeze_panes=ws.cell(r0+1,1)
ws=wb.active; ws.title='Read me'
lines=[('Actual stress versus model-detected stress',True),
('',False),
('Actual stress = the person\'s own answer to the STAI-6 questionnaire (six items, scored 20 to 80), cut into three classes across everyone: Low up to 46.7, Medium 50.0 to 53.3, High 56.7 and above.',False),
('Detected stress = what ContextGate predicted for the 60-second windows around that answer, combined to one prediction per answer. Score 0 to 2 = 1 x P(Medium) + 2 x P(High).',False),
('"signals only" = the model with the context switched off. "+ context" = the full ContextGate (hour, weekend, posture, activity count, activity class, body position, sleep).',False),
('',False),
('Every row was predicted by a model that did NOT train on that row, but what that means depends on the split:',True),
('  Random split: other windows of the same answer were in training. It measures recall, not prediction. The gain from context here comes from the clock (hour and weekend).',False),
('  Answers kept whole: all windows of an answer are held out together.',False),
('  A person never seen: the model is trained on four people and tested on the fifth, five times.',False),
('  Earlier train, later test: the first part of each person\'s answers trains, the later part tests (closest to real use, 26 answers).',False),
('',False),
('Sheets: Summary (one line per split and model) | Confusion (actual rows, detected columns) | Per person | one sheet of every answer for each split.',False),
('Chance for balanced accuracy with three classes is 0.333.',False)]
for i,(tx,b) in enumerate(lines,1):
    c=ws.cell(i,1,tx); c.font=Font(bold=b,size=14 if i==1 else 11); c.alignment=Alignment(wrap_text=True,vertical='top')
ws.column_dimensions['A'].width=140
ws=wb.create_sheet('Summary'); put(ws,summary)
ws=wb.create_sheet('Confusion'); put(ws,conf)
ws=wb.create_sheet('Per person'); put(ws,perp)
names={'random':'Random (88)','response':'Whole answers (57)','user':'Unseen person (88)','temporal':'Earlier-later (26)'}
for k,v in ans.items():
    ws=wb.create_sheet(names[k]); put(ws,v)
    cols=list(v.columns)
    for i in range(len(v)):
        for c in ('Actual stress','Detected (signals only)','Detected (+context)'):
            cell=ws.cell(i+2,cols.index(c)+1); cell.fill=CLS[cell.value]
        for c in ('Match (signals only)','Match (+context)'):
            cell=ws.cell(i+2,cols.index(c)+1); cell.fill=GREEN if cell.value=='yes' else RED
wb.save(f'{OUT}/Actual_vs_Detected_Stress.xlsx')

# ---------- figures
CC={'Low':'#3b7dd8','Medium':'#e3a72f','High':'#c0392b'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
# F1 confusion grid
fig,axs=plt.subplots(2,4,figsize=(13,6.2))
for j,(key,(title,t,m0,m1)) in enumerate(data.items()):
    for i,(tag,col) in enumerate((('signals only','pred0'),('signals + context','pred1'))):
        ax=axs[i,j]; cm=confusion_matrix(t.true,t[col],labels=[0,1,2]); ba=balanced_accuracy_score(t.true,t[col])
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(cm.max(),1))
        for a in range(3):
            for b in range(3): ax.text(b,a,cm[a,b],ha='center',va='center',color='white' if cm[a,b]>cm.max()*0.55 else '#222',fontsize=12,fontweight='bold')
        ax.set_xticks(range(3)); ax.set_xticklabels(CL); ax.set_yticks(range(3)); ax.set_yticklabels(CL)
        ax.set_title(f'{title}\n{tag}: balanced acc {ba:.2f}',fontsize=9)
        if j==0: ax.set_ylabel('Actual (STAI-6 class)')
        if i==1: ax.set_xlabel('Detected by the model')
fig.suptitle('Actual versus detected stress class, one prediction per survey answer',fontsize=12,y=0.99)
fig.tight_layout(); fig.savefig(f'{OUT}/figures/1_confusion_by_split.png',dpi=160); plt.close(fig)
# F2 scatter
fig,axs=plt.subplots(2,4,figsize=(13,6.4),sharey=True)
for j,(key,(title,t,m0,m1)) in enumerate(data.items()):
    for i,(tag,col) in enumerate((('signals only','score0'),('signals + context','score1'))):
        ax=axs[i,j]
        for k,c in enumerate(CL): s=t[t.true==k]; ax.scatter(s.STAI6,s[col],s=22,color=CC[c],label=c,alpha=.85,edgecolor='white',linewidth=.4)
        rho=spearmanr(t.STAI6,t[col])[0]
        ax.axvline(48.3,color='#999',lw=.8,ls=':'); ax.axvline(55,color='#999',lw=.8,ls=':')
        ax.set_title(f'{title}\n{tag}: rank corr {rho:+.2f}',fontsize=9); ax.set_ylim(-.05,2.05)
        if j==0: ax.set_ylabel('Model stress score (0 to 2)')
        if i==1: ax.set_xlabel('Actual STAI-6 score')
axs[0,0].legend(title='Actual class',frameon=False,fontsize=8,loc='upper left')
fig.suptitle('Model stress score against the actual STAI-6 score (one dot per answer)',fontsize=12,y=0.99)
fig.tight_layout(); fig.savefig(f'{OUT}/figures/2_score_vs_actual.png',dpi=160); plt.close(fig)
# F3 per person mix
fig,axs=plt.subplots(1,4,figsize=(13,3.9),sharey=True)
for j,(key,(title,t,m0,m1)) in enumerate(data.items()):
    ax=axs[j]; users=sorted(t.Subject_ID.unique()); x=np.arange(len(users)); w=.26
    for o,(lab,col) in enumerate((('Actual','true'),('Detected, signals only','pred0'),('Detected, + context','pred1'))):
        bot=np.zeros(len(users))
        for k,c in enumerate(CL):
            v=np.array([(t[t.Subject_ID==u][col]==k).mean() for u in users])
            ax.bar(x+(o-1)*w*1.1,v,w,bottom=bot,color=CC[c],edgecolor='white',linewidth=.5,label=c if (j==0 and o==0) else None); bot+=v
    ax.set_xticks(x); ax.set_xticklabels([u.replace('user','u') for u in users]); ax.set_title(title,fontsize=9)
    if j==0: ax.set_ylabel('Share of answers'); ax.legend(frameon=False,fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(0.5,-0.12))
fig.suptitle('Per person: actual class mix (left bar), detected without context (middle), detected with context (right)',fontsize=11,y=1.0)
fig.tight_layout(); fig.savefig(f'{OUT}/figures/3_per_person_mix.png',dpi=160); plt.close(fig)
print(summary.to_string()); print(perp[perp.Split=='Earlier train, later test'].to_string())
