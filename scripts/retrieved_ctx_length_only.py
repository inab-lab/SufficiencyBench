import json, numpy as np, re
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import os; R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
def ctx(p):
    return p.split('Context: ',1)[1].rsplit('\n\nQuestion:',1)[0]
def load(tag):
    out={}
    for sp in ['train','test']:
        d=json.load(open(f'{R}data/experiments/retrieved_ctx/{tag}/{sp}.json'))
        s={a['id']:a for a in json.load(open(f'{R}results/llama/retrieved_ctx_{tag}/{sp}_scores.json'))}
        rows=[]
        for a in d:
            c=ctx(a['prompt']); 
            sc=s.get(a['id'])
            rows.append(dict(id=a['id'],y=int(a['sufficient']),ysc=None if sc is None else sc['label'],chars=len(c),cc=int(a['context_chars']),words=len(c.split()),
              tok=None if sc is None else sc['n_tokens_prompt'],tr=None if sc is None else sc.get('truncated'),
              pw=len(ctx(a['prompt_plain']).split()), pc=len(ctx(a['prompt_plain']))))
        out[sp]=rows
    return out
def auc_sign(x,y):
    a=roc_auc_score(y,x); return a
def boot(f,x,y,B=2000,seed=0):
    rng=np.random.default_rng(seed); n=len(y); v=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if y[i].min()==y[i].max(): continue
        v.append(f(i))
    return np.percentile(v,[2.5,97.5]).tolist()
res={}
for tag in ['bm25_k5','e5_k5']:
    D=load(tag); te=D['test']; tr=D['train']
    y=np.array([r['y'] for r in te])
    info=dict(n_test=len(te),pos_rate=float(y.mean()),
      label_mismatch_with_scores=int(sum(r['ysc'] is not None and r['ysc']!=r['y'] for r in te)),
      n_train=len(tr),train_pos=float(np.mean([r['y'] for r in tr])),
      chars_vs_context_chars_mismatch=int(sum(r['chars']!=r['cc'] for r in te)))
    feats={'chars':'chars','words':'words','prompt_tokens(llama, whole prompt)':'tok'}
    for name,k in feats.items():
        x=np.array([r[k] for r in te],float)
        a=roc_auc_score(y,x)
        sgn='longer context => more sufficient' if a>=.5 else 'longer context => LESS sufficient (AUROC reported as 1-AUC)'
        s=x if a>=.5 else -x
        aa=roc_auc_score(y,s); ci=boot(lambda i:roc_auc_score(y[i],s[i]),s,y)
        info[name]=dict(auroc=aa,ci95=ci,raw_auroc_longer_is_sufficient=a,direction=sgn,
            mean_sufficient=float(x[y==1].mean()),mean_insufficient=float(x[y==0].mean()),
            sd_sufficient=float(x[y==1].std()),sd_insufficient=float(x[y==0].std()))
    # logreg on log chars, train split
    xt=np.log(np.array([r['chars'] for r in tr],float))[:,None]; yt=np.array([r['y'] for r in tr])
    lr=LogisticRegression(C=1e6,max_iter=1000).fit(xt,yt)
    xe=np.log(np.array([r['chars'] for r in te],float))[:,None]
    p=lr.decision_function(xe); a=roc_auc_score(y,p)
    info['logreg_logchars']=dict(auroc=a,ci95=boot(lambda i:roc_auc_score(y[i],p[i]),p,y),coef=float(lr.coef_[0,0]),intercept=float(lr.intercept_[0]),note='fit on train split, evaluated on test; AUROC unsigned (coef sign learned)')
    # also with type
    res[tag]=info
json.dump(res,open(R+'results/retrieved_ctx_length_only.json','w'),indent=1)
L=['# Length-only baseline on retrieved contexts (Llama-side labels; CPU only)\n',
'Label = support-sufficiency (`sufficient` in data/experiments/retrieved_ctx/<tag>/test.json). Bootstrap: 2,000 question resamples, seed 0, percentile 95% CI. "chars"/"words" = retrieved context text inside the generator prompt (BM25 prompt includes "Title: " prefixes). "prompt tokens" = Llama tokenizer length of the entire prompt (template+context+question) from test_scores.json.\n']
for tag,i in res.items():
    L.append(f"\n## {tag}: n_test={i['n_test']}, sufficient={100*i['pos_rate']:.1f}%, train n={i['n_train']} ({100*i['train_pos']:.1f}% sufficient)\n")
    L.append('| Predictor | AUROC [95% CI] | direction | mean suff. | mean insuff. |\n|---|---|---|---|---|')
    for k in ['chars','words','prompt_tokens(llama, whole prompt)']:
        v=i[k]; L.append(f"| {k} | {v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['direction']} | {v['mean_sufficient']:.1f} | {v['mean_insufficient']:.1f} |")
    v=i['logreg_logchars']; L.append(f"| LogReg on log(chars), fit on train | {v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | coef={v['coef']:.3f} | | |")
open(R+'results/retrieved_ctx_length_only.md','w').write('\n'.join(L)+'\n')
print('\n'.join(L)); print({t:(r['label_mismatch_with_scores'],r['chars_vs_context_chars_mismatch']) for t,r in res.items()})
