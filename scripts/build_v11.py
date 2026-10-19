#!/usr/bin/env python
"""Build SufficiencyBench v1.1 from v1.0: collapse all whitespace runs inside every context to a
single space (both conditions), rebuild the prompt by exact substring replacement, drop the
questions whose two contexts become identical after normalisation (whitespace-only pairs), and
update metadata. v1.0 is backed up to data/experiments/sufficiency_bench_v1.0/ (never modified).
Usage: python scripts/build_v11.py   (idempotent: refuses to run twice on v1.1 data)
"""
import json, re, shutil, sys, os, collections
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B=os.path.join(ROOT,"data","experiments","sufficiency_bench"); B0=B+"_v1.0"
def norm_ws(t): return re.sub(r"\s+"," ",t).strip()
meta=json.load(open(os.path.join(B,"metadata.json")))
meta=json.load(open(os.path.join(B0,"metadata.json"))) if os.path.exists(B0) else meta
if meta.get("version")!="1.0": sys.exit("v1.0 metadata not found; abort")
if not os.path.exists(B0): shutil.copytree(B,B0); print("backup ->",B0)
else: print("backup exists:",B0)
report={"dropped_ids":{},"split_sizes":{},"qtype_counts":collections.Counter(),"quadrant_counts":collections.Counter()}
ratios=[]
for split in ("train","val","test"):
    data=json.load(open(os.path.join(B0,f"{split}.json"))); keep=[]; dropped=[]
    for ex in data:
        conds=ex["conditions"]; assert len(conds)==2
        new=[]
        for c in conds:
            ctx=c["context"]; nctx=norm_ws(ctx); p=c["prompt"]
            assert p.count(ctx)==1, c["condition_id"]
            c=dict(c); c["context"]=nctx; c["prompt"]=p.replace(ctx,nctx); c["context_length"]=len(nctx); new.append(c)
        if re.sub(r"\s+","",new[0]["context"])==re.sub(r"\s+","",new[1]["context"]): dropped.append(ex["id"]); continue
        ex=dict(ex); ex["conditions"]=new; keep.append(ex)
        suf=[c for c in new if c["sufficient"]][0]; ins=[c for c in new if not c["sufficient"]][0]
        ratios.append(len(suf["context"])/len(ins["context"]))
        report["qtype_counts"][ex["question_type"]]+=1
        for c in new: report["quadrant_counts"][c["quadrant"]]+=1
    json.dump(keep,open(os.path.join(B,f"{split}.json"),"w"),indent=1)
    report["dropped_ids"][split]=dropped; report["split_sizes"][split]=len(keep)
    print(f"{split}: {len(data)} -> {len(keep)} questions (dropped {len(dropped)} whitespace-only pairs)")
import statistics
meta["version"]="1.1"
meta["total_questions"]=sum(report["split_sizes"].values()); meta["total_conditions"]=2*meta["total_questions"]
meta["splits"]=report["split_sizes"]; meta["quadrant_counts"]=dict(report["quadrant_counts"]); meta["question_type_counts"]=dict(report["qtype_counts"])
meta["length_ratio_mean"]=statistics.mean(ratios); meta["length_ratio_std"]=statistics.pstdev(ratios)
meta["changelog"]={"1.1":"2026-10-06. All runs of whitespace inside every context collapsed to a single space in both conditions and prompts rebuilt accordingly. In v1.0 the sufficient contexts kept the double spaces of the source passages while the edited insufficient contexts were re-joined with single spaces, which made the number of double spaces a label cue (0.775 AUROC overall, 0.991 on HotpotQA) that the hidden-state probes partly exploited. "+str(sum(len(v) for v in report['dropped_ids'].values()))+" questions whose two contexts were identical once all whitespace was removed (whitespace-only edits, mostly sentence-join differences such as \"grounds.[citation needed]\" against \"grounds. [citation needed]\") were removed. Questions, gold answers, ids, splits and parametric-knowledge labels are unchanged. v1.0 is kept in sufficiency_bench_v1.0/.",
                   "1.0":"Original release."}
json.dump(meta,open(os.path.join(B,"metadata.json"),"w"),indent=1)
json.dump({k:(dict(v) if isinstance(v,collections.Counter) else v) for k,v in report.items()},open(os.path.join(B,"v1.1_build_report.json"),"w"),indent=1)
print("metadata:",json.dumps({k:meta[k] for k in ("version","total_questions","splits","length_ratio_mean","length_ratio_std")}))
# verification
for split in ("train","val","test"):
    d=json.load(open(os.path.join(B,f"{split}.json")))
    assert all(not re.search(r"\s{2,}|[\t\n\r\xa0 ]",c["context"]) and c["prompt"].count(c["context"])==1 for ex in d for c in ex["conditions"])
    assert all(re.sub(r"\s+","",ex["conditions"][0]["context"])!=re.sub(r"\s+","",ex["conditions"][1]["context"]) for ex in d)
print("verification OK")
