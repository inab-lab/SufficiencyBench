import json, sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs.paths import MODEL_CONFIGS
import scripts.run_steering_mistral as S

device="cuda:0"
w_unit=S.compute_direction()
items=S.build_heldout(8, seed=1)
cfg=MODEL_CONFIGS["mistral"]
tok=AutoTokenizer.from_pretrained(cfg["path"]); tok.pad_token=tok.eos_token; tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(cfg["path"],torch_dtype=torch.float16,device_map=device).eval()
M=S.measure_residual_norm(model,tok,[it["prompt"] for it in items],device)
print("M=",round(M,2))
w_t=torch.tensor(w_unit,device=device,dtype=torch.float16)
hooks=[model.model.layers[l].register_forward_hook(S.make_hook()) for l in S.WINDOW]
prompts=[it["prompt"] for it in items]; golds=[it["gold"] for it in items]; quads=[it["quadrant"] for it in items]

def run(vecnorm):
    S._STATE["vec"]=None if vecnorm==0 else vecnorm*w_t
    gens=[]
    for i in range(0,len(prompts),16):
        enc=tok(prompts[i:i+16],return_tensors="pt",padding=True,truncation=True,max_length=2048).to(device)
        with torch.no_grad():
            out=model.generate(**enc,max_new_tokens=40,do_sample=False,pad_token_id=tok.pad_token_id)
        gens.extend(tok.batch_decode(out[:,enc["input_ids"].shape[1]:],skip_special_tokens=True))
    agg={q:{"n":0,"c":0,"a":0} for q in ["Q1","Q2","Q3","Q4"]}
    for g,gold,q in zip(gens,golds,quads):
        c,a=S.classify(g,gold); agg[q]["n"]+=1; agg[q]["c"]+=int(c); agg[q]["a"]+=int(a)
    return agg

# sweep signed injection norms as fractions of M
for f in [0.0, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, -0.05, -0.08, -0.12, -0.18, -0.25, -0.35]:
    agg=run(f*M)
    line=" ".join(f"{q}:c={agg[q]['c']/agg[q]['n']:.2f},a={agg[q]['a']/agg[q]['n']:.2f}" for q in ["Q1","Q2","Q3","Q4"])
    print(f"f={f:+.2f} (norm={f*M:+.2f}) {line}")
