import sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, WatermarkingConfig, WatermarkDetector

t0=time.time()
name="gpt2"
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name)
model.eval()
print(f"loaded {name} in {time.time()-t0:.1f}s", flush=True)

wm_config = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config=wm_config)

prompt = "The history of the Roman Empire is"
inputs = tok(prompt, return_tensors="pt")

def gen(watermark):
    torch.manual_seed(7)
    kw = dict(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7, pad_token_id=tok.eos_token_id)
    if watermark: kw["watermarking_config"]=wm_config
    t=time.time()
    with torch.no_grad():
        out = model.generate(**kw)
    dt=time.time()-t
    txt = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return txt, dt

wm_txt, wm_dt = gen(True)
print("WATERMARKED TEXT:", wm_txt, flush=True)
print(f"wm gen time {wm_dt:.2f}s", flush=True)
now_txt, now_dt = gen(False)
print("UNWATERMARKED TEXT:", now_txt, flush=True)
print(f"nowm gen time {now_dt:.2f}s", flush=True)

def detect(text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    out = detector(ids, return_dict=True)
    d={}
    for a in ("prediction","p_value","z_score","num_tokens_scored","num_green_tokens","green_fraction"):
        if hasattr(out,a):
            v=getattr(out,a)
            d[a]=v.tolist() if hasattr(v,"tolist") else v
    return d

human = "It was a quiet morning at the market, vendors stacking oranges and hosing the concrete floor while pigeons fought over crumbs near the fountain."

print("DETECT wm:", detect(wm_txt), flush=True)
print("DETECT nowm:", detect(now_txt), flush=True)
print("DETECT human:", detect(human), flush=True)
print(f"TOTAL TIME {time.time()-t0:.1f}s", flush=True)
