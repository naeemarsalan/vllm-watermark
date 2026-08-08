import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, SynthIDTextWatermarkingConfig

t0=time.time()
name="gpt2"
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name)
model.eval()

cfg = SynthIDTextWatermarkingConfig(
    keys=[654, 400, 836, 123, 340, 443, 597, 160, 57, 29],
    ngram_len=5,
)
prompt = "The history of the Roman Empire is"
inputs = tok(prompt, return_tensors="pt")
torch.manual_seed(7)
t=time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7,
                          watermarking_config=cfg, pad_token_id=tok.eos_token_id)
dt=time.time()-t
txt = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("SYNTHID STATUS: generation_ok", flush=True)
print("SYNTHID TEXT:", txt, flush=True)
print(f"gen time {dt:.2f}s, total {time.time()-t0:.1f}s", flush=True)
