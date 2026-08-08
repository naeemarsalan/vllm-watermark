"""
Local CPU proof: decode-time text watermarking (KGW green-list) + statistical
detection, using HuggingFace transformers' built-in WatermarkingConfig /
WatermarkLogitsProcessor / WatermarkDetector. This exercises the exact same
logits-processor mechanism a vLLM logits-processor plugin would use at
decode time.

transformers 4.57.6, torch 2.9.1, CPU only.
"""
import json
import time
import traceback

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WatermarkingConfig,
    WatermarkDetector,
)

MODEL_CANDIDATES = ["Qwen/Qwen2.5-0.5B-Instruct", "gpt2"]

PROMPTS = [
    "Explain in a short paragraph why regular exercise benefits mental health.",
    "Write a short paragraph describing the water cycle for a middle school student.",
    "Summarize the main causes of the fall of the Roman Empire in a short paragraph.",
]

HUMAN_TEXT = (
    "I walked down to the market this morning before the heat set in. The "
    "vendors were just opening their stalls, stacking oranges into small "
    "pyramids and hosing down the concrete. An old man played a battered "
    "accordion near the fountain, and a few pigeons fought over a dropped "
    "piece of bread. It was the kind of quiet, unremarkable morning that "
    "you only appreciate in hindsight, once the day has filled up with "
    "noise and errands and the small emergencies of ordinary life."
)

MAX_NEW_TOKENS = 200
RESULTS = {"max_new_tokens": MAX_NEW_TOKENS}


def load_model():
    last_err = None
    for name in MODEL_CANDIDATES:
        try:
            print(f"[load] trying {name} ...")
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
            model.eval()
            print(f"[load] loaded {name} in {time.time()-t0:.1f}s")
            return name, tok, model
        except Exception as e:  # noqa
            print(f"[load] FAILED for {name}: {e}")
            last_err = e
    raise last_err


def build_prompt(tok, model_name, user_text):
    if "Instruct" in model_name and hasattr(tok, "apply_chat_template"):
        messages = [{"role": "user", "content": user_text}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_text


def generate(tok, model, prompt_text, watermark_config, seed):
    inputs = tok(prompt_text, return_tensors="pt")
    torch.manual_seed(seed)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=1.0,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    if watermark_config is not None:
        gen_kwargs["watermarking_config"] = watermark_config
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    elapsed = time.time() - t0
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    n_tok = new_tokens.shape[0]
    return text, n_tok, elapsed, new_tokens


def main():
    model_name, tok, model = load_model()
    RESULTS["model_name"] = model_name

    wm_config = WatermarkingConfig(
        bias=2.5,
        seeding_scheme="selfhash",
    )
    RESULTS["watermark_config"] = {"bias": 2.5, "seeding_scheme": "selfhash"}

    detector = WatermarkDetector(
        model_config=model.config,
        device="cpu",
        watermarking_config=wm_config,
    )

    runs = []
    total_wm_tokens = total_wm_time = 0.0
    total_nowm_tokens = total_nowm_time = 0.0

    for i, p in enumerate(PROMPTS):
        prompt_text = build_prompt(tok, model_name, p)
        print(f"\n=== Prompt {i+1}: {p}")

        wm_text, wm_ntok, wm_time, wm_new_tokens = generate(tok, model, prompt_text, wm_config, seed=1000 + i)
        print(f"[watermarked] {wm_ntok} tok in {wm_time:.2f}s ({wm_ntok/wm_time:.1f} tok/s)")
        total_wm_tokens += wm_ntok
        total_wm_time += wm_time

        nowm_text, nowm_ntok, nowm_time, nowm_new_tokens = generate(tok, model, prompt_text, None, seed=1000 + i)
        print(f"[unwatermarked] {nowm_ntok} tok in {nowm_time:.2f}s ({nowm_ntok/nowm_time:.1f} tok/s)")
        total_nowm_tokens += nowm_ntok
        total_nowm_time += nowm_time

        runs.append(
            {
                "prompt": p,
                "watermarked_text": wm_text,
                "watermarked_ntok": wm_ntok,
                "watermarked_time_s": wm_time,
                "unwatermarked_text": nowm_text,
                "unwatermarked_ntok": nowm_ntok,
                "unwatermarked_time_s": nowm_time,
            }
        )

    RESULTS["runs"] = runs
    RESULTS["throughput"] = {
        "watermarked_tok_per_s": total_wm_tokens / total_wm_time if total_wm_time else None,
        "unwatermarked_tok_per_s": total_nowm_tokens / total_nowm_time if total_nowm_time else None,
    }
    if RESULTS["throughput"]["watermarked_tok_per_s"] and RESULTS["throughput"]["unwatermarked_tok_per_s"]:
        overhead_pct = (
            (RESULTS["throughput"]["unwatermarked_tok_per_s"] - RESULTS["throughput"]["watermarked_tok_per_s"])
            / RESULTS["throughput"]["unwatermarked_tok_per_s"]
            * 100.0
        )
        RESULTS["throughput"]["overhead_pct_slower"] = overhead_pct

    # ---------------- Detection ----------------
    def detect(text):
        inputs = tok(text, return_tensors="pt", add_special_tokens=False)
        ids = inputs["input_ids"]
        out = detector(ids, return_dict=True)
        # out is a WatermarkDetectorOutput; fields vary by version, dump what's there
        d = {}
        for attr in ("prediction", "p_value", "z_score", "num_tokens_scored", "num_green_tokens", "green_fraction"):
            if hasattr(out, attr):
                val = getattr(out, attr)
                try:
                    val = val.tolist() if hasattr(val, "tolist") else val
                except Exception:
                    pass
                d[attr] = val
        return d

    detection_results = []
    for i, r in enumerate(runs):
        detection_results.append({"label": f"prompt{i+1}_watermarked", "scores": detect(r["watermarked_text"])})
        detection_results.append({"label": f"prompt{i+1}_unwatermarked", "scores": detect(r["unwatermarked_text"])})
    detection_results.append({"label": "human_written", "scores": detect(HUMAN_TEXT)})
    RESULTS["detection_results"] = detection_results
    RESULTS["human_text"] = HUMAN_TEXT

    print("\n=== DETECTION RESULTS ===")
    for d in detection_results:
        print(d["label"], d["scores"])

    # ---------------- SynthID (generation-only, prove API exists) ----------------
    synthid_result = {}
    try:
        from transformers import SynthIDTextWatermarkingConfig

        synthid_config = SynthIDTextWatermarkingConfig(
            keys=[654, 400, 836, 123, 340, 443, 597, 160, 57, 29],
            ngram_len=5,
        )
        prompt_text = build_prompt(tok, model_name, PROMPTS[0])
        inputs = tok(prompt_text, return_tensors="pt")
        torch.manual_seed(4242)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                watermarking_config=synthid_config,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            )
        elapsed = time.time() - t0
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        synthid_result = {
            "status": "generation_ok",
            "config": {"ngram_len": 5, "num_keys": 10},
            "text": text,
            "ntok": new_tokens.shape[0],
            "time_s": elapsed,
            "note": (
                "SynthIDTextWatermarkingConfig generation succeeded (proves the API "
                "exists in transformers 4.57.6). Detection is NOT run here: SynthID's "
                "detector (BayesianDetector in "
                "transformers/generation/watermarking.py / the synthid_text research repo) "
                "must be TRAINED on a labelled corpus of watermarked vs. non-watermarked "
                "text produced under the same keys before it can classify. That training "
                "step is out of scope for this 15-minute CPU proof."
            ),
        }
        print("\n=== SynthID generation OK ===")
        print(text)
    except Exception as e:
        synthid_result = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        print("\n=== SynthID generation FAILED ===")
        print(synthid_result["error"])

    RESULTS["synthid"] = synthid_result

    with open("./raw_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

    print("\nDONE. Raw results written to raw_results.json")


if __name__ == "__main__":
    main()
