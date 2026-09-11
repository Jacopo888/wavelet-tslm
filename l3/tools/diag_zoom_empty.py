"""Diagnose empty second-turn generations in the zoom episode (run on the node from ROOT, PYTHONPATH=ROOT)."""
import sys, json, torch
sys.path.insert(0, ".")
from l3.zoom import (_load_eval_runtime, _model_inputs, _generate, oracle_zoom, heuristic_zoom,
                     prepare_zoom_batch, build_turn2_sample, with_action_instruction)
from l3.grpo import _autocast_for

CFG = "l3/configs/l3_zoom_z0.yaml"
CKPT = "results/anomaly_xl_precise/anomaly_xl_wavelet_va1/checkpoints/epoch_5_end.pt"
model, dataset, resolved = _load_eval_runtime(CFG, CKPT, "val")
tok = model.tokenizer
model.eval()


def show(tag, ids, prompt=None):
    row = ids[0].tolist()
    print(f"[{tag}] n_ids={len(row)} first={row[:8]} text_special={tok.decode(row[:20])!r}")
    print(f"[{tag}] text={tok.decode(row, skip_special_tokens=True)[:160]!r}")
    if prompt is not None:
        ii = prompt["input_ids"][0]
        print(f"[{tag}] prompt_len={ii.shape[0]} attn_sum={int(prompt['attention_mask'].sum())} "
              f"tail={tok.decode(ii[-60:].tolist())[-260:]!r}")


for index in (0, 1):
    sample = dataset[index]
    print("=== sample", index, sample["category"], "L", sample.get("length"), "gold", str(sample["answer"])[:90])
    # A: plain va1 path
    p = _model_inputs(model, sample)
    show("A_no_zoom", _generate(model, p, sample=False, max_new_tokens=80), p)
    # B: oracle zoom with separately encoded features (kit path)
    act = oracle_zoom(sample) or heuristic_zoom(sample)
    print("action", act.text())
    p2 = prepare_zoom_batch(model, sample, act, training=False)
    print("B features", tuple(p2["_l3_ts_features"].shape), "input_ids", tuple(p2["input_ids"].shape),
          "pads", int((p2["input_ids"] == tok(model.TS_PAD_TOKEN, add_special_tokens=False)["input_ids"][0]).sum()))
    show("B_oracle_features", _generate(model, p2, sample=False, max_new_tokens=80), p2)
    # C: same turn-2 prompt but through model.generate with prepare_batch's own (padded) time_series
    turn2 = build_turn2_sample(sample, act)
    p3 = _model_inputs(model, turn2)
    show("C_oracle_padded_generate", _generate(model, p3, sample=False, max_new_tokens=80), p3)
    # D: text-only variant: action instruction + history, no zoom block
    d = with_action_instruction(sample)
    d["post_prompt"] = d["post_prompt"] + "\nAssistant action: " + act.text() + "\nNow output the final answer in absolute coordinates."
    p4 = _model_inputs(model, d)
    show("D_text_only", _generate(model, p4, sample=False, max_new_tokens=80), p4)
    # E: action instruction only (turn-1 prompt): does va1 answer or act?
    p5 = _model_inputs(model, with_action_instruction(sample))
    show("E_turn1", _generate(model, p5, sample=False, max_new_tokens=80), p5)
    # F: B but greedy via llm_model.generate WITHOUT attention_mask (mask hypothesis)
    with _autocast_for(model):
        emb = model._merge_features(p2["_l3_ts_features"], p2["input_ids"])
        g = model.llm_model.generate(inputs_embeds=emb, max_new_tokens=40, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    show("F_no_mask", g)
