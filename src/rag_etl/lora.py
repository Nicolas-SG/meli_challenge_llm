"""LoRA fine-tuning of small Qwen2.5 models on the documentation sections.

Two tasks share the same rows and the same 80/10/10 split:

* ontology: next-token prediction over ``entity_instruct`` (instruction +
  section + its annotated entities/relationships as JSON), so the model learns
  to extract entities in the ontology's format.
* memory:   next-token prediction over the raw section ``text``, so the model
  absorbs the documentation itself.

For a token sequence ``t``, ``X = t[:-1]`` and ``Y = t[1:]`` (Y is X shifted
by one token); the error is the mean per-token cross-entropy of predicting Y.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 6037784
INSTRUCTION = (
    "Extract the entities and relationships from the following document "
    "in the specified JSON format."
)
ONTOLOGY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MEMORY_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def make_input(row) -> str:
    return (
        f"file_name: {row['file_name']} \n project: {row['project_name']} \n "
        f"version: {row['version']} \n section: {row['section']} \n text: {row['text']}"
    )


def make_entity_instruct(row) -> str:
    """Instruction record as a JSON string; ``output`` is the row's
    entities/relationships object (embedded as JSON, not as an escaped string)."""
    return json.dumps(
        {
            "instruction": INSTRUCTION,
            "input": make_input(row),
            "output": json.loads(row["entities"]),
        },
        ensure_ascii=False,
    )


def instruct_prompt(row) -> str:
    """The ``entity_instruct`` string up to where the model must write
    ``output`` (needs no ``entities``: usable on any section)."""
    head = json.dumps({"instruction": INSTRUCTION, "input": make_input(row)}, ensure_ascii=False)
    return head[:-1] + ', "output": '


_ENTITY_RE = re.compile(r'\{"name":\s*"((?:[^"\\]|\\.)*)",\s*"type":\s*"((?:[^"\\]|\\.)*)"\}')
_RELATION_RE = re.compile(
    r'\{"source":\s*"((?:[^"\\]|\\.)*)",\s*"relation":\s*"((?:[^"\\]|\\.)*)",\s*"target":\s*"((?:[^"\\]|\\.)*)"\}'
)


def parse_extraction(generated: str) -> dict | None:
    """Parse the ``output`` object the ontology adapter wrote.

    When the JSON is cut off (long sections can exhaust ``max_new_tokens``),
    the complete entity/relationship objects are recovered with a regex.
    Returns None only when nothing can be recovered."""
    start = generated.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(generated[start:])
    except json.JSONDecodeError:
        ents = [{"name": json.loads(f'"{n}"'), "type": json.loads(f'"{t}"')} for n, t in _ENTITY_RE.findall(generated)]
        rels = [
            {"source": json.loads(f'"{s}"'), "relation": json.loads(f'"{r}"'), "target": json.loads(f'"{t}"')}
            for s, r, t in _RELATION_RE.findall(generated)
        ]
        return {"entities": ents, "relationships": rels} if ents else None
    if not isinstance(obj, dict):
        return None
    ents = [e for e in obj.get("entities", []) if isinstance(e, dict) and "name" in e]
    rels = [r for r in obj.get("relationships", []) if isinstance(r, dict) and {"source", "relation", "target"} <= r.keys()]
    return {"entities": ents, "relationships": rels}


def add_entity_instruct(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entity_instruct"] = pd.Series(
        [make_entity_instruct(r) for _, r in out.iterrows()], index=out.index, dtype="string"
    )
    return out


def load_tokenizer(name: str = ONTOLOGY_MODEL):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def tokenize(texts, tokenizer, add_eos: bool = True) -> list[list[int]]:
    """Qwen2.5 BPE token ids per text; EOS is appended so the model learns
    where a document / an extraction ends."""
    ids = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    if add_eos:
        ids = [t + [tokenizer.eos_token_id] for t in ids]
    return ids


def add_token_columns(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    out = df.copy()
    out["entity_instruct_tok"] = tokenize(out["entity_instruct"], tokenizer)
    out["text_tok"] = tokenize(out["text"], tokenizer)
    return out


def split_dataset(df: pd.DataFrame, seed: int = SEED):
    """80% training_set, 10% test_set, 10% dev_set (in that order)."""
    from sklearn.model_selection import train_test_split

    training_set, rest = train_test_split(df, train_size=0.8, random_state=seed)
    test_set, dev_set = train_test_split(rest, test_size=0.5, random_state=seed)
    return training_set, test_set, dev_set


def make_xy(token_lists) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """X = tokens[:-1], Y = tokens[1:] for every sequence."""
    X = [np.asarray(t[:-1], dtype=np.int64) for t in token_lists]
    Y = [np.asarray(t[1:], dtype=np.int64) for t in token_lists]
    return X, Y


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_lora_model(
    model_name: str,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] = LORA_TARGET_MODULES,
    device: str | None = None,
):
    """Base model in bf16 with a fresh LoRA adapter (adapter weights in fp32)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    device = device or default_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    # Recompute activations in backward: needed to fit 5k-token sections.
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.config.use_cache = False
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )
    return get_peft_model(base, config)


def lora_state_dict(model) -> dict:
    from peft import get_peft_model_state_dict

    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def save_lora(model, state: dict, path: str | Path) -> Path:
    """Save LoRA weights to ``path`` (.bin) and the adapter config next to it."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    config = model.peft_config["default"].to_dict()
    config["target_modules"] = sorted(config["target_modules"])
    config["peft_type"] = str(config["peft_type"])
    config["task_type"] = str(config["task_type"])
    path.with_suffix(".json").write_text(json.dumps(config, indent=2, default=str))
    return path


def load_lora_weights(model_name: str, path: str | Path, device: str | None = None):
    """Rebuild base model + adapter from a .bin saved by :func:`save_lora`."""
    import torch
    from peft import set_peft_model_state_dict

    config = json.loads(Path(path).with_suffix(".json").read_text())
    model = load_lora_model(
        model_name,
        r=config["r"],
        alpha=config["lora_alpha"],
        dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        device=device,
    )
    set_peft_model_state_dict(model, torch.load(path, map_location="cpu"))
    return model


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


LOSS_CHUNK = 1024


def _chunk_loss_sum(lm_head, hidden, target):
    import torch.nn.functional as F

    return F.cross_entropy(lm_head(hidden).float(), target, reduction="sum")


def _sequence_loss_sum(model, x: np.ndarray, y: np.ndarray):
    """Summed cross-entropy of predicting ``y`` from ``x``.

    The vocabulary projection and the loss are computed in chunks of
    ``LOSS_CHUNK`` positions and recomputed in backward, so the full
    (seq_len x 151k) fp32 logits never sit in memory: long sections (5k
    tokens) fit on a 12 GB GPU without truncation."""
    import torch
    from torch.utils.checkpoint import checkpoint

    base = model.get_base_model()
    device = next(model.parameters()).device
    ids = torch.from_numpy(x).unsqueeze(0).to(device)
    target = torch.from_numpy(y).to(device)
    hidden = base.model(input_ids=ids).last_hidden_state[0]
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(y), LOSS_CHUNK):
        sl = slice(start, start + LOSS_CHUNK)
        if torch.is_grad_enabled():
            total = total + checkpoint(_chunk_loss_sum, base.lm_head, hidden[sl], target[sl], use_reentrant=False)
        else:
            total = total + _chunk_loss_sum(base.lm_head, hidden[sl], target[sl])
    return total


def evaluate(model, X, Y) -> float:
    """Mean per-token cross-entropy over a set of sequences."""
    import torch

    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in zip(X, Y):
            total += _sequence_loss_sum(model, x, y).item()
            n += len(y)
    model.train(was_training)
    return total / n


@dataclass
class TrainResult:
    history: pd.DataFrame
    best_step: int
    best_dev_loss: float
    steps_run: int
    stop_reason: str
    seconds: float
    best_state: dict = field(repr=False, default_factory=dict)


def train_lora(
    model,
    X_train,
    Y_train,
    X_dev,
    Y_dev,
    *,
    max_steps: int = 1000,
    target_dev_loss: float = 0.005,
    grad_accum: int = 4,
    lr: float = 2e-4,
    warmup_steps: int = 20,
    eval_every: int = 25,
    seed: int = SEED,
    save_path: str | Path | None = None,
    log=print,
) -> TrainResult:
    """Train the adapter until ``max_steps`` optimizer steps or until the dev
    error drops below ``target_dev_loss``.

    One step = ``grad_accum`` sequences (one sequence per forward pass, no
    padding or truncation), with the loss weighted per token. The dev error is
    measured every ``eval_every`` steps; the adapter weights from the step with
    the lowest dev error are kept, loaded back into ``model`` and saved to
    ``save_path`` (with the loss history as ``<name>_history.csv``).
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup_steps)
    )

    history = [
        {"step": 0, "train_loss": evaluate(model, X_train, Y_train), "dev_loss": evaluate(model, X_dev, Y_dev)}
    ]
    best_dev, best_step, best_state = history[0]["dev_loss"], 0, lora_state_dict(model)
    log(f"step    0 | train {history[0]['train_loss']:.4f} | dev {best_dev:.4f}")

    order, cursor = rng.permutation(len(X_train)), 0
    interval_loss, interval_tokens = 0.0, 0
    stop_reason, step, start = "max_steps", 0, time.perf_counter()
    model.train()
    for step in range(1, max_steps + 1):
        batch = []
        for _ in range(grad_accum):
            if cursor == len(order):
                order, cursor = rng.permutation(len(X_train)), 0
            batch.append(order[cursor])
            cursor += 1
        n_tokens = sum(len(Y_train[i]) for i in batch)
        for i in batch:
            loss_sum = _sequence_loss_sum(model, X_train[i], Y_train[i])
            (loss_sum / n_tokens).backward()
            interval_loss += loss_sum.item()
        interval_tokens += n_tokens
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % eval_every == 0 or step == max_steps:
            dev = evaluate(model, X_dev, Y_dev)
            train = interval_loss / interval_tokens
            history.append({"step": step, "train_loss": train, "dev_loss": dev})
            interval_loss, interval_tokens = 0.0, 0
            if dev < best_dev:
                best_dev, best_step, best_state = dev, step, lora_state_dict(model)
            log(f"step {step:4d} | train {train:.4f} | dev {dev:.4f} | best dev {best_dev:.4f} @ {best_step}")
            if dev < target_dev_loss:
                stop_reason = "target_dev_loss"
                break

    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model, best_state)
    history = pd.DataFrame(history)
    if save_path is not None:
        save_lora(model, best_state, save_path)
        history.to_csv(Path(save_path).with_name(Path(save_path).stem + "_history.csv"), index=False)
    return TrainResult(
        history=history,
        best_step=best_step,
        best_dev_loss=best_dev,
        steps_run=step,
        stop_reason=stop_reason,
        seconds=time.perf_counter() - start,
        best_state=best_state,
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Greedy continuation of ``prompt`` (stops at EOS)."""
    import torch

    device = next(model.parameters()).device
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def generate_batch(
    model, tokenizer, prompts: list[str], max_new_tokens: int = 512, batch_size: int = 8
) -> list[str]:
    """Greedy continuations for many prompts (left-padded batches, longest first)."""
    import torch

    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outputs: list[str] = [""] * len(prompts)
    model.eval()
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        enc = tokenizer([prompts[i] for i in idx], add_special_tokens=False, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
            )
        for i, seq in zip(idx, out[:, enc["input_ids"].shape[1]:]):
            outputs[i] = tokenizer.decode(seq, skip_special_tokens=True)
    return outputs
