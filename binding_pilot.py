"""
Interchange interventions on entity binding: separating positional, lexical and
reflexive retrieval in a small instruction-tuned LM.

Replication pilot for Gur-Arieh et al. (2025), "Mixing Mechanisms: How Language
Models Retrieve Bound Entities In-Context" (arXiv:2510.06182).

Design
------
A base context lists n groups "E_i loves A_i" at positions 1..n, then asks
"Who loves A_k?"; the correct answer is E_k.

A counterfactual context keeps the same entities at the same positions but
permutes the attributes by a permutation pi, and queries the attribute sitting
at position kp. We take the residual stream at the final token of the
counterfactual run and write it into the same layer and position of the base
run. Three candidate answers are then distinguishable:

    positional  -> E_{kp}        the entity at the counterfactual's query index
    lexical     -> E_{pi(kp)}    the entity paired in the base with the
                                 counterfactual's query attribute
    unchanged   -> E_k           the base answer

Trials are generated so that these three entities are always distinct, so a
single patched forward pass yields a three-way mechanism attribution.

Caveat this design cannot resolve on its own: a reflexive pointer to the target
token predicts the same answer as the positional mechanism. The reflexive
control (--reflexive) replaces the entity at position kp in the base context
with a fresh entity F. A positional signal then retrieves F, whereas a
reflexive pointer to E_{kp} has no referent and should fail.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

ENTITY_POOL = [
    "Ann", "Tim", "Joe", "Kate", "Mark", "Lucy", "Paul", "Sara", "Dan", "Emma",
    "Nick", "Rose", "Jack", "Mia", "Sam", "Beth", "Carl", "Jane", "Luke", "Ruth",
    "Adam", "Nina", "Eric", "Cara", "Ivan", "Lena", "Omar", "Tess", "Hugo", "Iris",
]

ATTR_POOL = [
    "pie", "tea", "jam", "soup", "cake", "rice", "beans", "bread", "corn", "milk",
    "ham", "fish", "eggs", "toast", "honey", "salad", "pasta", "juice", "cheese", "candy",
]


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    backend: str = "nnsight"
    n_groups: int = 6
    n_trials: int = 24
    n_shot: int = 2
    seed: int = 0
    dtype: str = "float16"
    n_sweep: tuple = (3, 4, 5, 6)
    out_dir: str = "results"


# ---------------------------------------------------------------- model access


def _unwrap(lm):
    """Return the underlying HF CausalLM from an nnsight LanguageModel."""
    for attr in ("_model", "_envoy", "model"):
        cand = getattr(lm, attr, None)
        if cand is None:
            continue
        cand = getattr(cand, "_module", cand)
        if hasattr(cand, "lm_head"):
            return cand
    raise RuntimeError("could not locate the underlying HF model on the nnsight wrapper")


def _val(x):
    """nnsight <0.4 returns a proxy with .value; >=0.4 returns the tensor."""
    return x.value if hasattr(x, "value") else x


class Runner:
    """Runs forward passes with optional residual-stream patches.

    Two interchangeable backends are provided: nnsight tracing (the primary
    path) and raw PyTorch forward hooks (a fallback, and a cross-check).
    """

    def __init__(self, model_name: str, dtype: str = "float16", prefer_nnsight: bool = True):
        torch_dtype = getattr(torch, dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch_dtype = torch.float32

        self.lm = None
        if prefer_nnsight:
            try:
                from nnsight import LanguageModel

                self.lm = LanguageModel(
                    model_name, device_map=device, torch_dtype=torch_dtype, dispatch=True
                )
                self.hf = _unwrap(self.lm)
                self.tok = self.lm.tokenizer
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] nnsight unavailable ({exc}); falling back to hooks")
                self.lm = None

        if self.lm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.hf = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype
            ).to(device)

        self.hf.eval()
        self.device = device
        self.dtype = torch_dtype
        self.blocks = self.hf.model.layers
        self.n_layers = len(self.blocks)

    # -- backends ---------------------------------------------------------

    def _run_nnsight(self, prompt, patch, save_resid):
        with self.lm.trace(prompt):
            if patch is not None:
                layer, pos, vec = patch
                self.lm.model.layers[layer].output[0][0, pos, :] = vec.to(
                    self.device, self.dtype
                )
            saved = {}
            if save_resid:
                for i in range(self.n_layers):
                    saved[i] = self.lm.model.layers[i].output[0][0, -1, :].save()
            logits = self.lm.output.logits[0, -1, :].save()
        resid = {i: _val(v).detach().float().cpu() for i, v in saved.items()}
        return _val(logits).detach().float().cpu(), resid

    def _run_hooks(self, prompt, patch, save_resid):
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        saved, handles = {}, []

        def make_hook(idx):
            def hook(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                if save_resid:
                    saved[idx] = hs[0, -1, :].detach().float().cpu()
                if patch is not None and patch[0] == idx:
                    _, pos, vec = patch
                    hs[0, pos, :] = vec.to(self.device, self.dtype)
                return out

            return hook

        try:
            for i, block in enumerate(self.blocks):
                handles.append(block.register_forward_hook(make_hook(i)))
            with torch.no_grad():
                out = self.hf(**ids)
        finally:
            for h in handles:
                h.remove()
        return out.logits[0, -1, :].detach().float().cpu(), saved

    def run(self, prompt, patch=None, save_resid=False, backend="nnsight"):
        if backend == "nnsight" and self.lm is not None:
            return self._run_nnsight(prompt, patch, save_resid)
        return self._run_hooks(prompt, patch, save_resid)

    # -- logit lens -------------------------------------------------------

    @torch.no_grad()
    def logit_lens(self, resid_by_layer):
        """Project residual-stream vectors through the final norm and unembedding."""
        norm, head = self.hf.model.norm, self.hf.lm_head
        out = {}
        for layer, vec in resid_by_layer.items():
            h = vec.to(self.device, self.dtype).unsqueeze(0)
            out[layer] = head(norm(h))[0].float().cpu()
        return out

    # -- tokenisation helpers --------------------------------------------

    def single_token_words(self, words):
        """Keep words whose ' word' form is exactly one token.

        Multi-token answers would make candidate-restricted decoding compare
        prefixes rather than whole names, which silently corrupts the readout.
        """
        keep = []
        for w in words:
            ids = self.tok(" " + w, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                keep.append(w)
        return keep

    def first_id(self, word):
        return self.tok(" " + word, add_special_tokens=False)["input_ids"][0]


# ------------------------------------------------------------------- prompting


def render(pairs, query_attr):
    facts = " ".join(f"{e} loves {a}." for e, a in pairs)
    return f"{facts}\nQuestion: Who loves {query_attr}?\nAnswer:"


def few_shot_prefix(rng, entities, attrs, n_groups, n_shot):
    blocks = []
    for _ in range(n_shot):
        ents = rng.sample(entities, n_groups)
        ats = rng.sample(attrs, n_groups)
        k = rng.randrange(n_groups)
        pairs = list(zip(ents, ats))
        blocks.append(render(pairs, ats[k]) + f" {ents[k]}")
    return "\n\n".join(blocks) + ("\n\n" if blocks else "")


@dataclass
class Trial:
    entities: list
    attrs: list
    perm: list
    k: int
    kp: int
    fresh: str
    base_prompt: str = ""
    cf_prompt: str = ""
    swapped_prompt: str = ""
    base_answer: str = ""
    positional: str = ""
    lexical: str = ""
    candidates: list = field(default_factory=list)


def make_trial(rng, n, entities, attrs, prefix_fn):
    # n = 2 is infeasible by construction: kp != k forces kp to be the other
    # index, and perm[kp] != kp then forces perm[kp] == k, which collides with
    # the base answer. Three distinct predictions require at least 3 groups.
    if n < 3:
        raise ValueError("need n_groups >= 3 to separate positional from lexical")

    ents = rng.sample(entities, n)
    ats = rng.sample(attrs, n)
    fresh = rng.choice([e for e in entities if e not in ents])
    k = rng.randrange(n)

    for _ in range(1000):
        perm = list(range(n))
        rng.shuffle(perm)
        kp = rng.randrange(n)
        if kp != k and perm[kp] != kp and perm[kp] != k:
            break
    else:
        raise RuntimeError("failed to sample a valid trial")

    base_pairs = [(ents[i], ats[i]) for i in range(n)]
    cf_pairs = [(ents[i], ats[perm[i]]) for i in range(n)]
    swapped = list(base_pairs)
    swapped[kp] = (fresh, ats[kp])

    t = Trial(entities=ents, attrs=ats, perm=perm, k=k, kp=kp, fresh=fresh)
    t.base_prompt = prefix_fn() + render(base_pairs, ats[k])
    t.cf_prompt = prefix_fn() + render(cf_pairs, ats[perm[kp]])
    t.swapped_prompt = prefix_fn() + render(swapped, ats[k])
    t.base_answer = ents[k]
    t.positional = ents[kp]
    t.lexical = ents[perm[kp]]
    t.candidates = ents + [fresh]
    return t


# ------------------------------------------------------------------ evaluation


def classify(logits, runner, trial, swapped=False):
    """Restrict to candidate answers and label the argmax by mechanism."""
    names = list(trial.candidates)
    ids = torch.tensor([runner.first_id(n) for n in names])
    probs = torch.softmax(logits[ids], dim=-1)
    winner = names[int(probs.argmax())]

    positional = trial.fresh if swapped else trial.positional
    label = "other"
    if winner == positional:
        label = "positional"
    elif winner == trial.lexical:
        label = "lexical"
    elif winner == trial.base_answer:
        label = "unchanged"

    return {
        "winner": winner,
        "label": label,
        "p_positional": float(probs[names.index(positional)]),
        "p_lexical": float(probs[names.index(trial.lexical)]),
        "p_unchanged": float(probs[names.index(trial.base_answer)]),
        "top1_vocab": runner.tok.decode([int(logits.argmax())]),
    }


def clean_accuracy(runner, trials, backend):
    hits = 0
    for t in trials:
        logits, _ = runner.run(t.base_prompt, backend=backend)
        res = classify(logits, runner, t)
        hits += res["winner"] == t.base_answer
    return hits / max(len(trials), 1)


def interchange_sweep(runner, trials, backend, layers=None, swapped=False, random_patch=False):
    """Patch the counterfactual final-token residual into the base run."""
    rows = []
    layers = list(range(runner.n_layers)) if layers is None else list(layers)
    for ti, t in enumerate(trials):
        _, cf_resid = runner.run(t.cf_prompt, save_resid=True, backend=backend)
        target_prompt = t.swapped_prompt if swapped else t.base_prompt
        for layer in layers:
            vec = cf_resid[layer]
            if random_patch:
                g = torch.randn_like(vec)
                vec = g * (vec.norm() / g.norm())
            logits, _ = runner.run(
                target_prompt, patch=(layer, -1, vec), backend=backend
            )
            res = classify(logits, runner, t, swapped=swapped)
            rows.append({"trial": ti, "layer": layer, "kp": t.kp, "k": t.k,
                         "n_groups": len(t.entities), **res})
    return rows


def logit_lens_curve(runner, trials, backend):
    """Per-layer probability and rank of the correct answer at the final token."""
    rows = []
    for ti, t in enumerate(trials):
        _, resid = runner.run(t.base_prompt, save_resid=True, backend=backend)
        lens = runner.logit_lens(resid)
        gold = runner.first_id(t.base_answer)
        for layer, logits in lens.items():
            probs = torch.softmax(logits, dim=-1)
            rank = int((probs > probs[gold]).sum())
            rows.append({"trial": ti, "layer": layer,
                         "p_gold": float(probs[gold]), "rank_gold": rank})
    return rows


def permutation_null(rows, n_perm=2000, seed=0):
    """Null for the positional rate: shuffle which trial each outcome belongs to.

    Breaks the pairing between a patched run and its own positional target
    while preserving the marginal distribution of predicted names.
    """
    rng = random.Random(seed)
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r)
    out = {}
    for layer, rs in by_layer.items():
        observed = sum(r["label"] == "positional" for r in rs) / len(rs)
        winners = [r["winner"] for r in rs]
        null = []
        for _ in range(n_perm):
            shuffled = winners[:]
            rng.shuffle(shuffled)
            null.append(sum(w == r["positional_name"] for w, r in zip(shuffled, rs)) / len(rs))
        out[layer] = {"observed": observed,
                      "null_mean": sum(null) / len(null),
                      "null_p95": sorted(null)[int(0.95 * len(null))]}
    return out


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=Config.model)
    ap.add_argument("--n-groups", type=int, default=Config.n_groups)
    ap.add_argument("--n-trials", type=int, default=Config.n_trials)
    ap.add_argument("--n-shot", type=int, default=Config.n_shot)
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--backend", default="nnsight", choices=["nnsight", "hooks"])
    ap.add_argument("--dtype", default=Config.dtype)
    ap.add_argument("--out-dir", default=Config.out_dir)
    ap.add_argument("--skip-cross-check", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runner = Runner(args.model, dtype=args.dtype,
                    prefer_nnsight=args.backend == "nnsight")
    ents = runner.single_token_words(ENTITY_POOL)
    ats = runner.single_token_words(ATTR_POOL)
    print(f"single-token entities: {len(ents)}/{len(ENTITY_POOL)}, "
          f"attributes: {len(ats)}/{len(ATTR_POOL)}")
    if len(ents) < args.n_groups + 2 or len(ats) < args.n_groups:
        raise SystemExit("not enough single-token names for this tokenizer")

    rng = random.Random(args.seed)

    def prefix_fn():
        return few_shot_prefix(rng, ents, ats, args.n_groups, args.n_shot)

    trials = [make_trial(rng, args.n_groups, ents, ats, prefix_fn)
              for _ in range(args.n_trials)]

    acc = clean_accuracy(runner, trials, args.backend)
    print(f"clean accuracy (n={args.n_groups}): {acc:.3f}")

    if not args.skip_cross_check and runner.lm is not None:
        t = trials[0]
        _, resid = runner.run(t.cf_prompt, save_resid=True, backend="nnsight")
        mid = runner.n_layers // 2
        a, _ = runner.run(t.base_prompt, patch=(mid, -1, resid[mid]), backend="nnsight")
        b, _ = runner.run(t.base_prompt, patch=(mid, -1, resid[mid]), backend="hooks")
        print(f"backend cross-check max|delta| = {(a - b).abs().max():.2e}")

    rows = interchange_sweep(runner, trials, args.backend)
    for r in rows:
        r["positional_name"] = trials[r["trial"]].positional
    (out / "interchange.json").write_text(json.dumps(rows, indent=1))

    rand = interchange_sweep(runner, trials, args.backend, random_patch=True)
    (out / "random_control.json").write_text(json.dumps(rand, indent=1))

    swapped = interchange_sweep(runner, trials, args.backend, swapped=True)
    (out / "reflexive_control.json").write_text(json.dumps(swapped, indent=1))

    lens = logit_lens_curve(runner, trials, args.backend)
    (out / "logit_lens.json").write_text(json.dumps(lens, indent=1))

    null = permutation_null(rows)
    (out / "permutation_null.json").write_text(json.dumps(null, indent=1))

    best = max(null.items(), key=lambda kv: kv[1]["observed"])
    print(f"peak positional rate: layer {best[0]} = {best[1]['observed']:.3f} "
          f"(null mean {best[1]['null_mean']:.3f}, null p95 {best[1]['null_p95']:.3f})")
    print(f"wrote results to {out.resolve()}")


if __name__ == "__main__":
    main()
