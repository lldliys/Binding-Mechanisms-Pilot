# Separating positional, lexical and reflexive binding with interchange interventions

A small pilot replicating the central distinction in Gur-Arieh et al. (2025),
*Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context*
([arXiv:2510.06182](https://arxiv.org/abs/2510.06182)), on a 1.5B
instruction-tuned model using [`nnsight`](https://nnsight.net).

**Start here:** [`binding_mechanisms_pilot.ipynb`](binding_mechanisms_pilot.ipynb) —
the design is in the first cell, the two figures worth looking at are
`figures/mechanism_rates.png` and `figures/controls.png`.

## The idea in one paragraph

A base context lists `n` groups `E_i loves A_i` and asks `Who loves A_k?`. A
counterfactual context keeps the same entities at the same positions but permutes
the attributes, and queries the attribute now at position `kp`. Copying the final
token's residual stream from the counterfactual run into the base run makes three
answers distinguishable at once: the entity at index `kp` (positional retrieval),
the entity paired in the base with the counterfactual's query attribute (lexical
retrieval), or the original answer (no effect). Trials are constructed so those
three entities are always distinct, so one patched forward pass yields a three-way
attribution instead of a binary one.

Two things fell out of building it that I did not expect going in:

- **`n = 2` is impossible.** With two groups the constraints force the lexical
  prediction to coincide with the base answer, so separating positional from
  lexical retrieval needs at least three groups. `test_trials.py` asserts this.
- **Positional and reflexive retrieval are not separable in this design**, because
  the entities occupy the same positions in both contexts, so a pointer to the
  target token predicts the same answer as a position index. The reflexive control
  replaces the entity at position `kp` in the base context with a fresh entity: a
  positional signal then retrieves the new entity, while a reflexive pointer has no
  referent.

## Controls

Three, because a raw intervention effect is easy to over-read:

1. **Norm-matched random direction** at the same layer and position — bounds how much
   of the effect is just perturbing the final token.
2. **Permutation null** on the positional rate — shuffles which outcome belongs to
   which trial, preserving the marginal distribution of predicted names, so
   name-frequency biases are priced in.
3. **Backend cross-check** — the same intervention is implemented twice, once with
   `nnsight` tracing and once with raw PyTorch forward hooks, and the notebook
   asserts the logits agree to floating-point noise. This is the cheapest guard
   against a silent off-by-one in the patch.

Candidate names are filtered to those that are a single token with a leading space;
otherwise candidate-restricted decoding silently degrades into a prefix match.

## Running it

```bash
pip install -r requirements.txt
python -m pytest test_trials.py -q          # design invariants, no model needed
python binding_pilot.py --n-groups 6 --n-trials 24
```

Defaults to `Qwen/Qwen2.5-1.5B-Instruct` (ungated); `--model
meta-llama/Llama-3.2-1B-Instruct` also works. Roughly ten minutes on a Colab T4.
Pass `--backend hooks` to skip `nnsight` entirely.

## Scope

One model, one prompt template, 24 trials per condition, and patching the full
residual stream rather than individual attention heads. This localises the effect in
depth, not in the heads that write it; attributing it to heads, and asking whether
positional and lexical signals land in shared or separate subspaces, needs DAS
rather than plain patching. The numbers here are indicative, not tight.
