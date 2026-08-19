"""Checks the trial generator without loading a model.

The three predicted answers must always be distinct, otherwise the patched
forward pass cannot attribute an outcome to a single mechanism.
"""

import random
import sys
import types

if "torch" not in sys.modules:  # the generator is pure Python; stub the import
    class _NoGrad:
        def __call__(self, fn):
            return fn

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    stub = types.ModuleType("torch")
    stub.no_grad = _NoGrad
    stub.float16 = "float16"
    stub.float32 = "float32"
    stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = stub

from binding_pilot import ATTR_POOL, ENTITY_POOL, make_trial  # noqa: E402


def test_two_groups_is_infeasible():
    rng = random.Random(0)
    try:
        make_trial(rng, 2, ENTITY_POOL, ATTR_POOL, lambda: "")
    except ValueError:
        return
    raise AssertionError("n=2 should be rejected: positional and lexical collide")


def test_three_way_separability():
    rng = random.Random(0)
    no_prefix = lambda: ""  # noqa: E731
    for n in (3, 4, 5, 6):
        for _ in range(2000):
            t = make_trial(rng, n, ENTITY_POOL, ATTR_POOL, no_prefix)
            assert len({t.base_answer, t.positional, t.lexical}) == 3
            assert t.fresh not in t.entities
            assert t.entities[t.perm[t.kp]] == t.lexical
            assert t.entities[t.kp] == t.positional
            assert t.entities[t.k] == t.base_answer
            # the reflexive swap must not disturb the lexical target
            assert t.perm[t.kp] != t.kp


if __name__ == "__main__":
    test_two_groups_is_infeasible()
    test_three_way_separability()
    rng = random.Random(7)
    t = make_trial(rng, 3, ENTITY_POOL, ATTR_POOL, lambda: "")
    print("BASE\n" + t.base_prompt)
    print("\nCOUNTERFACTUAL\n" + t.cf_prompt)
    print("\nREFLEXIVE CONTROL (base with entity at kp replaced)\n" + t.swapped_prompt)
    print(f"\nunchanged={t.base_answer}  positional={t.positional}  "
          f"lexical={t.lexical}  fresh={t.fresh}")
    print("all checks passed")
