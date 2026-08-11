# MNIST Phase 1 ledger — bring-up on a second dataset

Running log for the MNIST port. Plan and ground rules: `docs/mnist-plan.md`.

**Branch:** `mnist`. `main` holds the JSC study, tagged `jsc-complete`.

**What this phase is for:** proving the generator is general, not producing a good MNIST model.
The generalised tool is the deliverable; MNIST is the test that finds the JSC assumptions we
cannot see by inspection.

**Correct entries rather than appending to them.** When a measurement overturns an earlier
conclusion, strike the old one through and say what was retracted and why. A wrong turn that stays
visible is worth more than a tidy log — four conclusions in the JSC study were generalised from a
single configuration and had to be withdrawn, and every one was caught by measuring at a second
point.

---

## Status

| Step | What | Status |
|---|---|---|
| M1a | Derive the feature count; add `datasets/` descriptors | ⬜ next |
| M1b | Thread configurable precision through the flow | ⬜ |
| M1c | Train a small MNIST model (Kaggle, off-machine) | ⬜ |
| M1d | Export and pass Gate 1 bit-exact | ⬜ |
| M1e | Synthesize; measure core / encoder / top separately | ⬜ |
| M1f | Harness record format and vector-store capacity | ⬜ |
| M1g | Gate 1b on the board, full MNIST test set | ⬜ |

**The gate, after every generalisation commit** — `scripts/verify_phase1.py` at **12/12 with areas
108 / 1,519 / 1,619**, plus `run_gate1.py` on the two-layer `300-100` checkpoint. Identical, not
close. See `docs/mnist-plan.md` §1.2.

---

## Numbers to beat, or to fail against honestly

The JSC baseline this generalisation must not disturb:

| | |
|---|---|
| Gate 1 | bit-exact, 1,504 core vectors / 1,518 top vectors |
| `dwn_core` | **108 LUTs**, 73 FF |
| `thermometer_encoder` | **1,519 LUTs**, 0 FF |
| `dwn_top` | **1,619 LUTs**, 269 FF |
| On silicon | 166,000 / 166,000 exact |

What MNIST is walking into, for scale:

| | JSC | MNIST |
|---|---|---|
| input features | 16 | **784** |
| classes | 5 | **10** |
| paper's smallest model | 1 layer, 50 nodes | **2 layers, 1000 + 500** |
| board record | 33 bytes | **~1,569 bytes** at 16-bit |

⚠️ **The paper's MNIST configuration is not expected to fit.** At the per-comparator cost measured
on JSC, its encoder alone extrapolates to roughly 67,000 LUTs against this device's 20,800 — three
times over before the network. That is an extrapolation across a different feature count and
threshold distribution, so it is a reason to start small and measure, not a prediction to quote.

---

## Log

*(entries go here, newest first, dated)*

---

## Open questions

| Question | Status |
|---|---|
| How many thermometer thresholds per pixel is right? | ⬜ Open. 784 × z is the entire area problem; JSC's 200 is almost certainly impossible here |
| Are MNIST pixels standard-scaled or min-max normalised? | ⬜ Open. Decides how many integer bits the word needs, and M1b depends on the answer |
| Does the upstream MNIST recipe binarise differently from JSC? | ⬜ Open. `docs/checkpoint-format.md` was verified against JSC checkpoints only |
| How many MNIST vectors fit in the vector store? | ⬜ Open. Sets whether Gate 1b is one pass or many |
| Does the paper's `1000, 500` fit at any precision? | ⬜ Open. A negative answer is a result, not a failure |
