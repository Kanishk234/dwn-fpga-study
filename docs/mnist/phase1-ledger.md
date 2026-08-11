# MNIST Phase 1 ledger — bring-up on a second dataset

Running log for the MNIST port. Plan and ground rules: `docs/mnist/plan.md`.

**Branch:** `mnist`. `main` holds the JSC study, tagged `jsc-complete`.

**What this phase is for:** getting a real MNIST model onto this device, bit-exact, and
generalising the flow in the process. **Both outcomes count** — MNIST accuracy is a second
dataset's worth of results, comparable against published MNIST rows (PolyLUT 96% / 70,673 LUTs,
NeuraLUT 96% / 54,798) and, for the first time in this project, against the weightless lineage
itself: ULEEN and BTHOWeN benchmark on MNIST and report no JSC at all.

Phase 1 is bring-up only. Accuracy work belongs to the sweep that follows it, exactly as JSC's
did.

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
| M1b | Thread configurable precision through the flow | ⬜ **blocker** — nothing fits at 16-bit |
| M1c | Train a small MNIST model (Kaggle, off-machine) | ⬜ |
| M1d | Export and pass Gate 1 bit-exact | ⬜ |
| M1e | Synthesize; measure core / encoder / top separately | ⬜ |
| M1f | Harness record format and vector-store capacity | ⬜ |
| M1g | Gate 1b on the board, full MNIST test set | ⬜ |

**The gate, after every generalisation commit** — `scripts/verify_phase1.py` at **12/12 with areas
108 / 1,519 / 1,619**, plus `run_gate1.py` on the two-layer `300-100` checkpoint. Identical, not
close. See `docs/mnist/plan.md` §1.2.

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

⚠️ ~~The paper's MNIST configuration is not expected to fit — its encoder alone extrapolates to
roughly 67,000 LUTs.~~ **Retracted 2026-08-11, see the log.** That assumed every input slot reads a
distinct threshold; the learned mapping reuses them heavily. Corrected: the paper's configuration
is **over at 16-bit (102.5%) and fits comfortably at 11-bit (38.0%)**. Word width, not `z`, is what
decides whether MNIST runs here.

---

## Log

### 2026-08-11 — threshold analysis: `z` is not the constraint for MNIST, word width is

`experiments/mnist_threshold_analysis.py`. This reverses the intuition carried over from JSC and
retracts a number written into this ledger yesterday.

**Method.** Comparator counts come from `dse/area_model.predict_comparators`, an occupancy model
with a fitted correction. Validated first against **37 JSC checkpoints spanning z=8-800 and widths
65-3000: 3.9% mean error, 10.9% worst.** LUTs-per-comparator are *measured*, not modelled, from
`experiment_encoder_area.py` — the area model scales cost linearly with word width and that is
wrong, because of the cliff between 12 and 11 bits.

#### ⚠️ RETRACTION: the "~67,000 LUTs of encoder" estimate was wrong

This ledger's own scale table said the paper's MNIST configuration extrapolates to roughly 67,000
LUTs of encoder. That assumed every one of the first layer's `6 x 1000 = 6,000` input slots reads a
**distinct** thermometer bit. It does not. The learned mapping reuses bits heavily and saturates:
the model puts it at **2,425 distinct comparators at z=8, rising only to 5,638 at z=200**.

Corrected, for `1000, 500`:

| word | encoder | core | total | device |
|---|---|---|---|---|
| 16-bit, z=8 | 18,697 | 2,178 | 21,314 | **102.5% — over** |
| 16-bit, z=50 | 30,678 | 2,178 | 33,295 | 160.1% — over |
| **11-bit, z=50** | **5,292** | 2,178 | **7,909** | **38.0%** |
| 8-bit, z=50 | 3,502 | 2,178 | 6,118 | 29.4% |

#### The finding, and it inverts the JSC conclusion

**Comparators grow with `z` far more slowly than they did on JSC.** From z=8 to z=200 — a 25x
increase — the count rises only 2.3x, and total device use goes 28% to 49%.

The reason is which side of the ceiling each dataset sits on. On JSC, `16 x z` gave 3,200 bits
against a few hundred slots, so the *pool* was the binding limit and `z` set it directly — which is
why cutting z from 200 to 50 there saved 40% of the silicon. On MNIST, `784 x z` is 6,272 bits at
z=8 alone, against 6,000 slots, so the encoder is **slot-limited, not pool-limited.** Adding
thresholds mostly adds bits nobody reads.

**So the JSC lesson "z is the expensive axis" does not transfer.** For MNIST:

- **At 16-bit the paper's configuration does not fit at any z**, z=8 included, at 102.5%.
- **At 11-bit every configuration tested fits**, from 28% to 49% of the device.

**Configurable precision (M1b) is therefore a blocker, not an optimisation.** It is what decides
whether MNIST runs on this board at all. That is a change in its priority, and the plan's ordering
already puts it second, which now looks right for a reason we had not established.

#### Bring-up candidates, single layer, 11-bit

| config | z | comparators | encoder | core | total | device |
|---|---|---|---|---|---|---|
| `1x200` | 25 | 999 | 1,329 | 448 | 2,216 | 10.7% |
| **`1x300`** | **25** | **1,353** | **1,799** | **684** | **2,923** | **14.1%** |
| `1x500` | 25 | 1,990 | 2,647 | 1,178 | 4,264 | 20.5% |
| `1x1000` | 50 | 3,979 | 5,292 | 2,463 | 8,194 | 39.4% |

**Recommended for M1c: `1x300`, z=25, n=6.** Small enough that a failure means one thing, large
enough to exercise 784 features and 10 classes, and the width divides by 10 so the reduction is
even. `1x200` also works; `256` does not, because the final layer must divide by the class count.

#### What could make all of this wrong

- **A 49x extrapolation on the one axis never swept.** All 37 calibration configs have 16 input
  features. The occupancy mathematics is dimension-agnostic but the fitted correction is not.
- **MNIST pixel statistics are nothing like JSC's.** Most pixels are zero most of the time, so
  quantile-placed thresholds will collapse onto identical values in a way JSC's continuous physics
  features never did. Duplicate thresholds cost nothing, so the real comparator count is probably
  **lower** than modelled — this is likely pessimistic, but it is untested either way.
- **The 11-bit cliff was measured on JSC's threshold distribution.** Whether it sits at the same
  width for MNIST is unknown, and the JSC evidence is that the safe width *moved* between two
  configs of the same dataset.
- **8-bit is natural for MNIST** — pixels are natively 8-bit integers — but JSC lost 1.09 pp at 8
  bits, so it needs its own accuracy measurement, not an assumption.

**None of this replaces synthesizing one real MNIST configuration, which is M1e.**


---

## Open questions

| Question | Status |
|---|---|
| ~~How many thermometer thresholds per pixel is right?~~ | ✅ **Largely settled 2026-08-11: `z` is cheap here.** MNIST is slot-limited, not pool-limited, so z=8→200 costs only 2.3× in comparators. Use **z=25** for bring-up and treat z as an accuracy knob later, not an area one. |
| Are MNIST pixels standard-scaled or min-max normalised? | ⬜ Open. Decides how many integer bits the word needs, and M1b depends on the answer |
| Does the upstream MNIST recipe binarise differently from JSC? | ⬜ Open. `docs/checkpoint-format.md` was verified against JSC checkpoints only |
| How many MNIST vectors fit in the vector store? | ⬜ Open. Sets whether Gate 1b is one pass or many |
| Does the paper's `1000, 500` fit at any precision? | ⬜ Open. A negative answer is a result, not a failure |
