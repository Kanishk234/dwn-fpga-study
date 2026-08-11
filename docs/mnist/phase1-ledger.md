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
| M1a | Derive the feature count; add `datasets/` descriptors | ✅ done 2026-08-11 — JSC identical |
| M1b | Thread configurable precision through the flow | ✅ done 2026-08-11 — JSC identical, Gate 1 passes at 11-bit |
| M1c | Train a small MNIST model (Kaggle, off-machine) | ⬜ **next** — `1x300`, **z=3**, n=6, min-max (T3) |
| M1d | Export and pass Gate 1 bit-exact | 🟡 **mechanically proven 2026-08-11** on a synthetic 784/10 checkpoint; needs a real one |
| M1e | Synthesize; measure core / encoder / top separately | 🟡 **synthetic measured**: 3,233 LUTs (15.5%), **fails timing at 91.5 MHz** |
| M1f | Harness record format and vector-store capacity | ⬜ **in scope 2026-08-11** — MNIST goes on the board |
| M1g | Gate 1b on the board, full MNIST test set | ⬜ **in scope 2026-08-11** |

**Scope, 2026-08-11 — two questions, and only one is answered.**

**The tool ships generator-only:** synthesizable Verilog for the network, no harness, because the
harness changes with every application and dataset. That is a decision about a deliverable *after*
a successful port.

**Whether MNIST runs on our board here is still open**, and does not follow from it. M1a–M1e are
unaffected either way; M1f and M1g exist only if the answer is yes. See `docs/mnist/plan.md` §3 for
what they would cost — the short version is a 1,569-byte record against JSC's 33, a vector store
about 48× wider per entry, and roughly 23 minutes to stream the test set at 115,200 baud. Real
work, not prohibitive, and `LABEL_W` is already a parameter.

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

### 2026-08-11 — the generator works at MNIST's shape, measured, before any training

`experiments/make_test_checkpoint.py --dataset mnist` (added by the other machine in `42d9b0b`,
on top of the `datasets/` descriptors from M1a — the two tracks met without coordination). It
fabricates a checkpoint at any dataset shape, so **Gate 1 can run at 784 features and 10 classes
with nothing trained.**

That is legitimate because Gate 1 verifies the *emitter*, not the model: both sides are derived
from the same checkpoint, so random tables exercise the machinery as well as learned ones, and
better in one respect — they reach address patterns a trained model may never produce. It is the
argument `make_test_checkpoint.py` was written for, and it is what caught the `np.packbits` shift
at n<3.

#### Gate 1 PASSES at MNIST's shape

`1x300`, z=25, n=6, 784 features, 10 classes, Q0.8 nine-bit words:

```
1,734 distinct bits selected of 19,600
dwn_core : 1,504 vectors, 0 mismatches, PASS
dwn_top  : 2,199 vectors, 0 mismatches, PASS
```

**The mechanical half of M1d is answered.** Port widths, wiring indices, address ordering and the
4-bit class index are all correct at a shape nothing has been trained at.

#### And it synthesizes — with two findings

Place-and-routed, `xc7a35tcpg236-1`, 10 ns:

| | projected | **measured** |
|---|---|---|
| comparators | 1,353 | **1,734** |
| encoder | 1,434 | **2,607** |
| core | 684 | **641** |
| top | 2,557 | **3,233 (15.5%)** |
| timing | — | ❌ **FAILS — 91.5 MHz** |

**⚠️ The area projection was 26% low, and the encoder specifically was 82% low.** Two errors
compounded: 28% more comparators than the occupancy model predicted, and **1.50 LUTs per
comparator against the 1.06 measured on JSC at nine bits.** The 49x extrapolation on feature count
does not hold as well as the 2026-08-11 threshold analysis implied. Treat every number in that
entry as indicative, not predictive.

**⚠️ It misses the board clock at 300 nodes.** 91.5 MHz against the required 100, with the core
itself at 94.2. JSC's `1x50` closed 147 MHz. Ten classes make the popcount groups wider relative
to layer width, and MNIST's ladder goes to 2,000 nodes — so **timing, not area, is likely to be
MNIST's binding constraint too**, exactly as it was for JSC. Pipeline depth is a synthesis-side
sweep needing no retraining, and the harness already parameterises it.

#### What this does NOT establish

Real MNIST pixels are **mostly zero**, so quantile-placed thresholds will collapse onto duplicates
in a way uniform synthetic data never does. Duplicate thresholds cost nothing in hardware, so the
real design is likely **cheaper** than the numbers above. Nothing here says anything about
accuracy, about whether upstream's checkpoints have the structure we expect, or about Gate 1b.
The trained models are still required for every result.

### 2026-08-11 — T3: what upstream actually does for MNIST, and it is not what we assumed

Read from `third_party/DWN/examples/mnist.py` and `src/torch_dwn/`, at the pinned commit. Three
risks retired, one assumption overturned.

#### ✅ The checkpoint format holds

Upstream binarises MNIST with the **same `DistributiveThermometer`** class as JSC, and
`feature_wise=True` is the default, so thresholds are per-feature quantiles exactly as
`docs/checkpoint-format.md` describes. No new format work.

#### ✅ Both wiring representations are already exercised

Upstream's MNIST stacks `LUTLayer(..., mapping='learnable')` then `LUTLayer(...)` — and
`LUTLayer`'s default is **`mapping='random'`**, a fixed mapping. So layer 2 uses the `§3b` fixed
path, not the learnable one.

That could have been a nasty surprise. It is not: our `300-100` JSC checkpoint is **learnable then
fixed**, so both paths are already Gate 1 verified. Recorded so nobody re-audits it.

#### ✅ No StandardScaler, and nothing needs one

Upstream feeds `transforms.ToTensor()`, which puts pixels in **[0, 1]** — no scaler at all, unlike
our JSC path. Checked whether that breaks us: `exporter/`, `rtlgen/` and `tb/` never read
`ck['scaler']`; only a comment in `emit_encoder.py` mentions it. So an MNIST checkpoint without a
scaler flows through unchanged. The descriptor's `scaling='minmax'` is right.

#### ⚠️ Upstream uses `DistributiveThermometer(3)` — z=3, not the 25 we planned

This is the overturned assumption. **Three thresholds per pixel**, giving 784 x 3 = 2,352
thermometer bits, against the 19,600 our z=25 recommendation implied.

It is consistent with the 2026-08-11 threshold finding rather than contradicting it: MNIST is
slot-limited, so `z` buys little area either way. But it means the accuracy knob probably sits far
lower than JSC's 200, and upstream — who trained these models — chose 3.

Projected at Q0.8, nine bits:

| config | z=3 | z=8 | z=25 |
|---|---|---|---|
| `1x300` | **9.4%** | 10.6% | 12.3% |
| `1x1000` | 21.6% | 26.3% | 31.1% |
| `2000, 1000` (upstream's own) | **32.9%** | 41.7% | 52.1% |

**Upstream's full MNIST model projects to roughly a third of the device.** That is a far better
starting position than the paper's `1000, 500` at 16 bits, which was over the device entirely.

**Revised bring-up recommendation: `1x300`, z=3, n=6, min-max to [0,1].** Matching upstream's `z`
removes a variable — if accuracy disappoints, that is a training question, not a suspicion about
our binarisation. Sweep `z` afterwards, when it is an accuracy knob rather than a guess.

#### Also worth noting

Upstream's example is `2000, 1000`, not the paper's `1000, 500`, and uses `tau = 1/0.3`. Neither
blocks anything, but the two sources disagree about what "the MNIST model" is, and the paper's
Table 14 is the one `docs/paper-configs.md` records.

### 2026-08-11 — M1b: precision is a parameter, and the golden model moves with it

The fixed-point format was a constant in `exporter/extract.py` (`FRAC_BITS = 12`,
`WORD_BITS = 16`) that everything downstream read. Building the same checkpoint at another width
meant editing source. **At 16 bits nothing MNIST-shaped fits this device** — the paper's
configuration is 102.5% at its cheapest and 160% at z=50 — so this was a blocker, not a tidy-up.

**JSC is unchanged:** 12/12 with areas 108 / 1,519 / 1,619, the emitted RTL byte-identical to
`rtl/example-model-1x50/`, and the two-layer `300-100` checkpoint still bit-exact.

| file | change |
|---|---|
| `exporter/extract.py` | `required_int_bits()` — derives the exact integer-bit floor from the thresholds |
| `rtlgen/emit_encoder.py` | `--word-bits` / `--frac-bits`; refuses a word too narrow for the thresholds |
| `tb/gen_vectors.py` | the same flags, so the golden model quantises identically |
| `scripts/run_gate1.py` | takes both and passes **one shared list** to the encoder and the vector generator |
| `rtlgen/config.py` | comment only |

#### The one detail that matters

`run_gate1` builds the precision flags **once** and hands the same list to both the encoder emitter
and the golden-model generator. Give those two different widths and Gate 1 compares two *different
designs* and still reports PASS — a green light on a comparison that means nothing. Constructing
the list in one place makes that unrepresentable rather than merely unlikely.

#### What is derivable, and what is not

Following the V4 argument in `docs/tool-roadmap.md` §4, which is the right one:

**Integer bits are derivable exactly.** `required_int_bits()` returns the floor below which a
threshold cannot be represented at all, and the emitter refuses to build there:

```
ABORT: Q0.7 cannot represent this model. Its thresholds span +/-4.545 and need 3
integer bits; a 8-bit word with 7 fractional bits has 0.
```

**Fractional bits are not.** Whether quantisation changes predictions depends on the data, not the
checkpoint. `required_int_bits()`'s docstring says so and cites the scar: the encoder-narrowing
result in `REPORT.md` §5.6 was fitted and validated on the same 1,000 samples, and 8 of 15 features
came out too narrow. **The code reports a floor as a floor and never calls a width safe.**

#### Verification

| test | result |
|---|---|
| default vs the committed `1x50` reference | byte-identical, all files |
| **11-bit (Q3.7), Gate 1** | **PASS — bit-exact on 1,518 vectors** |
| two-layer `300-100`, Gate 1 | PASS |
| word too narrow for the thresholds | refused, with the numbers |
| `verify_phase1.py` | 12/12, areas identical |

**The 11-bit pass is the substantive one.** Everything else confirms nothing regressed; that one
shows the golden model and the RTL agree at a width neither was written for, which is the property
MNIST depends on.

#### A correction to `docs/tool-roadmap.md`

F1 read `rtlgen/config.py:186-187` as asserting that the config equals the module constants, and
therefore as an obstacle. It checks `HardwareConfig()` — the **default** instance — so it means
"the default agrees with the module that owns it", which is a drift guard worth keeping. Nothing
needed relaxing. A comment now says so, in case a later reader takes it for the blocker.

#### Not done here, deliberately

`scripts/host.py` still reads the module constants. It is the board path, which is decision-pending
under M1f, and threading it halfway would leave the host and the Verilog loader disagreeing about a
wire format. It moves as one piece or not at all.

### 2026-08-11 — M1a: the feature count is derived, and `datasets/` exists

Two generalisation changes, both gated. **JSC is unchanged: 12/12 with areas 108 / 1,519 / 1,619,
and the two-layer `300-100` checkpoint still passes Gate 1 bit-exact on 1,519 vectors.**

#### The defect

`rtlgen/emit_core.py` computed the core's input port width as `16 * cfg['thermometer_bits']` —
JSC's feature count written into the emitter. Now derived from the thresholds themselves
(`thresholds.size`, which is features × z), matching what `rtlgen/emit_encoder.py` already did.

**This was wrong on `main` today**, not merely wrong for MNIST. It is the dangerous kind: a wrong
port width elaborates and synthesizes, then disagrees with the encoder feeding it. Nothing errors.

`scripts/host.py`'s self-test had the same constant twice — `rec[:WORD_BITS * 16 // 8]` and a
literal `!= 33`. Both now derive from the record. `pack_record` itself was already general.

**A fast check worth reusing:** `rtl/example-model-1x50/` is the committed emitted RTL for the
Phase 1 config. Regenerating it and diffing gives a byte-exact answer in seconds, covering table
contents, wiring indices and port widths, long before Vivado has an opinion. All five files came
back identical.

#### `datasets/` — the contract, and it caught a bug immediately

One frozen `Dataset` descriptor per dataset: dimensions, scaling, fixed-point format, record
layout, sweep axes. `check_checkpoint()` raises if a descriptor disagrees with a trained model,
because a silently wrong descriptor produces an export that elaborates and is wrong.

| | features | classes | format | record |
|---|---|---|---|---|
| `jsc` | 16 | 5 | Q3.12 | **33 B** |
| `mnist` | 784 | 10 | Q0.8 | 1,569 B |

JSC's 33 bytes is **derived and matches the constant the UART loader was written around**, which
is the check that the derivation is correct rather than merely plausible.

⚠️ **The contract found a real bug within the hour.** `record_bytes()` first used
`word_bits // 8`, which returns **one byte for an 11-bit word** and silently truncates every
feature. Fixed with ceiling division. **`scripts/host.py`'s `pack_record` still makes the same
assumption** — deliberately left, because it shares a wire format with `harness/uart_loader.v` and
the two must change together, which is M1f.

#### That bug forced a better decision on MNIST precision

The descriptor initially carried 11 bits, taken from the JSC measurement. Wrong reasoning: MNIST
pixels are 8-bit integers and min-max scaled, so **only 256 distinct input values exist** and
**Q0.8 — nine bits, one sign and eight fractional — represents them exactly.** JSC's 0.4 pp loss at
nine bits came from truncating *continuous* features; there is nothing to truncate in data that
arrives already quantised. Nine bits also sits on the cheap side of the measured area cliff.

Still provisional. On JSC the accuracy-safe width moved between two configurations of the *same*
dataset, so it cannot be assumed to transfer to a different one.

#### Not done here, deliberately

`dse/area_model.py` still has `JSC_FEATURES = 16`. It is outside the directories §1.3 covers, and
it belongs to the R1 recalibration, but it should read from the descriptor when that happens.

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
| ~~How many thermometer thresholds per pixel is right?~~ | ✅ **Settled 2026-08-11.** `z` is cheap here — MNIST is slot-limited, so z=8→200 costs only 2.3× in comparators. **Upstream uses z=3** (T3), so start there rather than the 25 first guessed, and treat `z` as an accuracy knob afterwards. |
| ~~Are MNIST pixels standard-scaled or min-max normalised?~~ | ✅ **Closed 2026-08-11: min-max, [0,1]**, via `transforms.ToTensor()`. No scaler in the pipeline, and nothing in our flow requires one |
| ~~Does the upstream MNIST recipe binarise differently from JSC?~~ | ✅ **Closed 2026-08-11: no.** Same `DistributiveThermometer`, `feature_wise=True`. Both wiring paths (learnable, fixed) already Gate 1 verified by our `300-100` checkpoint |
| How many MNIST vectors fit in the vector store? | ⬜ Open. Sets whether Gate 1b is one pass or many |
| Does the paper's `1000, 500` fit at any precision? | ⬜ Open. A negative answer is a result, not a failure |
| ~~Does the **tool** ship a harness?~~ | ✅ **Closed 2026-08-11: no, generator-only.** Users bring their own, as hls4ml does. The encoder still ships — intrinsic to a DWN and most of the area, so omitting it would repeat the reporting failure `REPORT.md` §5.2 criticises |
| ~~Does MNIST run on our board in this repo?~~ | ✅ **Closed 2026-08-11: yes.** M1f and M1g are in scope. Separate from the tool question — the tool still ships generator-only; the board demo is this project's own deliverable, and a second dataset on real silicon is a far stronger result than a second dataset in simulation |
| How many MNIST vectors fit in the vector store? | ⬜ Open, and only matters if the board answer is yes. ~48× wider per vector than JSC, so `DEPTH` falls to order 100 |
