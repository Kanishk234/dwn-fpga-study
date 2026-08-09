# The datapath: what each stage actually does

A walkthrough of one jet passing through the design, stage by stage. Config is `1x50` — 50
nodes, n=6, z=200, 5 classes — the Phase 1 reference and the sweep's first ladder rung.

This is the explanatory companion to the RTL. For *what was measured*, see
`docs/phase2-report.md`; for the checkpoint fields each stage is built from, see
`docs/checkpoint-format.md`.

---

## The shape

The datapath does four different jobs, so there are four natural places for a register:

```
16 features (Q3.12)
      │
  ┌───▼──────────────┐
  │  ENCODER         │   202 comparators, 16-bit each        ← deep-ish
  └───┬──────────────┘
   [pipe_enc]  ────────────────────────────────── register #1
      │
  ┌───▼──────────────┐
  │  LUT LAYER       │   50 nodes, each ONE table lookup     ← very shallow
  └───┬──────────────┘
   [pipe_lut]  ────────────────────────────────── register #2
      │
  ┌───▼──────────────┐
  │  POPCOUNT        │   5 adder trees, 10 bits → 5 counts   ← DEEPEST
  └───┬──────────────┘
   [pipe_pop]  ────────────────────────────────── register #3
      │
  ┌───▼──────────────┐
  │  ARGMAX          │   compare 5 counts, pick the biggest  ← shallow
  └───┬──────────────┘
   [pipe_out]  ────────────────────────────────── register #4
      │
   class 0–4
```

Each register is a 0/1 switch (`rtl/pipe_reg.v`, `ENABLE=0` compiles it out to a plain wire).
Latency in cycles is their sum, derived in `HardwareConfig.latency` and never hand-copied —
`benchmark_fsm` aligns labels with it, and a drifted value silently scores every sample against
the wrong answer.

---

## What arrives

A jet, described by **16 numbers** — mass, multiplicity, substructure ratios like `zlogz` and
`c1_b2_mmdt`. Each is a Q3.12 fixed-point value: 16 bits, 12 of them fractional, range ±8. So
256 bits enter the chip.

The job is to turn that into one of 5 class labels, using **no arithmetic** — except in the
first stage, which exists precisely because the outside world speaks numbers and the network
speaks bits.

---

## Stage 1 — ENCODER: numbers → bits

**What it does:** compares each feature against a list of learned thresholds, producing a
thermometer bit pattern.

```
feature 14 (mass_mmdt) = 1.7500

  > -0.42 ?  → 1
  >  0.13 ?  → 1
  >  1.05 ?  → 1
  >  1.63 ?  → 1
  >  1.88 ?  → 0      ← the mercury stops here
  >  2.41 ?  → 0
   ...
```

**The hardware:** one comparator per threshold. z=200 means each feature *could* have 200 of
them, so 16 × 200 = **3,200 possible bits**. But only bits that some node actually reads need to
exist — the other comparators drive nothing and Vivado deletes them. At `1x50` the wiring
selects **202 distinct thresholds**, so 202 comparators get built.

**Cost:** ~7.5 LUTs per 16-bit comparator × 202 = **1,519 LUTs**. That's 94% of the whole
design. This stage is the entire reason the encoder discussion dominates the project.

**Why it's "deep-ish":** a 16-bit comparison is a carry chain — you can't know if `A > B` until
you've resolved the high bits. Several levels of logic, though Vivado maps it efficiently.

**Output:** the selected thermometer bits.

---

## Stage 2 — LUT LAYER: bits → learned pattern detections

**What it does:** 50 independent nodes. Each one looks at **6 specific bits** and answers
yes/no.

```verilog
assign out = TABLE[addr];        // rtl/lut_node.v — that's the whole module
```

That single line is the entire neuron. Two things were learned during training and are now
frozen into the RTL:

1. **Which 6 bits this node reads** — the *wiring*, from `argmax` over the mapping weights. At
   inference it's literally just wires; it costs zero logic.
2. **What it answers for each of the 64 possible input combinations** — the *table*, 64 bits
   baked in as a Verilog parameter.

So a node is asking a learned question like *"is mass above 1.6 AND multiplicity below 40
AND …"* — except it can express **any** boolean function of its 6 inputs, not just AND. All
2⁶⁴ possible functions are reachable, because the table is just 64 stored bits.

**Cost: exactly 1 LUT per node.** 50 nodes = 50 LUTs, measured directly
(`scripts/experiment_reduction.py`). A 6-input truth table *is* what a Xilinx LUT6 physically
is — this is the architectural bet the whole project rests on, and it's why n>6 is rejected
outright.

**Why it's "very shallow":** the signal passes through **one** lookup and it's done. This is the
fastest stage in the design by a wide margin, and the reason DWN is interesting at all.

> ⚠️ One load-bearing detail: address bits are **LSB-first** — mapping slot 0 is address bit 0,
> matching upstream's CUDA kernel. Reverse it and you get a design that elaborates, synthesizes,
> and is wrong on most inputs (`docs/checkpoint-format.md` §2).

**Output:** 50 bits, one per node.

---

## Stage 3 — POPCOUNT: votes → scores

**What it does:** this is where the class decision gets formed, and the intuition is **voting**.

The 50 nodes are split into 5 contiguous blocks of 10 — one block per class:

```
nodes  0– 9  →  votes for class 0
nodes 10–19  →  votes for class 1
nodes 20–29  →  votes for class 2
nodes 30–39  →  votes for class 3
nodes 40–49  →  votes for class 4
```

Each block just **counts how many of its 10 nodes fired**:

```
class 0:  1 0 1 1 0 0 1 0 1 1   →  6
class 1:  0 0 1 0 0 1 0 0 0 0   →  2
class 2:  1 1 1 1 0 1 1 1 0 1   →  8    ← strongest
class 3:  0 1 0 0 1 0 0 0 1 0   →  3
class 4:  1 0 0 1 0 0 1 1 0 0   →  4
```

Five scores, each 0–10, so 4 bits each. That's `GroupSum` from the software model.

**"Contiguous" is load-bearing:** the final layer width must divide evenly by 5, or PyTorch
silently zero-pads and hardware and software end up disagreeing about where the group boundaries
are. `ModelConfig.__post_init__` rejects it before a GPU run gets spent.

**And this is where `tau` disappears.** Software divides each sum by tau — but it's the *same*
constant on all five, so it cannot change which is largest. Hardware skips it entirely.

**Cost: ~58 LUTs — more than the 50 nodes themselves.** Measured standalone. The paper warns
about exactly this: in small models "the popcount circuit can be as large as the network."

**Why it's the DEEPEST stage:** summing 10 bits is an **adder tree**. You add pairs, then add
those results, then those — stacked levels, each waiting on the one below. And it gets worse
with width: at `1x2400` each group is 480 bits, so the tree is far taller.

**This is the critical path of the entire design.** It's why dropping `pipe_pop` costs **32 MHz**
while dropping `pipe_out` costs 5.

---

## Stage 4 — ARGMAX: scores → the answer

**What it does:** find which of the 5 scores is biggest, output its index.

```verilog
if (scores_flat[c*W +: W] > best) begin ... end     // rtl/argmax.v
```

Four comparisons on 4-bit numbers. Trivially cheap and shallow.

**But the `>` is a spec decision, not a detail.** With only 11 possible scores, ties are
*common* — **29 of the 1000 Gate 1 vectors (2.9%) have two classes sharing the top score**.
numpy and torch both return the lowest index on a tie, so hardware must too. Using `>=` would
keep the highest tied index instead, and the design would disagree with the golden model on ~3%
of inputs while looking perfect on the other 97%. That's exactly the class of bug Gate 1 exists
to catch and spot-checking never would.

**Output:** 3 bits — class 0 to 4. Done.

---

## The whole pipeline, with what each stage costs

| stage | job | LUTs @ `1x50` | logic depth |
|---|---|---|---|
| **ENCODER** | 16 numbers → thermometer bits | **1,519** (94%) | medium — carry chains |
| **LUT LAYER** | 50 learned 6-input questions | **50** | **1 lookup** — shallowest |
| **POPCOUNT** | count votes per class | **~58** | **adder trees — deepest** |
| **ARGMAX** | pick the winner | ~small | shallow |

Two things fall straight out of this table, and they're the two central findings of the project:

**The area story:** the *network* is 50 LUTs. Getting numbers into it costs 1,519. That's the
14× encoder ratio — and it's why Phase 2's discovery that the ratio **inverts to 2.8×** at large
widths mattered so much: as you add nodes, the core grows at 1 LUT each while comparators
saturate at the `16 × z` ceiling.

**The timing story:** the stage that does the actual neural network computation is the *fastest*
one. The two slow stages are the plumbing on either side — converting numbers to bits, and
counting the votes. That's why pipeline registers matter where they do, and why `pipe_pop` and
`pipe_out` are nothing alike despite both making it "3-stage."

---

# How much of this generalizes?

## The shape is general; three of the four blocks are not

The **four-block skeleton is the DWN architecture itself**, not a JSC choice:

```
binarize → LUT layers → reduce → decide
```

Every DWN in the paper has it. What is specific here is that each block was implemented in
exactly one way, chosen for one dataset.

| block | what is general | what is JSC-specific here |
|---|---|---|
| **binarize** | *something* must turn inputs into bits | thermometer only; assumes **continuous, bounded, tabular** features; Q3.12 hardcoded |
| **LUT layers** | `TABLE[addr]`, LSB-first, n ≤ 6 | **nothing** — already general (widths 20→2000, 1–3 layers, n=2/4/6, both wiring forms, all Gate 1 verified) |
| **reduce** | the final layer must collapse to per-class evidence | plain popcount only; `GroupSum`'s contiguous grouping |
| **decide** | classification needs a winner | argmax only; single-label, 5 classes |

The middle block — the actual neural network — is the one that already generalizes. The
plumbing on either side is where the assumptions live. That is the same asymmetry
`docs/reusable-generator.md` §3 found from a different direction.

## Where each block would need variants

**Binarize is the least general block, and it is also the most expensive one.** Thermometer
encoding assumes a feature is a continuous quantity where "above threshold t" is a meaningful
question. That is true of jet masses. It is not true of:

- **already-discrete inputs** — MNIST pixels are 8-bit integers, and image DWNs typically
  binarize with very few levels. A pass-through or 1-bit threshold is the right block, and a
  200-level thermometer would be absurd.
- **categorical features** — "detector region ∈ {A,B,C}" wants one-hot, not thermometer;
  ordering the categories would invent a relationship that is not there.
- **already-binary inputs** — no encoder at all. The block should be able to vanish.

A generalized front end is therefore **a choice of binarizer per feature**, not one scheme for
the whole input vector.

**Reduce has a second form the paper already defines.** `GroupSum` + popcount is the baseline;
the paper's **Learnable Reduction** replaces the adder tree with a pyramid of LUT nodes, on the
grounds that for tiny models the popcount can be as large as the network. This project measured
that at `sm`: reduction is 58 LUTs against 50 for the network, so the paper's concern is real —
but it is **3.6% of the whole design**, so it stays deferred here (`docs/phase2-report.md` §7).
A general tool would still want both, because the ratio depends entirely on the dataset's
encoder cost.

**Decide is not always argmax.** Multi-label classification wants a threshold per class;
regression wants the reduced sums themselves and no decide stage at all. This block should be
allowed to be empty.

## What a generalized datapath looks like

```
raw input
   │
┌──▼──────────────────────────────────────────┐
│ BINARIZE   per-feature, pluggable:          │   ← the dataset-dependent block
│   thermometer (distributive/gaussian/linear)│
│   one-hot (categorical)                     │
│   threshold (few-level, images)             │
│   pass-through (already binary)             │
│   word/frac bits chosen per dataset         │
└──┬──────────────────────────────────────────┘
   │  [pipe]
┌──▼──────────────────────────────────────────┐
│ LUT LAYER × L    n ≤ 6, learnable or fixed  │   ← already general
└──┬──────────────────────────────────────────┘      one [pipe] per layer
   │  [pipe]
┌──▼──────────────────────────────────────────┐
│ REDUCE     popcount (GroupSum)              │
│            learnable-reduction pyramid      │
│            none (raw sums out)              │
└──┬──────────────────────────────────────────┘
   │  [pipe]
┌──▼──────────────────────────────────────────┐
│ DECIDE     argmax (single-label)            │
│            per-class threshold (multi-label)│
│            none (regression)                │
└──┬──────────────────────────────────────────┘
   │
 output
```

**The pipeline structure generalizes for free.** Registers sit *between blocks*, so stage count
follows from the block list: `1 + L + 1 + 1`, which is exactly what `HardwareConfig.latency`
already computes. A 2-layer model is naturally 5 stages, which is why `2x100` reached 155.5 MHz
— the fastest design in the sweep.

## What this project already has, and what it would cost

**Already general and Gate 1 verified:** the LUT layer block entirely — arbitrary widths, 1–3
layers, n = 2/4/6, learnable *and* fixed wiring, class count from the checkpoint. Plus three
thermometer variants and z from 8 to 800.

**The gaps, in the order the evidence says they matter:**

| gap | cost | evidence it is needed |
|---|---|---|
| **Configurable precision** (`word_bits`/`frac_bits`) | 1–2 days, five consumers | **Proven.** `1x200 linear` and `1x360 linear` are *unbuildable* — evenly-spaced thresholds reach 8.906 against Q3.12's ±8 ceiling. The right format depends on the encoding, and that does not generalize. |
| **Binarizer variants** beyond thermometer | unscoped | Not yet tested. This is what the MNIST port would surface. |
| **Learnable Reduction** | new RTL + custom training | Measured as 3.6% of the design here; would matter more where the encoder is cheap. |
| **Decide variants** | small | No dataset here needs them. |
| **Harness generality** | 2–4 days | 33-byte records, 5 classes, `DEPTH=1024`. `uart_loader`'s `reg [5:0] byte_idx` would silently wrap on a wider record. |

Note the ordering: **precision is the only one with a measured failure behind it.**
`docs/reusable-generator.md` §4 lists it as ordinary packaging work alongside four other items;
the Phase 2 sweep upgraded it to a correctness prerequisite, because without it one of three
encodings cannot be built at all.

## One structural limit worth stating

**Four stages is the architectural maximum for a single-layer model** — there are exactly four
register sites and `pipe_reg.ENABLE` is 0/1. If a config misses timing at 4 stages, this RTL
cannot rescue it. `1x2000` failing at −0.538 ns is a hard failure, not a pipelining problem.

A generalized version would allow **registers *inside* a block** — most obviously partway up the
popcount adder tree, which is the critical path and grows with width. That is the natural fix
and it is not implemented. The current escape hatch is architectural rather than structural: a
multi-layer model of similar size gets an extra stage per layer for free.
