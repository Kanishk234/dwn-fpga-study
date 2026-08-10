# DSE Plan — what we sweep, and what that means for Phase 1 

**Status: implemented, 2026-08-07.** Written 2026-08-02 as planning, when nothing here ran yet.
The plan is now code — `dse/grid.py` (the slice), `dse/area_model.py` (§5), `dse/run.py` (the
loop), `dse/report.py` + `dse/plot.py` (§6's outputs). **Where this document and the code
disagree, the code is right and this file is history**; `docs/phase2-ledger.md` records what
changed and why. Two sections are explicitly superseded: §5's encoder assumption (see the box
there) and §7's open questions (most are now answered).

This document exists mainly to answer a question that has to be settled *before* Phase 1 starts:
**what does the sweep actually vary?** — because the answer is a hard requirement on how the Phase 1
code gets written.

---

## 1. The sweep modifies no code at all

This is the thing to understand first, and it's easy to get backwards.

When Phase 2 runs, we are **not** editing the exporter, the RTL generator, the Verilog templates, or
the harness. Every one of those is finished, verified, and frozen at Gate 1b. What changes between
sweep point #7 and sweep point #8 is **a config — a dict of numbers.** That's it.

```
config #7  ──▶ ┌─────────────────────────────────────────┐ ──▶ accuracy
{t:4, L:3,     │  train → export → rtlgen → Vivado →      │     LUTs / FF
 W:[512,256,   │  parse reports                           │     Fmax / latency
 128], ...}    │  (FIXED CODE, built in Phase 1)          │
               └─────────────────────────────────────────┘
```

So the real content of this document is a **specification for Phase 1**: every knob listed in §3 has
to be a *parameter threaded through the whole pipeline*, not a hardcoded value. Any axis that gets
hardcoded during Phase 1 is an axis that cannot be swept in Phase 2 without going back and rewriting
code you'd already verified — which means re-passing Gate 1.

**Concretely, by Gate 1b these must all be parameters:**

| Component | Must be parameterized on |
|---|---|
| training script | `n`, thermometer resolution + scheme, layer count, layer widths, reduction type, `λ` |
| `exporter/` | any model shape the training script can produce — no assumed `n`, layer count, or width |
| `rtlgen/` | `n`, layer count, widths, reduction type, **pipeline depth** |
| `rtl/` templates | `n`, width, and depth as Verilog parameters, not literals |
| `scripts/build.tcl` | target clock constraint, synthesis strategy, output report paths |
| `tb/` golden model | same shape parameters as the exporter, driven from the same config |

The single sanity check: **one config file should drive the entire flow end to end.** If running a
different configuration requires editing a `.py` or `.v` file by hand, Phase 1 isn't done.

### Do we need all of Phase 1 first?

Yes — with one narrow exception.

Gate 1 (bit-exactness in simulation) and Gate 1b (test-set accuracy on the physical board) both have
to pass before sweep results mean anything. A DSE point is a claim of the form *"this configuration
achieves X% accuracy in Y LUTs"* — the accuracy half comes from training, but the area/timing half is
only trustworthy if the generated RTL is known to implement the trained model faithfully. Sweeping an
unverified generator produces a Pareto frontier of numbers that describe nothing.

The exception: the **area prediction model** in §5 is pure Python arithmetic over a config. It can be
written and used any time, and it's genuinely useful during Phase 1 for sizing the toy models.

---

## 2. Not a full grid

A full factorial over the axes below — five axes, ~4 values each — is 1,024 configurations. At
10–20 minutes of Vivado per point on one machine (§3 of the brief: one machine, not two) that's
200–340 hours of serial synthesis. Not happening.

A DSE is not "try everything." It's choosing a **slice** of the space that maps the frontier's shape
at a cost you can actually pay. The strategy is §6.

---

## 3. The knobs

The organizing distinction is **which knobs require retraining.**

### Group A — model architecture

Changes accuracy *and* area. **Requires a training run per config** (GPU time).

| Knob | What it does | Candidate range |
|---|---|---|
| LUT input width `n` | Inputs per node. **Fixed at 6 through Phase 1**, a real axis in Phase 2 — see below | 6 first, then 4, 2 |
| Thermometer resolution `t` | Bits per input feature. JSC has 16 features → input width = `16 × t` | 2, 3, 4, 6, 8 |
| Encoding scheme | `Thermometer` (evenly spaced), `GaussianThermometer` (normal icdf), `DistributiveThermometer` (quantile-spaced) — all three ship upstream | 3 values |
| Layer count `L` | Depth of the LUT stack | 2, 3, 4 |
| Layer widths `W₁…W_L` | LUT nodes per layer — **the primary area dial** | ~50 → ~4,000 total |
| Reduction | `GroupSum` (popcount) only, for now — see below | **deferred**, 1 value |
| Spectral reg `λ` | Regularization strength (brief §4) | Tune for accuracy; **not** a hardware axis |

### Group B — RTL implementation

Changes timing and FF count. **Accuracy is invariant** — no retraining, synthesis only.

| Knob | What it does |
|---|---|
| Pipeline depth | Register placement between layers. Deeper → higher Fmax, more FFs, more cycles of latency |
| Clock constraint | The target period in the XDC |
| Synthesis strategy | Vivado area-vs-speed directives |

**This split is the most load-bearing fact in the plan.** Group B explores on top of an
already-trained model at zero GPU cost, which is why §6 defers it to the survivors.

#### On `n` — sweep it, but calibrate what you expect to find

`n` is **fixed at 6 for all of Phase 1.** It's the safest config to prove the pipeline with — least
likely to hit the routing congestion failure mode (brief §12, risk #2) — so an early pipeline bug
doesn't get mistaken for a hardware routing limit. In Phase 2, once Gate 1 has passed and the sweep
infrastructure works, `n` becomes a real axis.

**What to expect when you do sweep it:** not an area win. The intuition that "smaller tables are
cheaper" does not survive contact with the fabric. A Xilinx LUT6 is physically a 64-entry table with
a 6-bit address, so:

| `n` | Trained params/node | FPGA cost/node | Efficiency |
|---|---|---|---|
| 2 | 4 | ~1 LUT6 | 4 of 6 inputs wasted; needs many more nodes for equal capacity |
| 4 | 16 | ~1 LUT6 | 2 of 6 inputs wasted |
| **6** | **64** | **1 LUT6** | **exact fit — brief §4, the architectural premise** |
| 8 | 256 | several LUT6s + muxing | genuinely grows fast past the primitive |

Below 6 you pay full silicon price for a fraction of the capacity, *and* you need more nodes, *and*
more nodes means more wires means worse congestion — which is exactly why the paper's n=2 models
failed to route on a part far larger than ours.

So the likely finding is that n=2 and n=4 are **worse on both axes**. That is still worth running and
reporting: it empirically confirms the architecture/fabric match on an entry-level part, which nobody
has measured, and it locates the congestion wall. Frame it as confirmation, not as a search for a
better operating point. **A config that fails to route is a data point marking the frontier's edge,
not a mistake.**

#### On reduction — the axis upstream doesn't give us

**`third_party/DWN` ships no Learnable Reduction.** The only reduction in the repo is
`GroupSum(k, tau)` in `utils.py`: split the final layer's output bits into `k` groups, one per class,
and score each class by how many of its bits are 1. That's a popcount, and in hardware it's an
**adder tree** per class.

The paper's Learnable Reduction — a pyramid of shrinking LUT layers that *learns* how to combine the
final bits into class scores, e.g. 100 → 32 → 10 → 5 — is described in the paper (brief §4) but not
implemented upstream. Building it means stacking `LUTLayer`s ourselves.

Why it might matter: the adder trees are the **only arithmetic in an otherwise arithmetic-free
design**. The paper notes that in small models "the popcount circuit can be as large as the network
itself" — and small models on a constrained part is exactly our regime.

**Decision: deferred.** Phase 1 uses `GroupSum`, because it's what ships, it's less custom code
between us and Gate 1, and popcount-plus-argmax hardware has to be built either way. The trade can't
be evaluated without knowing what the popcount actually costs in LUTs on `xc7a35t` — and the first
end-to-end synthesis produces that number. If it's 40% of area, building the pyramid is obviously
worth it; if it's 3%, this was never an interesting axis. **Revisit after the first end-to-end
synthesis, not before.**

### Group C — fixed

Dataset = JSC, part = `xc7a35t`.

---

## 4. What one config concretely is

A single sweep point is a tuple:

```
t=4, distributive, L=3, W=[512, 256, 128], reduction=learnable, pipeline=2
```

Read as hardware: 16 features × 4 bits = **64 input bits** → layer 1 has **512 LUT6 nodes**, each
learning which 6 of those 64 bits to read → their 512 output bits feed layer 2's **256 nodes** →
**128** → reduction pyramid → 5 class scores → argmax.

---

## 5. Area is predictable without synthesizing

Because one DWN node **is** one LUT6 (brief §4), and inter-layer wiring is fixed after training and
therefore free:

```
core LUTs   ≈ W₁ + W₂ + ... + W_L  +  reduction cost
total LUTs  ≈ core LUTs  +  encoder cost
```

So a config's approximate area is computable **in Python, before Vivado ever launches.** Configs that
obviously overshoot 20,800 LUTs get discarded for free. Most ML architectures don't offer this; DWN
does, and it's what makes the filter step affordable.

> ⚠️ **Superseded, 2026-08-07 — this section is kept for its reasoning, not its numbers.** The
> implementation is `dse/area_model.py`, calibrated on Phase 1's measurements, and it corrects
> the estimate below by a factor of four. See `docs/phase2-ledger.md` for the derivation.
>
> - **The encoder is 14.06× the core, not "up to 3.2×".** Filtering with 3.2× would have
>   underestimated encoder area by **4.4×** at `sm` alone, passing configs that cannot fit.
> - **Encoder cost does not scale with the core.** It tracks the number of *distinct thresholds
>   the mapping selects*, which grows with node count and **saturates at `features × z` = 3200
>   comparators**. That makes `z`, not width, the ceiling on encoder area.
> - **The model cannot extrapolate over `z` or `n`.** Its selection ratio (67%) was measured at
>   one config and reflects *learned concentration*, not random collision. `dse/grid.py`
>   therefore refuses to filter out any config whose area estimate is extrapolated — otherwise
>   the filter would discard exactly the points Study 1 exists to measure.

Two caveats:
- **Encoder cost is the open unknown** (brief §6, risk #3) — up to 3.2× the core, per Mecik & Kumm.
  Measure it on the first end-to-end model and calibrate this formula against reality before trusting
  it to filter. *(Done: 14.06×. See the box above.)*
- The estimate ignores routing. A config that fits by LUT count can still fail to route (risk #2).
  **That failure is a result**, not a bug — it's where the congestion wall gets located.

---

## 6. Sweep strategy

1. **Baseline.** Find one config that trains well on JSC and clearly fits — the small-config range in
   brief §8 is the starting point (DWN sm-50: 311 LUTs at 74%).
2. **Size ladder.** Scale widths up from baseline in ~6–8 steps until the part runs out. This is the
   spine of the frontier and produces the headline number: *the largest DWN that fits an XC7A35T, and
   what it scores.* It's also where the congestion wall appears.
3. **One-factor-at-a-time on 2–3 rungs** of the ladder — vary `t`, then `L`, then reduction, holding
   everything else fixed. This is what reveals *which axis buys the most accuracy per LUT*, which is
   the actual scientific content of Study 1.
4. **Group B on survivors only** — pipeline/clock sweeps on ~5 already-trained models. Cheap.

**Estimated cost: ~40–70 synthesis runs, ~15–25 hours of serial Vivado.** Feasible on one machine.

Execution order, per brief §10: **train the whole Group A set first** (GPU-bound, batchable), *then*
filter with §5, *then* synthesize. Never interleave — interleaving spends serial Vivado time on
configs that would have been discarded for free.

---

## 7. Open questions — resolve during Phase 1

- ~~**Exact `torch_dwn` API.**~~ **RESOLVED 2026-08-02**, read off pinned commit `9f887a0`:
  - `LUTLayer(input_size, output_size, n, mapping=..., alpha, beta, ste, clamp_luts, lm_tau)`,
    with `mapping ∈ {'arange', 'random', 'learnable'}` or an explicit `int32` tensor of shape
    `[output_size, n]`.
  - LUTs are a float parameter of shape `[output_size, 2**n]`, clamped to [-1, 1] during training.
    **Only the sign matters at inference** — that's what the exporter reads to build tables.
  - `GroupSum(k, tau, randperm=False)` in `utils.py` is the reduction. `STE`/`STEFunction` there too.
  - `LearnableMapping` in `mapping.py`; `layer_mapping(input_size, n, output_size, random=)` builds
    fixed mappings.
  - Three thermometers in `binarization.py`, all with `.fit(x)` / `.binarize(x)`.
- ~~**Real width values** for the ladder in §6 step 2.~~ **RESOLVED 2026-08-07.** The ladder is
  `dse/grid.py`'s `SIZE_LADDER` = 50, 100, 200, 360, 500, 600, 800, 1200 — chosen to *bracket*
  the predicted wall (fit boundary near W≈550, encoder saturation near W≈800), not to stop short
  of it. The top rungs are expected to fail; that is what locates the frontier's edge.
- **Whether to build Learnable Reduction** — ⚠️ **REOPENED 2026-08-10.** The resolution below
  was measured at `sm` alone and does not survive scale: at `1x2400 z=50` the reduction is 4,450
  LUTs, **34.9% of the design**. The 40% bar this document set is essentially met. Superseded
  reasoning kept below.
- ~~**Whether to build Learnable Reduction**~~ ~~**RESOLVED 2026-08-07: no, stay deferred.**~~
  Measured standalone (`scripts/experiment_reduction.py`): reduction **58 LUTs**, LUT layer
  **50** — summing to exactly the 108 `dwn_core` measures. So it *is* 54% of the core, as the
  bar above asked. **But that bar measures the wrong ratio.** It was written before the encoder
  was known to cost 14× the core; against the whole design the reduction is **3.6%** at `sm` and
  ~3% at `md`. By this document's own "if it's 3%, this was never an interesting axis" test, it
  is not. `z` and per-feature comparator narrowing both move far more area.
- ~~**Encoder cost multiplier** for §5.~~ **RESOLVED 2026-08-07: 14.06×**, not the 3.2× assumed
  here. See the box in §5 — the correction is large enough to change which configs are viable.
- ~~**Does pipeline depth actually move Fmax here?**~~ **RESOLVED 2026-08-03: yes, nearly 2×.**
  84.2 MHz at one stage → 161.0 MHz at four, and **LUT count never changes** — pipelining costs
  flip-flops, not logic (196→269 FFs on a part with 41,600). So Group B does *not* collapse; it
  is a real axis and a cheap one, since accuracy is invariant and no retraining is needed.
  Three stages is the floor that closes the board's 100 MHz.

  One refinement Phase 2 added: rank pipeline variants on **latency in nanoseconds**, not cycles.
  4 stages at 161.0 MHz is 24.84 ns against 3 stages at 122.9 MHz at 24.4 ns — a whole cycle
  apart and effectively the same real latency. Cycles alone would call that a clear win.
