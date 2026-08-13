# MNIST Phase 2 — the design-space exploration on a second dataset

Twenty-five configurations trained, exported, Gate 1 verified bit-exact, and placed and routed on
a Digilent Basys 3 (`xc7a35tcpg236-1`) at a 10 ns target. The same generator that produced the JSC
sweep, run on a dataset with 49× the features and twice the classes.

This is the written-up account. `docs/mnist/phase2-ledger.md` is the running log it was written
from — dated, with the retractions in place. The JSC study this parallels is
`docs/jsc/phase2-report.md`; Phase 1 of this port is `docs/mnist/phase1-report.md`.

**What makes this phase worth reading is not the frontier.** It is that seven conclusions were
withdrawn — two of them Phase 1 predictions recorded specifically so they could be scored, one of
them this study's own headline. A sweep that only confirmed things would have been evidence that
the sweep was not sharp enough.

---

## 1. Headline results

| | |
|---|---|
| Configurations built | **25**, place-and-routed, **0 failures** |
| **Gate 1** | **25/25 bit-exact**, 1,504 core vectors each |
| **Gate 1b on silicon** | **MNIST 10,000/10,000, JSC 166,000/166,000** — both re-verified post-refactor (§5.7) |
| Accuracy range | **92.98% → 98.32%** across a 20× node range |
| **Measured noise floor** | **0.24 pp** — larger than JSC's 0.15 |
| Best accuracy | `1x2000` — **98.26%**, 6,294 LUTs (30.3%), 94.0 MHz ⚠️ misses the board clock |
| | *(`2x[2000,1000]`'s stored 98.32 is higher, but it is a single-seed figure and over four seeds it falls **below** `1x2000` — §5.1)* |
| **Best usable design** | **`2x[1000,500]` — 97.76%, 3,464 LUTs (16.7%), 103.8 MHz, 5 cycles** |
| Smallest design above 96% | `1x300` — 96.77% in **1,597 LUTs (7.7%)**, 107.5 MHz |
| Largest built | `1x2000` 2-stage — **6,877 LUTs, 33.06%** |
| Frontier edge | ⚠️ **not measured** — nothing failed to fit. See §7 |

**The single most useful result is that the paper's own configuration is the best design on this
board, and for a reason the accuracy study could not see.** `2x[1000,500]` and `1x1000` are the
same size (3,464 vs 3,490 LUTs) and statistically the same accuracy (97.76 vs 97.97, inside the
0.24 pp floor). But the two-layer model earns a fifth pipeline stage for free and reaches
**103.8 MHz where the single-layer manages 93.1**. At the board's real 100 MHz clock, one of them
runs and the other does not.

### Against the published MNIST rows

| design | accuracy | LUTs | part |
|---|---|---|---|
| **this work, `1x300`** | **96.77%** | **1,597** | `xc7a35t-1` |
| **this work, `2x[1000,500]`** | **97.76%** | **3,464** | `xc7a35t-1` |
| PolyLUT | 96% | 70,673 | `xcvu9p` |
| NeuraLUT | 96% | 54,798 | `xcvu9p` |

⚠️ **Read this with the same care Phase 1 asked for.** Our area convention includes the encoder,
which is the stricter choice (`docs/jsc/report.md` §5.2), but the published rows were measured on a
different part at a different clock target. It is evidence the approach is in the right range, not
a claim to have beaten either.

---

## 2. What the sweep covered

One-factor-at-a-time around `1x1000`, n=6, z=3, distributive — the same star shape JSC used.

| group | configs | axis |
|---|---|---|
| ladder | 6 | width: 100 → 2000 nodes |
| ofat-z | 4 | z = 1, 2, 8, 25 (z=3 is the ladder's `1x1000`) |
| ofat-n | 2 | n = 2, 4 (n=6 is the anchor) |
| corner | 2 | `2x[1000,500]` (the paper's), `2x[2000,1000]` (upstream's) |
| group-b | 11 | pipeline depth and clock target on three already-trained models |

Every model is `distributive` at Q0.8 (9-bit) unless noted. Group B varies only *hardware* on a
fixed checkpoint, which is why it needed no extra training.

**Four grid entries were never built** — `gaussian`, `linear`, `2x500`, `3x330`. No training
notebook ever produced them. Recorded as a gap rather than quietly dropped: the encoding axis in
particular is unmeasured on MNIST, and JSC found it worth 0.12 pp against a 0.15 pp floor, i.e.
nothing.

---

## 3. Complete results

### 3.1 The ladder

| rung | acc% | core | encoder | `dwn_top` | %dev | Fmax | latency |
|---|---|---|---|---|---|---|---|
| `1x100` | 92.98 | 234 | 611 | **845** | 4.06 | **123.8** | 4 |
| `1x200` | 95.93 | 420 | 839 | 1,264 | 6.08 | **111.3** | 4 |
| `1x300` | 96.77 | 630 | 971 | 1,597 | 7.68 | **107.5** | 4 |
| `1x500` | 97.70 | 1,168 | 1,079 | 2,246 | 10.80 | **108.4** | 4 |
| `1x1000` | 97.97 | 2,272 | 1,220 | 3,490 | 16.78 | 93.1 | 4 |
| `1x2000` | 98.26 | 4,821 | 1,315 | 6,294 | 30.26 | 94.0 | 4 |

**Bold Fmax meets the 100 MHz board clock — four of the six do.** `1x1000` and `1x2000` do not,
which is the constraint that decides §4.5.

### 3.2 The other axes

| group | config | acc% | core | encoder | `dwn_top` | %dev | Fmax |
|---|---|---|---|---|---|---|---|
| corner | `2x[1000,500]` | 97.76 | 2,168 | 1,302 | **3,464** | 16.65 | **103.8** |
| corner | `2x[2000,1000]` | 98.32 ⚠️ | 4,271 | 1,403 | 5,670 | 27.26 | 92.3 |
| ofat-z | `1x1000 z=1` | 97.91 | 2,272 | **846** | 3,118 | 14.99 | 95.4 |
| ofat-z | `1x1000 z=2` | 98.05 | 2,272 | 1,038 | 3,310 | 15.91 | 90.6 |
| ofat-z | `1x1000 z=8` | 98.13 | 2,272 | 1,618 | 3,883 | 18.67 | 91.5 |
| ofat-z | `1x1000 z=25` | 98.23 | 2,272 | 2,935 ⚠️ | 5,195 | 24.98 | 92.8 |
| ofat-n | `1x1000 n=2` | 96.30 | 2,257 | 996 | 3,240 | 15.58 | 95.3 |
| ofat-n | `1x1000 n=4` | 97.57 | 2,272 | 1,147 | 3,414 | 16.41 | 92.5 |

⚠️ `2x[2000,1000]`'s 98.32 is a single-seed figure and is **withdrawn** — §5.1.
⚠️ `z=25` was built at **Q1.8, a 10-bit word**, because its thresholds do not fit 9 bits. Its
encoder is therefore wider *per comparator* than every other row, and the z axis is confounded
with word width at that one point.

### 3.3 Group B — pipeline depth and clock target

| base | stages | core | `dwn_top` | Fmax |
|---|---|---|---|---|
| `1x300` | **4** | **630** | **1,597** | **107.5** |
| `1x300` | 3, no OUT reg | 641 | 1,611 | 96.6 |
| `1x300` | 3, no POP reg | 817 | 1,788 | 85.5 |
| `1x300` | 2 | 825 | 1,785 | 75.3 |
| `1x1000` | **4** | **2,272** | **3,490** | **93.1** |
| `1x1000` | 3, no OUT reg | 2,292 | 3,505 | 84.6 |
| `1x1000` | 3, no POP reg | 2,919 | 4,125 | 59.8 |
| `1x1000` | 2 | 2,919 | 4,138 | 54.9 |
| `1x2000` | **4** | **4,821** | **6,294** | **94.0** |
| `1x2000` | 3, no OUT reg | 4,815 | 6,433 | 82.6 |
| `1x2000` | 3, no POP reg | 5,560 | 6,871 | 53.2 |
| `1x2000` | 2 | 5,560 | 6,877 | 48.5 |

| clock target | `1x1000` core | `dwn_top` | Fmax |
|---|---|---|---|
| 8 ns (125 MHz) | 2,337 | 3,618 | **97.4** |
| 10 ns (100 MHz) | 2,272 | 3,490 | 93.1 |
| 12 ns (83 MHz) | 2,256 | 3,480 | 86.6 |

---

## 4. What the axes say

### 4.1 Width — accuracy saturates long before the device does

500 → 2000 nodes is **4× the core for +0.56 pp**. The knee is at 500, and above `1x500` every
further rung also loses the board clock. On the device axis the ladder's top rung is 30.3% of a
part it never came close to filling.

The core costs **2.10–2.41 LUTs per node** across the whole ladder — a 15% spread over a 20× size
range, and **not monotone** in width (2.34 / 2.10 / 2.10 / 2.34 / 2.27 / 2.41). That it is roughly
*constant* at all is the DWN premise holding: one node is one LUT6 plus its share of the
reduction, regardless of what the table contains. The marginal cost between the two widest rungs
is 2.549 LUT/node, and that is the figure §7 extrapolates the wall from — deliberately the
top-of-ladder slope rather than the average, since it is the regime the wall is in.

### 4.2 The encoder saturates, and that is what makes MNIST cheap

| rung | `1x100` | `1x300` | `1x1000` | `1x2000` |
|---|---|---|---|---|
| encoder LUTs | 611 | 971 | 1,220 | 1,315 |
| **share of `dwn_top`** | **72.3%** | 60.8% | 35.0% | **20.9%** |

Across a **20× node range the encoder grows only 2.2×**. The cause is structural: the learnable
mapping builds a comparator only for a threshold some node actually reads, and MNIST is
*slot-limited* — 784 × z bits far exceed the input slots of any layer that fits, so widening the
model buys new comparators only until the mapping saturates.

**This is the single biggest difference from JSC**, and it runs the opposite way. JSC has 16 wide
features and a 16-bit word: comparators cost 7.52 LUTs each and the encoder stays a large fraction
of the design at every width. MNIST has 784 narrow features and a 9-bit word, and the encoder
flattens.

### 4.3 `z` — nearly free, and the biggest area lever nobody needs

z=1 scores **97.91%** against z=25's **98.23%**: **0.32 pp across a 25× threshold count**, which
is 1.3 noise floors — it barely clears. The area difference is 846 vs 2,935 encoder LUTs.

⚠️ **But the z=25 point is confounded.** It was built at a 10-bit word because its thresholds do
not fit 9 bits, so part of that 2,935 is the extra bit per comparator rather than the extra
thresholds. The direction of the finding is safe; the magnitude is not exact.

The useful reading is the one Phase 1 reached from the other side: **word width, not `z`, decides
whether an MNIST model fits.** `z` is an accuracy knob with a mild area cost, not an area problem.

### 4.4 `n` — not a lever on MNIST, unlike JSC

| | acc% | core | `dwn_top` |
|---|---|---|---|
| n=2 | 96.30 | 2,257 | 3,240 |
| n=4 | 97.57 | 2,272 | 3,414 |
| n=6 | 97.97 | 2,272 | 3,490 |

`n=2` saves **7% of area for −1.67 pp**; `n=4` saves **2% for −0.40 pp**. The core is flat
(2,257 / 2,272 / 2,272) because **one node is one LUT6 regardless of `n`** — only the encoder
moves, because a smaller fan-in selects fewer distinct thresholds.

On JSC, `n=2` was on the frontier. Here it is not, and the reason is §4.2: when the encoder is a
small share of the design, an axis that only moves the encoder cannot move much.

### 4.5 Layers — depth pays, for timing

`2x[1000,500]` against `1x1000`: **3,464 vs 3,490 LUTs, 97.76 vs 97.97% (inside the floor), and
103.8 vs 93.1 MHz.** Same area, same accuracy, +10.7 MHz — because `PIPE_LUT` inserts a register
per layer, so depth buys a pipeline stage for free. Latency goes 4 → 5 cycles, and at II=1 that
costs throughput nothing.

**This does not contradict the reduction study's "depth does not pay."** That was an accuracy
claim and it stands: `2x[1000,500]` is not more accurate. Depth buys an axis the accuracy study
could not see.

⚠️ **And it is not new — JSC found it first** (`docs/jsc/phase2-report.md` §4.4: `2x100` at 155.5 MHz
against `1x200`'s 113.9). That makes it *more* valuable, not less: it is one of the few JSC
conclusions now **reproduced** on a second dataset rather than generalised from one. The two
retractions in §5.2 are what happens when that check is skipped.

### 4.6 Group B — four stages is both the maximum and the optimum

Removing the popcount register costs **41 MHz and 739 core LUTs** on `1x2000`. A 200-wide group
sum is a deep adder tree and that register is load-bearing; without it, the combinational path
grows and Vivado spends logic trying to meet timing it cannot meet. **Fewer stages is worse on
both axes at every width tested.**

The only lever that does move timing is the constraint itself: asking for 8 ns gives `1x1000`
**97.4 MHz** against 10 ns's 93.1 — **+4.3 MHz for 128 LUTs**. Over-constraining works, modestly.

---

## 5. What broke — seven retractions

### 5.1 ⚠️ The study's own best model was withdrawn by the noise floor

The reduction study's headline was *"`2x[2000,1000]` at 98.32% is the best model in the entire
study."* The noise-floor run — four configurations × four seeds — killed it:

| | mean over 4 seeds | range |
|---|---|---|
| `1x2000` | **98.290** | [98.20, 98.41] |
| `2x[2000,1000]` | **98.190** | [98.15, 98.22] |

**The claimed +0.06 pp advantage does not merely fail to clear the floor — it reverses.** 98.32
was one lucky draw; all four new runs sit below it. `1x2000` reaches at least the same accuracy
with 2,000 nodes instead of 3,000.

The measured floor is **0.24 pp**, larger than JSC's 0.15. No width dependence is detectable
(0.24 / 0.14 / 0.21 / 0.07 across four sizes is not monotone), so one number is quoted rather than
a schedule.

⚠️ **A same-seed rerun moved by up to 0.17 pp**, almost certainly GPU nondeterminism — `torch_dwn`
uses CUDA atomics and Kaggle does not guarantee the same accelerator between sessions. That does
not invalidate the floor; **it is part of it**. It does mean seed spread and run-to-run spread
cannot be separated with this data, and that **no MNIST accuracy in this study should be quoted to
more than one decimal.**

### 5.2 ⚠️ Two Phase 1 predictions, recorded in advance and both wrong

Written into the ledger before the sweep so they could be scored rather than reconstructed
favourably. Both failed.

**"Timing binds before area on MNIST."** The whole bottom half of the ladder clears 100 MHz
comfortably — `1x100` reaches 123.8 MHz. The 90–95 MHz figures are a property of `1x1000` and
wider, not of the dataset. **The z-sweep looked like confirmation only because every config in it
was `1x1000`.** Corrected: timing binds above roughly 500 nodes.

**"Pipeline depth is the lever, and it needs no retraining."** Wrong in the unhelpful direction —
§4.6. There is no lever to pull.

Both are the same error: **a conclusion generalised from a single width.** This project has now
retracted that shape of claim five times, and every one was caught by measuring at a second point.

### 5.3 ⚠️ "`device_pct` is measured against the wrong denominator" — self-retracted

The reasoning was: JSC's `1x2000` reports 102.80% *and* completed place-and-route; a design cannot
route into 102.8% of a part; therefore the denominator is wrong.

The actual explanation was already in JSC's own report:

```
post-synthesis : 20,126 LUTs  (96.76%)   <- fits, so placement proceeded
post-route     : 21,382 LUTs (102.80%)   <- physical optimization pushed it over
```

Placement starts from a netlist that fit. Physical optimisation then replicated logic chasing
timing and drove the routed design past the device. **The design does not fit**, `DEVICE_LUTS =
20800` is correct, and `device_pct` means what it says.

**The lesson is the inference, not the arithmetic:** the flow completing was treated as proof the
design fit, so a contradiction was resolved by doubting the denominator rather than by doubting
that assumption. Nothing was changed on the strength of the wrong claim.

⚠️ **The rule this restates: judge "did it fit" on measured routed area and timing, never on tool
exit status.**

### 5.4 ⚠️ "The JSC study has no `linear` encoding data" — overstated, withdrawn

Two JSC configs (`1x200 linear`, `1x360 linear`) are recorded as `gate1-failed`, the only 2
failures in 54. This was written up as a discovery; it is not one. `docs/jsc/phase2-ledger.md` already
records the cause and the decision — Q4.11 would represent them at identical area, but the
encoding axis spread is 0.12 pp against a 0.15 pp floor, so the plumbing was judged not worth it.
**A documented failure with a stated reason, not a silent gap.**

And rebuilding them would have been actively wrong: the widening helper moves the *word* (16 → 17
bits), so the linear points would stop being comparable to the gaussian and distributive points at
16 bits, confounding the encoding axis with word width.

### 5.5 The area model is not usable for MNIST

| rung | predicted | measured | error |
|---|---|---|---|
| `1x100` | 852 | 845 | +0.8% |
| `1x300` | 1,326 | 1,597 | **−17.0%** |
| `1x500` | 1,820 | 2,246 | **−19.0%** |
| `1x2000` | 5,802 | 6,294 | −7.8% |

It **under**-predicts by 8–19%, worst mid-range, and is accurate only at the smallest rung. Its
comparator model is separately off by **+108% at z=3**. It never blocked the sweep — nothing was
filtered on prediction — but **no projected MNIST area should be quoted.**

The root cause is that `lut_per_comparator_bit` is treated as a constant and is not one:

| dataset | config | comparators/feature | LUT/bit |
|---|---|---|---|
| JSC | `1x50 z=200` | 12.62 | 0.4700 |
| MNIST | `1x1000 z=1` | 0.55 | 0.2181 |
| MNIST | `1x1000 z=3` | 0.92 | 0.1417 |
| MNIST | `1x1000 z=25` | 4.34 | 0.0863 |

A **5.5× spread**. Within MNIST it falls as comparators-per-feature rises, which is logic sharing
between comparisons against different constants on the same input word. Between datasets that does
not explain it — JSC has the most comparators per feature and is still the most expensive per bit,
which points at threshold *values* (MNIST pixels are mostly zero, so quantile thresholds cluster
near zero and those comparisons collapse to a few bits). **Two mechanisms, previously conflated
into one falsified claim.** A z-dependent model is the right fix and is not done.

### 5.6 A `1x300` that is 49 LUTs larger than Phase 1's

Phase 1 reports `1x300` at **1,548 LUTs / 96.14%**; this sweep reports **1,597 / 96.77%**. Both
are correct and they are different checkpoints — Phase 2 uses the `tau`-corrected retrain.

The interesting part is *where* the 49 LUTs are:

| | Phase 1 | Phase 2 | Δ |
|---|---|---|---|
| `dwn_core` | 631 | 630 | **−1** |
| `thermometer_encoder` | 918 | 971 | **+53** |
| `dwn_top` | 1,548 | 1,597 | +49 |

**`tau` never reaches hardware, but a differently trained model is a different design** — the
mapping selects a different set of thresholds, so a different number of comparators gets built.
The core is invariant because one node is one LUT6 whatever the table holds; the encoder is not,
because *which* thresholds are wired is a learned property.

(The +53 and +49 do not have to agree: `dwn_top` is not the sum of its two submodules —
631 + 918 = 1,549 against a measured 1,548, and 630 + 971 = 1,601 against 1,597 — because
out-of-context synthesis of the whole optimises across the boundary. The submodule figures are
reported separately by requirement, not because they partition the design.)

Anyone comparing the two reports will hit this. It is not a regression.

### 5.7 ⚠️ A Gate 1b that failed at chance, and the two silent defects behind it

Both datasets were re-verified on silicon at the close of Phase 2, because the descriptor refactor
had touched the export path and only simulation had confirmed it:

| | Gate 1b | board | slack |
|---|---|---|---|
| **JSC** | **166,000 / 166,000** | 1,893 LUTs, 864 FF, 8 BRAM | +2.014 ns |
| **MNIST** | **10,000 / 10,000** | 4,586 LUTs, 10,746 FF, 0 BRAM | +0.292 ns |

Both reproduce their Phase 1 records exactly — MNIST to the LUT, the flip-flop *and* the same
slack. JSC's `core_cycles` landed on 166,815, which is the load-bearing figure: 166,000 samples
plus pipeline fill, so an exact match proves **II=1 and the label alignment both survived the
refactor.** A drifted latency scores every sample against the wrong label and still completes.

**But the first MNIST attempt failed at 1,035/10,000 — chance for ten classes.** The cause was
operator error, and the interesting part is that nothing in the flow objected.

`build_bitstream.py` derives the *harness* from the checkpoint and does **not** emit the network,
which comes from `build/rtl`. A `verify_phase1.py --with-board` run had just regenerated **JSC's**
RTL there, so the MNIST bitstream wrapped an MNIST-shaped 1,569-byte loader around JSC's 256-bit
core. **Verilog truncates a too-wide connection rather than erroring**, so the build succeeded,
met timing, programmed, ran, and produced confident garbage. Hierarchical utilization settled it:
`u_dwn` measured **1,621 LUTs — exactly JSC's `dwn_top`**, against MNIST's 1,597.

Two defects, both of this project's characteristic shape — silent, and reassuring:

1. **`build_bitstream.py` reported a model it was not building**, printing
   `dataset: mnist / 784 features x 9-bit, 10 classes` while building JSC's network. Now checked
   before Vivado launches: `x_flat` width against `features × word_bits`, `class_idx` against
   `ceil(log2(classes))`, and the `lut_node` count against `sum(layers)` — the last because two
   MNIST checkpoints have identical port widths, so widths alone would build either happily.
2. **`host.py` printed the software reference's accuracy under the label "on hardware."** On the
   failing run it displayed **96.1400% "on hardware"** from a board agreeing on 10% of samples.
   Now labelled conditionally.

⚠️ **JSC structurally cannot detect either defect.** Its 16-bit word is exactly two bytes, so the
byte-padding calculation gives the same answer whether it ceilings or floors, and its record is 33
bytes against MNIST's 1,569. This is the clearest case in the whole port of a second dataset
finding what inspection could not — and it was found by a *failing* run, which is the argument for
running Gate 1b rather than reasoning about it.

⚠️ **One diagnosis was retracted along the way.** The bad build reported 2,237 LUTs / 7 BRAM
against the record's 4,586 / 0, and the first explanation offered was that this machine's Vivado
inferred block RAM where the Phase 1 machine had not — which would have contradicted the standing
finding that a 12,544-bit-wide store cannot map to block RAM at all. **Wrong:** the difference was
entirely that the design contained JSC's network, and the rebuild returned 4,586 / 0 BRAM exactly.
`grep -c "lut_node #" build/rtl/dwn_core.v` would have answered it in a second. Same failure of
method as §5.3 — an anomaly explained by hypothesis instead of by inspection.

---

## 6. Reproducing this

```
.venv\Scripts\python.exe scripts\verify_phase1.py             # 12/12 -- JSC parity first
.venv\Scripts\python.exe dse\grid.py --dataset mnist          # the grid and its budget
.venv\Scripts\python.exe dse\run.py --dataset mnist --list    # what has a checkpoint
.venv\Scripts\python.exe dse\run.py --dataset mnist --all --impl
.venv\Scripts\python.exe dse\report.py --dataset mnist --snapshot
.venv\Scripts\python.exe dse\plot.py  --dataset mnist --snapshot
```

**Prove JSC parity before trusting a single sweep point.** Expect **110 / 1,519 / 1,621** — not the
108 / 1,519 / 1,619 that `docs/jsc/report.md` and `README.md` quote, which are the `jsc-complete` tag's
pre-argmax-tree figures. Both are right; they describe different commits.

Checkpoints are not in git (one JSC file exceeds GitHub's 100 MB limit). `docs/mnist/results/`
describes all 25 configurations in ~10 KB; the trained models are ~1.2 GB.

---

## 7. What Phase 2 deliberately did not do

**The frontier has no measured edge.** No config failed to fit; the largest built is 33.06% of the
device. So the supported sentence is *"the largest tried was 33.06% and it fit comfortably"* —
**never** *"the largest MNIST model that fits is…"*. From the marginal core cost (2.549 LUT/node)
plus a saturated ~1,315-LUT encoder, the wall is around **6,800–7,600 nodes**, roughly 3.5× beyond
the top rung.

Stopping was a decision, not an oversight. The edge answers *"how large can a DWN get on a Basys 3
before routing fails"* — a device-characterisation question, not an MNIST one, and not a
prerequisite for a generator that never needs to know where the device runs out. The paper's own
MNIST configuration fits at 16.65% and meets timing, so the largest model anyone would plausibly
build is already measured with room to spare. Reaching the wall costs ~1 h of GPU, 1.2 GB of
downloads and ~1.5 h of place-and-route that scales worse than linearly near high occupancy.

**The two frontiers are not merged, and should not be.**

| | JSC | MNIST |
|---|---|---|
| features / classes | 16 / 5 | 784 / 10 |
| word format | Q3.12 (16-bit) | Q0.8 (9-bit) |
| accuracy scale | ~73–76% | ~93–98% |
| widest built | `1x3000 z=50`, 13,972 LUTs, **67.2%** | `1x2000`, 6,294 LUTs, **30.3%** |
| cost per node there | **4.66 LUT/node** | **3.15 LUT/node** |
| encoder share there | **42%** | **21%** |

Accuracy scales differ by 20 points, so a shared axis would be meaningless, and the encoder
mechanism runs opposite in the two (§4.2). **JSC reaches high occupancy at a fraction of MNIST's
node count**, which is why JSC is the cheaper vehicle for probing the routing limit. A combined
figure would need a normalised axis and this reasoning stated alongside it.

**Also not done:** the `gaussian` and `linear` encoding axes on MNIST (never trained); a
z-dependent area model (§5.5); the `2x[2000,500]` monotone taper; extending the ladder to bracket
the wall.

~~🔴 **Outstanding and required: Gate 1b on silicon since the descriptor refactor.**~~
✅ **Done 2026-08-12, both datasets** — see §5.7. `verify_phase1.py --with-board` gives 22/22.

---

## 8. Pointers

- `docs/mnist/phase2-ledger.md` — the dated running log, with every retraction in place
- `docs/mnist/phase1-report.md` — the port that made this sweep possible
- `docs/mnist/reduction-ledger.md` — the learnable-reduction study and the `tau` confound
- `docs/mnist/results/` — the snapshot: `sweep-results.json`, `.csv`, and both figures
- `docs/jsc/phase2-report.md` — the JSC sweep this parallels, and the source of §5.3's rule
- `docs/reference/tool-roadmap.md` — what the generator still needs to become a tool
