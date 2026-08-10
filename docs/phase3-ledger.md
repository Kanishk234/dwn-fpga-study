# Phase 3 ledger — the Controlled Comparison (Study 2)

Running log for Phase 3. Plan: `docs/phase3-plan.md`. Handoff: `docs/phase3-handoff.md`.

**Split:** the *hands-on half* (conifer, hls4ml — plan §2) is being run by the other person on the
other machine. **This ledger currently covers the literature half only** (plan §3–§4), which needs
no board, no Vivado and no synthesis. When the two halves merge, say so here explicitly and note
which machine produced which rows.

---

## Status

| Step | What | Status |
|---|---|---|
| 3L-a | Refresh the literature list — brief §8 is stale (plan §3 ⚠️) | ✅ done 2026-08-10 |
| 3L-b | Settle the encoder-convention trap (plan §4.1) | ✅ done 2026-08-10 |
| 3L-c | Pull per-paper JSC numbers into a machine-readable table | ⬜ next |
| 3L-d | Combined comparison table + Pareto plot with our 15 frontier points | ⬜ |
| 3L-e | Phase 3 report — literature section | ⬜ |
| 3M-a | conifer (GBDT) — *other machine* | ⬜ |
| 3M-b | hls4ml (quantized MLP) — *other machine* | ⬜ |

---

## Log

### 2026-08-10 — the literature list, refreshed

Brief §8 was written at project start and plan §3 flags it as stale. It is. What it misses:

| Work | What it is | Status in brief §8 |
|---|---|---|
| **A Survey on LUT-based DNNs in FPGAs** (arXiv:2506.07367) | consolidated JSC tables for the whole family | **absent** — and it is the single most useful reference for us |
| **LLNN** (Ramirez et al., IEEE TCAS-I 2025) | LUT logic-based networks | listed as "if time allows"; now a standard row |
| **ReducedLUT** (Cassidy et al., FPGA 2025) | table decomposition with don't-cares | listed as "if time allows"; now a standard row |
| **AmigoLUT** (Weng et al., FPGA 2025) | ensembles of small LUT nets | listed as "if time allows"; now a standard row |
| **SparseLUT** (Lou et al., TCAD; arXiv:2601.09773, Jan 2026) | sparse connectivity optimisation, +0.94 pp on JSC | **absent** |
| **WARP Logic Neural Networks** (Gerlach et al., arXiv:2602.03527, Feb 2026) | Walsh-relaxation training for logic nets | **absent**; no JSC hardware numbers — training paper, cite as related work only |
| **BitLogic** (arXiv:2602.07400, Feb 2026) | gradient-based FPGA-native training | **absent**, unread |
| **FPGN** (Liang et al., arXiv:2607.08427, Jul 2026) | differentiable LUTs; claims up to 205× LUT efficiency over prior differentiable LUT-native nets, **and compares directly against DWN** | **absent** — the most directly threatening new baseline |
| **bit-flip resilience of logic/LUT nets** (arXiv:2603.22770) | robustness study | **absent**; not a JSC area/accuracy row |

The three "if time allows" entries are no longer optional — Mecik & Kumm's Table II carries all
three as ordinary rows. Treat brief §8's tier-2 list as superseded by this table.

**FPGN is the one to read properly.** It is the only new work that benchmarks *against DWN* on JSC,
and at 3,345 LUTs / 76.0% on VU9P it is in the same region as the DWN `lg` numbers. Whether that
count includes an input encoder is unknown and is exactly the trap below.

### 2026-08-10 — RESOLVED: the encoder-convention trap, and why Mecik & Kumm is our anchor paper

Plan §4.1 says every row must state whether its LUT count includes the input encoder, and that
this is unsettled. It is now settled for the DWN rows, and the answer is cleaner than expected.

Mecik & Kumm (arXiv:2512.15251, Asilomar 2025) report **two variants of every DWN config**:

- **DWN-TEN** ("thermometer-encoded numbers") — the model *expects* thermometer input. The encoder
  is **not** in the design. These numbers reproduce the original DWN paper's exactly (`lg-2400` =
  4,972 LUTs), which is the paper's own convention.
- **DWN-PEN+FT** ("positional encoded numbers" + fine-tuning) — takes ordinary binary features and
  **includes the thermometer encoder in hardware**, with thresholds quantised to 6–9 bits and the
  model fine-tuned to recover accuracy.

So `PEN+FT − TEN` is a published, measured encoder cost, at the same `z=200` we use:

| config | TEN (no encoder) | PEN+FT (with encoder) | encoder | ratio |
|---|---|---|---|---|
| `sm-10` | 20 | 64 (6-bit) | 44 | **3.20×** |
| `sm-50` | 110 | 311 (8-bit) | 201 | 2.83× |
| `md-360` | 720 | 1,697 (9-bit) | 977 | 2.36× |
| `lg-2400` | 4,972 | 7,011 (9-bit) | 2,039 | **1.41×** |

⚠️ **A misreading to avoid.** The abstract's "encoding can increase LUT usage by up to 3.20×" is
the *total-design* multiplier at the smallest model, not a per-component figure, and an automated
summary of this paper twice got the direction backwards (claiming positional encoding was the
expensive one). The numbers were read from the paper's own tables, not from a summary.

**Consequence for our tables:** `DWN-PEN+FT` is the row directly comparable to our totals — same
convention (encoder included), same `z=200`, same distributive thermometer, same architecture.
`DWN-TEN` and the original paper are core-only and must be labelled as such. This retires plan
§4.1 as an open trap for the DWN rows; it stays open for every non-DWN row, none of which state
their convention.

### 2026-08-10 — two of our Phase 2 findings are independently corroborated

Both were ours first from our own measurements; finding them in an independent implementation on
different silicon raises confidence that they are architectural, not artifacts of our generator.

1. **The encoder dominates small models and stops dominating large ones.** Ours: encoder/core
   **14.1× at `1x50` → 2.8× at `1x2400 z=50`.** Theirs: total/core **3.20× at `sm-10` → 1.41× at
   `lg-2400`**, with "for smaller models the thermometer encoders dominate the overall hardware
   costs … for larger models the encoder cost becomes less dominant."
2. **The reduction (popcount + argmax) dominates at scale.** Their future-work item (iv) is
   "optimizing the classification logic, since for large models such as DWN (lg-2400), the popcount
   and LUT layers dominate hardware utilization at smaller input bit-widths." That is the same
   conclusion as our Learnable Reduction retraction of 2026-08-10 (`docs/phase2-ledger.md`), which
   measured the reduction at **34.9% of the headline design**.

Their future-work item (i) — "reducing thermometer encoder outputs by decreasing the number of bits
per feature … 3,200 outputs are currently provided" — is **the `z` sweep we already ran**. We have
the measurement they propose: `z=200 → 50` at 2400 nodes cuts the encoder from a projected ~23k to
5,753 LUTs at a 0.2 pp accuracy cost. This is worth raising in the note to the authors.

### 2026-08-10 — ⚠️ OPEN: our encoder is ~7.6× more expensive than theirs on the same workload

The most actionable thing the literature half has produced. Same model size, same `z`, same
encoding scheme, both post-synthesis LUT counts:

| | ours (`1x50`) | theirs (`sm-50`) |
|---|---|---|
| nodes | 50 | 50 |
| `z` / feature | 200 | 200 |
| thermometer bits | 16 × 200 = 3,200 | 16 × 200 = 3,200 |
| encoding | distributive | distributive |
| accuracy | 73.84% | 74.0% |
| core | 108 | 110 |
| **encoder** | **1,519** | **201** |
| threshold precision | **Q3.12, 16-bit** | **8-bit** |
| part | `xc7a35t-1` | `xcvu9p-2` |
| clock target | 100 MHz | 700 MHz |

Cores agree to within 2 LUTs — strong evidence the two implementations are the same architecture
and that the gap is entirely in the encoder. **1,519 / 201 = 7.6×.**

Our cost is ~**7.5 LUTs per comparator** over 202 used thresholds, which is what a 16-bit
compare-against-constant costs on a carry chain — our encoder is not badly built *for what it is*.
Theirs is ≤1 LUT per comparator, which a 16-bit comparator cannot reach at any effort level.

**The likely mechanism, not yet verified:** they quantise the *input feature* to 6–9 bits before
comparing. At 6 bits a thermometer bit is a Boolean function of 6 inputs — **exactly one LUT6, by
definition**, the same argument that makes one DWN node one LUT. Our `rtlgen/emit_encoder.py`
compares a full 16-bit `WORD_BITS` word per threshold (`exporter/extract.py:118`), so no amount of
threshold-constant folding gets below the carry chain.

**This is a different lever from the one we tested.** Phase 1's per-feature *comparator narrowing*
(−17.1%, and over-narrowed — see `docs/phase2-report.md` §5.6) trimmed comparator widths while
keeping the Q3.12 input. Quantising the shared input word is a change to the datapath's precision,
which we have never swept: `q16.12` is fixed in every one of the 54 sweep configs.

Order-of-magnitude, if a 6-bit input made each thermometer bit one LUT6:

| | encoder LUTs | design total |
|---|---|---|
| `1x50` today | 1,519 | 1,619 |
| `1x50` projected at 6-bit input | ~202 | ~310 |
| `1x2400 z=50` today | 5,753 | 12,751 |
| `1x2400 z=50` projected | ~1,700 | ~8,700 (≈42% of device, from 61.3%) |

⚠️ **Projection from one published data point plus an argument, not a measurement.** It assumes
accuracy survives 6-bit inputs, which for *them* required fine-tuning at that precision — i.e. it
is a training-side change, not an RTL-side one, and would need new checkpoints. Do not put these
numbers in the report as results.

**Status: open, and the strongest candidate on the "what next" list** — bigger than Learnable
Reduction (projected −29% of the design) and, unlike it, corroborated by a published measurement.

---

## Numbers worth quoting

**Our two anchor configs** (measured, post-route, `xc7a35tcpg236-1`, 10 ns, out-of-context):

| config | acc | core | encoder | top | device | Fmax | cycles | ns |
|---|---|---|---|---|---|---|---|---|
| `1x50` (= paper `sm-50`) | 73.84% | 108 | 1,519 | 1,619 | 7.78% | 147.1 MHz | 4 | 27.2 |
| `1x2400 z=50` (headline) | 76.18% | 6,850 | 5,753 | 12,751 | 61.3% | 101.3 MHz | 4 | 39.5 |

**The literature, on JSC** — from Mecik & Kumm Table II, all on `xcvu9p-flgb2104-2-i`,
out-of-context, `Flow_PerfOptimized_high`, 700 MHz target. **DWN rows include the encoder; every
other row's convention is unstated and must be checked before publication.**

| Model | Acc. | LUT | FF | Fmax (MHz) | Lat (ns) |
|---|---|---|---|---|---|
| DWN-PEN+FT (lg-2400, 9-bit) | 76.3% | 7,011 | 961 | 947 | 2.1 |
| NeuraLUT-Assemble | 76.0% | 1,780 | 540 | 941 | 2.1 |
| TreeLUT | 76.0% | 2,234 | 347 | 735 | 2.7 |
| DWN-PEN+FT (md-360, 9-bit) | 75.6% | 1,697 | 198 | 696 | 2.6 |
| TreeLUT | 75.0% | 796 | 74 | 887 | 1.1 |
| PolyLUT-Add | 75.0% | 36,484 | 1,209 | 315 | 16 |
| NeuraLUT | 75.0% | 92,357 | 4,885 | 368 | 14 |
| PolyLUT | 75.0% | 236,541 | 2,775 | 235 | 21 |
| LLNN | 75.0% | 13,926 | 0 | 153 | 6.5 |
| ReducedLUT | 74.9% | 58,409 | 0 | 303 | 17 |
| AmigoLUT-NeuraLUT-S (32) | 74.4% | 42,742 | 4,717 | 520 | 9.6 |
| DWN-PEN+FT (sm-50, 8-bit) | 74.0% | 311 | 52 | 1,011 | 2.0 |
| LogicNets* | 73.1% | 36,415 | 2,790 | 390 | 6 |
| AmigoLUT-NeuraLUT-XS (16) | 72.9% | 1,243 | 1,240 | 1,008 | 5.0 |
| ReducedLUT | 72.5% | 2,786 | 0 | 409 | 4.9 |
| LogicNets* | 72.1% | 15,526 | 881 | 577 | 5 |
| PolyLUT | 72.0% | 12,436 | 773 | 646 | 5 |
| NeuraLUT | 72.0% | 4,684 | 341 | 727 | 3 |
| PolyLUT-Add | 72.0% | 895 | 189 | 750 | 4 |
| LLNN | 72.0% | 6,431 | 0 | 449 | 2.2 |
| DWN-PEN+FT (sm-10, 6-bit) | 71.2% | 64 | 18 | 1,307 | 1.6 |
| AmigoLUT-NeuraLUT-XS (4) | 71.1% | 320 | 482 | 1,445 | 3.5 |

`*` LogicNets rows are the updated numbers from github.com/Xilinx/logicnets, not the FPL 2020 paper.

**Not in the table, needs its own row once read:** FPGN (76.0% / 3,345 LUTs / 730 MHz / 5.5 ns on
VU9P, JSC-OpenML variant) and SparseLUT. FPGN also reports a JSC-CERNBox variant at 74.9% / 12,358
LUTs — ⚠️ **two different JSC variants, so its rows are not directly comparable to the table above
until we confirm which dataset split everyone else used.**

---

## Open questions

| Question | Status |
|---|---|
| **Why is our encoder 7.6× theirs at the same `z`?** | ⚠️ **Open, top priority.** Hypothesis: input-word quantisation to 6–9 bits makes each thermometer bit one LUT6. Ours compares a 16-bit Q3.12 word. Needs a training-side experiment, not an RTL change. See the 2026-08-10 entry. |
| Do the non-DWN rows include an input encoder? | ⚠️ **Open.** Settled for DWN only. LogicNets/PolyLUT/NeuraLUT take quantised inputs directly, so the question may not apply in the same form — but that itself has to be stated per row, not assumed. |
| Which JSC split does each paper use? | ⚠️ **Open.** FPGN reports JSC-CERNBox *and* JSC-OpenML with a 1.1 pp gap between them. If the family is not all on one split, accuracy comparisons at the 0.2 pp level are meaningless — and our whole frontier is argued at that resolution. |
| Is `q16.12` worth sweeping? | ⚠️ **Newly open.** Fixed across all 54 Phase 2 configs. The 7.6× gap suggests it is the largest unswept axis in the project. |
| Does FPGN beat us on our own comparison? | ⬜ Unread. Compares against DWN directly; the most likely paper to change what we can claim. |
| Fmax/latency comparability | ✅ **Closed by plan §4.2.** `xcvu9p-2` at 700 MHz vs `xc7a35t-1` at 100 MHz — LUT counts transfer, ns does not. Report cycles alongside ns. Our 4 cycles vs their 2–7 is the comparison that survives. |

---

## Sources

- Bacellar et al., *Differentiable Weightless Neural Networks*, ICML 2024 — arXiv:2410.11112
- Mecik & Kumm, *Implementation and Analysis of Thermometer Encoding in DWN FPGA Accelerators*,
  Asilomar 2025 — arXiv:2512.15251 — **anchor paper for the encoder convention**
- *A Survey on LUT-based Deep Neural Networks Implemented in FPGAs* — arXiv:2506.07367
- Liang et al., *FPGN*, Jul 2026 — arXiv:2607.08427
- Lou et al., *SparseLUT / connectivity optimisation*, TCAD — arXiv:2601.09773
- Gerlach et al., *WARP Logic Neural Networks*, Feb 2026 — arXiv:2602.03527
- *BitLogic*, Feb 2026 — arXiv:2602.07400
- Umuroglu et al., LogicNets, FPL 2020 · Andronic & Constantinides, PolyLUT (ICFPT 2023),
  NeuraLUT (FPL 2024), NeuraLUT-Assemble (arXiv:2504.00592) · Lou et al., PolyLUT-Add (FPL 2024) ·
  Khataei & Bazargan, TreeLUT (FPGA 2025) · Weng et al., AmigoLUT (FPGA 2025) ·
  Cassidy et al., ReducedLUT (FPGA 2025) · Ramirez et al., LLNN (IEEE TCAS-I 2025)
- Bacellar et al., *Distributive thermometer*, ESANN 2022 — the encoding we and they both use
