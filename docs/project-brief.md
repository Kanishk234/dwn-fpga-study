# DWN on FPGA — Project Brief

**Standalone document.** Everything needed to understand and start this project is here; no other
files or prior conversation required. Written 2026-08-01 (rev. 2). Intended to become the README of
the repo.

---

## 1. What this project is, in one paragraph

Deploy a **Differentiable Weightless Neural Network (DWN)** — a machine learning model made entirely of
lookup tables, published at ICML 2024 — onto a **Digilent Basys 3** FPGA board (Xilinx Artix-7
XC7A35T), writing the hardware in **hand-written Verilog** rather than an HLS toolchain, since no
open RTL implementation of DWN exists for small FPGAs. Then run two engineering studies on that
hardware: a **design-space exploration (DSE)** mapping the accuracy/area/latency frontier on a small
FPGA, and a **controlled comparison (CC)** against the standard FPGA-ML toolchains (hls4ml, conifer),
positioned against the published LUT-DNN literature. The goal is a deployed, benchmarked ML model on
real hardware with results that don't currently exist — not a port, and not a toy.

**Team:** two people, one shared repo, one shared implementation, divided by component and, within
each study, by axis (see §10–11).

---

## 2. What you're actually going to build

If someone asks "what are you making," this is the literal answer.

**A repo containing:**
- `exporter/` — Python: trained DWN checkpoint → LUT tables, wiring, and thresholds
- `rtlgen/` — Python: export → synthesizable Verilog
- `rtl/` — hand-written Verilog: LUT node, layer, pipeline registers, reduction, argmax
- `tb/` — bit-exact golden-model testbench (this is your only correctness signal — see Gate 1, §12)
- `harness/` — UART controller, BRAM test-vector store, cycle counter, FSM, 7-segment driver
- `dse/` — sweep automation, Vivado run scripts, report parsers, Pareto plotting
- `cc/` — the hls4ml project and the conifer project, both targeting `xc7a35t`, plus comparison scripts

**A working demo:** a bitstream running on a physical Basys 3, classifying JSC test vectors two ways —
**benchmark mode** (vectors preloaded in BRAM, cycle-accurate core throughput, comparable to the
paper) and **interactive mode** (one sample over UART, prediction on the 7-segment display, visibly
slow because of the I/O wall — see §6).

**Study 1 output (DSE):** Pareto plots (accuracy vs. LUTs, accuracy vs. latency, area vs. Fmax) across
the swept configuration space, plus a single headline number: the largest DWN that fits an XC7A35T,
and what it scores.

**Study 2 output (CC):** a same-silicon table (accuracy, LUT/FF/BRAM/DSP, Fmax, latency, throughput,
power estimate) comparing your DWN core to hls4ml and conifer designs synthesized for the identical
part — **plus** a literature-comparison table/plot positioning your results against published
LogicNets/PolyLUT/NeuraLUT/TreeLUT/etc. numbers on JSC (§8).

**Optional (stretch):** a short written report or workshop-style paper draft (§14).

That's the whole deliverable set. Everything else in this document is context and planning for how to
get there.

---

## 3. Who's doing this, and what you already know

Two undergrads (sophomores). Relevant prior experience:

- Both have already **deployed an MLP via hls4ml and a GBDT via conifer to an FPGA for inference.**
  CC's hands-on half (§10, Study 2) is largely familiar territory.
- That prior work leaned heavily on Claude Code, so understanding of *why* those flows work is
  shakier than the fact that they worked. Budget real time to actually understand the tools you're
  reusing, not just re-run them.
- Both completed an **intro digital logic design course** — combinational/sequential logic and basic
  Verilog are known, but neither of you has built a full RTL generation pipeline, a bit-exact
  testbench, or done FPGA bring-up before. Treat Phase 1 (§12) as genuinely new territory.

Net effect on planning: CC should go faster than a from-scratch team's would. The RTL core and DSE
automation should **not** be assumed faster just because there are two of you — that part is new to
both of you and has a real learning curve baked in.

---

## 4. Background: what a DWN actually is

A DWN is a stack of lookup tables and **nothing else**. No weights, no multiplications, no arithmetic
between layers.

```
real-valued inputs
   → thermometer/distributive binarization (each feature → a set of threshold bits)
   → bits grouped into n-bit tuples
   → each tuple addresses a LUT-n; the LUT emits one bit
   → those output bits form addresses into the next LUT layer
   → ... repeat for L layers ...
   → final layer reduced (popcount, or a learned LUT pyramid) into per-class scores
   → argmax
```

Each "neuron" is an n-input RAM node whose **2ⁿ table entries are the learned parameters**. `n` is a
hyperparameter; the paper evaluates n=2 and n=6.

### The four contributions of the paper

**Extended Finite Difference (EFD)** — the enabler. Earlier work (DiffLogicNet) trained LUT networks by
relaxing over *all possible* binary functions, costing O(2^(2ⁿ)) — for a LUT-6 that is
18,446,744,073,709,551,616 parameters to represent one node. EFD instead approximates the derivative of
the table's *addressing* function, weighting contributions from all table positions by Hamming distance
to the address in use. Cost drops to O(2ⁿ): **1.8×10¹⁹ → 64**. This is what makes multi-layer,
large-LUT weightless networks trainable at all.

**Learnable Mapping** — earlier weightless networks wired LUTs together pseudo-randomly. DWN learns the
connections via a weight matrix during training and takes `argmax` at inference. Since that argmax is
input-independent, the matrix is discarded afterwards: **better wiring at zero inference cost**.

**Learnable Reduction** — replaces the final popcount with a pyramid of shrinking LUT layers. In small
models the popcount circuit can be as large as the network itself.

**Spectral Regularization** — L2 norm over the Fourier coefficients of the pseudo-Boolean function.
Standard L1/L2 can't apply (only the *sign* of a table entry matters), and tiny DWNs memorize easily.
Effect: unvisited table entries inherit values from their Hamming-distance-1 neighbours.

### Why n=6 is the entire point on this hardware

A Xilinx LUT-6 (internally two LUT-5s plus a 2:1 MUX) implements **exactly one six-input RAM node**.
From the paper: *"DWNs with n=6 make efficient use of readily available FPGA resources."*

This is not a model that maps *well* onto the fabric. **A DWN-6 neuron is a Xilinx LUT6.** The Basys 3
has 20,800 of them. Consequence: every DWN result in the paper uses **zero DSPs and zero BRAM** — the
model lives entirely in logic.

---

## 5. Published DWN results (the numbers to reproduce and beat)

FPGA, Zynq Z-7045 @ 200 MHz, I/O limited to 112 bits/cycle:

| Dataset | Model | Accuracy | Latency | Throughput | Energy | LUTs |
|---|---|---|---|---|---|---|
| MNIST | FINN (BNN) | 98.40% | 2440 ns | 1.56M/s | 5445 nJ | 83,000 |
| MNIST | ULEEN (WNN) | 98.46% | 940 ns | 4.05M/s | 823 nJ | 123,100 |
| MNIST | **DWN (n=6, sm)** | 97.80% | **60 ns** | **50.0M/s** | **2.5 nJ** | **2,100** |
| MNIST | DWN (n=6, lg) | 98.31% | 125 ns | 25.0M/s | 19.0 nJ | 4,600 |
| KWS | ULEEN | 70.34% | 390 ns | 10.0M/s | 642 nJ | 141,100 |
| KWS | **DWN (n=6)** | **71.52%** | 235 ns | 10.5M/s | 42.3 nJ | **4,800** |
| CIFAR-10 | FINN | **80.10%** | 283000 ns | 21.9K/s | 150685 nJ | 46,300 |
| CIFAR-10 | DWN (n=6) | 57.42% | 2190 ns | 468K/s | 3972 nJ | 16,700 |

Geometric-mean improvement vs FINN: **20.7× latency, 12.3× throughput, 121.6× energy, 11.7× area.**
Vs ULEEN: 3.3× / 2.3× / 19.0× / 22.7×. Headline: **2522× energy-delay product vs FINN, 63× vs ULEEN.**

*(FINN wins CIFAR-10 because it is the only one of these that supports convolution.)*

Original DWN paper numbers on **JSC** (out-of-context synthesis, xcvu9p — the device the whole
LUT-DNN literature standardizes on, see §8):

| Dataset | Model | Accuracy | LUTs | FF | DSP | BRAM | Latency |
|---|---|---|---|---|---|---|---|
| JSC | hls4ml | 76.2% | 63,251 | 4,394 | **38** | 0 | 45.0 ns |
| JSC | **DWN (n=6, lg)** | **76.3%** | **4,972** | 3,305 | **0** | 0 | **7.3 ns** |
| JSC | DWN (n=6, sm) | 71.1% | **20** | 22 | 0 | 0 | **0.6 ns** |
| MNIST | PolyLUT | 96% | 70,673 | 4,681 | 0 | 0 | 16.0 ns |
| MNIST | NeuraLUT | 96% | 54,798 | 3,757 | 0 | 0 | 12.0 ns |
| MNIST | **DWN (n=6, sm)** | **97.1%** | **692** | 422 | 0 | 0 | **2.4 ns** |

On microcontrollers, bit-packed DWN beats **XGBoost by an average of 5.4% accuracy**.

---

## 6. The hardware, and what fits

| Resource | Basys 3 / XC7A35T |
|---|---|
| LUTs | 20,800 (6-input) |
| Flip-flops | 41,600 |
| Block RAM | 1,800 Kb ≈ **225 KB** |
| DSP slices | 90 |
| **External DRAM** | **none** |
| On-board clock | 100 MHz |
| Speed grade | **-1** (slow) |
| PC link | USB → FT2232HQ **UART bridge** (no SPI to host) |

Both of the paper's target FPGAs are LUT6 architectures like Artix-7, so **LUT counts transfer
honestly**:

| Model | LUTs | % of Basys 3 |
|---|---|---|
| DWN JSC (sm) | 20 | 0.1% |
| DWN MNIST (sm, out-of-context) | 692 | 3% |
| DWN MNIST (sm) | 2,100 | 10% |
| DWN MNIST (lg) | 4,600 | 22% |
| DWN KWS | 4,800 | 23% |
| DWN JSC (lg) | 4,972 | 24% |
| DWN CIFAR-10 | 16,700 | 80% — skip |

**Everything except CIFAR-10 fits with 75%+ headroom, using zero BRAM and zero DSPs.**

### Two constraints that shape the whole build

**Clock speed will not transfer.** The paper's 827–3030 MHz figures are speed-graded Z-7045/xcvu9p
parts. A -1 Artix-7 will land around **100–200 MHz**. Report latency in **cycles as well as
nanoseconds** so comparisons stay meaningful.

**You will be I/O-bound, not compute-bound.** Their Z-7045 was already limited by 112 bits/cycle. Over
UART at ~1 Mbaud, a 32-byte JSC sample takes ~320 µs to arrive while the model classifies in ~120 ns —
**I/O-bound by roughly 2,600×**. Design around this explicitly (see §9's harness section).

---

## 7. Dataset: JSC (jet substructure classification)

**Only dataset in core scope.** A second domain is an explicit stretch goal (§13), not part of the
plan you're committing to.

**What it is.** The Large Hadron Collider smashes protons together. Quarks and gluons produced in
those collisions can't exist alone — within ~10⁻²³ s they pull new particles out of the vacuum and
become a **spray of dozens or hundreds of particles flying in roughly the same direction**: a **jet**.
The task is to identify what particle *created* it:

| Origin | Jet appearance |
|---|---|
| light quark (q) | one tight core |
| gluon (g) | one core, wider and messier |
| W boson | decays to 2 quarks first → **two** sub-clusters in one jet |
| Z boson | same, two clusters, different mass |
| top quark (t) | decays to 3 quarks → **three** sub-clusters |

Heavy particles decay into multiple quarks *before* hadronizing — that's what "jet **substructure**"
means. The 16 features are physics summary statistics of it: jet mass, particle multiplicity,
N-subjettiness ratios (τ₂₁, τ₃₂), and energy correlation functions.

**Why latency is the product.** The LHC collides at **40 MHz** — petabytes per second, physically
unstorable. A **trigger** decides within a few microseconds whether to keep an event, and the Level-1
trigger is FPGAs with a hard latency budget. A classifier that misses the deadline is useless
regardless of accuracy. This is why hls4ml exists, and why JSC became the standard low-latency
FPGA-ML benchmark.

**Why it's primary:** 16 clean preprocessed features, 5 classes, public, and it's the shared benchmark
across essentially the entire low-latency FPGA-ML literature — see §8. You always know what the right
answer looks like, which is exactly what you want when bringing up an RTL flow for the first time.

---

## 8. The landscape — every baseline JSC gives you, and which ones you actually run

This section answers directly: **is hls4ml + conifer enough for the controlled comparison? No.**
JSC isn't just "the hls4ml benchmark" — it's the standard benchmark for an entire family of
LUT-based DNN architectures that are closer relatives to DWN than hls4ml or conifer are. Treat this
as two tiers, not one.

### Tier 1 — toolchains you actually run (same silicon, hands-on, §10 Study 2)

- **hls4ml** (CERN/Fermilab) — Keras/PyTorch/ONNX → HLS C++ for Vitis HLS. The de facto reference
  for low-latency FPGA ML. You've deployed an MLP with this before.
- **conifer** — hls4ml's sibling, for decision-tree ensembles (BDTs, random forests). You've deployed
  a GBDT with this before.

These stay the hands-on baseline: general-purpose, mainstream, actually resynthesizable on your part
in a reasonable amount of time, and tools you already have working experience with.

### Tier 2 — the LUT-DNN family you cite, not resynthesize (literature comparison, cheap and necessary)

Every one of these publishes accuracy/LUT/latency numbers on **JSC specifically**, mostly on the same
`xcvu9p` out-of-context setup the original DWN paper uses — meaning you can pull them straight into
your DSE/CC plots without re-running anything:

- **LogicNets** (2020) — origin of "map a trained neuron directly to a LUT." Open-source
  (github.com/Xilinx/logicnets).
- **PolyLUT** (2023) / **PolyLUT-Add** (2024) — polynomial feature expansion per LUT; PolyLUT-Add adds
  an adder tree to cut LUT count 2.0–13.9× over PolyLUT.
- **NeuraLUT** (2024) / **NeuraLUT-Assemble** (2025) — maps a whole small sub-network into one LUT;
  Assemble adds tree-structured fan-in. Open-source (github.com/MartaAndronic/NeuraLUT).
- **TreeLUT** (2025) — decision trees synthesized directly as LUT structures. Notably the closest
  *conceptual* relative to conifer (both target GBDTs), but via a specialized LUT-tree architecture
  rather than a generic HLS compile — worth citing even though you're not resynthesizing it.
  Open-source.
- **AmigoLUT** (2025), **LLNN** (2025), **ReducedLUT** (2025) — ensemble-of-simple-networks,
  logic-gate-based, and table-decomposition approaches respectively; newer and smaller-community than
  the above, include if time allows.

Representative published JSC numbers (xcvu9p, out-of-context, ~700 MHz), for scale — pull the full,
current set for your own comparison table rather than treating this as final:

| Model | Accuracy | LUTs | Latency (ns) |
|---|---|---|---|
| hls4ml | 76.2% | 63,251 | 45.0 |
| DWN (lg, original paper) | 76.3% | 4,972 | 7.3 |
| NeuraLUT-Assemble | 76.0% | 1,780 | 2.1 |
| TreeLUT | 76.0% | 2,234 | 2.7 |
| PolyLUT-Add | 75.0% | 36,484 | 16 |
| NeuraLUT | 75.0% | 92,357 | 14 |
| PolyLUT | 75.0% | 236,541 | 21 |
| LogicNets | 73.1% | 36,415 | 6 |
| DWN (sm-50) | 74.0% | 311 | 2.0 |
| DWN (sm-10) | 71.2% | 64 | 1.6 |

*(DWN numbers here include a fine-tuned thermometer encoder, per the related work below — slightly
larger than the encoder-excluded numbers in §5.)*

### Related work you need to know about and cite

**Implementation and Analysis of Thermometer Encoding in DWN FPGA Accelerators** (Mecik & Kumm, Fulda
University of Applied Sciences, Dec 2025, arXiv:2512.15251) — an independent group already built a
DWN hardware generator (via the FloPoCo framework) and benchmarked it on JSC, specifically measuring
the hardware cost of the thermometer encoder that DWN's original paper left out of its resource counts
(the encoder alone can add up to 3.2× more LUTs than the paper's numbers suggest).

**This doesn't replace your project — it changes what you need to say to differentiate it:**
- They target the **same large device as the original paper** (xcvu9p). Nobody has done this on an
  entry-level part like a Basys 3. That gap is untouched.
- Their contribution is **encoder-cost analysis on one architecture per size class**. Yours is a
  **Pareto frontier across architecture axes** (n, layer count/width, reduction method) plus a
  **controlled toolchain comparison** plus an **I/O-wall characterization on real interactive
  hardware**. Different questions.
- No public code was found for their generator — cite the paper, don't assume you can build on it.

Cite this paper explicitly in your writeup as the closest related work, and state the distinction
in those terms rather than treating it as a footnote — a reader who knows this space will find it
regardless, so get ahead of it.

---

## 9. What actually gets built (technical breakdown)

**Verified 2026-08-01:** the DWN repo (`github.com/alanbacellar/DWN`) contains **only the PyTorch
training library** (`src/torch_dwn/`, `examples/`). **No RTL, no Verilog, no HLS, no FPGA flow** from
the original authors, and (per §8) the one independent hardware generator that does exist targets a
large device with no public code. So the hardware is written from scratch, in Verilog, for a small
FPGA — that's still the real gap. It doesn't mean thousands of hand-typed lines:

| Component | What it is | Roughly |
|---|---|---|
| Model exporter | Python: checkpoint → LUT tables + wiring + thresholds | ~200 lines |
| RTL generator | Python: emits Verilog from the export | ~300 lines |
| Verilog templates | LUT node, layer, pipeline regs, reduction, argmax | ~200 lines by hand |
| Harness | UART, BRAM vector store, cycle counter, FSM, 7-seg | ~500 lines by hand |
| Verification | bit-exact golden model + testbench | ~300 lines |

The generated core is mostly this, repeated:

```verilog
localparam [63:0] TABLE = 64'h...;   // from export
assign out = TABLE[addr];            // 6-bit addr → exactly one LUT6
```

Inter-layer wiring is fixed after training, so it is **pure wire assignment — zero logic, zero cost**.
Register between layers for initiation interval 1: one classification per clock, latency = layer count.

### The dual-mode harness (this is where the I/O wall gets handled)

- **Benchmark mode** — preload test vectors into BRAM, run the pipeline flat out, count cycles in
  hardware. Measures the *core's* throughput rather than the serial link's. This is the number
  comparable to the paper.
- **Interactive mode** — one sample at a time over UART, prediction back over UART and onto the
  7-segment display. The demo path; honestly slow, and worth saying so.

Mode selected by a switch. LEDs for status, 7-seg for class or measured throughput.

---

## 10. The two studies, and how the shared repo is split

Both people work on **one implementation, one repo**. Work is divided by component and, within each
study, by axis or toolchain — not by duplicating the whole build twice. Both of you should be able to
run the full flow solo by the end of Phase 1; if only one person understands the pipeline end-to-end,
that's a silo to fix before moving on.

**Sequencing is fixed: DSE before CC.** To compare at iso-area (Study 2) you need to be able to
*produce* a DWN at a target area — which is exactly what the sweep (Study 1) gives you. Build CC
first and you'll redo it once the sweep exists.

### Study 1 — Design Space Exploration (DSE)

*Question: what is the accuracy/area/latency Pareto frontier for DWN on a fixed small FPGA?*

Build the core fully parameterized, plus a scripted flow: **train → export → generate RTL → Vivado →
parse reports**, looped over dozens of configurations.

Sweep axes: `n` (2/4/6), layer count, layer width, thermometer resolution (bits per feature), Learnable
Reduction vs plain popcount, pipeline depth.

**Split:** one person owns the architecture axes (n, layer count/width, reduction method), the other
owns encoding/timing axes (thermometer resolution, pipeline depth). Each of you runs your slice of the
grid — ideally on separate machines in parallel, since this is wall-clock-bound Vivado time, not
thinking time — then merge into one combined Pareto frontier.

Deliverable: Pareto plots (accuracy vs LUTs, accuracy vs latency, area vs Fmax) plus *"the largest DWN
that fits an XC7A35T, and what it scores."* **Failed routing runs are data** — a congestion wall at N
LUTs is a finding, not a dead end.

### Study 2 — Controlled Comparison (CC)

*Question: for the same task on the same silicon, how does hand-written weightless RTL compare to the
standard toolchains — and where does it sit against the wider LUT-DNN literature?*

**Hands-on half.** Generate an **hls4ml** design and a **conifer** design for JSC, synthesized for the
same `xc7a35t` part, same clock constraint, same synthesis strategy. Compare at **iso-accuracy** and
**iso-area** — not "our one model vs their one model."

**Split:** one person owns the hls4ml design, the other owns conifer — a clean seam, since they're
genuinely separate toolchains with different setup and different failure modes (see §12, risk #5:
hls4ml may not fit the part at all). Analysis and writeup happen together once both designs are
synthesized.

Measure: accuracy, LUT/FF/BRAM/DSP, Fmax, latency (cycles + ns), throughput, and Vivado power estimate
(flag it as an estimate).

A likely finding is baked in: hls4ml's published JSC design is **63,251 LUTs against your 20,800 — it
does not fit.** Shrinking it until it does (pruning, fewer bits) *is* the experiment, and "the
standard flow doesn't fit; the weightless one uses 24%" is a strong result.

**Literature half.** Build one combined table/plot placing your DWN results alongside the published
LUT-DNN family from §8 (LogicNets, PolyLUT, PolyLUT-Add, NeuraLUT, NeuraLUT-Assemble, TreeLUT, and the
Mecik & Kumm thermometer-encoding DWN numbers) on JSC. No resynthesis required — this is a citation
and plotting exercise, roughly a few days of joint work, done once both of you have your own numbers
in hand. This is what actually answers "how does DWN compare" for a reader who knows the space, and
it's the part that's missing if you stop at hls4ml/conifer alone.

---

## 11. Master plan: how the work actually breaks down

```
Phase 1 — CORE (joint, both learn the whole flow)
├─ Weeks 1–2:  toolchain setup, reproduce DWN training in PyTorch,
│              tiny 2-layer sanity model, verify Vivado maps LUT6s correctly (risk #1)
├─ Weeks 2–4:  exporter (Person A) + RTL generator (Person B), built against
│              a jointly-agreed export format
├─ Weeks 4–6:  Verilog templates + golden-model testbench — pair on this,
│              Gate 1 is the highest-stakes checkpoint in the project
└─ Weeks 6–8:  harness — benchmark mode (Person A) + interactive mode (Person B),
               board bring-up together
        │
        ▼
Phase 2 — DSE (split by axis, run in parallel)
├─ Person A: architecture axes   (n, layer count/width, reduction method)
├─ Person B: encoding/timing axes (thermometer resolution, pipeline depth)
└─ Merge into one combined Pareto frontier
        │
        ▼
Phase 3 — CC (split by toolchain, converge on analysis)
├─ Person A: hls4ml design, same part/clock/strategy as conifer track
├─ Person B: conifer design, same part/clock/strategy as hls4ml track
├─ Joint: literature-comparison table against the LUT-DNN family (§8)
└─ Joint: iso-accuracy / iso-area writeup
        │
        ▼
   [Gate 2 — stop here with a complete project, or continue to stretch]
        │
        ▼
Stretch — second dataset (§13), explicitly optional
```

| Phase | Estimate | Split |
|---|---|---|
| 1 — Core | 5–8 wk | Joint; both understand the full flow before Gate 1 |
| 2 — DSE | 2–3 wk | By axis (architecture vs. encoding/timing), merged |
| 3 — CC | 1–2 wk, likely faster given prior hls4ml/conifer experience | By toolchain, joint analysis + literature table |

**~8–13 weeks**, calendar time. Treat Phase 1's low end as optimistic given neither of you has built a
full RTL generation pipeline before — better to budget the high end and be pleasantly surprised.

**Gate 1:** do not write the harness until the RTL core matches the bit-exact golden model in
simulation on every test vector. This matters *more* here than a duplicate-build design would, since
there's no second independent implementation to cross-check against — the golden-model testbench is
your only correctness signal. Don't cut corners on it.

**Gate 2:** a second dataset (§13) is an explicit **stretch goal**. After Phase 3 you already have a
complete, defensible project — a working DWN on hardware, a Pareto frontier nobody has published for a
small FPGA, and a controlled comparison against both the standard toolchains and the published
LUT-DNN literature. If time runs short, stop there with something whole.

---

## 12. Risks, in order

**1. Vivado may not map tables to LUT6s.** It might infer distributed RAM or BRAM instead. Check the
post-synthesis netlist **early**, on a tiny two-layer model, before building anything else. Mitigation:
`(* rom_style = "distributed" *)`, or restructure as explicit constants.

**2. Routing congestion — the paper hit this.** Several DiffLogicNet and DWN *n=2* models "could not be
implemented on our target FPGA," because "it would be infeasibly expensive for FPGAs to implement a
full crossbar interconnect." Critically: **"all DWNs with n=6 were successfully routed and
implemented."** Learnable Mapping produces irregular wiring, and irregular wiring congests. The
Artix-7 is far smaller than their part, so this is the **top technical risk**. Mitigation: **n=6
only**; start with the smallest working model and scale up.

**3. Clock/latency claims won't transfer** (§6). Report cycles alongside nanoseconds.

**4. Exporter format drift.** The repo's checkpoint format is whatever the authors chose. Budget real
time for the exporter and validate it via the golden model, not by inspection.

**5. hls4ml may not fit the target part at all.** This is a finding, not a blocker — but budget time
for shrinking it to get a valid comparison point.

**6. Silo risk from the split-by-axis structure.** With a shared repo split by axis/toolchain rather
than duplicated, there's no automatic cross-check if one person's slice has a bug. Mitigate with code
review on each other's parts, not just your own, and treat Gate 1 as a joint responsibility.

**7. Closely related work exists and will be found.** The Mecik & Kumm paper (§8) means "we built the
first DWN hardware" is not an accurate claim — "first on an entry-level FPGA, first with this
Pareto-frontier scope" is. State it that way from the start rather than getting caught flat-footed by
a reviewer or classmate who's read the same paper.

---

## 13. Stretch goal: a second dataset

Out of core scope. If Phases 1–3 land with time to spare, both studies (DSE + CC) get re-run on a
second domain with no published LUT-network baseline — i.e. all comparisons would be ones you build.
Two candidates, evaluated but not committed to:

| Dataset | One-liner | Verdict |
|---|---|---|
| **ECG arrhythmia** (MIT-BIH) | Beat classification from segmented features; implant-scale framing ("small enough to live in a pacemaker"). Severe class imbalance — needs per-class sensitivity/F1, not raw accuracy. | Viable; no front-end signal-processing dependency beyond beat segmentation. |
| **Limit order book** (FI-2010 / LOBSTER) | 40 features (price+volume, top 10 levels/side), predict mid-price up/down/flat. XGBoost is the genuine incumbent, so the conifer comparison is meaningful. | Real data-wrangling effort; noisy labels compress the accuracy axis. Frame as a *latency demonstration*, not a trading model. |

Other candidates considered and set aside: Speech Commands/KWS (needs an MFCC front-end — a second
project), intrusion detection (needs a Pmod NIC for real packets — added scope, though note UNSW-NB15
is also a published NeuraLUT-Assemble/PolyLUT-Add benchmark if you want a literature-comparison-only
version), RadioML (raw IQ favors CNNs, DWN likely underwhelms — higher risk), MNIST/CIFAR-10 (sanity
targets or too large, not projects on their own).

---

## 14. What makes this a contribution rather than a port

The paper already deployed DWN on an FPGA, and (per §8) someone has since built a second DWN hardware
generator too. What's still new here:

- **Entry-level portability.** Every existing DWN hardware result — the original paper's and Mecik &
  Kumm's — targets parts costing 20–100× a Basys 3. *"Does this work on the FPGA students actually
  own?"* is a real, unanswered question.
- **The Pareto frontier on a constrained part.** The paper gives three configs; the frontier —
  including where the congestion wall sits — doesn't exist anywhere, for any device.
- **A controlled comparison on identical silicon, against both toolchains and the literature.**
  Published hls4ml/conifer numbers are on large FPGAs, and the LUT-DNN family (§8) has never been
  compared to DWN on a shared small-FPGA baseline. Hand-written RTL vs. generated HLS vs. the
  published LUT-DNN landscape, same chip, same clock, iso-accuracy — nobody has run it.
- **The I/O wall, quantified.** A 50M-samples/s core behind a 1 Mbaud link is 99.9% idle. Measuring it
  and architecting around it, on real interactive hardware, is real, undocumented engineering.

**If you reach the stretch goal (§13) and want to turn this into a paper:** the natural target is an
FPGA-ML workshop venue (FCCM/FPL/FPT workshops, ReConFig, or a student research venue) rather than a
flagship conference, given the scope. Structure follows this section's contribution list directly:
intro → DWN background → related work (§8, cite Mecik & Kumm explicitly) → implementation → DSE
results → CC results (toolchain + literature) → discussion, with an author-contributions note
reflecting the axis/toolchain split in §10–11.

---

## 15. References

- **DWN paper:** [Differentiable Weightless Neural Networks, Bacellar et al., ICML 2024](https://arxiv.org/abs/2410.11112)
  — Federal University of Rio de Janeiro, **UT Austin**, ISCTE Lisbon, UT San Antonio, IT Porto
- **DWN code (PyTorch only):** https://github.com/alanbacellar/DWN
- **DWN hardware generator + thermometer encoding cost analysis:** Mecik & Kumm,
  [arXiv:2512.15251](https://arxiv.org/abs/2512.15251), Fulda University of Applied Sciences, 2025
  — closely related work; see §8
- **ULEEN:** Susskind et al., 2023 (UT Austin) — the single-layer WNN predecessor
- **LogicNets:** Umuroglu et al., FPL 2020 — github.com/Xilinx/logicnets
- **PolyLUT:** Andronic & Constantinides, ICFPT 2023 · **PolyLUT-Add:** Lou et al., FPL 2024
- **NeuraLUT:** Andronic & Constantinides, FPL 2024 · **NeuraLUT-Assemble:** arXiv:2504.00592, 2025 —
  github.com/MartaAndronic/NeuraLUT
- **TreeLUT:** Khataei & Bazargan, FPGA 2025
- **AmigoLUT:** Weng et al., FPGA 2025 · **LLNN:** Ramirez et al., 2025 · **ReducedLUT:** Cassidy et
  al., FPGA 2025
- **hls4ml:** https://fastmachinelearning.org/hls4ml/ · **conifer:** https://github.com/thesps/conifer
- **FINN:** https://xilinx.github.io/finn/
- **Basys 3 Reference Manual:** https://digilent.com/reference/basys3/refmanual
- **MIT-BIH Arrhythmia Database:** https://physionet.org/content/mitdb/
- **FI-2010 LOB benchmark** / **LOBSTER:** https://lobsterdata.com/

---

*Resource and timing figures for the Basys 3 are estimates derived from datasheets and the published
work above. Nothing here has been synthesized on hardware yet.*
