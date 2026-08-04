# Phase 1 — what we built, what broke, and how to reproduce it

A Differentiable Weightless Neural Network running on a Digilent Basys 3, in hand-written
Verilog, verified bit-exact against the software model on every one of the 166,000 JSC test
samples.

This is the written-up account. `docs/phase1-ledger.md` is the running log it was written from —
dated, with the dead ends. `releases/phase1/MANIFEST.md` describes the frozen artifact.

---

## 1. Headline results

| | |
|---|---|
| **Gate 1** (simulation, bit-exact) | core **1504/1504**, full datapath **1518/1518** |
| **Gate 1b** (hardware, full test set) | **166,000 / 166,000** |
| Software accuracy | 73.8361% float32 · 73.8349% Q3.12 on hardware |
| Paper's `sm` config | 74.0% — reproduced |
| **DWN core area** | **108 LUTs** (the paper reports **110**) |
| Thermometer encoder | **1519 LUTs** — 14× the core |
| Whole board design | 2058 LUTs (9.89%), 865 FF, 8 BRAM, **0 DSP** |
| Latency / throughput | **4 cycles, II=1** → 99.5 M classifications/s, measured on-chip |
| Fmax | 147.1 MHz out-of-context; +1.753 ns slack at the board's 100 MHz |
| Full test set over UART | **11.2 s** at 5 Mbaud (was 480 s at 115200) |

Zero DSPs and zero BRAM in the model itself — it lives entirely in logic, as the paper claims.

## 2. What actually got built

```
training/          Kaggle notebooks: train a DWN, and dump the full test set without retraining
exporter/          checkpoint -> LUT tables, wiring, thresholds -> Verilog
  extract.py         reads the checkpoint; also the numpy golden model
  emit_core.py       emits the 50-node LUT core
  emit_encoder.py    emits the 202-comparator thermometer encoder + top
  analyze_*.py       precision and encoder-area studies
rtl/               hand-written: lut_node, popcount, argmax, pipe_reg
rtl/gen/           generated: dwn_core, thermometer_encoder, dwn_top
harness/           uart_rx/tx, uart_loader, vector_store, benchmark_fsm, seg7, board top
tb/                Gate 1 testbenches + unit tests, and the vector generator
scripts/           run_gate1, run_tb, run_synth, build_bitstream, program, host
constraints/       basys3.xdc
releases/phase1/   the exact bitstream that passed Gate 1b
```

The model is **50 LUT6 nodes**. That is the whole network: 50 × 64-bit truth tables (400 bytes)
plus 300 wire indices. Everything else is getting data to it.

## 3. The five things that actually cost time

### 3.1 The model was 2pp short, and three plausible explanations were all wrong

Early runs plateaued at ~72.5% against the paper's 74.0%. We tested, in order:

| Hypothesis | Result |
|---|---|
| Thermometer resolution too low (4 → 8 bits) | +0.22pp. Rejected. |
| Batch size wrong (256 → 32) | +0.07pp. Rejected. |
| Learning rate too high | Rejected **without running it** |

The third was killed by reading `third_party/DWN/examples/mnist.py`: our optimizer, schedule,
epochs, tau and mapping pattern were already byte-for-byte the authors' recipe. The one
deviation was batch size, which we had just tested.

**The answer was in the paper's appendix, not in tuning.** Table 14 gives the JSC configs:
**z=200** thermometer bits (we were at 4–8) and a **single** LUT layer (we had two, the second
randomly wired). With a single learnable-mapped layer the thermometer is a *feature pool* the
mapping selects from — 50 nodes × 6 inputs choose 300 slots from 3200 candidates, and we had
been offering 64.

Fixing both reproduced 74.06% on the first try. **50 nodes on a rich input beat 400 nodes on a
starved one.**

*Lesson: three tuning experiments cost more than one careful read of the appendix would have.*

### 3.2 The checkpoint format is full of traps

Read out of the pinned submodule, never inferred (`docs/checkpoint-format.md`):

- **`_LUTLayer__dummy_mapping` is a decoy.** It sits in `state_dict` with the *same shape and
  dtype* as a real wiring tensor, but is only `arange()` reshaped. Exporting it yields a
  structurally valid, completely wrong model. Key off whether `.mapping.weights` exists, never
  off shape.
- **Address bits are LSB-first.** The CUDA kernel does `addr |= bit << l`, so mapping slot 0 is
  the address LSB. Reversing it produces a design that elaborates, synthesizes, and is wrong on
  most inputs.
- **Learnable wiring is `weights.argmax(dim=0)`**, indexed `[j*n + k]`.
- **Table bit is `luts[j][addr] > 0`** — strictly greater.
- **GroupSum zero-pads silently** if the final layer width isn't divisible by the class count.

### 3.3 Three alignment bugs, all of which looked like working designs

Every one produced plausible output rather than an obvious failure:

- **Argmax ties.** With 10 nodes per class there are only 11 possible scores, and **29 of 1000**
  test vectors have a tied top score. `>=` instead of `>` would pass 97% of vectors and silently
  disagree on the rest.
- **`benchmark_fsm` label delay.** The valid flag entered the delay line at address-issue time
  but the label two cycles later, so every prediction was scored against the wrong sample's
  label — producing a believable accuracy number.
- **`uart_loader` write address.** `wr_en` is a registered pulse (asserts the *next* cycle) but
  `wr_addr` incremented in the same cycle, so every vector landed one slot late and slot 0 was
  never written.

*Lesson: each was caught only because a test asserted an exact expected value with deliberately
wrong data mixed in. "It ran without errors" would have passed all three.*

### 3.4 A timing methodology bug that inflated our headline number

The pipeline-depth sweep reported a **2-stage design as 3× faster than the 4-stage one**.
Impossible — removing registers cannot improve timing.

Cause: `build.tcl` created a clock but set no `set_input_delay`/`set_output_delay`, so
input-to-first-register and last-register-to-output paths were **unconstrained and omitted from
the report**. Removing a register moved a long path into the unanalyzed set, so slack *improved*
as the design got worse.

It had also been quietly inflating the single-design number: post-route Fmax was **147.1 MHz,
not the 155.6 MHz** first reported.

*Lesson: the bug was only visible because a sweep produced an impossible ordering. On one design
it was merely optimistic, which is much harder to notice.*

### 3.5 Gate 1b "failed" because the reference was wrong

The first full run reported 5 disagreements in 20,480 — and the hardware was correct.

`pred` in the test set came from PyTorch on **float32** features; the hardware implements
**Q3.12**, a deliberate part of the spec. Scoring hardware against the float model measures the
quantization decision, not the hardware.

Confirmed in software before changing anything: the numpy **float** model agreed with the
reference 20,480/20,480, and the numpy **Q3.12** model disagreed on exactly 5 — matching the
hardware. Across the full set the two differ on **30 of 166,000** samples, worth **−0.0012 pp**.

The same run exposed a second issue: **one feature value in 2,656,000 overflows Q3.12** (8.08 in
scaled space, past the ±8 range). Fixed by saturation, which is lossless here and proven so —
thresholds span [−18618, +17782] inside a [−32768, +32767] word, so a clamped value stays on the
same side of every threshold. Encoder bits verified identical with and without clamping across
all 166,000 samples.

⚠️ **Q3.15 would not have fixed the overflow** — same 3 integer bits, same range. It buys
precision, not headroom. We had recorded it as *the* fallback; that was wrong for range.

## 4. Findings worth carrying into Phase 2

**The encoder costs 14× the core** — 1519 LUTs against 108. Brief §12 risk #3 anticipated at
most 3.2× (Mecik & Kumm). It is not our sloppiness: 1519/202 = 7.5 LUTs per 16-bit
compare-against-constant, which is what a carry chain costs. Per-feature width narrowing was
measured at −17.1% and **not adopted** — 260 LUTs does not justify a spec change at this size.

**The encoder, not the model, decides what fits.** Comparator count grows with node count and
saturates at `features × z` = 3200. The paper's `lg` config (2400 nodes) would need ~24,000 LUTs
of encoder on a **20,800-LUT device**, against a 4,972-LUT core. *DWN-`lg` almost certainly does
not fit on a Basys 3, and the network is not the reason.*

**Learnable Mapping does feature selection for free.** Of 202 comparators, `mass_mmdt` takes 46,
`multiplicity` 39, `zlogz` 37 — and **`d2_b2_mmdt` gets zero**, never read by any node. The
favoured features are the physically obvious discriminators.

**The I/O wall, measured on both sides:**

| Baud | Full 166k run | I/O wall | Link efficiency |
|---|---|---|---|
| 115,200 | 480.0 s | 287,770× | 99.1% |
| 1,000,000 | 55.1 s | 33,060× | 99.4% |
| **5,000,000** | **11.2 s** | **6,707×** | 97.8% |
| 10,000,000 | ✗ no response | — | — |

43× faster end to end with the core untouched. 10 Mbaud divides exactly on both ends and the
FT2232H is rated to 12 M, yet gives nothing while 4 M and 5 M are perfect — **the wall is the
Windows VCP driver**, not the design.

---

## 5. Reproducing this from scratch

Assumes a Basys 3, Windows, and Vivado 2025.2. Every command is copy-pasteable and the expected
output is given so you can tell whether it worked.

### 5.1 Toolchain

```bat
git clone https://github.com/Kanishk234/dwn-fpga.git
cd dwn-fpga
git submodule update --init third_party/DWN
```

Install **Python 3.12** (not newer — Phase 3's hls4ml/conifer lag new releases; not older —
`numpy>=2.3` needs ≥3.11), ticking "py launcher" in the installer. Then:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

> On Windows, bare `python` may hit the Microsoft Store alias stub and fail with "Python was not
> found". Use `py -3.12` to create the venv; inside it, `python` works normally.

If Vivado is not at `C:\AMDDesignTools\2025.2\Vivado\bin`, pass `--vivado-bin <path>` to any
script below.

### 5.2 Gate 1 — prove the RTL in simulation (no board needed)

Regenerates the RTL from the committed checkpoint, regenerates the test vectors, and simulates
both levels:

```bat
.venv\Scripts\python.exe scripts\run_gate1.py
```

Expect:

```
read-back check: 50/50 nodes match the checkpoint
read-back check: 202/202 comparators match the checkpoint
GATE 1 -- dwn_core       vectors tested : 1504   mismatches : 0   RESULT : PASS
GATE 1 -- dwn_top        vectors tested : 1518   mismatches : 0   RESULT : PASS
GATE 1 PASSED (both levels)
```

Harness unit tests:

```bat
.venv\Scripts\python.exe scripts\run_tb.py
```

Expect `ALL PASSED (uart, benchmark, loader, top)`, including a board-level integration test
that drives real golden vectors through a real UART model: `correct : 64 / 64`.

### 5.3 Area and timing (no board needed)

```bat
.venv\Scripts\python.exe scripts\run_synth.py --impl
```

Expect, out-of-context on `xc7a35tcpg236-1`:

```
dwn_core               108   0.52%    73 FF   +3.708 ns   158.9 MHz
thermometer_encoder   1519   7.30%     0 FF   +7.084 ns   342.9 MHz
dwn_top               1619   7.78%   269 FF   +3.200 ns   147.1 MHz
```

Placement varies run to run, so slack may differ by a few tens of picoseconds. LUT and FF counts
should match exactly.

### 5.4 The full test set (needs a Kaggle GPU session, once)

Gate 1b needs all 166,000 samples; only 1,000 are committed. Upstream `torch_dwn` has **no CPU
path**, so even inference needs a GPU.

1. Kaggle → Datasets → New Dataset, upload both files from `training/artifacts/`:
   `..._checkpoint.pt` and `..._testvectors.npz`
2. Import `training/dump_testset_kaggle.ipynb`, **Add Input →** that dataset
3. Accelerator → **GPU**, Internet → **On**, then **Run All**
4. Download `..._testset_full.npz` into `training/artifacts/`

It runs **inference only** — no retraining, so the model stays bit-identical to the one every
number here was measured against. It self-checks before saving: recomputed accuracy must equal
the recorded `final_acc`, and the first 1000 predictions must match the committed vectors
sample-for-sample. Expect `166000 samples, software accuracy 73.8361%`.

### 5.5 Build and program

```bat
.venv\Scripts\python.exe scripts\build_bitstream.py
.venv\Scripts\python.exe scripts\program.py
```

Expect `2058 LUTs (9.89%)`, `865 FF`, `8 BRAM`, `WNS +1.7xx ns -> MEETS timing`, then
`PROGRAM_DONE`. Close Vivado's Hardware Manager first — it holds the JTAG interface.

Programming is volatile: it is lost on power cycle.

To flash the exact frozen artifact instead:

```bat
.venv\Scripts\python.exe scripts\program.py --bit releases\phase1\dwn_basys3_top.bit
```

### 5.6 Gate 1b

```bat
.venv\Scripts\python.exe scripts\host.py --ping
.venv\Scripts\python.exe scripts\host.py --gate1b --limit 2048
.venv\Scripts\python.exe scripts\host.py --gate1b
```

The COM port is auto-detected by USB ID and confirmed with a ping, so no port number is
hardcoded — the same board is COM3 on one laptop and COM8 on another.

Expect:

```
samples              : 166000
hardware == software : 166000/166000
core cycles          : 166815
wall clock           : ~11 s
accuracy, float32 model      : 73.8361%
accuracy, Q3.12 (on hardware): 73.8349%
RESULT: PASS -- hardware reproduces the golden model to the sample
```

`166815 = 166000 + 162 × (LATENCY+1)`, which is how you know II=1 held on silicon.

> **On a machine that has never run this**, set the FTDI latency timer to 1 ms or the run is
> ~21% slower with no other symptom. `host.py` detects and warns; fix it from an Administrator
> shell with `--port COM3 --set-latency 1`, then replug the board.

### 5.7 Using the board

`sw[1:0]` selects the seven-segment: `00` last class, `01` cycle count, `10` correct count,
`11` cycle count high word. Classes are alphabetical — **`0=g, 1=q, 2=t, 3=w, 4=z`** — *not* the
physics ordering, so index 2 is top, not W.

`led[0]` busy · `led[1]` a run finished · **`led[2]` UART framing error (sticky)** · `led[3:4]`
echo the switches · `led[15:8]` low byte of the correct count (wraps — use `sw=10` for the real
value).

### 5.8 Optional: re-train from scratch

Only if you want a *different* model. `training/dwn_jsc_kaggle.ipynb` trains one; expect ~74% in
about 20 minutes on a T4. **It will not reproduce the committed checkpoint bit-for-bit** — CUDA
kernels accumulate non-deterministically — so every measurement above would need re-taking.
`scripts/run_gate1.py` regenerates the RTL from whatever checkpoint you point it at.

---

## 6. What Phase 1 deliberately did not do

- **One config only.** `sm` (1×50) was chosen as the smallest safe bring-up target so a pipeline
  bug could not be confused with a routing limit. The frontier is Phase 2.
- **No encoder optimization.** Measured at −17.1%, not adopted; it matters at `md`/`lg` sizes,
  not here.
- **4 pipeline stages, not 3.** 3 closes 100 MHz post-synthesis but was not re-verified
  post-route, and Gate 1 is verified against 4.
- **`rtlgen/` is empty.** The generator exists as `exporter/emit_*.py` and targets one model
  shape; Phase 2 needs it generalized over a sweep grid.
- **5 Mbaud, not 12.** The VCP driver wall; FTDI's D2XX API might get past it, logged as
  optional in the ledger.
