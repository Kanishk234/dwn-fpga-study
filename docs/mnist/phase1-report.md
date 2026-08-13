# MNIST Phase 1 — porting the flow to a second dataset

A Differentiable Weightless Neural Network trained on MNIST, exported by the same generator that
produced the JSC design, running on the same Digilent Basys 3, and verified bit-exact against the
software model on all 10,000 test samples.

This is the written-up account. `docs/mnist/phase1-ledger.md` is the running log it was written
from — dated, with the retractions. The JSC study it generalises is `docs/phase1-report.md`.

**The deliverable of this phase is the generalised flow, not the accuracy number.** A port that
did not fit the board would still have been a success if the generator came out general. It fits,
which is a bonus rather than the point.

---

## 1. Headline results

| | |
|---|---|
| **Gate 1** (simulation, bit-exact) | core **1504/1504**, full datapath **1865/1865** |
| **Gate 1b** (hardware, full test set) | **10,000 / 10,000** |
| Accuracy | **96.14%** — float32 and Q0.8 identical, to four decimals |
| **DWN core area** | **631 LUTs** (300 nodes) |
| Thermometer encoder | **918 LUTs** — 1.5× the core |
| Model total (`dwn_top`) | **1,548 LUTs (7.44%)**, 906 FF, 0 BRAM, **0 DSP** |
| Whole board design | **4,586 LUTs (22.05%)**, 10,746 FF, 0 BRAM, 0 DSP |
| Latency / throughput | **4 cycles, II=1** → 76.2 M classifications/s, measured on-chip |
| Fmax | 108.0 MHz for the model; 103.0 MHz for the whole board design |
| Full test set over UART | **31.9 s** at 5 Mbaud, 625 batches |

**Q0.8 is exactly lossless.** 56,835 feature values saturate at the word boundary and not one
flips an encoder bit, so the fixed-point and float32 models agree on all 10,000 samples. On JSC
they differ on 30 of 166,000. The reason is structural rather than lucky: MNIST pixels are
natively 8-bit, so there are only 256 distinct input values and nine bits cannot lose anything.
JSC's features are continuous, and quantising them genuinely discards information.

### Against the published MNIST rows

| design | accuracy | LUTs |
|---|---|---|
| **this work, `1x300`** | **96.14%** | **1,548** |
| PolyLUT | 96% | 70,673 |
| NeuraLUT | 96% | 54,798 |

⚠️ **Read this comparison with care.** The area convention is ours (encoder included, which is the
stricter choice — see `docs/jsc-report.md` §5.2), but the published rows were measured on different parts
at different clock targets, and this is a bring-up configuration rather than a tuned one. It is
evidence that the approach is in the right range, not a claim to have beaten either.

## 2. What actually got built

Nothing in this list is new. Every path already existed for JSC; the work was removing the places
where it had quietly assumed JSC's shape.

```
datasets/          per-dataset descriptors -- THE change that makes the flow general
exporter/          checkpoint -> LUT tables, wiring, thresholds  (no dataset constants left)
rtlgen/            emit_core, emit_encoder, config
rtl/               hand-written: lut_node, popcount, argmax, pipe_reg
harness/           uart_rx/tx, uart_loader, vector_store, benchmark_fsm, seg7, board top
tb/                Gate 1 testbenches + the vector generator
scripts/           run_gate1, run_synth, build_bitstream, program, host, dump_testset
docs/mnist/        this phase: plan, ledger, report, results
```

The model is **300 LUT6 nodes** in one layer, reading a 3-threshold-per-pixel thermometer
encoding of 784 pixels. `z=3` comes from upstream's own MNIST example, not from us.

## 3. The four things that actually cost time

### 3.1 One bug, seven times: a dataset constant where a derived value belongs

Every real failure in this phase was the same defect wearing different clothes.

| # | site | wrong for MNIST because |
|---|---|---|
| 1 | `emit_core.py` `16 * thermometer_bits` | 784 features, not 16 |
| 2 | `record_bytes()` `word_bits // 8` | 9 bits is 2 bytes; floor gives 1 |
| 3 | `uart_loader.v` `reg [5:0]` | caps at 63; the record is 1,568 bytes |
| 4 | testbenches `[2:0]` class index | 10 classes need 4 bits |
| 5 | `dwn_basys3_top.v` `correct_count[7:0]` | part-select past the end at `ADDR_W=4` |
| 6 | `extract.py` `FRAC_BITS`/`WORD_BITS` | Q3.12 imported as a default by six modules |
| 7 | `host.py` `WORD_BITS // 8`, `'33 bytes'` | the same two defects again, in the host |

They surfaced one crash at a time because **`datasets/` existed but nothing imported it.** It was
written in the first step of this phase as the single place dataset facts live, and then every
consumer kept a private copy of JSC's numbers anyway. A descriptor nothing reads is documentation,
not a boundary.

The fix that ended the sequence was `datasets.identify(ck)` — resolve the dataset from the
checkpoint's own shape — plus **deleting `extract.py`'s module constants entirely** and making the
widths required arguments. That last part is what makes it durable: a missing argument is a
`TypeError` at the call site, whereas a default is a plausible wrong number that reaches the FPGA.

Matching is on `(features, classes)`, deliberately **not** on the filename. A slug is a naming
convention; resolving behaviour through one means a renamed checkpoint silently changes its
quantisation.

### 3.2 The argmax was a linear chain, and it held the design to 87.5 MHz

`argmax.v` reduced classes sequentially, each iteration reading the previous one's best. That is a
chain of K−1 dependent compare-and-selects: 4 deep at JSC's five classes and invisible, 9 deep at
MNIST's ten, where it synthesised to 17 logic levels and missed the board's 100 MHz.

A balanced tree makes the depth `ceil(log2 K)`. The tie-break — lowest index wins, matching numpy
and torch — is preserved by a strict `>` at every merge, so equal scores always keep the lower
index at every level and therefore overall.

**The interesting part was the wrong first fix.** It shipped as `K <= 5 ? chain : tree`, to hold
JSC's published 108-LUT core, since the tree costs +2 there. That branch encoded a fact about this
project's git history, not about the target — nothing in the FPGA changes at five classes — and
branches that encode history do not compose, because the next dataset brings a K with no
principled place to go. It was made unconditional and the published figures pinned to the
`jsc-complete` tag instead, which is what tags are for.

### 3.3 The harness cost four times the network, and the cause was one line

The first MNIST board build came out at 10,460 LUTs (50.3% of the device), against a model of
1,543. `uart_loader.v` held the record with a variable indexed write:

```verilog
wr_data[byte_idx*8 +: 8] <= rx_data;
```

which synthesises to a `VEC_BYTES`-way address decoder driving a write enable on every byte lane.
At JSC's 32 lanes that is 439 LUTs and nobody notices. At MNIST's 1,568 it was **6,343 LUTs and
5,228 FF**. Bytes arrive in order and are never revisited, so the decoder buys nothing:

```verilog
wr_data <= {rx_data, wr_data[DATA_W-1:8]};
```

| | before | after |
|---|---|---|
| MNIST board design | 10,460 LUTs (50.29%) | **4,586 (22.05%)** |
| MNIST `u_loader` | 6,343 LUTs, 5,228 FF | **463 LUTs, 6,404 FF, 274 SRL** |
| JSC board design | 2,060 LUTs | **1,893 LUTs** |
| JSC `u_loader` | 439 LUTs | **77 LUTs** |

**The prediction going in was wrong and the mechanism is worth keeping.** The change was expected
to move cost from LUTs to flops — a 12,544-bit shift register still needs 12,544 flops — making it
a modest win. LUTs fell 13.5× and flops rose 1.2×, because a shift register can map into
**SRL16/SRL32 primitives** (one LUT holding 16 or 32 bits of shift depth) and an indexed write
forbids that mapping, since any element may be written at any time. The access pattern decides
which primitives are reachable.

This also retired a planned redesign. A streaming harness had been scoped to make MNIST fit; after
the one-line fix, fit was no longer the binding constraint and streaming became a convenience
change. It would still remove the 2,416-LUT distributed-RAM store — a 12,544-bit-wide store cannot
use block RAM, since a BRAM36 is at most 72 bits wide and one cycle of that width would need ~175
of the 50 on the device — but it would also give up the `cycle_count` measurement, because a
UART-bound harness measures the UART.

### 3.4 A 3-LUT discrepancy that a tolerance check would have hidden

`verify_phase1.py` came back 20/22 after the loader change, reporting 1,893 board LUTs against an
expected 1,896. Two builds of the same design then proved byte-identical, so nothing was
stochastic. The expected value had been filled in from `utilization.rpt` (post-**synthesis**)
while `build_bitstream.py` parses `utilization_routed.rpt` (post-**route**) — the source every
historical figure came from.

The wrong number was within 0.2% of the right one. A tolerance-based check would have passed it
silently and the two report stages would now be mixed through every table built from that file.
`verify_phase1.py` demands exact equality precisely so a 3-LUT discrepancy is loud.

## 4. Findings worth carrying into Phase 2

- **Word width, not `z`, decides whether MNIST fits.** MNIST is slot-limited rather than
  pool-limited: 784 × z far exceeds the input slots of any layer that fits, so `z=8` → `z=200` is
  only 2.3× the comparators. At 16 bits the paper's configuration is over the device at every `z`;
  at 11 bits it fits comfortably. This inverts the JSC conclusion, where `z` dominated.
- **The harness is not a fixed overhead.** "≈439 LUTs, independent of the model" held only while
  JSC was the only dataset. It is 3,038 LUTs here and scales with record width. Any area model
  that treats it as a constant is wrong for the second dataset onward.
- **The I/O wall is 243,271×**, against JSC's ~5,100×. The core classifies one vector per clock
  regardless of dataset; the link carries 1,569 bytes per record against 33. Any throughput claim
  must come from `cycle_count`, never from wall-clock.
- **Timing, not area, is the binding constraint.** The model uses 7.44% of the device and closes
  100 MHz with 0.742 ns to spare, but the whole board design has only 0.292 ns — 2.9% margin. The
  ladder goes to 2,000 nodes, and timing will bite before area does.
- **A prediction from two points is not a finding.** An "amortisation" explanation for encoder
  cost-per-comparator was derived from two measurements and falsified by the third. So was the
  claim that MNIST would tolerate more encoder narrowing than JSC; it tolerated less.

## 5. Reproducing this

Assumes the JSC Phase 1 toolchain setup (`docs/phase1-report.md` §5.1) — same Vivado, same venv,
plus `scikit-learn` and `pandas` for the test-set dump.

### 5.1 Gate 1 — prove the RTL in simulation (no board needed)

```
.venv\Scripts\python.exe scripts\run_gate1.py ^
  --checkpoint training\artifacts\mnist_n6_z3_distributive_w300_checkpoint.pt ^
  --rtl-dir build\mnist\rtl --work build\mnist\gate1
```

Expect `GATE 1 PASSED (both levels)`, 1504 core vectors and 1865 top vectors. **No precision
flags** — Q0.8 comes from the descriptor. Passing `--word-bits 9 --frac-bits 8` explicitly
produces byte-identical RTL, which is how that equivalence was verified.

### 5.2 Area and timing (no board needed)

```
.venv\Scripts\python.exe scripts\run_synth.py --impl --rtl-dir build\mnist\rtl
```

Expect `dwn_core` 631, `thermometer_encoder` 918, `dwn_top` 1,548 LUTs, WNS +0.742 ns.

### 5.3 The full test set — locally, no Kaggle session

```
.venv\Scripts\python.exe scripts\dump_testset.py ^
  training\artifacts\mnist_n6_z3_distributive_w300_checkpoint.pt
```

Writes `..._testset_full.npz` (10,000 samples, 3.0 MB). Unlike JSC's, this needs no GPU: the
float32 and fixed-point golden models differ only in the thermometer comparison, and everything
after it is exact integer work. It **validates before writing** — regenerates the rows already in
the committed `_testvectors.npz` and requires an exact match on features and predictions, because
a test set with the wrong split or scaling produces a Gate 1b run that looks entirely normal and
means nothing.

### 5.4 Build, program, Gate 1b

```
.venv\Scripts\python.exe scripts\build_bitstream.py ^
  --checkpoint training\artifacts\mnist_n6_z3_distributive_w300_checkpoint.pt ^
  --rtl-dir build\mnist\rtl --outdir build\mnist\board
.venv\Scripts\python.exe scripts\program.py --bit build\mnist\board\basys3\dwn_basys3_top.bit
.venv\Scripts\python.exe scripts\host.py --gate1b --depth 16 ^
  --checkpoint training\artifacts\mnist_n6_z3_distributive_w300_checkpoint.pt
```

Expect `4586 LUTs (22.05%)`, `WNS +0.292 ns`, then `10000/10000` in about 32 s at 5 Mbaud.

⚠️ **`--depth 16` is required.** `host.py` defaults to JSC's 1024, and the MNIST bitstream's
vector store holds 16 records — a 12,544-bit-wide store cannot use block RAM at all. Loading more
than the store holds wraps silently and every result after the wrap is wrong.

### 5.5 Confirm JSC still reproduces

```
.venv\Scripts\python.exe scripts\verify_phase1.py --with-board
```

**22/22, exactly.** This is the gate every generalisation commit had to pass: areas 110 / 1,519 /
1,621, board 1,893 / 864, Gate 1b 166,000/166,000. Identical, not close.

## 6. What Phase 1 deliberately did not do

- **No accuracy work.** `1x300` at `z=3` is a bring-up configuration chosen to be safe, not good.
  The ladder, the `z` sweep and the multilayer configurations belong to Phase 2.
- **No baud sweep.** 5 Mbaud works and a full pass takes 31.9 s. JSC swept baud because 480 s →
  11.2 s made the test set practical to run at all; here the I/O wall is already 243,271×, so a
  faster link changes no reported number.
- **No streaming harness.** Scoped, then retired by §3.3 — it is a convenience change now, and it
  would cost the throughput measurement.
- **The `tau` correction is not on silicon.** A retrained `..._tau1p678` checkpoint scores 96.77%,
  but it is not on this machine and has never been exported, synthesized or run. **Everything in
  this report is the 96.14% model**, which is the one in the bitstream. The two must not be merged
  into a single claim without redoing the chain.
