# Phase 3 ledger — the Controlled Comparison (Study 2)

Running log for Phase 3. Plan: `docs/phase3-plan.md`. Handoff: `docs/phase3-handoff.md`.

**Split:** the *hands-on half* (conifer, hls4ml — plan §2) runs on the machine with the Phase 1/2
toolchain; the *literature half* (plan §3–§4) needs no board, no Vivado and no synthesis and runs
on the other. **Both halves are now logged here** — see the note under the status table for which
machine produced which rows. Where an entry corrects an earlier one written from the other side,
it says so rather than editing it away.

---

## Status

| Step | What | Status |
|---|---|---|
| 3L-a | Refresh the literature list — brief §8 is stale (plan §3 ⚠️) | ✅ done 2026-08-10 |
| 3L-b | Settle the encoder-convention trap (plan §4.1) | ✅ done 2026-08-10 |
| 3L-c | Pull per-paper JSC numbers into a machine-readable table | ✅ done 2026-08-10 — `cc/literature/` |
| 3L-d | Combined comparison table + Pareto plot with our 15 frontier points | ✅ done 2026-08-11 — `cc/literature/plot.py`, figures in `docs/results-cc/` |
| 3X-a | Encoder input-word width: accuracy floor + area curve | ✅ done 2026-08-10 — **5.9x smaller encoder** |
| 3X-b | Re-run 3X-a at `1x2400 z=50` before quoting anything | ✅ done 2026-08-10 — **the width limit moved; the saving held** |
| 3L-e | Phase 3 report | ✅ done 2026-08-11 — `docs/phase3-report.md`, covers both halves |
| 3M-a | conifer (GBDT) — *hands-on machine* | ✅ done 2026-08-10 — **14 configs, `docs/results-cc/`** |
| 3M-b | hls4ml (quantized MLP) — *hands-on machine* | ✅ done 2026-08-10 — **6 configs, it fits at 16/8/8** |

**Both halves now have entries in this ledger.** The literature half (3L-\*) runs on the machine
that wrote this file; the hands-on half (3M-\*) runs on the machine that holds the Phase 1/2
toolchain — Vivado/Vitis 2025.2, and the venv `scripts/verify_phase1.py` validates. Every
hands-on row below was produced there, through `scripts/build.tcl` at `xc7a35tcpg236-1` / 10 ns,
so it is directly comparable to the 54 DWN results.

---

## Log

### 2026-08-10 — ✅ 3M-b RUN AFTER ALL: hls4ml fits an XC7A35T at 16/8/8, and the DSP claim is now measured

*(hands-on machine)* `cc/hls4ml/run_hls4ml.py`. Supersedes the "scoped out" entry below, which
stays visible because its *reasoning* about cost was right — it was the toolchain assessment that
was wrong.

**None of the predicted blockers were real.** hls4ml 1.3.0 needs **no WSL, no second Vivado
install, no `vitis_hls` shim and no TensorFlow**: its Vitis backend already issues
`vitis-run --tcl build_prj.tcl --mode hls`, guards its POSIX tool-detection with
`if 'linux' in sys.platform`, passes build options through a *file* rather than trailing
arguments, and `convert_from_pytorch_model` uses the torch already pinned. Install cost was five
small packages; numpy 2.3.4 and torch 2.13.0 were untouched.

#### The shrink sequence — the result plan §2.2 was actually after

Published architecture is **16→64→32→32→5** (Duarte/Fahim). Trained in PyTorch on the same split,
same seed, same scaler as everything else. All rows post-route on `xc7a35tcpg236-1` at 10 ns.

| config | reuse | precision | acc (float) | LUTs needed | vs 20,800 | DSPs |
|---|---|---|---|---|---|---|
| `64/32/32` | 1 | `<16,6>` | 76.69% | **259,492** | 12.5× over | 3,214 |
| `64/32/32` | 4 | `<16,6>` | 76.69% | 189,608 | 9.1× over | 1,064 |
| `32/16/16` | 4 | `<16,6>` | 76.33% | 52,927 | 2.5× over | 340 |
| `64/32/32` | 16 | `<16,6>` | 76.69% | 45,844 | 2.2× over | 266 |
| `32/16/16` | 4 | `<12,6>` | 76.33% | 26,883 | 1.3× over | 340 |
| **`16/8/8`** | 4 | `<12,6>` | **75.67%** | **8,749** | ✅ **42.1% — FITS** | **53** |

**It takes quarter-width layers plus 12-bit precision plus 4× time-multiplexing to fit.** The
published architecture does not fit at *any* reuse tested — still 2.2× over at reuse=16, at full
accuracy.

**Reuse is sub-linear, and it beats width.** 1→4 bought only 1.4×; 4→16 bought 4.1×. And
`64/32/32` at reuse 16 (45,844) is *smaller* than `32/16/16` at reuse 4 (52,927) while scoring
0.36 pp better — so time-multiplexing dominates narrowing as an area lever, and it costs latency
rather than accuracy.

#### Against DWN at matched area — DWN wins every column

| | LUTs | DSP | BRAM | latency | accuracy |
|---|---|---|---|---|---|
| hls4ml `16/8/8` | 8,749 | **53** | 2 | **34 cyc, II=4** | 75.67% *(float upper bound)* |
| **DWN `1x1200 z=50`** | **8,444** | **0** | **0** | **4 cyc, II=1** | **76.05%** |

Smaller, more accurate, no DSPs, no BRAM, **8.5× lower latency and 4× the throughput**. And DWN's
headline `1x2400 z=50` reaches 76.18% at 12,751 LUTs — an accuracy hls4ml never reaches at any
size that fits here.

**The DSP argument is no longer unexercised.** 53 DSPs is **59% of this part's 90**, against 0 for
all 52 DWN and all 14 conifer configs.

#### ⚠️ Two silent failures caught, and one nearly shipped

**1. Weight ROMs read as zero — a 235-LUT "result".** At `reuse=1` hls4ml bakes weights in as
inline constants; at `reuse>1` it switches to the Resource strategy and loads them from ROMs via
`$readmemh("./x.dat", rom0)` — a **relative** path. Staging the Verilog into another directory
while Vivado ran from the repo root left every ROM unresolved, so the weights read as zero, the
synthesizer folded the dead arithmetic away, and a 64/32/32 MLP reported **235 LUTs, 0 DSP,
`status: ok`**. A second row (`reuse=16`, 264 LUTs) recorded the same way. Both deleted and
re-measured.

Staging now uses `impl/verilog`, copies the `.dat` files alongside, and rewrites `$readmemh` to
absolute paths. `verify_staged()` refuses to synthesize RTL whose instantiated modules are not all
defined or whose ROM targets do not exist — either check would have caught this immediately.
**conifer was verified unaffected** (inline constants, no `$readmemh` anywhere).

**2. Latency read 0, and the unit was why.** Latency-strategy reports are in **ns**;
Resource-strategy reports switch to **us**, and their Pipeline Type is `dataflow`, not `yes`. A
regex requiring `' ns|'` matched nothing, so every `reuse>1` row recorded **0 cycles** — hiding
the II=4 finding entirely. The old regex also failed to stop at `+ Detail:`, so the `reuse=1` row
reported **4 cycles when the true figure is 30**. Both fixed; **all 14 conifer rows re-verified
against the corrected parser and unchanged** (2–9 cycles, II=1).

#### Still not measured

**The accuracy column is the float model's, an upper bound.** The real `ap_fixed<12,6>` number
needs `hls_model.predict()`, which compiles the generated C++, and there is no C++ compiler on
this machine — the same wall conifer's `compile()` hit. conifer escaped it because its model is a
declarative ensemble JSON that could be evaluated in numpy; hls4ml's is a C++ dataflow program
with no equivalent artifact. **So the true hls4ml accuracy is lower than 75.67%, and every
comparison above already favours it.**

### 2026-08-10 — ~~⚠️ 3M-b: hls4ml is SCOPED OUT~~ SUPERSEDED, see above

Not "incomplete" — a decision, recorded so the writeup states it rather than omitting the row.

**What running it would have added, and it is one thing only:** what accuracy hls4ml retains when
*shrunk to fit* an XC7A35T. Nobody has published that, and it is the actual content of plan §2.2.

**What is already established without running it**, from the primary-source row verified in the
literature half — same OpenML split as ours:

| | hls4ml JSC MLP | ours (`1x2400 z=50`) |
|---|---|---|
| accuracy | 76.0% | **76.18%** |
| LUTs | **63,251** | **12,751** |
| DSPs | **38** | **0** |
| part | `xcvu9p` | `xc7a35t-1` |

63,251 against a 20,800-LUT device is **3× over** — arithmetic, not something that needs
measuring. The DSP contrast is two published numbers. Plan §4.2 already sanctions the cross-part
LUT comparison ("LUT counts roughly transfer; Fmax and ns do not").

**Wording for the report**, so the boundary is explicit:

> *hls4ml was not re-synthesized on our part. Its published JSC design (76.0%, 63,251 LUTs, 38
> DSPs on xcvu9p) exceeds the XC7A35T by 3× on LUTs alone, so the comparison is made from
> published numbers with the part difference stated. What we do not measure is what accuracy
> hls4ml retains when shrunk to fit this device.*

**The cheap version, if time reappears:** one config, not a sweep. Because the control is *our*
synthesis flow, hls4ml's version and OS never enter the comparison — generate Verilog anywhere
(WSL, another machine, an older Vitis that still ships `vitis_hls`) and synthesize it here at
2025.2. That is an afternoon; the shrink sequence is the part that costs a day.

⚠️ **One thing this weakens, and it should be said plainly.** The 0 BRAM / 0 DSP column is the
central claim against hls4ml, and conifer also measures 0/0 across all 14 configs — trees do not
spend DSPs either. So on our own silicon the DSP argument is currently **unexercised**: it rests
entirely on the published hls4ml numbers.

### 2026-08-10 — 3M-a COMPLETE: 14 conifer configs, and a GBDT does not reach DWN on this part

*(hands-on machine)* Full sweep through `cc/conifer/run_conifer.py --sweep`. Snapshot committed
to `docs/results-cc/` — `build/` is gitignored and these are most of a day of HLS + place-and-route.

**10 of 14 fit; 4 exceed the device.** All post-route, `xc7a35tcpg236-1`, 10 ns, out-of-context.

| config | trees | acc % | LUT | dev % | cyc | II | Fmax |
|---|---|---|---|---|---|---|---|
| `gbdt_d4_n5` | 25 | 73.63 | 4,019 | 19.3% | 2 | 1 | 477.3 |
| `gbdt_d3_n10` | 50 | 73.64 | 3,774 | 18.1% | 3 | 1 | 477.3 |
| `gbdt_d4_n10` | 50 | 74.19 | 8,005 | 38.5% | 3 | 1 | 477.3 |
| `gbdt_d3_n20` | 100 | 74.24 | 7,602 | 36.6% | 5 | 1 | 477.3 |
| `gbdt_d5_n5` | 25 | 74.36 | 7,836 | 37.7% | 5 | 1 | 477.3 |
| `gbdt_d6_n3` | 15 | 74.50 | 9,376 | 45.1% | 8 | 1 | 477.3 |
| `gbdt_d4_n20` | 100 | 74.75 | 16,052 | 77.2% | 5 | 1 | 477.3 |
| `gbdt_d5_n10` | 50 | 74.77 | 15,605 | 75.0% | 6 | 1 | 477.3 |
| `gbdt_d6_n5` | 25 | 74.80 | 15,898 | 76.4% | 8 | 1 | 477.3 |
| **`gbdt_d3_n40`** | 200 | **74.88** | 15,363 | 73.9% | 7 | 1 | 477.3 |
| `gbdt_d6_n10` ❌ | 50 | 75.09 | — | over | 9 | 1 | — |
| `gbdt_d5_n20` ❌ | 100 | 75.26 | — | over | 9 | 1 | — |
| `gbdt_d4_n40` ❌ | 200 | 75.41 | — | over | 7 | 1 | — |
| `gbdt_d3_n80` ❌ | 400 | 75.50 | — | over | 8 | 1 | — |

**Every config: 0 BRAM, 0 DSP**, matching all 52 DWN configs.

#### The result: a GBDT does not reach DWN's accuracy on this part

- Best that **fits**: **74.88%** at 73.9% of device
- Best over-device: 75.50% — still **0.68 pp short** of DWN, while already exceeding a part DWN
  fits inside at **61.3%**
- **`1x2400 z=50`: 76.18% at 61.3%, 4 cycles** — more accurate, smaller, *and* lower latency

**The ceiling is sharply defined.** Four configs at 15.4–16.1k LUTs, spanning depths 3–6 and
25–200 trees, all land in **74.75–74.88%** — a 0.13 pp spread. Four structurally different models,
same accuracy, same area. That is an architectural wall, not undersampling.

**A clean iso-area pair worth quoting:** `gbdt_d4_n10` at **8,005** LUTs against `1x360` at
**8,006** — one LUT apart, same part, same flow. 74.19% vs 75.85%, **+1.67 pp to DWN**.

#### Latency: cycles is the only meaningful metric here, exactly as brief §6 says

Every conifer row reports **Fmax 477.3 MHz**, identical. That is not a bug and not a fast design:
HLS *chooses* pipeline depth to meet the 10 ns target, so it always lands near the same critical
path and Fmax carries no information. Latency in **cycles** does vary — **2 to 9, always II=1** —
and it comes from HLS's `<top>_csynth.rpt`, not from any Vivado report, which is why it was
missed on the first pass.

Against DWN's 4 cycles / II=1: conifer's *small* configs are faster (2–3 cycles) but 1.5–2 pp less
accurate; every config that gets near its accuracy ceiling costs **5–9 cycles**. So DWN wins the
latency comparison in the region that matters.

#### ⚠️ This measures conifer, not trees

**TreeLUT reports 76.0% at 2,234 LUTs** on the same OpenML split — roughly **7× denser** than
conifer's best fitting config, at 1.1 pp *better* accuracy. So the defensible claim is
**"DWN beats conifer's HLS implementation of GBDTs on this part"**, not "DWN beats GBDTs". The
tree-architecture comparison lives in the literature table, where TreeLUT already sits.

Also worth a column rather than a footnote: **FF count varies 3.9k–11.8k at essentially fixed
LUT area** (15.4–16.1k). Depth drives registers hard; area alone hides it.

### 2026-08-10 — 3M-a: the conifer flow runs end to end at 2025.2, and the first GBDT row

*(hands-on machine — Vivado/Vitis 2025.2, the venv `verify_phase1.py` validates)*

`cc/conifer/run_conifer.py`. sklearn/xgboost → conifer → HLS → Verilog → `scripts/build.tcl`,
place-and-routed at `xc7a35tcpg236-1` / 10 ns — the same flow all 54 DWN rows came from.

| config | acc | LUT | FF | BRAM | DSP | device | WNS | Fmax |
|---|---|---|---|---|---|---|---|---|
| `gbdt_d4_n10` (depth 4, 10 rounds = 50 trees) | **74.19%** | **8,005** | 1,418 | 0 | **0** | 38.5% | +7.905 | 477.3 MHz |

One point, not a curve — `--sweep` (depth 3-6 × 10-80 rounds) is what the plan actually asks for.
For orientation against the headline DWN config (76.18%, 12,751 LUTs, 61.3%): cheaper, and 2 pp
behind. **conifer also uses 0 DSP**, so the DSP argument is specifically against hls4ml's MLPs,
not against tree methods.

#### ⚠️ Corrections to the toolchain entry below, from running it

That entry was written before this machine had tried the flow. Two of its three recommendations
do not survive contact:

1. **"conifer's direct-to-RTL backend needs no HLS at all … the better controlled comparison"** —
   **it cannot run on Windows.** `FixedPointConverter` compiles a pybind11 helper with a
   hard-coded POSIX command: `g++ -O3 -shared -fPIC $(python3 -m pybind11 --includes) … -o X.so`.
   `os.system` runs that through `cmd.exe`, where `$( )` never expands; `-fPIC`/`.so` are not MSVC
   concepts; and no compiler is installed. It also dies earlier still, on
   `np.random.randint(0, 2**32)` — fine where the default int is 64-bit, `ValueError: high is out
   of bounds for int32` on Windows. **So the VHDL/`read_vhdl` question is moot**, and the Verilog
   convention holds for every row.
2. **"HLS Classic was removed in 2025.1 … hls4ml and conifer shell out to `vitis_hls`"** — the
   removal is real, but **HLS is present and works**: a trivial design synthesized to Verilog on
   `xc7a35tcpg236-1` at 370 MHz estimated. The entry point is `vitis-run --mode hls --tcl <script>`
   (`--mode hls` alone errors; the positional form the `--help` advertises does not satisfy it).
   There *is* a `vitis_hls.exe` under `Vitis/bin/unwrapped/win64.o/`, but it prints nothing for
   `-h`/`-version` and is not usable as a CLI. **No second Vivado install is needed.**

What is true is that conifer's own `build()` cannot work here — it detects the tool with
`os.system('type X > /dev/null')`, a POSIX builtin, then invokes `vitis_hls`. That does not
matter, because `docs/phase3-handoff.md` §2.1 forbids the vendor's default project flow anyway.
**Driving HLS ourselves is the method, not a workaround.**

#### Four more things that had to be fixed, all silent

- **`vitis-run` accepts no trailing arguments** (*"option '--input_file' cannot be specified more
  than once"*), but conifer's `build_hls.tcl` reads its flags from `$argv`. Injected via a
  generated wrapper Tcl that sets `argv`/`argc` and then sources conifer's script.
- **`csim=0` is mandatory.** conifer defaults to csim=1; its C++ testbench fails to link and the
  run dies ~20 s in, before synthesis starts.
- **"The command line is too long."** HLS emits one module per tree — 100 files and 14,341
  characters of `-tclargs` for the *smallest* sweep config, against cmd.exe's ~8,191 limit, and
  80 rounds would be ~8× worse. The sources are concatenated into one `.v` before synthesis
  rather than changing `build.tcl`, which every Phase 1/2 number depends on. Verified safe: no
  `include` directives, no duplicate module names.
- **`HistGradientBoostingClassifier` cannot be used**, though `docs/phase3-plan.md` §2.1 names it.
  conifer dispatches on `'GradientBoosting' in class name`, which it matches — so it is accepted
  and *then* dies on `n_estimators`, because it stores `_predictors`/`TreePredictor` rather than
  `estimators_`/`tree_`. Use xgboost (fast) or classic `GradientBoostingClassifier` (slow at 830k).

#### ⚠️ The one that would have poisoned every row: a NaN base score

conifer warns on import that *"prediction disagreements are observed for xgboost versions >=
2.0.0"*. It is right, and the mechanism is specific: **xgboost ≥ 2.0 auto-fits a per-class base
score, and conifer 1.9 cannot read it**, emitting

```
init_predict = [-4.965, NaN, -4.965, -5.742, -4.862]
```

One NaN makes that class's score NaN for every sample and sends the argmax arbitrary —
**127,034 of 166,000 predictions wrong**, while producing entirely plausible-looking HDL. Setting
`base_score=0.5` explicitly (xgboost's own pre-2.0 default) yields `init_predict` all zeros and a
clean conversion. Not caused by passing `objective`/`num_class`; reproduced with bare defaults.

Caught by an **independent numpy evaluator of conifer's own emitted ensemble JSON** — the same
golden-model pattern Gate 1 uses, and it needs no C++ compiler, unlike `conifer.model.compile()`.

**The gate was initially set wrong, and that is worth recording.** It first tested *prediction
identity* and refused the row at 0.61% mismatch. But those mismatches sit where the model is
nearly indifferent — median top-2 margin **0.082 against 1.773** for agreeing samples, 21× smaller
— and they move accuracy by **0.029 pp**, well under the 0.15 pp noise floor. The gate now tests
the accuracy delta, which is what actually goes in the table, and **the recorded accuracy is
conifer's (74.1861%), not xgboost's (74.2151%)** — because conifer's is the model the HDL
implements.

### 2026-08-10 — ⚠️ JSC is TWO DATASETS, and the standard comparison table conflates them

The largest comparability problem in this literature, and it invalidates part of what this ledger
said four hours ago. Everyone writes "JSC". There are two:

| | JSC-**OpenML** | JSC-**CERNBox** |
|---|---|---|
| source | `hls4ml_lhc_jets_hlf`, OpenML 42468 | CERNBox LHC Jets |
| instances | ~830,000 | 986,806 |
| features / classes | 16 / 5 | 16 / 5 |
| **who uses it** | **DWN, TreeLUT, hls4ml, — and us** | LogicNets, PolyLUT, PolyLUT-Add, NeuraLUT, AmigoLUT, ReducedLUT, SparseLUT |

NeuraLUT-Assemble (arXiv:2504.00592 §5) states it outright: *"Both datasets target the same jet
classification task, but the CERNBox version contains 986806 instances, while the OpenML version
has about 830000. … Experimentally, we observed that models trained on the OpenML dataset achieve
higher accuracy."* And: *"For the JSC dataset from CERNBox, we focused on comparisons with other
LUT-based approaches, as they are the only prior works utilizing this data source."*

**The offset is ~1.05 pp, and it is measured within-method, twice** — the only clean way to get it.
FPGN (arXiv:2607.08427 Table V) runs both datasets:

| method | OpenML | CERNBox | Δ |
|---|---|---|---|
| NeuraLUT-Assemble | 76.0% | 75.0% | 1.0 pp |
| FPGN | 76.0% | 74.9% | 1.1 pp |

**1.05 pp is seven times our 0.15 pp noise floor** and larger than nearly every accuracy difference
our Pareto frontier argues about. Mixing the two datasets in one table does not blur a comparison,
it reverses it.

**Which side are we on: OpenML.** Three independent confirmations —
`third_party/DWN/examples/DWN_Tutorial.ipynb:272` calls `openml.datasets.get_dataset(42468)`;
`training/dwn_jsc_kaggle.ipynb` calls `fetch_openml('hls4ml_lhc_jets_hlf')`; and our test set is
166k, which is 20% of ~830k. So **we and the DWN paper are on the same dataset**, and our core
claim against DWN is safe.

#### ⚠️ CORRECTION to this ledger's own table

The "Numbers worth quoting" table below was copied wholesale from Mecik & Kumm's Table II earlier
today and presented as a single comparable ranking. **It is not one ranking — it silently mixes
both datasets.** So does Mecik & Kumm's Table II, and so does the survey (arXiv:2506.07367), which
splits JSC only by accuracy band (JSC-H / JSC-L) and never mentions that two data sources exist.
The table has been replaced below with a dataset-split version. Nothing in it was mis-transcribed;
the error was presenting it as comparable.

**What this changes:** most of the "we are competitive with the LUT-DNN family" comparison was
against **CERNBox** rows — PolyLUT's 236,541, NeuraLUT's 92,357, LogicNets' 36,415. Those are all
on the harder dataset and are **not** ours to compare against directly. On our own dataset the
field is much smaller and much tougher: DWN, TreeLUT, NeuraLUT-Assemble, FPGN, hls4ml.

#### Consequences

- **3L-d produces two plots, never one.** Any figure mixing datasets is wrong.
- **`cc/literature/table.py` refuses to print rows from different datasets in one table.** This is
  the enforcement mechanism, not a convention to remember.
- **TreeLUT is on our dataset**, and it is the strongest tabular competitor: 76.0% at **2,234
  LUTs**, 0 BRAM, 0 DSP. Its dataset assignment is from TreeLUT §4 — *"the same JSC dataset as used
  by (Alsharari 2024; Fahim 2021; Summers 2020)"*, the hls4ml papers, deliberately **not** the
  PolyLUT lineage it cites for NID. This matters directly to the partner's conifer track: TreeLUT
  is the published GBDT-on-LUTs result that conifer will be measured against.
- **The best published LUT count at ≥76% on our dataset is NeuraLUT-Assemble at 1,780.** Our
  headline is 12,751. That gap is the honest framing, and it is much less flattering than the
  CERNBox rows made it look — see the caveat below before drawing conclusions from it.

⚠️ **Do not read that as "we are 7× worse."** Those are core-only-or-unstated designs on a
`xcvu9p-2` at 700 MHz; ours is encoder-included on a `xc7a35t-1` at 100 MHz, on a board that costs
~1.5% of theirs. The right comparison is against `DWN-PEN+FT lg-2400` at **7,011** LUTs, the only
row with the same convention and the same dataset. Resolving convention per row is what 3L-d needs
and what the `encoder_included` field exists for.

### 2026-08-10 — toolchain: which HLS command the comparison tools expect

`docs/phase3-plan.md` and `phase3-handoff.md` listed the toolchain as "Vivado 2025.2 **and Vitis
HLS**". Corrected, because the second half is not a thing you can simply install alongside 2025.2.

**Observed on this machine:** no `vitis_hls` or `vivado_hls` executable exists anywhere under
`C:\AMDDesignTools\2025.2` -- not in `Vivado/bin`, not in `Vitis/bin`. What exists is `v++`,
`vitis-run` and `vitis`. This matches AMD's note that **Vitis HLS Classic was removed in 2025.1
and later**, with HLS driven through `v++ --mode hls` instead. hls4ml and conifer's HLS backends
shell out to `vitis_hls`.

⚠️ **This is NOT recorded as a blocker.** Kanishk has run both hls4ml and conifer against this
Vivado version before, first-hand. Treat the missing binary as something to resolve at setup, not
as a known failure -- and if it does bite, the fixes in order of cost are:

1. **conifer's direct-to-RTL backend needs no HLS at all** -- Vivado only, which we have. It is
   also the better controlled comparison (RTL vs RTL, no HLS in the middle), and plan §2.1
   already runs conifer first.
2. **HLS only has to produce RTL.** Plan §2 synthesizes everything through *our* `build.tcl`, so
   the HLS tool's version and OS never enter the comparison -- generate RTL anywhere, synthesize
   at 2025.2 here. A second install (2024.1-2025.1, which still ships `vitis_hls`) is therefore
   harmless to the control.
3. WSL2 only if 1 and 2 both fail. Switching OS alone fixes nothing, since the constraint is the
   tool *version*, not the platform.

Stated support at time of writing: hls4ml up to **2025.1**; conifer's latest validated is
**2024.1**.

**Convention recorded in plan §2.4: generated HDL is Verilog.** Everything in `rtl/`, everything
`rtlgen/` emits and the Gate 1 testbench are Verilog, and `build.tcl` reads `.v`. Vitis HLS emits
Verilog by default -- keep it. The one unavoidable exception is conifer's direct-to-RTL backend,
which is VHDL by construction; if used, `build.tcl` needs a `read_vhdl` branch and the results
table must say which row it is.

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

### 2026-08-10 — RESOLVED: the 7.6x encoder gap is entirely input word width

`experiments/experiment_encoder_width.py` (accuracy, all 166k) and
`experiments/experiment_encoder_area.py` (area, out-of-context synthesis), both on `1x50`.

**Accuracy first.** The harness validates itself: a float encoder reproduces the software model
**exactly** (73.8361%), and Q3.12 costs 0.001 pp. Two schemes:

| scheme | narrowest word holding accuracy | vs today |
|---|---|---|
| **in-place** — same scaling, drop fractional bits | **10 bits** (-0.110 pp) | 6 narrower |
| **renorm** — per-feature affine into [-1,1) | **8 bits** (-0.099 pp) | 8 narrower |

Renorm needs **no retraining**: `x > t` is invariant under a monotonic affine map applied to both
sides, so it only changes the host's scaler constants. Mecik & Kumm got the same range by
*training* on [-1,1); we can get it as an export-time transform.

**Then area** — encoder only, `xc7a35tcpg236-1`, 10 ns:

| word | format | distinct constants | LUTs | per comparator | vs today |
|---|---|---|---|---|---|
| 16 | Q3.12 | 197/202 | **1,519** | 7.52 | 1.00x |
| 12 | Q3.8 | 196 | 1,121 | 5.55 | 1.36x |
| **10** | Q3.6 | 185 | **257** | 1.27 | **5.91x** |
| 9 | Q3.5 | 170 | 224 | 1.11 | 6.78x |
| 8 | Q3.4 | 149 | 182 | 0.90 | 8.35x |
| 6 | Q3.2 | 82 | 82 | 0.41 | 18.52x |

The `w=16` row reproduces the Phase 1/2 measurement of **1,519** exactly, which is the control.

**There is a cliff between 12 and 10 bits** — 1,121 to 257 LUTs, 4.4x for two bits. Above it a
comparator is a carry chain; below it the whole per-feature encoder collapses into LUT logic.
**Do not interpolate across it**: 12 bits buys almost nothing, 10 buys nearly everything.

**Mecik & Kumm's `sm-50` encoder is 201 LUTs at 8-bit. Ours at 8-bit is 182.** So the gap was
**entirely comparator width** — no FloPoCo trick, no sharing scheme we lacked. We were carrying
16-bit comparators where 8 sufficed. The 2026-08-10 open question is closed.

**The defensible result: 10 bits, in-place, 1,519 -> 257 LUTs (5.91x), -0.110 pp (inside the
0.15 pp noise floor), no retraining.** Both halves measured under the same scheme.

⚠️ **Do NOT pair "renorm holds accuracy at 8 bits" with "182 LUTs at 8 bits".** The area run used
in-place constants, where 53 of 202 comparators collapsed onto duplicates. Renorm keeps more
distinct constants and would cost more. Measuring it needs `emit_encoder` to accept per-feature
affine constants, which it does not.

**Incidental:** 5 wired thresholds are exact duplicates of another threshold on the same feature
even at full precision -- removable today, no downside.

#### What it does and does not change about Phase 2

**Measurements: nothing.** Every Phase 2 config is a real design at a recorded configuration --
`q16.12` is in every config name. `docs/results/` stays valid as evidence.

**The headline probably survives, and for a reason worth stating.** Every config that scored
above `1x2400 z=50`'s 76.18% failed on **timing**, not area:

| config | acc | LUTs | device | Fmax | failed on |
|---|---|---|---|---|---|
| `1x2000` | 76.43% | 21,382 | 102.8% | 94.9 | area **and** timing |
| `1x2400 z=100` | 76.39% | 16,681 | **80.2%** | 96.2 | **timing only** |
| `1x3000 z=50` | 76.16% | 13,972 | **67.2%** | 96.2 | **timing only** |

`1x2400 z=100` already had area to spare and still missed 100 MHz. **A smaller encoder gives it
more of what it did not need.** And the encoder is not the critical path -- on the headline
config it closed at 347 MHz against the core's 101.2. So Phase 2's central finding, *the wall is
timing and not area*, is not merely unaffected: making the encoder ~6x smaller does not move it.
That is the strongest confirmation of it we have.

*(Caveat: less area usually means less routing congestion, which can help timing indirectly.
`1x2000` at 102.8% was certainly congested. Unquantified -- do not claim it either way.)*

**What does change:**

1. **Every design gets much smaller.** Headline `1x2400 z=50` projects from 12,751 to **~7,990**
   LUTs -- **~38.4% of the device, from 61.3%** -- using the *measured* 11-bit encoder of 992 LUTs
   (3X-b, done). The claim "the paper's `lg` fits on a $150 board" holds and its supporting number
   improves a lot. ⚠️ Still a projection: `dwn_top` itself was never rebuilt at 11 bits.
2. **The `z` exchange rate moves a lot.** Phase 2 chose `z=50` over `z=200` because the encoder
   dominated. Scaling all encoders by ~1/5.9 shrinks that penalty: at `1x360`, `z=200` costs
   3,181 more LUTs than `z=50` today for +0.24 pp, but only ~540 more at 10 bits. The conclusion
   is **weakened, not reversed** -- and it matters least at the top of the frontier, where the
   binding constraint is timing anyway.
3. **The encoder stops being the dominant block, and that reshuffles what to optimize next.**
   At 10 bits the headline splits roughly: reduction 4,450 (~57%), core-minus-reduction 2,400
   (~31%), encoder ~975 (~12%). The 14x-encoder framing that has driven this project since
   Phase 1 would simply stop being true -- and **the Learnable Reduction reopening gets much
   stronger**, since the reduction becomes the single largest block by a wide margin.
4. **`dse/area_model.py` is calibrated against 16-bit encoders** and would need refitting before
   it predicts anything at a narrower word.

**Recommended handling: do not re-sweep.** Phase 2 stands as measured, with the fixed datapath
precision stated as a limitation, and this recorded as a Phase 3 finding. If a refresh is ever
wanted, re-run the **frontier points only** (~15 configs, not 54) -- a full re-sweep is not
affordable on one machine.

#### Does this transfer to the big configs? Probably, and here is the evidence

Cost per wired comparator at 16-bit, across configs:

| config | encoder LUTs | wired comparators (max) | per comparator |
|---|---|---|---|
| `1x50` z=200 | 1,519 | 202 | 7.52 |
| `1x2400 z=50` | 5,753 | <=800 | >=7.19 |

The per-comparator cost is a property of the comparator, not of design size, and it is flat.
Still **measure before quoting** (3X-b): at `z=50` the thresholds are further apart, so fewer
collapse at narrow widths -- likely *better* accuracy retention and slightly *more* area.

### 2026-08-11 — 3L-d: the comparison figures, and DWN vs conifer on identical silicon

`cc/literature/plot.py` -> `docs/results-cc/jsc-openml.png`, `jsc-cernbox.png`. `table.py` now
also ingests `docs/results-cc/conifer-results.json`, so the combined table carries all three
sources: 41 DWN configs, 10 conifer configs, 32 published rows.

**The head-to-head, both measured through `scripts/build.tcl` at `xc7a35tcpg236-1` / 10 ns.**

Iso-area — best accuracy within a LUT budget:

| budget | DWN | conifer | gap |
|---|---|---|---|
| 4,000 | `1x360 z=25` 75.27% | `gbdt_d3_n10` 73.64% | **+1.63 pp** |
| 8,000 | `1x800 z=50` 75.95% | `gbdt_d5_n5` 74.36% | **+1.59 pp** |
| 12,751 | `1x2400 z=50` 76.18% | `gbdt_d6_n3` 74.50% | **+1.68 pp** |
| 20,800 | `1x1600` 76.35% | `gbdt_d3_n40` 74.88% | **+1.48 pp** |

Iso-accuracy — smallest design reaching a target:

| target | DWN | conifer | ratio |
|---|---|---|---|
| 73.6% | 1,619 | 3,774 | **2.3x** |
| 74.2% | 2,541 | 7,602 | **3.0x** |
| 74.5% | 2,541 | 15,363 | **6.0x** |
| >=74.9% | 3,381 | **never reaches it** | — |

**A GBDT does not reach DWN's accuracy on this part at any size that fits.** conifer tops out at
74.88% using 73.9% of the device; DWN reaches that at 3,381 LUTs and continues to 76.35%.

⚠️ **The counterweight, which belongs in the report next to the above.** conifer wins latency
outright: **477.3 MHz against our 101-147 MHz**, and 4.2-16.8 ns against our 27-40 ns. In *cycles*
they are comparable (2-8 vs our 4) — which is exactly why brief §6 requires cycles alongside ns.
The fair one-line summary is **DWN wins accuracy-per-LUT; conifer wins speed.** Also note conifer
is 0 BRAM / 0 DSP on all 14 configs, so the DSP argument is against hls4ml's MLPs only.

#### Two things the figures enforce, rather than leave to a caption

- **One figure per dataset.** The script will not put OpenML and CERNBox on one axis. A combined
  plot would show a gap that is partly architecture and partly the ~1.05 pp dataset offset.
- **No Pareto curve spans two accounting conventions.** Series are keyed by
  *(method, encoder convention)*, so `DWN (core only)` and `DWN (+encoder)` are drawn as two
  curves. The first version of this plot did connect them, producing a frontier no single
  accounting produces — the same class of error as the dataset mixing, committed inside the very
  figure built to prevent it. Marker fill encodes the convention: filled = encoder included,
  hollow = core only, square = no separate encoder stage.

#### Closed while building it: the encoder convention for non-DWN rows

`encoder_included: "n/a"` on the LogicNets/PolyLUT/NeuraLUT/TreeLUT/AmigoLUT rows was an
assumption, flagged 2026-08-10 as unverified. It holds:

- **TreeLUT** quantises inputs *"as a pre-processing step"* (§2.1) — host-side, exactly as our
  StandardScaler is. Its reported LUTs are the tree comparators and adders.
- **Mecik & Kumm settle it for the family**: *"In previous performance evaluations, only the
  resource usage of the LUT layer and the classification logic was reported, which makes
  meaningful comparison with LUT-based architectures difficult."* That is said **of DWN** — DWN
  was the outlier omitting its encoder; the LUT-based architectures report complete designs.

These architectures feed quantised inputs straight into LUT address lines and have no expansion
stage. DWN is the one that needs an encoder, because thermometer coding turns 16 features into
3,200 bits. `n/a` is correct and now sourced.

### 2026-08-10 — the weightless lineage has no JSC numbers, and the DWN paper's own JSC table mixes datasets

Two results from closing the "no weightless related work" gap.

#### 1. ULEEN / BTHOWeN / WiSARD are related work, not table rows

Checked the DWN paper directly (v5). ULEEN appears only in its **MNIST/KWS/FashionMNIST** FPGA
tables -- never in the JSC table -- and the DWN paper's framing is explicit: *"All datasets were
chosen due to their use in prior work."* The weightless lineage was benchmarked on the BTHOWeN
suite (MNIST plus small UCI sets), **not on JSC**.

So the gap is real but it is a **citation gap, not a missing row**: brief §8 should name WiSARD ->
BTHOWeN -> ULEEN -> DWN as the lineage this project sits in, and no JSC comparison changes. Cheap
to fix, and it removes the "weightless project with no weightless related work" criticism.

The number worth carrying: DWN reports a **63x energy-delay improvement over ULEEN**, the prior
WNN state of the art -- that is the lineage's own measure of what DWN contributed.

#### 2. ⚠️ Which JSC dataset are the DWN paper's own numbers on? Unresolved, and load-bearing

The paper (v5, §4.1) says: *"We use the MNIST and Jet Substructure (JSC) datasets, **as in the
NeuraLUT paper**, and compare our models against published results."* NeuraLUT is **CERNBox**
(established 2026-08-10 from NeuraLUT-Assemble §5 and SparseLUT's own `jsc-cernbox` label).

But everything else points to **OpenML**:

| evidence | points to |
|---|---|
| paper text, "as in the NeuraLUT paper" | CERNBox |
| `third_party/DWN` tutorial calls `openml.datasets.get_dataset(42468)` | OpenML |
| FPGN (arXiv:2607.08427) Table V independently files DWN under **JSC-OpenML** | OpenML |
| **our own reproduction**: we trained on OpenML and got 76.18% at 2400 nodes vs their 76.3% | OpenML |
| DWN's accuracies (73.7-76.3%) sit **above every published CERNBox result** (max 75.1%) | OpenML |

Four independent lines against one sentence. The likely reading is that the sentence means "the
same *task* as NeuraLUT" loosely, not the same data file -- but that is inference, not evidence.

**Why it matters to us:** our headline comparison is `1x2400 z=50` at 76.18% against DWN `lg` at
76.3%. That is only a valid comparison if DWN `lg` is OpenML. If it is CERNBox, we are comparing
across a ~1.05 pp dataset gap and the claim has to be withdrawn. **Our reproduction landing within
0.12 pp of their number is the strongest evidence we have that we are on the same data** -- had
they been on the harder CERNBox set, our OpenML training should have come out ~1 pp *above* theirs,
not slightly below.

**This is the single best question for the authors' email.** It is cheap for them to answer and it
decides whether our central comparison stands.

#### 3. The conflation starts in the primary literature

DWN v5's JSC table lists, in one block: `hls4ml` (76.2%, 63,251, 38 DSP -- **OpenML**, via Fahim
et al.) alongside `PolyLUT` (236,541) and `NeuraLUT` (92,357) -- both **CERNBox**. So the mixing we
found in Mecik & Kumm's Table II and in the survey **originates upstream and propagates by
copying**. Every downstream table inherits it.

That strengthens the literature half's contribution: the dataset split is not a footnote we
noticed, it is a defect running through the whole comparison chain.

#### ⚠️ Version note: cite v5, not v1

arXiv v1 and v5 report **different** JSC LUT counts -- v1 has DWN `sm` at 134 LUTs and `md` at
2,144; v5 has **110** and **720**. `docs/paper-configs.md` was read from v5 and is correct, and v5
matches Mecik & Kumm's reproduction exactly. Anything scraped from `arxiv.org/html/2410.11112v1`
is superseded -- pin the version when quoting.

### 2026-08-10 — 3X-b: confirmed at `1x2400 z=50`, but the safe width moved and the cliff is elsewhere

Re-run of both halves on the headline config. Control passes again: the emitted encoder at 16 bits
measures **5,753 LUTs**, exactly the Phase 2 figure, and the float encoder reproduces the
checkpoint's recorded **76.1837%** to four decimals.

**The accuracy limit is one bit worse here, not better.** This was predicted the wrong way round:

| scheme | `1x50` | `1x2400 z=50` |
|---|---|---|
| in-place | 10 bits | **11 bits** (-0.142 pp) |
| renorm | 8 bits | **9 bits** (-0.120 pp) |

**Mechanism: fan-out.** Here 746 comparators feed 2400x6 = 14,400 node slots, ~19 slots per
comparator; at `1x50` it is 202 feeding 300, ~1.5 each. One wrong encoder bit propagates into ~19
nodes instead of ~1.5, so quantization error amplifies far more in a wide network. Expect the
usable width to keep creeping up with layer width -- **do not assume 11 bits transfers to a bigger
model either.**

**And the cliff is between 12 and 11, not 12 and 10.** `1x50` only sampled 12 and 10, which
located it too coarsely:

| word | distinct | LUTs | per comparator | vs 16-bit |
|---|---|---|---|---|
| 16 | 745/746 | 5,753 | 7.71 | 1.00x |
| 13 | 739 | 4,916 | 6.59 | 1.17x |
| 12 | 734 | 4,157 | 5.57 | 1.38x |
| **11** | 731 | **992** | 1.33 | **5.80x** |
| 10 | 720 | 891 | 1.19 | 6.46x |
| 9 | 687 | 794 | 1.06 | 7.25x |
| 8 | 596 | 655 | 0.88 | 8.78x |

The accuracy-safe width lands on the **cheap** side of the cliff by one bit. That is luck, not
design: had the cliff sat between 11 and 10, the usable saving would have been 1.38x instead of
5.80x. Any future config needs both halves re-measured, because the two limits are adjacent and
neither is predictable.

**Headline result: `in-place` at 11 bits, 5,753 -> 992 LUTs (5.80x), -0.142 pp, no retraining.**

⚠️ **-0.142 pp is 95% of the 0.15 pp bar.** "Within noise" is technically true and practically
marginal, and the conservative fallback is bad: 12 bits costs -0.111 pp and saves only 1.38x. The
whole win rides on one bit.

**The better operating point is `renorm` at 11 bits: -0.022 pp**, seven times inside the noise
floor, at the same comparator width and therefore approximately the same area. It also needs no
retraining. ⚠️ Its area is **not** directly measured -- renorm keeps more distinct constants (746
vs 731), so expect slightly more than 992. Measuring it needs `emit_encoder` to accept per-feature
affine constants, which it does not.

#### Corrected projection for the headline design

| | measured today | projected at 11-bit encoder |
|---|---|---|
| encoder | 5,753 | **992** |
| core | 6,850 | 6,850 (unchanged) |
| top | **12,751** | **~7,990** |
| device | **61.3%** | **~38.4%** |

Still a projection -- only the encoder was synthesized standalone; `dwn_top` was not rebuilt, and
top has historically run ~148 LUTs above core+encoder. **Accuracy is unchanged and the design is
not faster**, so this buys area on a design whose binding constraint is timing.

### 2026-08-10 — ~~⚠️ OPEN~~ ✅ CLOSED: our encoder is ~7.6× more expensive than theirs on the same workload

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

**The literature, on JSC** — ⚠️ **split by dataset; see the correction above.** Full machine-readable
set with per-row source and convention: `cc/literature/jsc_literature.json`. Render with
`.venv\\Scripts\\python.exe cc\\literature\\table.py [--dataset cernbox] [--markdown]`.

**JSC-OpenML — our dataset. These rows are comparable to ours.**

| Method | Model | Acc. | LUT | FF | Fmax | Lat (ns) | Encoder | Part |
|---|---|---|---|---|---|---|---|---|
| **this project** | `1x2400 z=50` | **76.18%** | **12,751** | 3,131 | 101 | 39.5 | **incl** | `xc7a35t-1` |
| **this project (conifer)** | `gbdt_d4_n10` | 74.19% | 8,005 | 1,418 | 477 | — | n/a | `xc7a35t-1` |
| DWN | lg-2400 (TEN) | 76.3% | 4,972 | 3,305 | 827 | 7.3 | core only | `xcvu9p-2` |
| DWN | lg-2400 (PEN+FT 9-bit) | 76.3% | 7,011 | 961 | 947 | 2.1 | **incl** | `xcvu9p-2` |
| DWN | lg (FPGN re-impl) | 76.3% | 6,302 | 4,128 | 695 | 14.4 | ? | `vu9p` |
| hls4ml | JSC MLP | 76.2% | 63,251 | — | — | 45.0 | n/a | `xcvu9p` |
| TreeLUT | I | 76.0% | 2,234 | 347 | 735 | 2.7 | n/a | `xcvu9p-2` |
| NeuraLUT-Assemble | OpenML | 76.0% | **1,780** | 540 | 941 | 2.1 | n/a | `xcvu9p-2` |
| FPGN | MLP 1000/1000/500 | 76.0% | 3,345 | 1,703 | 730 | 5.5 | ? | `vu9p` |
| DWN | md-360 (PEN+FT) | 75.6% | 1,697 | 198 | 696 | 2.6 | **incl** | `xcvu9p-2` |
| TreeLUT | II | 75.0% | 796 | 74 | 887 | 1.1 | n/a | `xcvu9p-2` |
| DWN | sm-50 (PEN+FT) | 74.0% | 311 | 52 | 1,011 | 2.0 | **incl** | `xcvu9p-2` |
| DWN | sm-10 (PEN+FT) | 71.2% | 64 | 18 | 1,251 | 1.6 | **incl** | `xcvu9p-2` |

**Three different LUT counts exist for DWN `lg`** — 4,972 (core only), 7,011 (with encoder), 6,302
(FPGN's re-implementation, convention unstated). Any comparison must say which one it uses.

**JSC-CERNBox — the harder dataset. NOT comparable to ours.** LogicNets, PolyLUT (236,541 at
75.0%), PolyLUT-Add, NeuraLUT (92,357 at 75.0%), AmigoLUT, ReducedLUT, SparseLUT, plus
NeuraLUT-Assemble at 8,539 and FPGN at 12,358. Full rows in the JSON.

**Dataset unknown:** LLNN (75.0% / 13,926 and 72.0% / 6,431). Paywalled at IEEE TCAS-I; on neither
plot until resolved.

---

## Open questions

| Question | Status |
|---|---|
| ~~Why is our encoder 7.6× theirs at the same `z`?~~ | ✅ **Closed 2026-08-10. Entirely comparator width.** At 8 bits ours is 182 LUTs against their 201. The hypothesis was right in mechanism and wrong about needing training: `in-place` at 10 bits gives **5.91×** with no retraining at all. |
| ~~Does the width result hold at `1x2400 z=50`?~~ | ✅ **Closed 2026-08-10, with two corrections.** The saving holds (**5.80×**) but the accuracy-safe width moved from 10 to **11 bits** (fan-out amplifies encoder error ~19x more), and the cliff is between **12 and 11**, not 12 and 10. Both limits are adjacent and neither extrapolates — re-measure per config. |
| Is `q16.12` worth sweeping? | ✅ **Answered: yes, it was the largest unswept axis.** 6 of 16 bits were doing nothing. See the entry above and the Phase 2 impact note. |
| Do the non-DWN rows include an input encoder? | ⚠️ **Open.** Settled for DWN only. LogicNets/PolyLUT/NeuraLUT take quantised inputs directly, so the question may not apply in the same form — but that itself has to be stated per row, not assumed. |
| ~~Which JSC split does each paper use?~~ | ✅ **Closed 2026-08-10, and it was the right thing to ask.** Two datasets, ~1.05 pp apart, and the standard tables conflate them. We are on OpenML with DWN, TreeLUT and hls4ml; the PolyLUT/NeuraLUT/LogicNets lineage is on CERNBox. Enforced in code by `cc/literature/table.py`. |
| Does LLNN use OpenML or CERNBox? | ⚠️ **Open.** IEEE TCAS-I, paywalled. Two rows parked until resolved. Try an author preprint. |
| ~~Which convention do the non-DWN rows use?~~ | ✅ **Closed 2026-08-11, `n/a` confirmed.** TreeLUT quantises host-side "as a pre-processing step"; Mecik & Kumm state that it was **DWN** whose evaluations reported only the LUT layer and classification logic, not the LUT-based family. See the 2026-08-11 entry. |
| Is the hls4ml 76.2% / 63,251 number real? | ⚠️ **Open, and load-bearing.** It is the headline comparison in our README and `docs/phase3-plan.md` §5, and it currently traces only to our own brief §8 with no primary citation. Must be found or the claim dropped. Marked `unverified` in the JSON. |
| Does FPGN beat us on our own comparison? | ⬜ Unread. Compares against DWN directly; the most likely paper to change what we can claim. |
| **Which JSC dataset are the DWN paper's numbers on?** | ⚠️ **Open, load-bearing, top question for the authors — tagged `PENDING-AUTHOR-REPLY`.** `grep -rn PENDING-AUTHOR-REPLY docs/ cc/ README.md` lists every claim that has to be revisited if the answer is CERNBox rather than OpenML. Asked 2026-08-10; no reply as of 2026-08-11. Paper says "as in the NeuraLUT paper" (CERNBox); code, FPGN, our reproduction and the accuracy level all say OpenML. Our headline comparison against DWN `lg` is only valid if OpenML. See the 2026-08-10 entry. |
| ~~Does the weightless lineage (ULEEN/BTHOWeN/WiSARD) need JSC rows?~~ | ✅ **Closed 2026-08-10, no.** None of them report JSC — they use the BTHOWeN suite. They belong in brief §8 as the lineage DWN descends from, which is a citation fix, not a table change. |
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
