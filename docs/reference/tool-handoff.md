# Starting the tool — a cold-start handoff

**Read this first, then `tool-roadmap.md` §1–§8.** This file exists so a fresh session in a new
repository can begin without reconstructing anything from chat history. Everything below is either
settled or explicitly open; nothing is left implied.

---

## 1. What the tool is

A **generator**: a trained DWN goes in, synthesizable Verilog comes out, plus the means to prove
that Verilog matches the model. Nothing else.

| in scope | out of scope |
|---|---|
| exporter (checkpoint → tables, wiring, thresholds) | the board harness — UART, vector store, benchmark FSM |
| RTL generator (core, encoder, top) | the design-space sweep (`dse/`) |
| the numpy golden model | the controlled comparisons (`cc/`) |
| **self-checking testbenches + golden vectors** | the area model — see roadmap Q7 |
| optional yosys resource estimate | any vendor toolchain dependency |

**The encoder always ships**, and its area is always reported separately from the network's. It is
intrinsic to a DWN, not preprocessing a user supplies — and on the smallest JSC model it is
**fourteen times** the network it feeds. Emitting the network alone would commit exactly the
reporting defect `docs/jsc/report.md` §5.2 criticises in published work.

## 2. Settled, with the reasoning

| | decision |
|---|---|
| **New repo, not a fork** | The tool is ~2,300 of ~19,600 lines — 12%. A fork's first commit deletes 88%. Knowledge lives in code comments, which travel with a copy; the ledgers should not travel |
| **Generator-only** | Users bring their own harness, as hls4ml does. The harness changes per application |
| **Vendor-neutral** | The emitted RTL instantiates **no vendor primitives** and is Verilog-2001. Verified by inspection across the whole tree |
| **Verification ships** | Self-checking testbenches the *user* runs in *their* simulator. That makes bit-exactness reproducible by them, not a claim they must trust |
| **No area model** | It filtered zero configs across two completed studies. hls4ml ships none either |
| **This repo is the evidence** | "77 configurations, all bit-exact, two datasets, one verified on silicon per dataset" is a claim nothing else in this space can make. Link to it; do not reproduce it |

## 3. Open — decide before writing a CLI

**Q8. There is no upstream checkpoint format.** Upstream trains and discards; the format this
project reads is our own. And a DWN is *two objects* — thermometer thresholds live outside the
`state_dict`, so `torch.save(model.state_dict())` silently loses the encoder. The tool must define
the format, own both ends, and fail loudly on a bare `state_dict`.
`docs/reference/checkpoint-format.md` is most of the specification already.

**Q9. Fractional bits are not derivable.** Integer bits are, exactly and per-config. Fractional
bits depend on data the tool does not have. Ask, default loudly, or measure — all defensible, all
produce different CLIs.

**Q3. Which upstream commits to support.** One pin is honest and cheap; a range needs a
compatibility layer.

**Q4. The name.** `dwn2rtl` is a placeholder used in discussion only.

## 4. The file inventory

Verified: the Gate 1 path compiles **nine files, none from `harness/`**, and the testbenches
instantiate only `dwn_top`/`dwn_core`. The core deliverable is already cleanly separable.

| source | lines | fate |
|---|---|---|
| `rtl/{lut_node,popcount,argmax,pipe_reg}.v` | 194 | **copy verbatim** |
| `tb/{dwn_core,dwn_top}_tb.v` | 177 | **copy** — already self-checking (`PASS`/`FAIL`, `$finish`) |
| `exporter/extract.py` | 296 | copy — it is the golden model too |
| `rtlgen/emit_core.py`, `emit_encoder.py` | 559 | copy |
| `tb/gen_vectors.py` | 215 | **rewrite** — drop the real-data half, keep the random half |
| `rtlgen/config.py` | 254 | **rewrite** — sweep-shaped (`build_dir`, `pipe_slug`, sweep names) |
| `datasets/__init__.py` | 352 | **extract ~40 lines** — precision only; the sweep axes are not tool material |
| `scripts/run_gate1.py` | 282 | **rewrite** — hardcodes xsim paths; must become simulator-agnostic |

≈2,300 copied, ≈1,100 rewritten.

⚠️ **Two files in this repo are not UTF-8** — `rtl/example-model-1x50/{dwn_core,dwn_top}.v`. They
contain no paths, but a repo-wide text tool will crash on them.

## 5. Vectors come from the model, not from data

The tool cannot assume a dataset either. Generate testbench vectors by drawing **random quantised
inputs**, running them through the numpy golden model, and emitting those.

This is not a compromise: **Gate 1 is RTL-versus-golden-model, not RTL-versus-dataset.** Random
vectors are arguably better, since they hit tie-break and saturation edges real data does not.
`tb/gen_vectors.py` already mixes 500 random vectors with real ones; the tool keeps the random half.

**The invariant that must not break:** the testbench vectors and the RTL must derive from the *same*
checkpoint. Otherwise you ship a testbench that passes against wrong RTL — worse than shipping none.

## 6. First real gate

**Run the emitted testbench under `iverilog`.** Neither iverilog nor verilator is installed on the
study machine, so "the RTL is portable" is currently an *inspection* result, not a measurement.
Make it the tool's first CI check rather than an assumption.

## 7. Suggested first steps

1. Decide **Q9** (fractional bits) — it determines the CLI.
2. Create the repo; copy the eight verbatim files; `pyproject.toml`; a CLI entry point.
3. Port the exporter and emitters unchanged, then the vector generator in its rewritten form.
4. Get `iverilog` green on one emitted design end to end.
5. Only then: yosys estimates, multi-version support, docs.

## 8. Pointers back

| | |
|---|---|
| `docs/reference/tool-roadmap.md` | the audited work list — defects, generality gaps, packaging, order |
| `docs/reference/reusable-generator.md` | earlier scoping; kept for the reasoning |
| `docs/reference/checkpoint-format.md` | the schema, verified against the pinned upstream commit |
| `docs/reference/datapath.md` | what each stage costs and how much generalises |
| `REPORT.md` | the evidence the tool's README should cite |
| `docs/mnist/report.md` §2 | the seven hard-coded dataset facts, and why none was findable by inspection |
