# Scoping: turning the generator into a reusable tool

**Status: not part of this project.** `CLAUDE.md` names two studies — the DSE frontier (Phase 2)
and the controlled comparison (Phase 3). The RTL generator exists to serve those. This document
scopes what it would take to turn it into something other people can use, in the way hls4ml is
used, and is written so the decision can be made later with real numbers rather than enthusiasm.

Nothing here should be started before Phase 3 finishes. It is recorded now because the shape of
the work is clearest while the code is fresh.

---

## 1. Why it might be worth doing

The project exists because **no open RTL implementation of DWN targets small FPGAs** — that is
the first line of `CLAUDE.md`, and it is still true. Upstream `third_party/DWN` ships PyTorch
training only: no RTL, no HLS, no FPGA flow (Phase 1 ledger, 2026-08-02).

So there are two possible outputs from this project:

| | durability |
|---|---|
| **A Pareto frontier for DWN on an XC7A35T** | a result: cite it, or reproduce it |
| **A working DWN → Verilog generator** | a tool: other people build on it |

The second is arguably the more reusable contribution, and it is nearly a by-product of the
first. hls4ml is the obvious model — framework-agnostic front end, HDL out, used by people who
did not write it.

## 2. What already generalizes

More than you would guess. This is **verified**, not assumed:

| dimension | tested range | evidence |
|---|---|---|
| layer width | 20 → 2400 nodes | Gate 1 at 20; emission + synthesis at 2400 |
| layer count | 1, 2, 3 | Gate 1 on a `[300,100]` model, 2026-08-08 |
| `n` (LUT inputs) | 2, 4, 6 | Gate 1 at each, via synthetic checkpoints |
| `z` (thermometer bits) | 8 → 800 | measured configs across the sweep |
| encoding | distributive, gaussian, linear | all three trained in the Phase 2 grid |
| wiring representation | learnable + fixed | both paths exercised (checkpoint-format §3a/§3b) |
| class count | driven by the checkpoint | `GroupSum` divisibility asserted at config build |

`exporter/extract.py` reads shape from the checkpoint; `rtlgen/emit_core.py` loops over layers.
Neither has a hardcoded model size.

The hand-written primitives (`rtl/lut_node.v`, `popcount.v`, `argmax.v`, `pipe_reg.v`) are
already parameterized and are the whole of the generated design's structure.

## 3. What is specific to this project

Three layers, in increasing order of coupling:

**a. Precision — one constant, several consumers.** `Q3.12 signed, 16-bit` is hardcoded in
`exporter/extract.py` (`FRAC_BITS`, `WORD_BITS`) and consumed by the golden model, the encoder
emitter, the testbench vector generator and the host. It was *measured* for JSC
(`exporter/analyze_precision.py`), so the analysis to pick it exists — it is the plumbing that
assumes one answer.

**b. The harness — thoroughly JSC-shaped.** `harness/` is the board plumbing: UART loader, BRAM
vector store, benchmark FSM, 7-segment display, top level. It assumes 16 features × 16 bits +
1 label = 33-byte records, `LABEL_W=3` (5 classes), and `DEPTH=1024` sized for 259-bit vectors.
The Phase 1 ledger's MNIST scoping note lists the three things that break, and they are all
width constants — `uart_loader`'s `reg [5:0] byte_idx` would silently wrap on a 1569-byte
record rather than fail.

**c. The checkpoint format — coupled to one upstream commit.** `docs/checkpoint-format.md`
records what the exporter reads, verified against pinned commit `9f887a0`. A different upstream
version could move `mapping.weights` or change the LUT sign convention, and the failure would be
a valid-looking wrong export (the `__dummy_mapping` trap in `extract.py` is exactly this).

## 4. What "reusable" would require

| work | effort | notes |
|---|---|---|
| **MNIST port** | 1–2 days | proves dataset-generality; already scoped in the Phase 1 ledger |
| **Configurable precision** | 1–2 days | thread word/frac through the five consumers, keep `analyze_precision.py` as the way to choose them |
| **Generalize or drop the harness** | 2–4 days | either parameterize the widths, or ship generator-only and let users bring their own I/O |
| **Packaging + CLI + docs + licence** | 2–4 days | `pip install`, one entry point, worked examples |
| **Tests on a second dataset + CI** | 2–3 days | the claim is generality; it needs a second dataset to be a claim at all |
| **total** | **~1–2 weeks** | almost none of it is new RTL |

The striking part is how little is RTL work. The generator is close to done; what is missing is
generality in the plumbing and everything around the edges that makes a tool usable by someone
who did not write it.

### The single most valuable piece

**The MNIST port**, because it is the only item that tests the claim rather than asserting it.
Everything else is packaging around a generator that might still turn out to be JSC-shaped in
ways nobody noticed. Phase 2 already found several: a table-packing bug at n<3, a tau schedule
that was wrong at every interpolated width, and an encoder narrowing result that was fitted and
tested on the same 1000 samples. A second dataset is how the next one gets found.

⚠️ **MNIST's 97–98% is not a better result** — it is an easier dataset. Its value here is that it
stresses the encoder and the I/O path rather than the LUT core (brief §13).

## 5. How to split it off

**Fork, then prune.** Not a fresh repo with files copied across.

The reason is the history. This repository's value is not only its code — it is the record of
*why* the code is shaped the way it is, and most of that reasoning was bought with debugging
time:

- address bit order is LSB-first, and reversing it "yields a design that elaborates,
  synthesizes, and is wrong on most inputs"
- `np.packbits` silently shifts tables shorter than 8 entries, which broke n=2
- `__dummy_mapping` has the same shape and dtype as a real mapping
- `benchmark_fsm` label alignment, which has produced a plausible-looking wrong accuracy twice

A fresh repo loses every one of those, and they are precisely the things a new contributor would
otherwise rediscover the hard way.

**Suggested shape:**

```
keep      exporter/  rtlgen/  rtl/  tb/  scripts/run_gate1.py  scripts/build.tcl
keep      docs/checkpoint-format.md   (what the exporter reads, and why)
optional  harness/   -- only if the board flow is in scope
drop      docs/phase*-ledger.md, docs/project-brief.md, dse/, cc/, releases/
```

Dropped files stay in history, so nothing is lost — but the tool's README should not open with a
Basys 3 project plan.

**Do the split after Phase 3, not before.** Phase 2 and 3 will keep changing the emitters, and
maintaining a fork in parallel means merging changes twice while the numbers that go in the
writeup depend on the original.

## 6. Open questions to settle before starting

- **Generator-only, or generator + board flow?** Generator-only is far smaller and far more
  general. But the board flow is the part that proves the RTL actually runs, and Gate 1b is the
  strongest evidence in this project. Shipping the generator alone means users get RTL nobody
  has put on silicon.
- **Which upstream DWN versions to support?** Currently exactly one pinned commit. Supporting a
  range means a compatibility layer plus a way to detect format drift — and the drift failure
  mode is silent.
- **Does Gate 1 ship with it?** It should. The golden-model testbench is what makes generated
  RTL trustworthy, and a generator without it is a generator whose output nobody can check.
  This is the single strongest thing the tool would have over rolling your own.
- **What is the precision story for a new dataset?** `analyze_precision.py` measures the required
  format, but somebody has to run it and thread the answer through. Automating that end to end
  is a real feature, not a detail.

## 7. What this is not

Not a replacement for hls4ml or conifer. Those cover broad model families; this covers exactly
one architecture. The right framing is narrower and more honest: **the reference RTL
implementation of DWN**, with a verification harness, for people who want to put a weightless
network on an FPGA without writing the Verilog themselves.
