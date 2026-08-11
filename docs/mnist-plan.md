# MNIST — the second dataset, and the rules that keep this branch clean

**Why this exists.** Every result so far is jet substructure classification. The generator is
*claimed* to be general; nothing has tested that. MNIST is the test. A working MNIST port turns
"our generator is parameterised" into "our generator has been shown to work on a second problem
with different dimensions" — which is the difference between an assertion and a result.

**The generalised tool is the real deliverable here, not the MNIST accuracy number.** MNIST is how
we find the JSC assumptions we cannot see. If the port fails to fit the board, that is still a
successful outcome as long as the generator came out general.

Work happens on the `mnist` branch. `main` holds the JSC study, tagged `jsc-complete`.

---

## 1. Ground rules for this branch

These exist because this branch touches code that every JSC result depends on.

### 1.1 Two kinds of commit, never mixed

| kind | what it is | can it be verified today? |
|---|---|---|
| **generalisation** | removing a JSC assumption from shared code | **yes** — JSC must still reproduce exactly |
| **MNIST-specific** | record formats, checkpoints, configs, results | no — needs MNIST to exist first |

**Never put both in one commit.** Not because they belong on different branches — they do not,
and separating branches ahead of time means guessing where the boundary is before you have hit
it. The reason is recoverability: if MNIST turns out not to fit this device, the generalisation
commits are cherry-picked onto `main` and the rest is abandoned. A commit that both derives the
feature count *and* rewrites the UART record format cannot be split later.

Commit after each change that leaves JSC passing, rather than batching.

### 1.2 The gate: JSC must keep reproducing, exactly

After **every** generalisation commit:

```
.venv\Scripts\python.exe scripts\verify_phase1.py
```

**12/12, with areas at 108 / 1,519 / 1,619.** Not "close" — identical. A generalisation that
changes a single LUT has changed behaviour, and every number in `REPORT.md` was measured on the
old one.

`verify_phase1.py` only exercises one single-layer config, and MNIST is two layers, so also run:

```
.venv\Scripts\python.exe scripts\run_gate1.py --checkpoint training\artifacts\dwn_jsc_t8_distributive_300-100_lr_checkpoint.pt
```

**If either breaks, stop and fix it before doing anything else.** Past that point you can no
longer tell whether an MNIST failure is MNIST's fault or ours.

### 1.3 Shared code must not know which dataset it is

If a file under `exporter/`, `rtlgen/`, `rtl/`, `tb/` or `scripts/` contains the number 16, the
number 5, or the string "JSC", that is a defect — whether or not anything currently fails.

Dataset facts live in **`datasets/`** as data, not as code paths. Adding a third dataset should
mean adding a descriptor, not editing an emitter.

The test to apply to any generalisation: *would this still be right for a dataset we have not
thought of?* Not: *does this work for MNIST?*

### 1.4 Never move or rename an existing JSC artifact

`REPORT.md`, `README.md` and the `jsc-complete` tag all reference current paths —
`docs/results/`, `docs/results-cc/`, `rtl/example-model-1x50/`, `training/artifacts/`.

MNIST is **additive**: new directories alongside, never a reorganisation of what is there. A tidier
layout is not worth breaking the published artifact, and a reorganisation is a separate change to
make deliberately, on its own, if it is ever wanted.

### 1.5 Layout for new material

| | |
|---|---|
| `datasets/` | one descriptor per dataset: dimensions, record layout, defaults. **New, shared.** |
| `docs/results-mnist/` | MNIST measurements, mirroring `docs/results/` |
| `docs/mnist-*.md` | MNIST plan, ledger, report |
| `training/artifacts/` | checkpoints, already dataset-tagged by filename |

Nothing else new at the repo root.

### 1.6 What "clean at merge time" means

When this branch merges, someone reading `main` should not be able to tell it was ever two
branches. Concretely:

- No file named for a branch, a phase of this work, or a person.
- No dataset name in shared code (1.3).
- `README.md` and `REPORT.md` describe JSC exactly as they do now, plus MNIST where it belongs —
  JSC results are not rewritten, reworded or renumbered because a second dataset arrived.
- The ledger stays a dated log. It is not tidied into a summary before merging; the wrong turns
  are the useful part.

---

## 2. What has to change before MNIST runs

Audited on the code as it stands, not guessed. Ordered by when it bites.

### Blockers — the port cannot start without these

**B1. The feature count is hardcoded in the core emitter.**
`rtlgen/emit_core.py`: `input_bits = 16 * cfg['thermometer_bits']`. This is wrong *today*; it is
merely masked because JSC has 16 features. `rtlgen/emit_encoder.py` already derives it correctly
from `thresholds.shape[0]`, so the fix is to match the file next to it. Dangerous because it does
not error — it emits a core with the wrong input width and fails Gate 1 confusingly.

**B2. The board record format is fixed at 33 bytes.** 32 feature bytes plus one label, hardcoded
in `harness/uart_loader.v` and `scripts/host.py`. MNIST at 784 features is 1,569 bytes.

**B3. On-board vector capacity collapses.** `harness/vector_store.v` is already parameterised and
already batches, which is the good news. But it is sized at 259 bits per vector with `DEPTH=1024`
costing ~15% of the device's block RAM. MNIST at 16-bit inputs is ~12,548 bits per vector, roughly
48× more, so `DEPTH` falls to order 100 and a full test-set pass needs far more UART round trips.
Slower, not impossible.

### The fit-enabler

**F1. Configurable precision — already half-built.** `rtlgen/config.py`'s `HardwareConfig` already
carries `word_bits` and `frac_bits`, and `emit_encoder()` already accepts them as arguments. What
is missing is threading: `exporter/extract.py`'s `quantize()` and the golden model still read the
module-level constants, and `config.py` asserts `hw.word_bits == extract.WORD_BITS`, which has to
become "the default matches" rather than "always equal".

This probably decides whether MNIST fits at all. MNIST inputs are natively 8-bit pixels, so a
16-bit word is roughly double what is needed, and the encoder is where MNIST's area problem lives.

### Recalibration — results are quietly wrong without these

**R1. `dse/area_model.py`** has `JSC_FEATURES = 16` and self-tests at 5 classes. Its encoder cost
model is fitted to a 16-feature encoder at 16-bit words; both change.

**R2. `dse/grid.py`** — size ladder, tau values and model slugs are all JSC-shaped.

### Off-machine

**T1. An MNIST checkpoint**, trained on Kaggle. The paper's MNIST `sm` is **two layers, 1000 and
500** — multi-layer is already verified in this flow, so no new RTL capability is needed.

**T2. A test-vector dump** in the same shape the exporter expects.

**T3. Confirm the checkpoint structure** matches `docs/checkpoint-format.md`. Probably fine, and
the failure mode if not is a valid-looking wrong export, which is the expensive kind.

---

## 3. Phase 1 — bring-up, mirroring what JSC did

Same shape as the JSC bring-up: get one small model bit-exact in simulation, then on silicon.
**Deliberately not the paper's config.** Start at the smallest thing that exercises the new
dimensions, so a failure means one thing.

| Step | What | Done when |
|---|---|---|
| **M1a** | Fix B1; add `datasets/` with JSC and MNIST descriptors | JSC still 12/12, areas identical |
| **M1b** | Thread precision through (F1) | JSC still 12/12 at the default; a non-default width builds |
| **M1c** | Train a **small** MNIST model on Kaggle — one layer, few hundred nodes | checkpoint + vectors in `training/artifacts/` |
| **M1d** | Export and run Gate 1 | **bit-exact on every vector** |
| **M1e** | Synthesize out-of-context, report core / encoder / top | area known; whether it fits is a *result* either way |
| **M1f** | Harness: record format and vector-store capacity (B2, B3) | JSC board path still works |
| **M1g** | Bitstream, program, Gate 1b on the full MNIST test set | hardware matches software exactly |

**M1a and M1b are generalisation** and land as their own commits, gated on §1.2. **M1c onward is
MNIST-specific.**

### What would make this fail, and what each failure means

| Failure | What it tells us |
|---|---|
| JSC stops reproducing after M1a/M1b | our generalisation is wrong — fix before continuing, do not proceed |
| Gate 1 fails at M1d | a real bug in the general path that JSC's shape happened to hide. **This is the point of the exercise** |
| Does not fit at M1e | a device limit, not a defect. Record it and try narrower precision and fewer thresholds |
| Fits but misses timing | expected — JSC's wall was timing too, and MNIST's popcount is wider |

A negative result at M1e or M1g is publishable and belongs in the report. A silent wrong answer is
the only genuinely bad outcome, which is why Gate 1 comes before any area number.

---

## 4. Open questions to settle early

| Question | Why it matters now |
|---|---|
| How many thermometer thresholds per pixel? | 784 features × z is the whole area problem. JSC used 200; MNIST almost certainly cannot |
| Are MNIST pixels standard-scaled or min-max? | decides how many integer bits the word needs, and F1 depends on it |
| Does the upstream MNIST recipe binarise differently? | `docs/checkpoint-format.md` was verified against JSC checkpoints only |
| How many test vectors can the board actually hold? | sets whether Gate 1b is one pass or many (B3) |

---

## 5. Pointers

- `docs/mnist-phase1-ledger.md` — the dated log for this work
- `REPORT.md` — the JSC study this is being generalised away from
- `docs/checkpoint-format.md` — what the exporter reads, verified against JSC only
- `docs/phase1-ledger.md` — how the JSC bring-up actually went, including what broke
