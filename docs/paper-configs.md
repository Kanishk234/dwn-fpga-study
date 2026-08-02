# The paper's JSC configurations

From **Bacellar et al., *Differentiable Weightless Neural Networks*, ICML 2024**
([arXiv:2410.11112](https://arxiv.org/abs/2410.11112), v5). Read from the PDF directly — Table 14
(model configurations) and Table 2 (FPGA results), not from a summary.

These are the numbers we are trying to reproduce before building anything on top of a checkpoint.

---

## Table 14 — JSC model configurations

| Model | z | Layers | tau | BS | Learning rate |
|---|---|---|---|---|---|
| DWN (n=6; sm) | 200 | 1× 10 | 1/0.7 | 100 | 1e-2(14), 1e-3(14), 1e-4(4) |
| DWN (n=6; sm) | 200 | **1× 50** | **1/0.3** | 100 | 1e-2(14), 1e-3(14), 1e-4(4) |
| DWN (n=6; md) | 200 | 1× 360 | 1/0.1 | 100 | 1e-2(14), 1e-3(14), 1e-4(4) |
| DWN (n=6; lg) | 200 | 1× 2400 | 1/0.03 | 100 | 1e-2(14), 1e-3(14), 1e-4(4) |

`z` is thermometer bits per feature. Binarization is the Distributive Thermometer for every dataset
in the paper (§4, "Binary Encoding").

⚠️ **Table 14's caption says "All models were trained for a total of 100 Epochs." That is wrong for
JSC** — the JSC rows' schedules sum to 32 epochs (14 + 14 + 4), while other datasets' sum to 100.
The per-row schedule is authoritative.

## Table 2 — JSC results (xcvu9p, out-of-context, `Flow_PerfOptimized_high`)

| Model | Accuracy | LUT | FF | Fmax | Latency |
|---|---|---|---|---|---|
| DWN (n=6; sm) 1× 10 | 71.1% | 20 | 22 | 3030 MHz | 0.6 ns |
| **DWN (n=6; sm) 1× 50** | **74.0%** | **110** | 72 | 1094 MHz | 1.5 ns |
| DWN (n=6; md) 1× 360 | 75.6% | 720 | 457 | 827 MHz | 3.6 ns |
| DWN (n=6; lg) 1× 2400 | 76.3% | 4972 | 3305 | 827 MHz | 7.3 ns |
| hls4ml | 76.2% | 63251 | 4394 | 200 MHz | 45.0 ns |

**`sm`/`md`/`lg` are informal size labels, not defined architectures.** They name points on the size
axis and are chosen per dataset — MNIST's `sm` is a *two-layer* `1000, 500`. The paper reuses `sm`
for two different JSC models (1× 10 and 1× 50); our `sm-10` / `sm-50` naming disambiguates by width.

**Out-of-context synthesis** means no I/O, no surrounding system. Those Fmax figures are not
achievable in a real design, which is part of why a working board deployment measures something this
table does not (brief §14).

---

## What this changed for us

### Every JSC model is a single layer

Not one uses a second LUT layer. Our `[300, 100]` inserted a **random-wired** second layer between
the encoder and the popcount, discarding most of layer 0's output. 50 nodes on a rich input beat our
400 nodes on a starved one, 74.0% vs 72.6%.

### z=200, and the thermometer is a feature pool

With a single learnable-mapped layer, the thermometer's job is to offer *candidates*. Learnable
Mapping picks `output_size × n` of them — 300 out of 3200 for the `1× 50` model. Our t=4 and t=8 runs
offered 64 and 128 candidates for those same 300 slots.

### Correction: "encoder resolution is a flat axis" was wrong

`training/README.md` previously concluded from t=4 → t=8 (+0.22pp) that thermometer resolution
barely matters. That is measuring the flat part of a curve whose operating point is 25× further out
and concluding the curve is flat. The paper's own numbers show `sm` at 71.1% and 74.0% differing by
**layer width at fixed z=200**, so z's true effect on accuracy is still unmeasured — by us *and* by
the paper, which fixes z=200 everywhere and never sweeps it.

### The thing the paper never had to ask

**z=200 does not mean a 3200-bit encoder in hardware.** At most `output_size × n` thermometer bits
are wired to anything; the rest feed no node and should vanish. For `1× 50` that is ≤300 comparators,
consistent with Table 2's 110 total LUTs against a 50-LUT core.

But each surviving bit is still a comparison against a constant, and ~300 comparators is not
obviously cheap on an Artix-7 — which is exactly the cost Mecik & Kumm measured at up to 3.2×
(brief §8, §12 risk #3). On an xcvu9p it never mattered. On a 20,800-LUT part it might dominate.

**Accuracy and area versus z, on hardware where z actually binds, is unpublished.** It is a Phase 2
sweep axis we can now aim properly, and it is one of the project's genuine contributions rather than
a reproduction (brief §14).
