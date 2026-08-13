# Documentation index

Start with a **report** if you want results, a **ledger** if you want the reasoning and the wrong
turns, and the **brief** if you want the plan the whole thing follows.

```
docs/
  jsc/         the JSC study    — reports, ledgers, and its measurements
  mnist/       the MNIST study  — the same, for the second dataset
  reference/   facts established once and relied on everywhere, both datasets
```

The combined write-up covering both datasets is [`REPORT.md`](../REPORT.md) at the repository root.

⚠️ **The two studies are not comparable on shared axes**, and are deliberately never merged into
one table or one figure: their accuracy scales differ by ~20 points, their encoder economics run
opposite, and their figures are pinned to different tags (`jsc-complete`, `mnist-complete`). Per-
dataset dimensions and settings live in `datasets/`, never in the shared code.

---

## The JSC study — [`jsc/`](./jsc/)

Jet substructure classification: 16 features, 5 classes. The original study.

| | |
|---|---|
| [**`jsc/report.md`**](./jsc/report.md) | **Start here** — the standalone study, with appendices and a glossary |
| [`jsc/phase1-report.md`](./jsc/phase1-report.md) | Getting one model onto a Basys 3, and the five things that cost time |
| [`jsc/phase2-report.md`](./jsc/phase2-report.md) | The design-space exploration: 52 configurations, the frontier, and the six things that broke |
| [`jsc/phase3-report.md`](./jsc/phase3-report.md) | The controlled comparison: conifer and hls4ml on identical silicon, plus two defects in the literature |
| [`jsc/results/`](./jsc/results/) | Phase 2 evidence — all 54 records, both figures, the grid as trained |
| [`jsc/results-cc/`](./jsc/results-cc/) | Phase 3 evidence — 14 conifer and 6 hls4ml configurations, plus the comparison figures |

Ledgers — the dated record, with corrections left visible rather than edited away:
[`phase1-`](./jsc/phase1-ledger.md) · [`phase2-`](./jsc/phase2-ledger.md) ·
[`phase3-`](./jsc/phase3-ledger.md).
Plans and handoffs: [`dse-plan.md`](./jsc/dse-plan.md) ·
[`phase2-handoff.md`](./jsc/phase2-handoff.md) · [`phase3-plan.md`](./jsc/phase3-plan.md) ·
[`phase3-handoff.md`](./jsc/phase3-handoff.md).

## The MNIST study — [`mnist/`](./mnist/)

784 features, 10 classes — a deliberately different second target. **The port is the result**: the
exporter, generator and testbenches became dataset-agnostic, and both datasets now reproduce from
the same code.

| | |
|---|---|
| [**`mnist/report.md`**](./mnist/report.md) | **Start here** — the standalone study, in the same form |
| [`mnist/phase1-report.md`](./mnist/phase1-report.md) | Bring-up: one bug pattern found seven times, and what generalising required |
| [`mnist/phase2-report.md`](./mnist/phase2-report.md) | The sweep: 25 configurations, and two Phase 1 predictions retracted |
| [`mnist/phase3-report.md`](./mnist/phase3-report.md) | The comparison: DWN against its own weightless family for the first time |
| [`mnist/results/`](./mnist/results/) | Phase 2 evidence — 25 measured records, both figures, the grid as trained |
| [`mnist/results-cc/`](./mnist/results-cc/) | Phase 3 evidence — the conifer measurements and the comparison figure |

Ledgers: [`phase1-`](./mnist/phase1-ledger.md) · [`phase2-`](./mnist/phase2-ledger.md) ·
[`phase3-`](./mnist/phase3-ledger.md) ·
[`reduction-ledger.md`](./mnist/reduction-ledger.md), which measured the 0.24 pp noise floor that
withdrew three claims — including the study's own headline.
Ground rules the port was held to: [`mnist/plan.md`](./mnist/plan.md).

## Reference — [`reference/`](./reference/)

Dataset-independent. Established once, relied on by both studies.

| | |
|---|---|
| [`project-brief.md`](./reference/project-brief.md) | The full technical plan: resource budgets, the two studies, phases, risks |
| [`datapath.md`](./reference/datapath.md) | What each stage does — encoder, LUT layer, popcount, argmax — and what each costs |
| [`checkpoint-format.md`](./reference/checkpoint-format.md) | What the exporter reads, verified against the pinned upstream commit, including the traps |
| [`paper-configs.md`](./reference/paper-configs.md) | The original paper's configurations, and what they corrected |
| [`probe-results.md`](./reference/probe-results.md) | Does `TABLE[addr]` map to a single LUT6? The evidence the area claim rests on |
| [**`tool-handoff.md`**](./reference/tool-handoff.md) | **Starting the tool** — a cold-start handoff: what is settled, what is open, and the file inventory |
| [`tool-roadmap.md`](./reference/tool-roadmap.md) | The audited work list behind it — defects, generality gaps, packaging, order |
| [`reusable-generator.md`](./reference/reusable-generator.md) | Earlier scoping of the same question, kept for the reasoning |
