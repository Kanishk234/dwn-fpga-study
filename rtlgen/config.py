"""A sweep point, as a first-class object.

Phase 1 had exactly one configuration, so "config" meant the `CONFIG` dict baked into a
checkpoint and everything else was a module-level constant. Phase 2 runs 40-70 configurations,
which breaks that in two ways:

  1. The checkpoint knows only about TRAINING. Pipeline depth, clock target, part and synthesis
     strategy are hardware choices made after training, and Group B sweeps exactly those on an
     already-trained model (docs/dse-plan.md §3). They have to live somewhere, and the
     checkpoint is the wrong place -- the same checkpoint feeds several hardware configs.
  2. Every emitted path was a constant. Sweep point #7 would overwrite #8.

So a Config is (model params from the checkpoint) + (hardware params chosen per run) + the
derived output directory that keeps those runs apart.

DEFAULTS REPRODUCE PHASE 1 EXACTLY. That is the whole point of this step: the object can be
introduced without changing a single emitted bit, so `scripts/verify_phase1.py` stays at 22/22
while the plumbing moves underneath it. `python rtlgen/config.py` checks that claim against the
live constants in the modules that still own them -- if someone edits `PIPE_LUT` in emit_core.py
and not here, this fails rather than silently desynchronizing.

Usage:
    python rtlgen/config.py                     # self-test: defaults match Phase 1
    python rtlgen/config.py <checkpoint.pt>     # show the config that checkpoint implies
"""

import os
import sys
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class ModelConfig:
    """What was trained. Read from the checkpoint, never invented here."""
    n: int                          # LUT inputs per node -- a Phase 2 sweep axis
    thermometer_bits: int           # z -- sets the encoder's saturation ceiling
    thermometer: str                # 'distributive' | 'gaussian' | 'linear'
    layers: Tuple[int, ...]         # LUT nodes per layer
    num_classes: int
    tau: float
    run_name: str = ''

    @property
    def nodes(self) -> int:
        return sum(self.layers)

    def __post_init__(self):
        # GroupSum zero-pads silently when this does not hold, and hardware and software then
        # disagree about group boundaries (docs/checkpoint-format.md §4). The emitter asserts
        # it too; catching it here means a bad sweep point dies before Vivado is ever launched.
        if self.layers[-1] % self.num_classes:
            raise ValueError(
                f'final layer {self.layers[-1]} is not divisible by num_classes '
                f'{self.num_classes} -- GroupSum would zero-pad and hardware would disagree '
                f'with software about group boundaries')


@dataclass(frozen=True)
class HardwareConfig:
    """How it gets built. Not in the checkpoint -- these are Group B sweep axes.

    Every default is the Phase 1 shipped value.
    """
    pipe_enc: int = 1               # register after the encoder
    pipe_lut: int = 1               # after the LUT layer
    pipe_pop: int = 1               # after the popcounts
    pipe_out: int = 1               # after the argmax
    word_bits: int = 16             # Q3.12 signed: 16-bit word...
    frac_bits: int = 12             # ...12 fractional bits
    part: str = 'xc7a35tcpg236-1'
    clock_ns: float = 10.0          # the Basys 3's 100 MHz
    strategy: Optional[str] = None  # Vivado directive; None = tool default

    @property
    def latency(self) -> int:
        """Cycles from feature-in to class-out.

        `benchmark_fsm` aligns labels using this, and a drifted value silently scores every
        sample against the wrong answer -- a bug that has already happened once in this project.
        It is derived, never hand-copied.
        """
        return self.pipe_enc + self.pipe_lut + self.pipe_pop + self.pipe_out

    @property
    def pipe_slug(self) -> str:
        return f'{self.pipe_enc}{self.pipe_lut}{self.pipe_pop}{self.pipe_out}'


@dataclass(frozen=True)
class Config:
    """One sweep point: a trained model plus the hardware choices used to build it."""
    model: ModelConfig
    hw: HardwareConfig = field(default_factory=HardwareConfig)
    checkpoint: str = ''

    @property
    def name(self) -> str:
        """Filesystem-safe slug, deterministic, carrying every axis that distinguishes a point.

        Two configs that differ in any swept axis must get different names, or one overwrites
        the other's build directory -- which is the failure this whole object exists to prevent.
        """
        w = 'x'.join(str(x) for x in self.model.layers)
        return (f'n{self.model.n}'
                f'_z{self.model.thermometer_bits}'
                f'_{self.model.thermometer}'
                f'_w{w}'
                f'_q{self.hw.word_bits}.{self.hw.frac_bits}'
                f'_p{self.hw.pipe_slug}'
                f'_c{self.hw.clock_ns:g}')

    @property
    def build_dir(self) -> str:
        """Everything this config generates. Under build/, which is gitignored in full."""
        return os.path.join(REPO, 'build', 'configs', self.name)

    @property
    def rtl_dir(self) -> str:
        """Emitted Verilog for this config -- the replacement for the committed rtl/gen/."""
        return os.path.join(self.build_dir, 'rtl')

    def with_hw(self, **kw) -> 'Config':
        """A Group B variant: same trained model, different hardware choices, no retraining."""
        return replace(self, hw=replace(self.hw, **kw))

    @classmethod
    def from_checkpoint(cls, path: str, **hw_kw) -> 'Config':
        import torch
        ck = torch.load(path, map_location='cpu', weights_only=False)
        c = ck['config']
        model = ModelConfig(
            n=c['n'],
            thermometer_bits=c['thermometer_bits'],
            thermometer=c['thermometer'],
            layers=tuple(c['layers']),
            num_classes=c['num_classes'],
            tau=c['tau'],
            run_name=ck.get('run_name', ''),
        )
        return cls(model=model, hw=HardwareConfig(**hw_kw), checkpoint=path)


# --------------------------------------------------------------------------------------------
# Self-test. The defaults above duplicate constants that still live in the modules that own
# them; this is what stops the two from drifting apart before those modules are converted.
# --------------------------------------------------------------------------------------------

def _selftest() -> int:
    # emit_core is here in rtlgen/; extract stays in exporter/ (brief §11 splits them).
    sys.path.insert(0, os.path.join(REPO, 'rtlgen'))
    sys.path.insert(0, os.path.join(REPO, 'exporter'))
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    import emit_core
    import extract
    import run_synth

    hw = HardwareConfig()
    checks = [
        ('pipe_lut  vs emit_core.PIPE_LUT', hw.pipe_lut, emit_core.PIPE_LUT),
        ('pipe_pop  vs emit_core.PIPE_POP', hw.pipe_pop, emit_core.PIPE_POP),
        ('pipe_out  vs emit_core.PIPE_OUT', hw.pipe_out, emit_core.PIPE_OUT),
        ('word_bits vs extract.WORD_BITS', hw.word_bits, extract.WORD_BITS),
        ('frac_bits vs extract.FRAC_BITS', hw.frac_bits, extract.FRAC_BITS),
        ('part      vs run_synth.DEFAULT_PART', hw.part, run_synth.DEFAULT_PART),
        ('latency   vs Phase 1 measured', hw.latency, 4),
    ]
    bad = 0
    for label, got, want in checks:
        ok = got == want
        bad += not ok
        print(f'  {"OK  " if ok else "FAIL"}  {label:38s} {got!r:22s} (expected {want!r})')

    print()
    if bad:
        print(f'{bad} MISMATCH(ES) -- a hardware default drifted from the module that owns it.')
        print('Fix before generating anything: emitted RTL and this config would disagree.')
        return 1
    print('defaults reproduce Phase 1 exactly.')
    return 0


def main() -> int:
    if len(sys.argv) > 1:
        cfg = Config.from_checkpoint(sys.argv[1])
        print(f'run       : {cfg.model.run_name}')
        print(f'model     : n={cfg.model.n} z={cfg.model.thermometer_bits} '
              f'{cfg.model.thermometer} layers={list(cfg.model.layers)} '
              f'classes={cfg.model.num_classes} ({cfg.model.nodes} nodes)')
        print(f'hardware  : Q{cfg.hw.word_bits-1-cfg.hw.frac_bits}.{cfg.hw.frac_bits} '
              f'pipe={cfg.hw.pipe_slug} latency={cfg.hw.latency} '
              f'part={cfg.hw.part} clock={cfg.hw.clock_ns}ns')
        print(f'name      : {cfg.name}')
        print(f'rtl_dir   : {os.path.relpath(cfg.rtl_dir, REPO)}')
        print()
        print('a Group B variant (no retraining):')
        v = cfg.with_hw(pipe_out=0)
        print(f'  name    : {v.name}')
        print(f'  latency : {v.hw.latency} cycles')
        return 0

    print('=== config defaults vs the constants still owned by other modules ===')
    return _selftest()


if __name__ == '__main__':
    sys.exit(main())
