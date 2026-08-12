"""Per-dataset facts, in one place, as data.

The rule this package exists to enforce: **nothing under `exporter/`, `rtlgen/`, `rtl/`, `tb/`,
`scripts/` or `harness/` may contain a dataset's dimensions.** Not 16, not 5, not 784. Those files
describe how a DWN becomes hardware; they must not know which problem it was trained on.

The test that it is working: adding a third dataset should mean adding a `Dataset` below and a
`docs/<name>/` directory, and editing no shared code. If a third dataset would require touching an
emitter, the boundary has leaked and the fix belongs here.

    from datasets import get
    ds = get('jsc')
    ds.record_bytes()        # what one board record costs

Descriptors are *defaults and metadata*, not the source of truth for a trained model. Where a
checkpoint states something itself -- feature count, threshold count, class count -- the checkpoint
wins, and `Dataset.check_checkpoint()` exists to catch a mismatch loudly rather than let a wrong
descriptor produce a valid-looking wrong export.
"""
import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainingRecipe:
    """Hyperparameters a sweep trains at. NOT swept -- they do not change the hardware.

    Dataset-specific in practice: JSC reproduces the paper at 32 epochs, MNIST's grid uses 30.
    Kept together so a notebook consumes one object and cannot drift from the grid it came from.
    """
    batch_size: int = 100
    epochs: int = 32
    lr: float = 1e-2
    lr_step: int = 14
    lr_gamma: float = 0.1
    seed: int = 20260802


@dataclass(frozen=True)
class Dataset:
    """Everything the flow needs to know about a dataset that is not in the checkpoint."""

    name: str
    features: int                      # input features per sample
    classes: int                       # output classes
    openml_name: str = ''              # how the raw data is fetched, for the training notebooks
    scaling: str = 'standard'          # 'standard' (zero mean, unit variance) or 'minmax'

    # Fixed-point format for the encoder's input word. Defaults are the JSC-era Q3.12. Encoder
    # area is dominated by comparator width, so a dataset whose inputs are natively narrow should
    # say so -- see docs/mnist/phase1-ledger.md, 2026-08-11.
    word_bits: int = 16
    frac_bits: int = 12

    # How the canonical test split is taken from the fetched data, for scripts/dump_testset.py.
    # 'tail:N' = the last N rows, which is the standard split for datasets published in
    # train-then-test order. Empty means the split is not reproducible from the raw data alone
    # (e.g. a seeded random split done in a training notebook) -- the dump script then refuses
    # rather than inventing a different split and calling it the test set.
    test_split: str = ''

    # Sweep axes, as data. dse/ walks these; it does not define them.
    size_ladder: tuple = ()
    z_values: tuple = ()
    default_z: int = 200

    # Baseline point the one-factor-at-a-time axes vary around.
    base_n: int = 6
    base_encoding: str = 'distributive'

    # ---- tau schedule ----
    # tau tracks layer width; it is not a constant to copy from the smallest config. Getting it
    # wrong does not fail loudly, it just trains a worse model, and the sweep point then reports
    # an accuracy that says more about tau than about the architecture.
    #
    # `tau_anchors` is ((width, tau), ...) ascending. With two or more, tau_for() interpolates
    # GEOMETRICALLY between them -- log(tau) linear in log(width) -- because the schedule is a
    # power law in width, not a linear function of it. With exactly one anchor there is nothing
    # to interpolate, so `tau_exponent` supplies the slope: tau = anchor * (w/anchor_w)**exp.
    # MNIST has exactly one published anchor, and the reduction study confirmed JSC's measured
    # 0.57 exponent transfers to it (docs/mnist/reduction-ledger.md).
    tau_anchors: tuple = ()
    tau_exponent: float = 0.57

    # ---- area-model calibration ----
    # LUTs per comparator BIT in the thermometer encoder. dse/area_model.py multiplies this by
    # comparators x word_bits.
    #
    # ⚠️ THIS IS NOT A CONSTANT AND THE SINGLE NUMBER HERE IS A DEFAULT, NOT A LAW. Measured:
    #
    #   dataset  config       comps  comps/feature  LUT/bit
    #   JSC      1x50 z=200     202          12.62   0.4700
    #   MNIST    1x1000 z=1     431           0.55   0.2181
    #   MNIST    1x1000 z=3     720           0.92   0.1417
    #   MNIST    1x1000 z=25   3400           4.34   0.0863
    #
    # Within MNIST it falls monotonically as comparators-per-feature rises -- logic sharing
    # between comparisons against different constants on the SAME input word. Between datasets
    # that does not explain it: JSC has the most comparators per feature and is still the most
    # expensive per bit, which points at threshold VALUES (MNIST pixels are mostly zero, so
    # quantile thresholds cluster near zero and those comparisons collapse to a few bits).
    #
    # Two mechanisms, previously conflated into one falsified "amortisation" claim. A z-dependent
    # model is the right fix and is NOT done -- so an MNIST prediction away from `default_z` is
    # wrong by up to 2.5x. See docs/mnist/phase2-ledger.md.
    lut_per_comparator_bit: float = 1519 / (202 * 16)

    # WHICH width feeds tau_for(). 'final' is the physically motivated choice -- GroupSum's
    # logit range is (final_layer / classes) / tau, so the final layer is what sets it.
    # ⚠️ JSC uses 'total' because that is what its published sweep was built with, and changing
    # it would silently move every multi-layer JSC config. Preserved for reproducibility, not
    # because it is right. For a single-layer config the two are identical.
    tau_basis: str = 'total'

    training: TrainingRecipe = field(default_factory=TrainingRecipe)

    # Sweep strategy. Widths the one-factor-at-a-time axes and the Group B pipeline variants are
    # run at, and (layers, z) corners where two axes are taken at their knees together.
    ofat_rungs: tuple = ()
    group_b_rungs: tuple = ()
    corners: tuple = ()

    notes: str = ''

    def tau_for(self, width) -> float:
        """The training tau for a layer of this width. See `tau_anchors`."""
        if not self.tau_anchors:
            raise ValueError(f'{self.name} has no tau anchors; a sweep cannot pick tau for it')
        if len(self.tau_anchors) == 1:
            w0, t0 = self.tau_anchors[0]
            return t0 * (width / w0) ** self.tau_exponent
        if width <= self.tau_anchors[0][0]:
            return self.tau_anchors[0][1]
        if width >= self.tau_anchors[-1][0]:
            return self.tau_anchors[-1][1]
        for (w0, t0), (w1, t1) in zip(self.tau_anchors, self.tau_anchors[1:]):
            if w0 <= width <= w1:
                f = (math.log(width) - math.log(w0)) / (math.log(w1) - math.log(w0))
                return math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))
        return self.tau_anchors[-1][1]

    def tau_width(self, layers) -> int:
        """The width to hand tau_for(), per this dataset's `tau_basis`."""
        return sum(layers) if self.tau_basis == 'total' else layers[-1]

    @property
    def int_bits(self) -> int:
        """Integer bits, excluding the sign bit."""
        return self.word_bits - 1 - self.frac_bits

    @property
    def fixed_point(self) -> str:
        return f'Q{self.int_bits}.{self.frac_bits}'

    def bytes_per_feature(self, word_bits=None) -> int:
        """Wire width of one feature, in whole bytes.

        CEILING, not floor. A word that is not a byte multiple pads on the wire -- 11 bits
        travels as 2 bytes. Using `// 8` here gives 1 byte for an 11-bit word and silently
        truncates every feature. `scripts/host.py`'s pack_record still makes that assumption and
        is fixed in M1f, together with the Verilog loader it shares a wire format with.
        """
        w = self.word_bits if word_bits is None else word_bits
        return -(-w // 8)

    def record_bytes(self, word_bits=None) -> int:
        """One board record: every feature padded to whole bytes, then a single label byte.

        Derived, never a constant. JSC at 16 features and a 16-bit word gives the 33 bytes the
        UART loader was originally written around.
        """
        return self.features * self.bytes_per_feature(word_bits) + 1

    def thermometer_bits(self, z=None) -> int:
        """Total encoder output width: one bit per threshold per feature."""
        return self.features * (self.default_z if z is None else z)

    def check_checkpoint(self, ck) -> None:
        """Fail loudly if a checkpoint disagrees with this descriptor.

        A silently wrong descriptor is worse than a missing one: it produces an export that
        elaborates, synthesizes, and is wrong.
        """
        thr = ck['thermometer']['thresholds']
        feats = thr.shape[0]
        classes = ck['config']['num_classes']
        if feats != self.features:
            raise ValueError(
                f'{self.name}: checkpoint has {feats} features, descriptor says {self.features}')
        if classes != self.classes:
            raise ValueError(
                f'{self.name}: checkpoint has {classes} classes, descriptor says {self.classes}')


JSC = Dataset(
    name='jsc',
    features=16,
    classes=5,
    openml_name='hls4ml_lhc_jets_hlf',
    scaling='standard',
    word_bits=16,
    frac_bits=12,
    size_ladder=(50, 100, 200, 360, 500, 600, 800, 1200, 1600, 2000),
    z_values=(8, 25, 50, 100, 200, 400, 800),
    default_z=200,
    base_n=6,
    base_encoding='distributive',
    # The paper's four JSC anchors (docs/paper-configs.md). Their log-log slopes are 0.526,
    # 0.557 and 0.635 -- near-constant, which is where the 0.57 exponent comes from.
    tau_anchors=((10, 1 / 0.7), (50, 1 / 0.3), (360, 1 / 0.1), (2400, 1 / 0.03)),
    tau_basis='total',        # see the field comment -- preserved, not endorsed
    training=TrainingRecipe(batch_size=100, epochs=32, lr=1e-2, lr_step=14, lr_gamma=0.1,
                            seed=20260802),
    ofat_rungs=(200, 360),
    group_b_rungs=(50, 360, 600, 1600),
    corners=(((800,), 50), ((1200,), 50), ((1600,), 100),
             ((2400,), 100), ((2400,), 50), ((3000,), 50)),
    notes=('Jet substructure classification, the OpenML distribution (dataset 42468) -- NOT the '
           'CERNBox one, which scores about 1.05 pp lower and is what the LogicNets/PolyLUT/'
           'NeuraLUT line of work uses. See REPORT.md section 5.1.'),
)

MNIST = Dataset(
    name='mnist',
    features=784,
    classes=10,
    openml_name='mnist_784',
    scaling='minmax',
    # mnist_784 is published in train-then-test order, so the last 10,000 rows are the canonical
    # test split every published MNIST number is measured on. Do NOT shuffle before slicing.
    test_split='tail:10000',
    # Q0.8: one sign bit, eight fractional bits, range [-1, 1). Chosen because it represents
    # min-max scaled 8-bit pixels EXACTLY -- there are only 256 distinct input values, so nine
    # bits loses nothing, unlike on JSC where narrowing truncated continuous features and cost
    # 0.4 pp. It also sits on the cheap side of the measured area cliff (that is between 12 and
    # 11 bits on JSC). Provisional until measured here: on JSC the safe width moved between two
    # configurations of the same dataset, so it cannot be assumed to transfer.
    word_bits=9,
    frac_bits=8,
    # The trained ladder goes to 2000; z=1 and z=2 are in the grid because the reduction
    # study found z nearly free (0.32 pp across a 25x encoder), making the low end the
    # interesting end rather than an afterthought.
    size_ladder=(100, 200, 300, 500, 1000, 2000),
    z_values=(1, 2, 3, 8, 25),
    default_z=3,              # upstream's own MNIST example uses DistributiveThermometer(3)
    base_n=6,
    base_encoding='distributive',
    # ONE anchor: upstream's `examples/mnist.py` uses tau = 1/0.3 on a 1000-wide final layer.
    # One anchor cannot fit an exponent, so JSC's measured 0.57 is borrowed -- and the reduction
    # study confirmed it transfers, beating a flat-logit-range schedule by 1.62 and 0.76 pp on
    # two architectures. Reproduces the `tau1p678` checkpoint's value at width 300 exactly.
    tau_anchors=((1000, 1 / 0.3),),
    tau_exponent=0.57,
    # Measured at z=3 (= default_z), 1x1000: 918 LUTs / (720 comparators x 9 bits).
    lut_per_comparator_bit=918 / (720 * 9),
    tau_basis='final',        # GroupSum's logit range is set by the FINAL layer, not the sum
    training=TrainingRecipe(batch_size=100, epochs=30, lr=1e-2, lr_step=14, lr_gamma=0.1,
                            seed=20260811),
    ofat_rungs=(1000,),       # the tau anchor; every off-anchor rung needs its own tau
    group_b_rungs=(300, 1000, 2000),
    corners=(((2000, 1000), 3), ((1000,), 1)),
    notes=('MNIST is slot-limited rather than pool-limited: 784 x z far exceeds the input slots '
           'of any layer that fits, so z costs far less area here than on JSC (z=8 to z=200 is '
           'only 2.3x the comparators). Word width, not z, decides whether a model fits -- at '
           '16 bits the paper configuration is over the device at every z. See '
           'docs/mnist/phase1-ledger.md, 2026-08-11.'),
)

_ALL = {d.name: d for d in (JSC, MNIST)}


def get(name: str) -> Dataset:
    try:
        return _ALL[name.lower()]
    except KeyError:
        raise KeyError(f'unknown dataset {name!r}; known: {sorted(_ALL)}') from None


def names() -> list:
    return sorted(_ALL)


def identify(ck) -> Dataset:
    """Which dataset a checkpoint belongs to, from its own shape.

    THIS IS THE FUNCTION THAT MAKES THE PACKAGE LOAD-BEARING. Before it existed, `datasets/` held
    the right facts and nothing imported it: every consumer kept a private copy of JSC's numbers,
    so each new dataset surfaced them one crash at a time -- six separate sites on this branch,
    all the same defect wearing different clothes.

    Matching is on (features, classes), which are structural: they are fixed by the trained model
    and cannot be set wrong without the export failing anyway. Deliberately NOT on the filename --
    a slug is a naming convention, and resolving behaviour through one means a renamed checkpoint
    quietly changes its quantisation.

    An unknown shape is an error with instructions, not a silent fallback to JSC's Q3.12. A
    fallback is exactly how a dataset gets exported at another dataset's precision and produces a
    model that elaborates, synthesizes, and is wrong.
    """
    thr = ck['thermometer']['thresholds']
    feats, classes = thr.shape[0], ck['config']['num_classes']

    hits = [d for d in _ALL.values() if d.features == feats and d.classes == classes]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError(
            f'no dataset descriptor matches this checkpoint ({feats} features, {classes} '
            f'classes). Known: ' + ', '.join(f'{d.name} ({d.features}x{d.classes})'
                                             for d in _ALL.values()) +
            '.\nAdd a Dataset to datasets/__init__.py -- that is the whole change; no emitter, '
            'testbench or script should need editing.')
    raise KeyError(
        f'{feats} features x {classes} classes matches more than one descriptor '
        f'({", ".join(d.name for d in hits)}), so precision cannot be resolved from shape alone. '
        'Give the dataset explicitly.')
