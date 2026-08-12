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
from dataclasses import dataclass


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

    notes: str = ''

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
    size_ladder=(100, 200, 300, 500, 1000),
    z_values=(8, 25, 50, 100),
    default_z=25,
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
