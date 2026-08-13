"""Run Gate 1 end to end: checkpoint -> core RTL -> test vectors -> xsim -> pass/fail.

Gate 1 is the only correctness signal in this project (CLAUDE.md), so it must be trivial to
re-run and must rebuild everything it checks. It regenerates the core and the vectors from the
checkpoint every time rather than trusting whatever is on disk -- a stale dwn_core.v passing
against stale vectors would be worse than no check at all.

This is Python rather than PowerShell because Phase 2's sweep automation (`dse/`) is Python and
has to drive Vivado the same way. `run_xsim()` and `find_vivado_bin()` are importable so the
sweep can reuse them instead of shelling out to a script and parsing its stdout.

Usage:
    .venv\\Scripts\\python.exe scripts/run_gate1.py
    .venv\\Scripts\\python.exe scripts/run_gate1.py --checkpoint training/artifacts/<other>.pt
    .venv\\Scripts\\python.exe scripts/run_gate1.py --vivado-bin D:/path/to/Vivado/bin
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = os.path.join(
    'training', 'artifacts', 'dwn_jsc_t200_distributive_50_l_b100_checkpoint.pt')

# Vivado is NOT at C:\Xilinx on this machine, and its bin directory is not on PATH.
VIVADO_CANDIDATES = [
    r'C:\AMDDesignTools\2025.2\Vivado\bin',
    r'C:\Xilinx\Vivado\2025.2\bin',
]

# Where the emitters write by default. Phase 2 overrides this per config (`Config.rtl_dir`),
# which is what stops sweep point #7 from overwriting #8.
DEFAULT_RTL_DIR = os.path.join(REPO, 'build', 'rtl')

# Hand-written primitives -- identical for every config, so not under rtl_dir.
PRIMITIVES = ['lut_node.v', 'popcount.v', 'argmax.v', 'pipe_reg.v']
GENERATED = ['dwn_core.v', 'thermometer_encoder.v', 'dwn_top.v']
TESTBENCH_SRC = ['dwn_core_tb.v', 'dwn_top_tb.v']


def rtl_sources(rtl_dir=None):
    """Full source list for Gate 1, with generated RTL taken from `rtl_dir`.

    Importable so `dse/` can build a per-config source list without duplicating the split
    between hand-written primitives and emitted files.
    """
    rtl_dir = rtl_dir or DEFAULT_RTL_DIR
    return ([os.path.join(REPO, 'rtl', f) for f in PRIMITIVES] +
            [os.path.join(rtl_dir, f) for f in GENERATED] +
            [os.path.join(REPO, 'tb', f) for f in TESTBENCH_SRC])

# Both levels run. The core testbench drives pre-binarized bits; the top testbench drives
# quantized features through the encoder as well. Keeping them separate is what makes a
# failure localize itself instead of just saying "the design is wrong".
TESTBENCHES = [
    ('dwn_core_tb', 'core  (pre-binarized bits -> class)'),
    ('dwn_top_tb', 'top   (quantized features -> encoder -> class)'),
]


def find_vivado_bin(explicit=None):
    """Locate Vivado's bin directory, or raise with something actionable."""
    if explicit:
        if not os.path.isdir(explicit):
            raise SystemExit(f'--vivado-bin path does not exist: {explicit}')
        return explicit
    for path in VIVADO_CANDIDATES:
        if os.path.isdir(path):
            return path
    on_path = shutil.which('xvlog') or shutil.which('xvlog.bat')
    if on_path:
        return os.path.dirname(on_path)
    raise SystemExit(
        'Vivado not found. Looked in:\n  ' + '\n  '.join(VIVADO_CANDIDATES) +
        '\nPass --vivado-bin <path> if it lives somewhere else.')


def python_exe():
    """The venv interpreter -- not necessarily the one running this script."""
    if sys.prefix != sys.base_prefix:
        return sys.executable          # already inside a venv
    for rel in (os.path.join('.venv', 'Scripts', 'python.exe'),
                os.path.join('.venv', 'bin', 'python')):
        cand = os.path.join(REPO, rel)
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        'No .venv found. Run:\n'
        '  py -3.12 -m venv .venv\n'
        '  .venv\\Scripts\\activate\n'
        '  pip install -r requirements.txt')


def tool(vivado_bin, name):
    """Full path to a Vivado tool, .bat on Windows."""
    exe = os.path.join(vivado_bin, name + ('.bat' if os.name == 'nt' else ''))
    return exe if os.path.exists(exe) else os.path.join(vivado_bin, name)


def run(cmd, cwd=None, env=None, capture=False):
    # .bat files are not directly executable via CreateProcess, so they go through cmd /c.
    if os.name == 'nt' and str(cmd[0]).lower().endswith('.bat'):
        cmd = ['cmd', '/c'] + list(cmd)
    # Subprocesses inherit stdout and write immediately; our own prints are buffered. Without
    # this flush the step headers appear after the output they are labelling.
    sys.stdout.flush()
    return subprocess.run(cmd, cwd=cwd, env=env, text=True,
                          capture_output=capture, check=False)


def xvlog(vivado_bin, work, sources, include=None):
    """Analyze all sources once. Returns (ok, output).

    `include` may be a list: the testbenches pull vector counts from build/gate1 and pipeline
    latency from the emitted RTL directory, so both have to be on the include path.
    """
    env = dict(os.environ)
    env['PATH'] = vivado_bin + os.pathsep + env.get('PATH', '')
    os.makedirs(work, exist_ok=True)
    cmd = [tool(vivado_bin, 'xvlog')]
    for inc in ([include] if isinstance(include, str) else (include or [])):
        cmd += ['-i', inc]
    cmd += list(sources)
    r = run(cmd, cwd=work, env=env, capture=True)
    return r.returncode == 0, r.stdout + r.stderr


def run_xsim(vivado_bin, work, top, snapshot=None):
    """Elaborate and run one top. Returns (ok, combined_output).

    Runs with `work` as the working directory: xsim drops xsim.dir/, logs and .pb files into
    the cwd, and nothing outside build/ should accumulate generated artifacts (CLAUDE.md).
    """
    env = dict(os.environ)
    env['PATH'] = vivado_bin + os.pathsep + env.get('PATH', '')
    snapshot = snapshot or (top + '_sim')

    r = run([tool(vivado_bin, 'xelab'), top, '-s', snapshot, '-debug', 'off'],
            cwd=work, env=env, capture=True)
    if r.returncode != 0:
        return False, r.stdout + r.stderr

    r = run([tool(vivado_bin, 'xsim'), snapshot, '-runall'],
            cwd=work, env=env, capture=True)
    return r.returncode == 0, r.stdout + r.stderr


def gate1(ckpt, vivado_bin, rtl_dir=None, work=None, pipe=None, quiet=False,
          word_bits=None, frac_bits=None):
    """Run Gate 1 for ONE config. Returns (ok, info).

    Importable so `dse/` can gate every sweep point on correctness without shelling out and
    scraping stdout. That matters more here than convenience: Gate 1 is what makes an area
    number mean anything (CLAUDE.md), so the sweep must be able to refuse to synthesize a
    config whose RTL has not been proven to match the golden model.

    `pipe` is a dict with any of lut/pop/out/enc; omitted keys keep the emitter defaults.
    `info` carries the per-level vector counts so a caller can record them per config.

    `word_bits`/`frac_bits` set the input fixed-point format; None keeps the exporter default.
    They are passed to the encoder emitter AND the vector generator together, because those two
    must quantise identically -- give them different widths and Gate 1 compares two different
    designs and still reports PASS.
    """
    py = python_exe()
    pipe = pipe or {}
    rtl_dir = os.path.abspath(rtl_dir) if rtl_dir else DEFAULT_RTL_DIR
    work = os.path.abspath(work) if work else os.path.join(REPO, 'build', 'gate1')

    def say(*a):
        if not quiet:
            print(*a)

    def pipe_flag(name, key):
        v = pipe.get(key)
        return [] if v is None else [f'--{name}', str(v)]

    core_flags = (pipe_flag('pipe-lut', 'lut') + pipe_flag('pipe-pop', 'pop') +
                  pipe_flag('pipe-out', 'out'))
    # One list, used for both the encoder and the golden model. Building it once is deliberate:
    # it makes it impossible to widen one and not the other.
    prec_flags = ([] if word_bits is None else ['--word-bits', str(word_bits)]) +                  ([] if frac_bits is None else ['--frac-bits', str(frac_bits)])
    info = {'rtl_dir': rtl_dir, 'work': work}

    say('=== 1/4  emit core RTL from checkpoint ===')
    if run([py, os.path.join(REPO, 'rtlgen', 'emit_core.py'), ckpt,
            '--outdir', rtl_dir] + core_flags, capture=quiet).returncode != 0:
        return False, dict(info, error='emit_core.py failed')

    say('\n=== 2/4  emit encoder + top RTL from checkpoint ===')
    if run([py, os.path.join(REPO, 'rtlgen', 'emit_encoder.py'), ckpt,
            '--outdir', rtl_dir] + pipe_flag('pipe-enc', 'enc') + prec_flags,
           capture=quiet).returncode != 0:
        return False, dict(info, error='emit_encoder.py failed')

    say('\n=== 3/4  generate test vectors ===')
    if run([py, os.path.join(REPO, 'tb', 'gen_vectors.py'), ckpt,
            '--outdir', work] + prec_flags, capture=quiet).returncode != 0:
        return False, dict(info, error='gen_vectors.py failed')

    # Latency is read from the emitted header, so a swept pipeline depth needs no special
    # handling here -- the golden model follows the RTL automatically.
    m = re.search(r'`define DWN_TOP_LATENCY (\d+)',
                  open(os.path.join(rtl_dir, 'dwn_top_params.vh')).read())
    info['latency'] = int(m.group(1)) if m else None

    say('\n=== 4/4  simulate (xsim) ===')
    ok, output = xvlog(vivado_bin, work, rtl_sources(rtl_dir), include=[work, rtl_dir])
    if not ok:
        say(output.strip())
        say('\nGATE 1 FAILED (compile error)')
        return False, dict(info, error='compile error')

    failures = []
    for top, label in TESTBENCHES:
        say(f'\n--- {label} ---')
        ok, output = run_xsim(vivado_bin, work, top)
        # The testbench prints its own verdict. $finish always exits 0, so pass/fail has to be
        # read out of the output rather than the exit code.
        verdict = re.search(r'RESULT\s+:\s+(PASS|FAIL)', output)
        n = re.search(r'vectors tested\s*:\s*(\d+)', output)
        info[f'{top}_vectors'] = int(n.group(1)) if n else None
        for line in output.splitlines():
            if re.search(r'====|GATE 1|vectors tested|mismatches|RESULT|MISMATCH', line):
                say(line)
        if not ok or not verdict or verdict.group(1) != 'PASS':
            failures.append(top)
            if not ok:
                say(output.strip()[-2000:])

    say()
    if failures:
        say(f'GATE 1 FAILED ({", ".join(failures)})')
        return False, dict(info, error=f'gate1 failed: {", ".join(failures)}')
    say('GATE 1 PASSED (both levels)')
    return True, info


def main():
    ap = argparse.ArgumentParser(description='Run Gate 1 (golden-model testbench).')
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    ap.add_argument('--vivado-bin', default=None)
    ap.add_argument('--rtl-dir', default=None,
                    help='where to emit and read generated RTL (default: build/rtl)')
    ap.add_argument('--work', default=None,
                    help='simulation working directory (default: build/gate1)')
    # Forwarded to the emitters. Gate 1 has to be runnable on a SWEPT pipeline depth, or a
    # Group B point gets synthesized without ever being proven correct -- the golden model
    # reads latency from the emitted params, so it follows automatically.
    ap.add_argument('--pipe-lut', type=int, default=None)
    ap.add_argument('--pipe-pop', type=int, default=None)
    ap.add_argument('--pipe-out', type=int, default=None)
    ap.add_argument('--pipe-enc', type=int, default=None)
    ap.add_argument('--word-bits', type=int, default=None,
                    help='signed input word width (default: the exporter constant, 16)')
    ap.add_argument('--frac-bits', type=int, default=None,
                    help='fractional bits in the input word (default: 12)')
    args = ap.parse_args()

    vivado_bin = find_vivado_bin(args.vivado_bin)
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) \
        else os.path.join(REPO, args.checkpoint)
    if not os.path.exists(ckpt):
        raise SystemExit(f'checkpoint not found: {ckpt}')

    rtl_dir = os.path.abspath(args.rtl_dir) if args.rtl_dir else DEFAULT_RTL_DIR
    if rtl_dir != DEFAULT_RTL_DIR:
        print(f'rtl dir : {os.path.relpath(rtl_dir, REPO)}')

    ok, _ = gate1(ckpt, vivado_bin, rtl_dir=args.rtl_dir, work=args.work,
                  word_bits=args.word_bits, frac_bits=args.frac_bits,
                  pipe={'lut': args.pipe_lut, 'pop': args.pipe_pop,
                        'out': args.pipe_out, 'enc': args.pipe_enc})
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
