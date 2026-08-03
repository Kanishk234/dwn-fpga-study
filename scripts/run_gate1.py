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

RTL_SOURCES = [
    ('rtl', 'lut_node.v'),
    ('rtl', 'popcount.v'),
    ('rtl', 'argmax.v'),
    ('rtl', 'gen', 'dwn_core.v'),
    ('rtl', 'gen', 'thermometer_encoder.v'),
    ('rtl', 'gen', 'dwn_top.v'),
    ('tb', 'dwn_core_tb.v'),
    ('tb', 'dwn_top_tb.v'),
]

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
    """Analyze all sources once. Returns (ok, output)."""
    env = dict(os.environ)
    env['PATH'] = vivado_bin + os.pathsep + env.get('PATH', '')
    os.makedirs(work, exist_ok=True)
    cmd = [tool(vivado_bin, 'xvlog')]
    if include:
        cmd += ['-i', include]
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


def main():
    ap = argparse.ArgumentParser(description='Run Gate 1 (golden-model testbench).')
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    ap.add_argument('--vivado-bin', default=None)
    args = ap.parse_args()

    py = python_exe()
    vivado_bin = find_vivado_bin(args.vivado_bin)
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) \
        else os.path.join(REPO, args.checkpoint)
    if not os.path.exists(ckpt):
        raise SystemExit(f'checkpoint not found: {ckpt}')
    work = os.path.join(REPO, 'build', 'gate1')

    print('=== 1/4  emit core RTL from checkpoint ===')
    if run([py, os.path.join(REPO, 'exporter', 'emit_core.py'), ckpt]).returncode != 0:
        raise SystemExit('emit_core.py failed')

    print('\n=== 2/4  emit encoder + top RTL from checkpoint ===')
    if run([py, os.path.join(REPO, 'exporter', 'emit_encoder.py'), ckpt]).returncode != 0:
        raise SystemExit('emit_encoder.py failed')

    print('\n=== 3/4  generate test vectors ===')
    if run([py, os.path.join(REPO, 'tb', 'gen_vectors.py'), ckpt]).returncode != 0:
        raise SystemExit('gen_vectors.py failed')

    print('\n=== 4/4  simulate (xsim) ===')
    sources = [os.path.join(REPO, *parts) for parts in RTL_SOURCES]
    ok, output = xvlog(vivado_bin, work, sources, include=work)
    if not ok:
        print(output.strip())
        print('\nGATE 1 FAILED (compile error)')
        return 1

    failures = []
    for top, label in TESTBENCHES:
        print(f'\n--- {label} ---')
        ok, output = run_xsim(vivado_bin, work, top)
        # The testbench prints its own verdict. $finish always exits 0, so pass/fail has to be
        # read out of the output rather than the exit code.
        verdict = re.search(r'RESULT\s+:\s+(PASS|FAIL)', output)
        for line in output.splitlines():
            if re.search(r'====|GATE 1|vectors tested|mismatches|RESULT|MISMATCH', line):
                print(line)
        if not ok or not verdict or verdict.group(1) != 'PASS':
            failures.append(top)
            if not ok:
                print(output.strip()[-2000:])

    print()
    if failures:
        print(f'GATE 1 FAILED ({", ".join(failures)})')
        return 1
    print('GATE 1 PASSED (both levels)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
