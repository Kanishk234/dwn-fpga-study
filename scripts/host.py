"""Host-side driver for the DWN board: quantize, load, run, and score over UART.

This is the other half of harness/uart_loader.v. It exists to make Gate 1b runnable: the full
JSC test set is 166,000 samples and the device holds ~1024, so the run has to be batched, with
accuracy accumulated on-chip and only totals coming back.

    python scripts/host.py --selftest                     # no board needed
    python scripts/host.py --port COM4 --ping
    python scripts/host.py --port COM4 --gate1b

--selftest is the important one today. It rebuilds the exact byte stream this script would put
on the wire and checks it against the vectors the RTL testbenches were verified with. If the
host packs features differently from tb/gen_vectors.py, every feature arrives transposed and
the board reports garbage that looks like a hardware fault -- so the encoding is checked
against a known-good reference rather than against my reading of the protocol.

Protocol (see harness/uart_loader.v):
    'P'                              -> 0xA5
    'L' n_lo n_hi  + n x 33 bytes    load n records from address 0
    'R' n_lo n_hi                    run a batch
    'S'                              -> 9 bytes: cycles[4] correct[4] flags[1], little-endian

A record is 32 feature bytes (little-endian Q3.12, feature f at bytes 2f, 2f+1) then 1 label
byte.
"""

import argparse
import os
import struct
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'exporter'))
from extract import (FRAC_BITS, WORD_BITS, load_checkpoint, quantize,  # noqa: E402
                     quantize_thresholds, fits_in_word)

DEFAULT_CHECKPOINT = os.path.join(
    'training', 'artifacts', 'dwn_jsc_t200_distributive_50_l_b100_checkpoint.pt')

# Must match harness/vector_store.v's DEPTH. The device cannot hold more than this per batch.
DEVICE_DEPTH = 1024


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def pack_record(features_q, label, word_bits=WORD_BITS):
    """Quantized features + label -> the 33 bytes the loader expects.

    Little-endian per feature and in feature order, so byte k lands at x_flat[k*8 +: 8]. Any
    other order silently transposes the input; --selftest is what catches that.
    """
    out = bytearray()
    mask = (1 << word_bits) - 1
    for f in features_q:
        out += int(f & mask).to_bytes(word_bits // 8, 'little')
    out.append(int(label) & 0xFF)
    return bytes(out)


def quantize_features(x_raw, frac_bits=FRAC_BITS):
    """Scaled float features -> Q3.12 integers, the same truncation the golden model uses."""
    q = quantize(x_raw, frac_bits)
    if not fits_in_word(q, WORD_BITS):
        raise SystemExit(
            f'features do not fit {WORD_BITS}-bit signed: range [{q.min()}, {q.max()}]. '
            'Q3.15 is the documented fallback -- see docs/phase1-ledger.md.')
    return q


# ---------------------------------------------------------------------------
# board link
# ---------------------------------------------------------------------------

class Board:
    def __init__(self, port, baud=115200, timeout=5.0):
        try:
            import serial
        except ImportError:
            raise SystemExit('pyserial not installed. pip install -r requirements.txt')
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.baud = baud

    def close(self):
        self.ser.close()

    def ping(self):
        self.ser.reset_input_buffer()
        self.ser.write(b'P')
        return self.ser.read(1) == b'\xA5'

    def load(self, records):
        n = len(records)
        if n > DEVICE_DEPTH:
            raise ValueError(f'{n} records exceeds the device store ({DEVICE_DEPTH})')
        self.ser.write(b'L' + struct.pack('<H', n))
        self.ser.write(b''.join(records))
        self.ser.flush()

    def run(self, n):
        self.ser.write(b'R' + struct.pack('<H', n))
        self.ser.flush()

    def status(self):
        self.ser.reset_input_buffer()
        self.ser.write(b'S')
        raw = self.ser.read(9)
        if len(raw) != 9:
            raise SystemExit(f'status returned {len(raw)} bytes, expected 9 '
                             '(link dead, or baud mismatch between host and bitstream?)')
        cycles, correct = struct.unpack('<II', raw[:8])
        return cycles, correct, bool(raw[8] & 1)

    def wait_idle(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, _, busy = self.status()
            if not busy:
                return True
        return False


# ---------------------------------------------------------------------------
# self-test: verify the encoding without a board
# ---------------------------------------------------------------------------

def selftest(checkpoint):
    work = os.path.join(REPO, 'build', 'host_selftest')
    os.makedirs(work, exist_ok=True)

    # Regenerate the reference vectors from the checkpoint so this can never pass against a
    # stale set.
    r = subprocess.run([sys.executable, os.path.join(REPO, 'tb', 'gen_vectors.py'),
                        checkpoint, '--outdir', work],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        raise SystemExit('gen_vectors.py failed')

    ck = load_checkpoint(checkpoint)
    npz = np.load(checkpoint.replace('_checkpoint.pt', '_testvectors.npz'))
    x_raw = npz['x_raw']

    hex_lines = open(os.path.join(work, 'x_quant.hex')).read().split()
    exp_lines = open(os.path.join(work, 'expected_top.hex')).read().split()

    q = quantize_features(x_raw)
    n_check = min(len(q), 200)
    errors = 0

    for i in range(n_check):
        rec = pack_record(q[i], int(exp_lines[i], 16))
        # x_quant.hex is one 256-bit word per line, LSB-last. Reassembling the record's
        # feature bytes little-endian must reproduce it exactly.
        got = int.from_bytes(rec[:WORD_BITS * 16 // 8], 'little')
        want = int(hex_lines[i], 16)
        if got != want:
            if errors < 3:
                print(f'  MISMATCH record {i}\n    got  {got:064x}\n    want {want:064x}')
            errors += 1
        if len(rec) != 33:
            print(f'  MISMATCH record {i}: {len(rec)} bytes, expected 33')
            errors += 1

    # Status reply parsing, against a frame built the way the RTL builds it.
    frame = struct.pack('<II', 0xDEADBEEF, 77) + bytes([1])
    cycles, correct = struct.unpack('<II', frame[:8])
    if (cycles, correct, bool(frame[8] & 1)) != (0xDEADBEEF, 77, True):
        print('  MISMATCH status decode')
        errors += 1

    thr_q = quantize_thresholds(ck['thermometer']['thresholds'].numpy())
    print()
    print(f'  records checked : {n_check} (against tb/gen_vectors.py output)')
    print(f'  record size     : 33 bytes (32 feature + 1 label)')
    print(f'  format          : Q{WORD_BITS-1-FRAC_BITS}.{FRAC_BITS} signed, '
          f'threshold range [{thr_q.min()}, {thr_q.max()}]')
    print(f'  mismatches      : {errors}')
    print(f'  RESULT          : {"PASS" if errors == 0 else "FAIL"}')
    return 0 if errors == 0 else 1


# ---------------------------------------------------------------------------
# Gate 1b
# ---------------------------------------------------------------------------

def gate1b(board, checkpoint, batch=DEVICE_DEPTH, limit=None):
    """Batch the test set through the board and accumulate accuracy.

    The device store holds ~1024 vectors against a 166,000-sample test set, so this is the
    only way Gate 1b can run at all. Only totals come back per batch.
    """
    npz = np.load(checkpoint.replace('_checkpoint.pt', '_testvectors.npz'))
    x_raw, y_true, y_ref = npz['x_raw'], npz['y'], npz['pred']
    if limit:
        x_raw, y_true, y_ref = x_raw[:limit], y_true[:limit], y_ref[:limit]

    q = quantize_features(x_raw)
    total = len(q)

    # Labels loaded are the SOFTWARE MODEL's predictions, not the ground truth. Gate 1b asks
    # whether hardware reproduces the software model to the sample -- a mismatch against y_ref
    # is a hardware bug, whereas a mismatch against y_true is just the model being wrong.
    agree = 0
    cycles_total = 0
    t0 = time.time()

    for start in range(0, total, batch):
        chunk = slice(start, min(start + batch, total))
        recs = [pack_record(q[i], y_ref[i]) for i in range(chunk.start, chunk.stop)]
        board.load(recs)
        board.run(len(recs))
        if not board.wait_idle():
            raise SystemExit('board never went idle -- run stalled')
        cycles, correct, _ = board.status()
        agree += correct
        cycles_total += cycles
        print(f'  batch {start:6d}..{chunk.stop-1:6d}  agree {correct}/{len(recs)}  '
              f'{cycles} cycles')

    elapsed = time.time() - t0
    print()
    print(f'  samples              : {total}')
    print(f'  hardware == software : {agree}/{total}')
    print(f'  core cycles          : {cycles_total}')
    print(f'  wall clock           : {elapsed:.1f} s')
    if elapsed > 0:
        print(f'  effective rate       : {total/elapsed:,.0f} samples/s over the link')
    print()
    print('  Software accuracy on this set: '
          f'{100*(y_ref == y_true).mean():.2f}%')
    if agree == total:
        print('  RESULT: PASS -- hardware reproduces the software model to the sample')
    else:
        print(f'  RESULT: FAIL -- {total-agree} disagreements')
    return 0 if agree == total else 1


def main():
    ap = argparse.ArgumentParser(description='Drive the DWN board over UART.')
    ap.add_argument('--port', help='serial port, e.g. COM4 or /dev/ttyUSB1')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    ap.add_argument('--selftest', action='store_true', help='verify encoding, no board needed')
    ap.add_argument('--ping', action='store_true')
    ap.add_argument('--gate1b', action='store_true')
    ap.add_argument('--limit', type=int, help='only run the first N samples')
    args = ap.parse_args()

    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) \
        else os.path.join(REPO, args.checkpoint)
    if not os.path.exists(ckpt):
        raise SystemExit(f'checkpoint not found: {ckpt}')

    if args.selftest:
        print('=== host encoding self-test (no board) ===')
        return selftest(ckpt)

    if not args.port:
        raise SystemExit('--port is required (or use --selftest)')

    board = Board(args.port, args.baud)
    try:
        if not board.ping():
            raise SystemExit('ping failed -- no 0xA5. Check the port, the baud rate, and '
                             'that the bitstream is loaded.')
        print(f'ping OK on {args.port} at {args.baud} baud')

        if args.gate1b:
            print('\n=== Gate 1b ===')
            return gate1b(board, ckpt, limit=args.limit)
        return 0
    finally:
        board.close()


if __name__ == '__main__':
    sys.exit(main())
