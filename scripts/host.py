"""Host-side driver for the DWN board: quantize, load, run, and score over UART.

This is the other half of harness/uart_loader.v. It exists to make Gate 1b runnable: the full
JSC test set is 166,000 samples and the device holds ~1024, so the run has to be batched, with
accuracy accumulated on-chip and only totals coming back.

    python scripts/host.py --selftest        # no board needed
    python scripts/host.py --list-ports      # what is actually plugged in
    python scripts/host.py --ping            # auto-detects the board
    python scripts/host.py --gate1b --limit 1024
    python scripts/host.py --gate1b          # the full test set

The COM number is assigned by Windows per machine and per USB socket -- the same board is COM3
on one laptop and COM8 on another -- so --port is optional and detection is the default.

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

# The Basys 3 talks through an FT2232HQ. The chip exposes two interfaces -- one for JTAG
# programming, one for the UART -- so more than one port can appear for a single board, and
# which is which is not guaranteed. That is why detection pings rather than trusting the IDs.
FTDI_VID = 0x0403
FT2232_PID = 0x6010


def _serial():
    try:
        import serial
        return serial
    except ImportError:
        raise SystemExit('pyserial not installed. pip install -r requirements.txt')


def list_candidate_ports():
    """Every serial port, FTDI ones first.

    The COM number is assigned by Windows per machine and per USB port -- COM3 on one laptop,
    COM8 on another -- so it can never be hardcoded. Ranking by USB vendor/product ID finds the
    board wherever it landed.
    """
    from serial.tools import list_ports
    ports = list(list_ports.comports())

    def rank(p):
        if p.vid == FTDI_VID and p.pid == FT2232_PID:
            return 0
        if p.vid == FTDI_VID:
            return 1
        return 2

    return sorted(ports, key=rank)


def autodetect(baud, timeout=1.0, verbose=True):
    """Find the board by pinging candidates. Returns (Board, port_info).

    Pinging is the detection: a port with the right USB IDs may still be the JTAG interface, or
    held open by Vivado's Hardware Manager, or belong to some other FTDI device entirely. Only
    a 0xA5 coming back proves it is the DWN bitstream on the other end.
    """
    cands = list_candidate_ports()
    if not cands:
        raise SystemExit(
            'No serial ports found at all.\n'
            '  - Is the board plugged in and powered on?\n'
            '  - Windows: Device Manager -> Ports (COM & LPT)')

    for p in cands:
        if verbose:
            print(f'  trying {p.device:8s} {p.description}')
        try:
            board = Board(p.device, baud, timeout=timeout)
        except SystemExit:
            raise
        except Exception as e:
            if verbose:
                print(f'    cannot open ({type(e).__name__})')
            continue
        try:
            if board.ping():
                return board, p
        except Exception:
            pass
        board.close()
        if verbose:
            print('    no response to ping')

    raise SystemExit(
        'No board responded to a ping on any port.\n'
        '  - Is the bitstream loaded? Programming is lost on power cycle.\n'
        '  - Is Vivado\'s Hardware Manager holding the port open? Close it.\n'
        f'  - Baud mismatch? Host is at {baud}; the bitstream is built with the BAUD\n'
        '    parameter in harness/dwn_basys3_top.v (default 115200).\n'
        '  Ports tried: ' + ', '.join(p.device for p in cands))


class Board:
    def __init__(self, port, baud=115200, timeout=5.0):
        serial = _serial()
        try:
            self.ser = serial.Serial(port, baud, timeout=timeout)
        except serial.SerialException as e:
            raise SystemExit(
                f'Could not open {port}: {e}\n'
                '  Run with --list-ports to see what is actually present, or omit --port\n'
                '  entirely to auto-detect.')
        self.port = port
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
    ap.add_argument('--port', help='serial port, e.g. COM3 or /dev/ttyUSB1. '
                                   'Omit to auto-detect.')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    ap.add_argument('--selftest', action='store_true', help='verify encoding, no board needed')
    ap.add_argument('--list-ports', action='store_true', help='show serial ports and exit')
    ap.add_argument('--ping', action='store_true')
    ap.add_argument('--gate1b', action='store_true')
    ap.add_argument('--limit', type=int, help='only run the first N samples')
    args = ap.parse_args()

    if args.list_ports:
        _serial()
        ports = list_candidate_ports()
        if not ports:
            print('no serial ports found')
            return 1
        print(f'{"port":10s} {"VID:PID":10s} description')
        for p in ports:
            ids = f'{p.vid:04X}:{p.pid:04X}' if p.vid is not None else '-'
            mark = '  <- FT2232H (Basys 3)' if (p.vid, p.pid) == (FTDI_VID, FT2232_PID) else ''
            print(f'{p.device:10s} {ids:10s} {p.description}{mark}')
        return 0

    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) \
        else os.path.join(REPO, args.checkpoint)
    if not os.path.exists(ckpt):
        raise SystemExit(f'checkpoint not found: {ckpt}')

    if args.selftest:
        print('=== host encoding self-test (no board) ===')
        return selftest(ckpt)

    if args.port:
        board = Board(args.port, args.baud)
        if not board.ping():
            board.close()
            raise SystemExit(
                f'No 0xA5 from {args.port}. The port opened, so it exists -- but nothing on\n'
                '  the other end answered. Check the bitstream is loaded and the baud matches,\n'
                '  or omit --port to auto-detect.')
        print(f'ping OK on {args.port} at {args.baud} baud')
    else:
        print(f'auto-detecting board at {args.baud} baud...')
        board, info = autodetect(args.baud)
        print(f'ping OK on {info.device} ({info.description}) at {args.baud} baud')

    try:
        if args.gate1b:
            print('\n=== Gate 1b ===')
            return gate1b(board, ckpt, limit=args.limit)
        return 0
    finally:
        board.close()


if __name__ == '__main__':
    sys.exit(main())
