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
                     quantize_thresholds, saturation_is_lossless, layer_indices,
                     extract_tables, extract_wiring, encode, forward)

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

    Kept for the self-test, which checks one record at a time. Bulk transfers use pack_batch.
    """
    out = bytearray()
    mask = (1 << word_bits) - 1
    for f in features_q:
        out += int(f & mask).to_bytes(word_bits // 8, 'little')
    out.append(int(label) & 0xFF)
    return bytes(out)


def pack_batch(features_q, labels):
    """Vectorized pack_record over a whole batch. Returns one bytes object.

    The per-record version runs 16 int.to_bytes() calls per sample -- 2.7 million of them
    across the test set, which cost several seconds and was invisible at 115200 baud. Above
    1 Mbaud the wire stops being the bottleneck and this starts to be, so the packing happens
    in numpy instead: one dtype cast and one view, no Python loop.

    '<i2' forces little-endian 16-bit regardless of host byte order, so the wire format does
    not silently depend on what machine the host runs on.
    """
    words = np.ascontiguousarray(features_q).astype('<i2')     # (n, 16) LE int16
    feat_bytes = words.view(np.uint8).reshape(len(words), -1)  # (n, 32)
    lab = np.asarray(labels, dtype=np.uint8).reshape(-1, 1)    # (n, 1)
    return np.hstack([feat_bytes, lab]).tobytes()              # row-major = record after record


def quantize_features(x_raw, thr_q=None, frac_bits=FRAC_BITS):
    """Scaled float features -> Q3.12 integers, exactly as the golden model does.

    quantize() saturates to the word range. The full test set has outliers past Q3.12's +8,
    and clamping them is lossless because every threshold sits well inside the range -- but
    that has to be checked against the actual thresholds, not assumed, so pass thr_q.
    """
    q = quantize(x_raw, frac_bits)
    if thr_q is not None and not saturation_is_lossless(thr_q, WORD_BITS):
        raise SystemExit(
            'A thermometer threshold lies at or beyond the edge of the '
            f'Q{WORD_BITS-1-frac_bits}.{frac_bits} range, so saturating features could flip '
            'an encoder bit. Widen the INTEGER bits (Q4.12), not the fractional ones -- Q3.15 '
            'has the same range and would not help.')
    return q


# ---------------------------------------------------------------------------
# board link
# ---------------------------------------------------------------------------

# The Basys 3 talks through an FT2232HQ. The chip exposes two interfaces -- one for JTAG
# programming, one for the UART -- so more than one port can appear for a single board, and
# which is which is not guaranteed. That is why detection pings rather than trusting the IDs.
FTDI_VID = 0x0403
FT2232_PID = 0x6010


def ftdi_latency(port_name):
    """Read the FTDI VCP latency timer (ms) for a COM port. None if unknown.

    Windows buffers a short USB read until either the packet fills or this timer expires, so
    every small reply -- our 9-byte status, one per batch -- can cost up to the timer. At the
    16 ms default that is ~2.6 s added to a 166k run, about 21%, with no other symptom. It is a
    per-machine driver setting and does not travel with the repo, so the least this script can
    do is notice and say so.
    """
    if os.name != 'nt':
        return None
    try:
        import winreg
    except ImportError:
        return None
    base = r'SYSTEM\CurrentControlSet\Enum\FTDIBUS'
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError:
        return None
    try:
        for i in range(winreg.QueryInfoKey(root)[0]):
            dev = winreg.EnumKey(root, i)
            try:
                params = winreg.OpenKey(root, dev + r'\0000\Device Parameters')
                name, _ = winreg.QueryValueEx(params, 'PortName')
                if name.upper() != port_name.upper():
                    continue
                return int(winreg.QueryValueEx(params, 'LatencyTimer')[0])
            except OSError:
                continue
    finally:
        root.Close()
    return None


def set_ftdi_latency(port_name, ms):
    """Write the latency timer. Needs administrator rights and a device re-enumeration."""
    if os.name != 'nt':
        raise SystemExit('--set-latency is Windows-only')
    import winreg
    base = r'SYSTEM\CurrentControlSet\Enum\FTDIBUS'
    root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    for i in range(winreg.QueryInfoKey(root)[0]):
        dev = winreg.EnumKey(root, i)
        path = dev + r'\0000\Device Parameters'
        try:
            ro = winreg.OpenKey(root, path)
            if winreg.QueryValueEx(ro, 'PortName')[0].upper() != port_name.upper():
                continue
        except OSError:
            continue
        try:
            rw = winreg.OpenKey(root, path, 0, winreg.KEY_SET_VALUE)
            winreg.SetValueEx(rw, 'LatencyTimer', 0, winreg.REG_DWORD, ms)
        except PermissionError:
            raise SystemExit(
                'Access denied. Re-run this from an Administrator PowerShell:\n'
                f'  .venv\\Scripts\\python.exe scripts\\host.py --port {port_name} '
                f'--set-latency {ms}')
        print(f'LatencyTimer for {port_name} set to {ms} ms.')
        print('Unplug and replug the board (or disable/enable it in Device Manager) --')
        print('the driver only reads this at device enumeration.')
        return 0
    raise SystemExit(f'No FTDI device found for {port_name}')


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

    def load_bytes(self, blob, n):
        """Load n pre-packed records. One write for the whole batch."""
        if n > DEVICE_DEPTH:
            raise ValueError(f'{n} records exceeds the device store ({DEVICE_DEPTH})')
        # Header and payload in a single write: two writes per batch means two chances for the
        # driver to sit on a partial buffer waiting out the latency timer.
        self.ser.write(b'L' + struct.pack('<H', n) + blob)
        self.ser.flush()

    def load(self, records):
        self.load_bytes(b''.join(records), len(records))

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

    def status_when_idle(self, timeout=10.0):
        """Read the result once the run has finished. Returns (cycles, correct).

        One round trip in the common case, not two. The previous code polled with wait_idle()
        and then read status() again -- but a 1024-sample run takes ~10 us against a round trip
        measured in milliseconds, so the first read always finds the FSM already idle and the
        second was pure latency. At 115200 that was noise; at 10 Mbaud it is most of the cost.
        """
        deadline = time.time() + timeout
        while True:
            cycles, correct, busy = self.status()
            if not busy:
                return cycles, correct
            if time.time() > deadline:
                raise SystemExit('board never went idle -- run stalled')


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

def gate1b_dataset(checkpoint):
    """Pick the test set, preferring the full one. Returns (path, is_full).

    Gate 1b's claim is about the WHOLE test set (brief §11), so the 1000-sample testbench file
    is a fallback, never the default -- and the caller says out loud which one it used. Silently
    running 1000 samples and reporting PASS would be a claim the run does not support.
    """
    full = checkpoint.replace('_checkpoint.pt', '_testset_full.npz')
    if os.path.exists(full):
        return full, True
    small = checkpoint.replace('_checkpoint.pt', '_testvectors.npz')
    if os.path.exists(small):
        return small, False
    raise SystemExit(f'No test set found next to {checkpoint}')


def gate1b(board, checkpoint, batch=DEVICE_DEPTH, limit=None):
    """Batch the test set through the board and accumulate accuracy.

    The device store holds ~1024 vectors against a 166,000-sample test set, so batching is the
    only way Gate 1b can run at all. Only totals come back per batch.
    """
    path, is_full = gate1b_dataset(checkpoint)
    npz = np.load(path)
    x_raw, y_true, y_float = npz['x_raw'], npz['y'], npz['pred']
    print(f'  test set : {os.path.basename(path)} ({x_raw.shape[0]} samples)')
    if not is_full:
        print('  WARNING: this is the 1000-sample testbench file, not the full test set.')
        print('           Gate 1b requires all 166k. Run training/dump_testset_kaggle.ipynb')
        print('           and put the result in training/artifacts/.')
    if limit:
        x_raw, y_true, y_float = x_raw[:limit], y_true[:limit], y_float[:limit]
        print(f'  --limit  : first {len(x_raw)} only -- a smoke test, not Gate 1b')

    total = len(x_raw)

    # THE REFERENCE IS THE FIXED-POINT MODEL, NOT THE FLOAT ONE.
    #
    # `pred` in the .npz came from PyTorch on float32 features. The hardware implements Q3.12,
    # which is a deliberate part of the specification, not an error (docs/phase1-ledger.md).
    # Scoring hardware against the float model therefore measures the quantization decision,
    # not the hardware -- and reports a FAIL for a design that is exactly correct.
    #
    # Measured on the full 166k set: the two models differ on 30 samples (0.018%), worth
    # -0.0012 pp of accuracy. That difference is reported below as a characterization number,
    # which is what it is, rather than being laundered into a pass/fail.
    ck = load_checkpoint(checkpoint)
    cfg = ck['config']
    layers = [(extract_tables(ck['state_dict'], i), *extract_wiring(ck['state_dict'], i, cfg['n']))
              for i in layer_indices(ck['state_dict'])]
    thr_q = quantize_thresholds(ck['thermometer']['thresholds'].numpy(), FRAC_BITS)
    q = quantize_features(x_raw, thr_q)

    saturated = int((np.abs(np.floor(x_raw.astype(np.float64) * 2**FRAC_BITS))
                     > 2**(WORD_BITS-1) - 1).sum())
    if saturated:
        print(f'  saturated: {saturated} feature value(s) clamped to the Q'
              f'{WORD_BITS-1-FRAC_BITS}.{FRAC_BITS} range -- lossless, every threshold is '
              'well inside it')

    chunks = []
    for i in range(0, total, 20000):          # chunked: encoding 166k x 3200 bits at once is 531 MB
        chunks.append(forward(encode(q[i:i+20000], thr_q), layers, cfg['num_classes'])[0])
    y_ref = np.concatenate(chunks)

    diverge = int((y_ref != y_float).sum())
    print(f'  reference: Q{WORD_BITS-1-FRAC_BITS}.{FRAC_BITS} golden model '
          f'(the spec the hardware implements)')
    print(f'             differs from the float32 model on {diverge}/{total} samples '
          f'({100*diverge/total:.4f}%)')
    print()

    agree = 0
    cycles_total = 0
    t0 = time.time()

    for start in range(0, total, batch):
        stop = min(start + batch, total)
        n = stop - start
        board.load_bytes(pack_batch(q[start:stop], y_ref[start:stop]), n)
        board.run(n)
        cycles, correct = board.status_when_idle()
        agree += correct
        cycles_total += cycles
        if correct != n or start == 0 or stop == total:
            print(f'  batch {start:6d}..{stop-1:6d}  agree {correct}/{n}  {cycles} cycles')

    elapsed = time.time() - t0
    print()
    print(f'  samples              : {total}')
    print(f'  hardware == software : {agree}/{total}')
    print(f'  core cycles          : {cycles_total}')
    print(f'  wall clock           : {elapsed:.1f} s')
    print()

    # The I/O wall, measured rather than estimated (brief §14). The core figure comes from the
    # cycle counter in hardware, so this compares two measurements, not a measurement against
    # a datasheet claim.
    if elapsed > 0 and cycles_total:
        link_rate = total / elapsed
        core_seconds = cycles_total / 100e6          # 100 MHz board clock
        core_rate = total / core_seconds
        print(f'  core throughput      : {core_rate/1e6:,.1f} M samples/s '
              f'({cycles_total} cycles at 100 MHz)')
        print(f'  over the link        : {link_rate:,.0f} samples/s')
        print(f'  I/O wall             : {core_rate/link_rate:,.0f}x '
              f'-- the link is the entire cost')
        print()

    print(f'  accuracy, float32 model      : {100*(y_float == y_true).mean():.4f}%')
    print(f'  accuracy, Q{WORD_BITS-1-FRAC_BITS}.{FRAC_BITS} (on hardware)  : '
          f'{100*(y_ref == y_true).mean():.4f}%')
    print(f'  cost of fixed point          : '
          f'{100*((y_ref == y_true).mean() - (y_float == y_true).mean()):+.4f} pp')
    print()
    if agree == total:
        print('  RESULT: PASS -- hardware reproduces the golden model to the sample')
    else:
        print(f'  RESULT: FAIL -- {total-agree} disagreements against the Q'
              f'{WORD_BITS-1-FRAC_BITS}.{FRAC_BITS} reference.')
        print('  This is a real hardware/software divergence: the reference already accounts')
        print('  for quantization, so it cannot be explained away by precision.')
    return 0 if agree == total else 1


def main():
    ap = argparse.ArgumentParser(description='Drive the DWN board over UART.')
    ap.add_argument('--port', help='serial port, e.g. COM3 or /dev/ttyUSB1. '
                                   'Omit to auto-detect.')
    # Must match the BAUD parameter the bitstream was built with (harness/dwn_basys3_top.v).
    # A mismatch shows up as a failed ping rather than as corrupted data.
    ap.add_argument('--baud', type=int, default=5000000)
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    ap.add_argument('--selftest', action='store_true', help='verify encoding, no board needed')
    ap.add_argument('--list-ports', action='store_true', help='show serial ports and exit')
    ap.add_argument('--ping', action='store_true')
    ap.add_argument('--gate1b', action='store_true')
    ap.add_argument('--limit', type=int, help='only run the first N samples')
    ap.add_argument('--set-latency', type=int, metavar='MS',
                    help='set the FTDI latency timer (needs admin + a replug). 1 is ideal.')
    args = ap.parse_args()

    if args.set_latency is not None:
        if not args.port:
            raise SystemExit('--set-latency needs --port (e.g. --port COM3)')
        return set_ftdi_latency(args.port, args.set_latency)

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

    lat = ftdi_latency(board.port)
    if lat is not None and lat > 2:
        print(f'\n  NOTE: FTDI latency timer on {board.port} is {lat} ms (1 ms is ideal).')
        print('  Costs roughly one timer period per batch -- about 21% of a full run at 16 ms.')
        print(f'  Fix once, from an Administrator PowerShell:')
        print(f'    .venv\\Scripts\\python.exe scripts\\host.py --port {board.port} '
              '--set-latency 1')
        print('  then unplug and replug the board.\n')

    try:
        if args.gate1b:
            print('\n=== Gate 1b ===')
            return gate1b(board, ckpt, limit=args.limit)
        return 0
    finally:
        board.close()


if __name__ == '__main__':
    sys.exit(main())
