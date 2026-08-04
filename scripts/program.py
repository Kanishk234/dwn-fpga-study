"""Program the Basys 3 over JTAG, without opening the Vivado GUI.

Phase 2 will rebuild and reprogram dozens of times, so this loop has to be one command. It is
also less error-prone than the Hardware Manager: no chance of picking last week's bitstream out
of a file dialog.

The FT2232HQ exposes JTAG and the UART as two interfaces of one chip. Vivado's hardware server
grabs the JTAG side; scripts/host.py opens the UART side. They coexist -- but the GUI Hardware
Manager holding the target open WILL block this, so close it first.

Programming is volatile: it lives in FPGA configuration RAM and is lost on power cycle or on
pressing PROG. Nothing is written to flash.

Usage:
    .venv\\Scripts\\python.exe scripts/program.py
    .venv\\Scripts\\python.exe scripts/program.py --bit path/to/other.bit
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gate1 import REPO, find_vivado_bin, run, tool  # noqa: E402

DEFAULT_BIT = os.path.join('build', 'bitstream', 'basys3', 'dwn_basys3_top.bit')

TCL = """
open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
current_hw_device [lindex [get_hw_devices] 0]
refresh_hw_device -update_hw_probes false [current_hw_device]
puts "DEVICE [get_property PART [current_hw_device]]"
set_property PROGRAM.FILE {BITFILE} [current_hw_device]
program_hw_devices [current_hw_device]
refresh_hw_device [current_hw_device]
puts "PROGRAM_DONE [get_property PROGRAM.FILE [current_hw_device]]"
close_hw_manager
"""


def main():
    ap = argparse.ArgumentParser(description='Program the Basys 3 over JTAG.')
    ap.add_argument('--bit', default=DEFAULT_BIT)
    ap.add_argument('--vivado-bin', default=None)
    args = ap.parse_args()

    bit_rel = args.bit.replace('\\', '/')
    bit_abs = os.path.join(REPO, args.bit)
    if not os.path.exists(bit_abs):
        raise SystemExit(f'bitstream not found: {bit_abs}\n'
                         '  Build it: .venv\\Scripts\\python.exe scripts/build_bitstream.py')

    vivado_bin = find_vivado_bin(args.vivado_bin)
    work = os.path.join(REPO, 'build', 'program')
    os.makedirs(work, exist_ok=True)
    tcl_path = os.path.join(work, 'program.tcl')
    with open(tcl_path, 'w') as f:
        f.write(TCL.replace('{BITFILE}', bit_rel))

    print(f'bitstream : {bit_rel} ({os.path.getsize(bit_abs)/1e6:.2f} MB)')
    print('programming...')

    env = dict(os.environ)
    env['PATH'] = vivado_bin + os.pathsep + env.get('PATH', '')
    r = run([tool(vivado_bin, 'vivado'), '-mode', 'batch', '-notrace',
             '-log', 'build/program/vivado.log', '-journal', 'build/program/vivado.jou',
             '-source', 'build/program/program.tcl'],
            cwd=REPO, env=env, capture=True)
    out = (r.stdout or '') + (r.stderr or '')

    for line in out.splitlines():
        if line.startswith(('DEVICE ', 'PROGRAM_DONE ')):
            print('  ' + line)

    if 'PROGRAM_DONE' not in out:
        tail = '\n'.join(out.strip().splitlines()[-25:])
        print(tail)
        print('\nFAILED. Most likely causes:')
        print('  - Vivado Hardware Manager is open and holding the target. Close it.')
        print('  - Board not powered on, or USB cable is charge-only.')
        return 1

    print('\nprogrammed OK (volatile -- lost on power cycle)')
    print('Check the link:  .venv\\Scripts\\python.exe scripts\\host.py --ping')
    return 0


if __name__ == '__main__':
    sys.exit(main())
