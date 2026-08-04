# Basys 3 (Artix-7 XC7A35T-1CPG236C) constraints for dwn_basys3_top.
#
# Pin assignments are taken from Digilent's Basys-3 master XDC. They are board facts -- the
# FPGA ball each peripheral is wired to -- and do not change.
#
# Written fresh rather than copied: the master file in circulation had been edited for an
# unrelated lab project (ports named `signal`, `outedge`, `slow_clk`, `sseg`, `reset`), so
# copying it wholesale would have brought someone else's port names along. Only the pin
# letters were reused.
#
# Ports here must match harness/dwn_basys3_top.v exactly. Only the peripherals the design
# actually uses are constrained; sw[15:2], the other buttons and the Pmods are deliberately
# absent because the top level does not declare them.

## ---------------------------------------------------------------------------
## Clock -- 100 MHz oscillator
## ---------------------------------------------------------------------------
set_property PACKAGE_PIN W5 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports clk]

## ---------------------------------------------------------------------------
## Switches -- sw[0] selects what the 7-segment shows (class vs cycle count)
## ---------------------------------------------------------------------------
set_property PACKAGE_PIN V17 [get_ports {sw[0]}]
set_property PACKAGE_PIN V16 [get_ports {sw[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[*]}]

## ---------------------------------------------------------------------------
## Buttons -- btnC is reset (active high)
## ---------------------------------------------------------------------------
set_property PACKAGE_PIN U18 [get_ports btnC]
set_property IOSTANDARD LVCMOS33 [get_ports btnC]

## ---------------------------------------------------------------------------
## USB-RS232 through the FT2232HQ bridge. The only path to the host: no SPI, no
## external DRAM (brief §6), which is why the I/O wall is a headline result.
## ---------------------------------------------------------------------------
set_property PACKAGE_PIN B18 [get_ports RsRx]
set_property IOSTANDARD LVCMOS33 [get_ports RsRx]
set_property PACKAGE_PIN A18 [get_ports RsTx]
set_property IOSTANDARD LVCMOS33 [get_ports RsTx]

## ---------------------------------------------------------------------------
## LEDs -- status. led[0] busy, led[1] done, led[2] framing error (sticky),
## led[3] display mode, led[15:8] low byte of correct_count.
## ---------------------------------------------------------------------------
set_property PACKAGE_PIN U16 [get_ports {led[0]}]
set_property PACKAGE_PIN E19 [get_ports {led[1]}]
set_property PACKAGE_PIN U19 [get_ports {led[2]}]
set_property PACKAGE_PIN V19 [get_ports {led[3]}]
set_property PACKAGE_PIN W18 [get_ports {led[4]}]
set_property PACKAGE_PIN U15 [get_ports {led[5]}]
set_property PACKAGE_PIN U14 [get_ports {led[6]}]
set_property PACKAGE_PIN V14 [get_ports {led[7]}]
set_property PACKAGE_PIN V13 [get_ports {led[8]}]
set_property PACKAGE_PIN V3  [get_ports {led[9]}]
set_property PACKAGE_PIN W3  [get_ports {led[10]}]
set_property PACKAGE_PIN U3  [get_ports {led[11]}]
set_property PACKAGE_PIN P3  [get_ports {led[12]}]
set_property PACKAGE_PIN N3  [get_ports {led[13]}]
set_property PACKAGE_PIN P1  [get_ports {led[14]}]
set_property PACKAGE_PIN L1  [get_ports {led[15]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]

## ---------------------------------------------------------------------------
## Seven-segment display. Segments and anodes are BOTH ACTIVE LOW on this board;
## driving them active high gives an inverted display that looks like a fault.
## seg[6:0] = {g,f,e,d,c,b,a}.
## ---------------------------------------------------------------------------
set_property PACKAGE_PIN W7 [get_ports {seg[6]}]
set_property PACKAGE_PIN W6 [get_ports {seg[5]}]
set_property PACKAGE_PIN U8 [get_ports {seg[4]}]
set_property PACKAGE_PIN V8 [get_ports {seg[3]}]
set_property PACKAGE_PIN U5 [get_ports {seg[2]}]
set_property PACKAGE_PIN V5 [get_ports {seg[1]}]
set_property PACKAGE_PIN U7 [get_ports {seg[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {seg[*]}]

set_property PACKAGE_PIN V7 [get_ports dp]
set_property IOSTANDARD LVCMOS33 [get_ports dp]

set_property PACKAGE_PIN U2 [get_ports {an[0]}]
set_property PACKAGE_PIN U4 [get_ports {an[1]}]
set_property PACKAGE_PIN V4 [get_ports {an[2]}]
set_property PACKAGE_PIN W4 [get_ports {an[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {an[*]}]

## ---------------------------------------------------------------------------
## Timing exceptions for I/O
##
## Every port here is either asynchronous to clk or orders of magnitude slower
## than it, so constraining them against the 100 MHz clock would report failures
## that mean nothing and hide the ones that do.
##
##   RsRx, btnC, sw   asynchronous. Each is passed through a two-flop
##                    synchronizer in RTL (uart_rx.v, dwn_basys3_top.v), which
##                    is what actually makes them safe -- not a timing
##                    constraint.
##   RsTx             one bit per ~8.7 us at 115200 baud, against a 10 ns clock.
##   led, seg, an, dp human-visible; the display refreshes at ~190 Hz.
##
## These are false paths, not tightened constraints, because there is no real
## deadline to meet. The paths that DO matter are register-to-register inside
## the design, and those stay fully analyzed.
## ---------------------------------------------------------------------------
set_false_path -from [get_ports {RsRx btnC}]
set_false_path -from [get_ports {sw[*]}]
set_false_path -to   [get_ports {RsTx dp}]
set_false_path -to   [get_ports {led[*]}]
set_false_path -to   [get_ports {seg[*]}]
set_false_path -to   [get_ports {an[*]}]

## ---------------------------------------------------------------------------
## Configuration
## ---------------------------------------------------------------------------
set_property CFGBVS VCCO        [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
