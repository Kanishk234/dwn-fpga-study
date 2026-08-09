# Non-project-mode synthesis for one module. Driven by scripts/run_synth.py; the DSE sweep
# will drive the same script in Phase 2, which is why everything is an argument and nothing is
# hardcoded.
#
# OUT-OF-CONTEXT on purpose. Two reasons:
#   1. It is what the paper did (Table 2: xcvu9p, out-of-context), so the LUT numbers are
#      comparable. Their Fmax figures are NOT achievable in a real system -- no I/O, no
#      surrounding logic -- which is part of why a real board deployment measures something
#      their table does not (brief §14).
#   2. It needs no pin constraints, so area can be measured before the harness exists.
#
# Bitstream builds later will NOT be out-of-context and will need constraints/basys3.xdc.
#
# Usage (via run_synth.py):
#   vivado -mode batch -source scripts/build.tcl -tclargs <top> <part> <out_dir> <src>...

if {$argc < 8} {
    puts "ERROR: expected <top> <part> <out_dir> <period_ns> <impl:0|1> <generics> <xdc> <sources...>"
    exit 1
}

# Threads for the placer and router. Vivado's default on this machine was 2, on a 16-core CPU --
# the whole Phase 2 sweep ran at ~12% of available capacity.
#
# NOT raised blindly. Vivado's placer and router are multithreaded, and the tool documents that
# results may differ between thread counts. This project's entire comparability argument is that
# every sweep point comes from one machine and one flow, so a change that silently altered
# placement would split the frontier in two. Overridable so the effect can be MEASURED against
# an already-known config before being adopted; see docs/phase2-ledger.md.
# VERIFIED IDENTICAL 2026-08-09: `1x600` re-synthesized at 8 threads reproduced 1312 / 9367 /
# 10631 LUTs and WNS +0.400 exactly, matching the 2-thread run that is in the results. So
# raising this does not split the frontier, and configs measured before and after the change
# remain comparable.
set dwn_threads 8
if {[info exists ::env(DWN_VIVADO_THREADS)]} {
    set dwn_threads $::env(DWN_VIVADO_THREADS)
}
set_param general.maxThreads $dwn_threads
puts "general.maxThreads = $dwn_threads"

set top      [lindex $argv 0]
set part     [lindex $argv 1]
set out_dir  [lindex $argv 2]
set period   [lindex $argv 3]
set do_impl  [lindex $argv 4]
# Generics arrive as NAME:VALUE+NAME:VALUE, or "-" for none. Neither ' ' nor '=' nor ',' can
# be used as separators here: Vivado splits -tclargs on whitespace, and cmd.exe (which has to
# launch vivado.bat on Windows) treats '=' and ',' as argument delimiters, so "PIPE_POP=0"
# arrives as two arguments and Vivado tries to open a file called "0".
set generics [lindex $argv 5]
# Constraints file, or "-" for none. Its presence is what switches this from an out-of-context
# area/timing measurement into a real pin-assigned design that can become a bitstream.
set xdc      [lindex $argv 6]
set sources  [lrange $argv 7 end]

set is_ooc [expr {$xdc eq "-" || $xdc eq ""}]

file mkdir $out_dir

foreach f $sources {
    puts "read_verilog $f"
    read_verilog $f
}

if {!$is_ooc} {
    puts "read_xdc $xdc"
    read_xdc $xdc
}

# -flatten_hierarchy none keeps module boundaries so report_utilization -hierarchical can
# attribute LUTs to the encoder vs the core. Brief §6 requires reporting them separately in
# every table we publish, so the flow has to be able to tell them apart.
set synth_args [list -top $top -part $part -flatten_hierarchy none]
if {$is_ooc} {
    lappend synth_args -mode out_of_context
}
# synth_design does not search anywhere for `include by default, and the harness pulls the
# pipeline latency out of the GENERATED header rather than hardcoding it -- deliberately, so
# benchmark_fsm and the emitted pipeline cannot disagree about depth.
#
# The include path is DERIVED from the sources rather than hardcoded or passed separately.
# Phase 2 emits each config into its own directory, so a fixed path here would compile a swept
# config against some other config's DWN_TOP_LATENCY -- silently, and the only symptom would be
# a wrong number. Deriving it means the headers always come from the same place as the RTL that
# was actually read.
set inc_dirs {}
foreach f $sources {
    set d [file dirname $f]
    if {$d ne "" && [file isdirectory $d] && [lsearch -exact $inc_dirs $d] < 0} {
        lappend inc_dirs $d
    }
}
if {[llength $inc_dirs]} {
    puts "include_dirs: $inc_dirs"
    lappend synth_args -include_dirs $inc_dirs
}
if {$generics ne "-" && $generics ne ""} {
    foreach g [split $generics "+"] {
        set kv [split $g ":"]
        set assign "[lindex $kv 0]=[lindex $kv 1]"
        lappend synth_args -generic $assign
        puts "generic: $assign"
    }
}
synth_design {*}$synth_args

# Constrain at the board clock (100 MHz), so the slack reported is directly the question that
# matters: does this run on a Basys 3 as-is? Fmax is then derived from WNS by run_synth.py.
#
# A design with a clk port gets a real clock on it. A purely combinational module (the encoder
# on its own) has no clock port, so it gets a virtual clock plus zero I/O delays, which turns
# the report into an input-to-output path delay instead. Both are useful; they are just not
# the same measurement, and the reports say which is which.
set clk_ports [get_ports -quiet clk]
if {!$is_ooc} {
    # The XDC owns all timing for a real design: it creates the clock on the actual pin and
    # declares false paths for the asynchronous and human-speed I/O. Adding the constraints
    # below on top would double-create the clock and re-constrain paths the XDC deliberately
    # excluded, so this branch does nothing.
    puts "TIMING_MODE constrained by $xdc"
} elseif {[llength $clk_ports] > 0} {
    create_clock -name clk -period $period $clk_ports

    # I/O delays are NOT optional here. Without them, input-to-first-register and
    # last-register-to-output paths are UNCONSTRAINED and simply vanish from the timing
    # report -- so a design with fewer pipeline stages reports BETTER slack, because the
    # long paths it just created are the ones no longer being checked. That artifact made a
    # 2-stage variant look 3x faster than the 4-stage one it is strictly slower than.
    #
    # Zero delay is the strict choice: it budgets the entire period to on-chip logic and
    # assumes the outside world is infinitely fast. Pessimistic for a real board, but it is
    # the same assumption for every variant, which is what makes them comparable.
    # get_ports -filter, not remove_from_collection: the latter is Synopsys-style and does
    # not exist in Vivado's Tcl.
    set_input_delay  0.000 -clock clk [get_ports -filter {DIRECTION == IN && NAME != "clk"}]
    set_output_delay 0.000 -clock clk [all_outputs]
    puts "TIMING_MODE registered (real clock, period $period ns, I/O constrained)"
} else {
    create_clock -name vclk -period $period
    set_input_delay  0.000 -clock vclk [all_inputs]
    set_output_delay 0.000 -clock vclk [all_outputs]
    puts "TIMING_MODE combinational (virtual clock, period $period ns)"
}

report_utilization              -file $out_dir/utilization.rpt
report_utilization -hierarchical -file $out_dir/utilization_hier.rpt
report_timing_summary -file $out_dir/timing_summary.rpt
report_timing -delay_type max -max_paths 10 -nworst 10 -file $out_dir/timing.rpt
report_design_analysis -logic_level_distribution -file $out_dir/logic_levels.rpt

write_checkpoint -force $out_dir/post_synth.dcp

# Place and route. Post-synthesis timing uses ESTIMATED routing delays and is systematically
# optimistic; routing is where designs actually fail. Area moves too, since placement enables
# optimizations synthesis only guesses at. Any number that goes in a paper comes from here,
# not from the synth-only reports above.
if {$do_impl} {
    opt_design
    place_design
    phys_opt_design
    route_design

    if {!$is_ooc} {
        write_bitstream -force $out_dir/$top.bit
        puts "BUILD_TCL_BITSTREAM $out_dir/$top.bit"
    }

    report_utilization               -file $out_dir/utilization_routed.rpt
    report_utilization -hierarchical -file $out_dir/utilization_routed_hier.rpt
    report_timing_summary            -file $out_dir/timing_summary_routed.rpt
    report_timing -delay_type max -max_paths 10 -nworst 10 -file $out_dir/timing_routed.rpt
    report_power                     -file $out_dir/power_routed.rpt

    write_checkpoint -force $out_dir/post_route.dcp
    puts "BUILD_TCL_IMPL_DONE $top"
}

puts "BUILD_TCL_DONE $top"
