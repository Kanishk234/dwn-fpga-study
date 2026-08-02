# Phase 1a LUT6 mapping probe -- Vivado batch script.
#
# Question: does a DWN LUT node (`assign out = TABLE[addr]`) map to exactly one LUT6?
# Everything downstream -- rtlgen's output format, the DSE area model, the "one neuron
# is one LUT6" claim -- rests on the answer.
#
# Out-of-context synthesis only: no XDC, no pins, no implementation. We are asking what
# primitive the construct becomes, not whether a design fits the board.
#
# Usage:
#   vivado -mode batch -source scripts/probe/probe.tcl -notrace
# Writes: build/probe/<variant>/{utilization.rpt,netlist.v}

set part      xc7a35tcpg236-1
set repo_root [file normalize [file dirname [info script]]/../..]
set rtl_dir   $repo_root/rtl/probe
set out_root  $repo_root/build/probe

# variant -> expected LUT count if the architectural premise holds (see gen_probe.py)
set variants {
    probe_a_baseline 37
    probe_b_param    37
    probe_c_romstyle 37
    probe_d_twolayer 60
}

# ---------------------------------------------------------------- part sanity check
if {[llength [get_parts -quiet $part]] == 0} {
    puts "\nERROR: part $part is not available in this Vivado install."
    puts "Artix-7 device support may not be installed. Check the installer's device list.\n"
    exit 1
}
puts "\n=== probe: part $part, Vivado [version -short] ===\n"

file mkdir $out_root
set summary {}

foreach {top expected} $variants {
    set out_dir $out_root/$top
    file mkdir $out_dir

    puts "\n--- synthesizing $top (expect $expected LUT6) ---"

    # Fresh in-memory project per variant so utilization is attributable to this top only.
    close_project -quiet
    create_project -in_memory -part $part
    read_verilog $rtl_dir/$top.v

    # -mode out_of_context: keep ports as ports, no I/O buffer insertion, no pin
    # constraints required. -flatten_hierarchy none so variant B's node instances stay
    # visible in the netlist rather than being dissolved before we can count them.
    synth_design -top $top -part $part -mode out_of_context -flatten_hierarchy none

    report_utilization -file $out_dir/utilization.rpt
    write_verilog -force -mode design $out_dir/netlist.v

    # ------------------------------------------------------ count primitives directly
    # The utilization report rolls LUT1..LUT6 into one "Slice LUTs" line, which would
    # hide the thing we care about. Count REF_NAMEs instead.
    set counts [dict create]
    foreach cell [get_cells -quiet -hier -filter {IS_PRIMITIVE}] {
        set ref [get_property REF_NAME $cell]
        dict incr counts $ref
    }

    set lut6   [expr {[dict exists $counts LUT6] ? [dict get $counts LUT6] : 0}]
    set total_lut 0
    set breakdown {}
    foreach ref [lsort [dict keys $counts]] {
        set n [dict get $counts $ref]
        lappend breakdown "$ref=$n"
        if {[string match "LUT*" $ref]} { incr total_lut $n }
    }

    set verdict [expr {$lut6 == $expected && $total_lut == $expected ? "PASS" : "MISMATCH"}]
    puts "$top: LUT6=$lut6  all-LUTs=$total_lut  expected=$expected  -> $verdict"
    puts "  primitives: [join $breakdown { }]"

    lappend summary [list $top $expected $lut6 $total_lut $verdict [join $breakdown " "]]
}

close_project -quiet

# ------------------------------------------------------------------------- summary
puts "\n\n================ PROBE SUMMARY ================"
puts [format "%-18s %9s %6s %9s  %s" variant expected LUT6 all-LUTs verdict]
foreach row $summary {
    lassign $row top expected lut6 total verdict _bd
    puts [format "%-18s %9d %6d %9d  %s" $top $expected $lut6 $total $verdict]
}
puts "=============================================="
puts "\nprimitive breakdowns:"
foreach row $summary {
    lassign $row top _e _l _t _v bd
    puts [format "  %-18s %s" $top $bd]
}
puts "\nreports + netlists under build/probe/<variant>/\n"
