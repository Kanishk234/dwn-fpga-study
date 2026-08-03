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

if {$argc < 4} {
    puts "ERROR: expected <top> <part> <out_dir> <sources...>"
    exit 1
}

set top     [lindex $argv 0]
set part    [lindex $argv 1]
set out_dir [lindex $argv 2]
set sources [lrange $argv 3 end]

file mkdir $out_dir

foreach f $sources {
    puts "read_verilog $f"
    read_verilog $f
}

# -flatten_hierarchy none keeps module boundaries so report_utilization -hierarchical can
# attribute LUTs to the encoder vs the core. Brief §6 requires reporting them separately in
# every table we publish, so the flow has to be able to tell them apart.
synth_design -top $top -part $part -mode out_of_context -flatten_hierarchy none

# A virtual clock so the purely combinational path gets a timing number. There are no
# registers yet, so this measures input-to-output delay -- which is exactly what decides how
# many pipeline stages are needed to reach a target clock (brief §9 wants II=1).
create_clock -name vclk -period 10.000
set_input_delay  0.000 -clock vclk [all_inputs]
set_output_delay 0.000 -clock vclk [all_outputs]

report_utilization              -file $out_dir/utilization.rpt
report_utilization -hierarchical -file $out_dir/utilization_hier.rpt
report_timing -delay_type max -max_paths 10 -nworst 10 -file $out_dir/timing.rpt
report_design_analysis -logic_level_distribution -file $out_dir/logic_levels.rpt

write_checkpoint -force $out_dir/post_synth.dcp

puts "BUILD_TCL_DONE $top"
