// GATE 1, top level -- thermometer encoder + LUT core, end to end.
//
// tb/dwn_core_tb.v checks the core against pre-binarized inputs. This checks the whole
// datapath from quantized features: encoder -> core -> class index. Running both means a
// failure localizes itself -- if this fails while the core testbench passes, the encoder is at
// fault and nothing else needs re-examining.
//
// The golden model quantizes exactly as this design does (Q3.12 signed), so "bit-exact" is
// still absolute here. Quantization is part of the spec, not a tolerance.
//
// Combinational end to end -- no pipeline registers yet, so a settle delay per vector is
// sufficient. When brief §9's II=1 registers land, this testbench changes with them.

`timescale 1ns / 1ps
`default_nettype none

`include "top_params.vh"

module dwn_top_tb;

    localparam integer N_TOP = `N_TOP;
    localparam integer X_W   = `X_W;

    reg [X_W-1:0] vectors  [0:N_TOP-1];
    reg [7:0]     expected [0:N_TOP-1];

    reg  [X_W-1:0] x_flat;
    wire [2:0]     class_idx;

    dwn_top dut (.x_flat(x_flat), .class_idx(class_idx));

    integer i;
    integer errors;
    integer first_bad;

    initial begin
        $readmemh("x_quant.hex",     vectors);
        $readmemh("expected_top.hex", expected);

        errors    = 0;
        first_bad = -1;

        for (i = 0; i < N_TOP; i = i + 1) begin
            x_flat = vectors[i];
            #1;
            // !== not != : an x or z must count as a failure rather than propagating
            // silently into a comparison that returns x.
            if (class_idx !== expected[i][2:0]) begin
                if (first_bad == -1) first_bad = i;
                errors = errors + 1;
                if (errors <= 10)
                    $display("  MISMATCH vector %0d: rtl=%0d golden=%0d",
                             i, class_idx, expected[i][2:0]);
            end
        end

        $display("");
        $display("========================================");
        $display("GATE 1 -- dwn_top (encoder + core) vs golden model");
        $display("  vectors tested : %0d", N_TOP);
        $display("  mismatches     : %0d", errors);
        if (errors == 0) begin
            $display("  RESULT         : PASS (bit-exact on every vector)");
        end else begin
            $display("  RESULT         : FAIL (first mismatch at vector %0d)", first_bad);
        end
        $display("========================================");
        $display("");
        $finish;
    end

endmodule

`default_nettype wire
