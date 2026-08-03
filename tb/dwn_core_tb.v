// GATE 1 -- the one non-negotiable rule (CLAUDE.md).
//
// The RTL core is not complete until it bit-exact matches the golden software model on every
// test vector including edge cases. This testbench is that check, and it is the ONLY
// correctness signal in the project: there is no second independent implementation to
// cross-check against.
//
// Scope: this tests the LUT CORE against pre-binarized inputs. It does not test the
// thermometer encoder, which is not built yet. That is deliberate -- the training notebook
// saves both x_raw and x_binarized precisely so a failure can be attributed to the encoder or
// to the core rather than to "somewhere in the design". The encoder gets its own Gate 1 pass.
//
// dwn_core is purely combinational (LUT lookups, popcount, argmax -- no state, no clock), so
// a settle delay per vector is sufficient; there is nothing to reset and nothing to pipeline
// yet. Pipeline registers come with the harness, and change this testbench when they do.

`timescale 1ns / 1ps
`default_nettype none

`include "vec_params.vh"

module dwn_core_tb;

    localparam integer N_VEC = `N_VEC;
    localparam integer VEC_W = `VEC_W;

    reg [VEC_W-1:0] vectors  [0:N_VEC-1];
    reg [7:0]       expected [0:N_VEC-1];

    reg  [VEC_W-1:0] x;
    wire [2:0]       class_idx;

    dwn_core dut (.x(x), .class_idx(class_idx));

    integer i;
    integer errors;
    integer first_bad;

    initial begin
        // Bare filenames: xsim runs with build/gate1 as its working directory, which is also
        // where tb/gen_vectors.py writes these.
        $readmemh("x_binarized.hex", vectors);
        $readmemh("expected.hex",    expected);

        errors    = 0;
        first_bad = -1;

        for (i = 0; i < N_VEC; i = i + 1) begin
            x = vectors[i];
            #1;
            // !== not != : an x or z on class_idx must count as a failure rather than
            // propagating silently into a comparison that returns x.
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
        $display("GATE 1 -- dwn_core vs golden model");
        $display("  vectors tested : %0d", N_VEC);
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
