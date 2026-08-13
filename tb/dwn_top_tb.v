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
// Pipelined, one vector per clock. Streaming back-to-back and getting every result right is
// what actually proves II=1 (brief §9); a design that computed correctly but stalled would
// pass a one-vector-at-a-time testbench and fail this one.

`timescale 1ns / 1ps
`default_nettype none

`include "top_params.vh"
`include "dwn_top_params.vh"

module dwn_top_tb;

    localparam integer N_TOP   = `N_TOP;
    localparam integer X_W     = `X_W;
    localparam integer LATENCY = `DWN_TOP_LATENCY;

    reg [X_W-1:0] vectors  [0:N_TOP-1];
    reg [7:0]     expected [0:N_TOP-1];

    reg            clk = 1'b0;
    reg  [X_W-1:0] x_flat;
    wire [`IDX_W-1:0]     class_idx;

    always #5 clk = ~clk;          // 100 MHz, the Basys 3 board clock

    dwn_top dut (.clk(clk), .x_flat(x_flat), .class_idx(class_idx));

    integer i, j;
    integer errors;
    integer first_bad;

    initial begin
        $readmemh("x_quant.hex",      vectors);
        $readmemh("expected_top.hex", expected);

        errors    = 0;
        first_bad = -1;
        x_flat    = {X_W{1'b0}};

        for (i = 0; i < N_TOP + LATENCY; i = i + 1) begin
            @(negedge clk);
            if (i >= LATENCY) begin
                j = i - LATENCY;
                if (class_idx !== expected[j][`IDX_W-1:0]) begin
                    if (first_bad == -1) first_bad = j;
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("  MISMATCH vector %0d: rtl=%0d golden=%0d",
                                 j, class_idx, expected[j][`IDX_W-1:0]);
                end
            end
            x_flat = (i < N_TOP) ? vectors[i] : {X_W{1'b0}};
        end

        $display("");
        $display("========================================");
        $display("GATE 1 -- dwn_top (encoder + core) vs golden model");
        $display("  vectors tested : %0d", N_TOP);
        $display("  latency        : %0d cycles, II=1 (new vector every clock)", LATENCY);
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
