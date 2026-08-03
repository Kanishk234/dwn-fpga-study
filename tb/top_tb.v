// Board-level integration test: Gate 1b, in simulation.
//
// Everything the real board does, except the silicon: a host drives the actual UART, loads
// REAL golden test vectors through the real protocol into the real BRAM, runs the real
// classifier, and reads back the accuracy count. Nothing is stubbed.
//
// The assertion is the strong one -- `correct_count` must equal the number of vectors loaded,
// because the labels loaded ARE the golden model's own predictions (expected_top.hex). So the
// hardware has to agree with the software model on every single sample, end to end, through
// the serial protocol. That is Gate 1b's claim, minus the physical board.
//
// What this still does NOT prove, and why Gate 1b remains open: real UART timing against a
// real FT2232HQ, bitstream-level behaviour, clock/reset on actual silicon, and timing closure.
// Brief §12 risk #7 exists precisely because those are not simulatable.
//
// Vectors come from build/tb/top/, put there by scripts/run_tb.py, which regenerates them from
// the checkpoint so this can never run against a stale set.

`timescale 1ns / 1ps
`default_nettype none

// Included explicitly rather than relying on the define leaking from dwn_basys3_top.v earlier
// in the compilation unit -- that works today and breaks the moment file order changes.
`include "dwn_top_params.vh"

module top_tb;

    localparam integer CLK_HZ       = 1_600_000;
    localparam integer BAUD         = 100_000;    // CLKS_PER_BIT = 16
    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;

    localparam integer DATA_W  = 256;
    localparam integer LABEL_W = 3;
    localparam integer ADDR_W  = 6;
    localparam integer DEPTH   = 64;
    localparam integer N_VEC   = 64;

    reg clk = 1'b0;
    reg rst_btn = 1'b1;
    always #5 clk = ~clk;

    // ---- host side ----
    reg        h_start = 1'b0;
    reg  [7:0] h_data  = 8'd0;
    wire       h_busy;
    wire       h2d;

    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_host_tx (
        .clk(clk), .rst(1'b0), .start(h_start), .data(h_data), .tx(h2d), .busy(h_busy));

    wire       d2h;
    wire [7:0] h_rx_data;
    wire       h_rx_valid;
    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_host_rx (
        .clk(clk), .rst(1'b0), .rx(d2h),
        .data(h_rx_data), .valid(h_rx_valid), .frame_err());

    // ---- device under test: the actual board top level ----
    reg  [1:0]  sw = 2'b00;
    wire [15:0] led;
    wire [6:0]  seg;
    wire [3:0]  an;
    wire        dp;

    dwn_basys3_top #(
        .CLK_HZ(CLK_HZ), .BAUD(BAUD), .DATA_W(DATA_W), .LABEL_W(LABEL_W),
        .DEPTH(DEPTH), .ADDR_W(ADDR_W), .REFRESH_BITS(4)
    ) dut (
        .clk(clk), .btnC(rst_btn), .sw(sw),
        .RsRx(h2d), .RsTx(d2h),
        .led(led), .seg(seg), .an(an), .dp(dp));

    // ---- golden data ----
    reg [DATA_W-1:0] vec   [0:N_VEC-1];
    reg [7:0]        label [0:N_VEC-1];

    reg [7:0]  reply [0:15];
    integer    reply_n = 0;
    always @(posedge clk) begin
        if (h_rx_valid) begin
            reply[reply_n] = h_rx_data;
            reply_n        = reply_n + 1;
        end
    end

    integer errors = 0;
    integer i, v;
    integer got_cycles, got_correct;

    task send(input [7:0] b);
        begin
            @(posedge clk);
            while (h_busy) @(posedge clk);
            h_data  <= b;
            h_start <= 1'b1;
            @(posedge clk);
            h_start <= 1'b0;
            @(posedge clk);
            while (h_busy) @(posedge clk);
        end
    endtask

    initial begin
        $readmemh("x_quant.hex",      vec);
        $readmemh("expected_top.hex", label);

        repeat (4) @(posedge clk);
        rst_btn = 1'b0;
        repeat (4) @(posedge clk);

        // ---- 1. ping: prove the link before trusting anything else ----
        reply_n = 0;
        send("P");
        repeat (CLKS_PER_BIT * 14) @(posedge clk);
        if (reply_n != 1 || reply[0] !== 8'hA5) begin
            $display("  ERROR: ping failed (%0d bytes, first = %02h)", reply_n, reply[0]);
            errors = errors + 1;
        end

        // ---- 2. load the golden vectors with the golden predictions as labels ----
        send("L");
        send(N_VEC[7:0]);
        send(8'h00);
        for (v = 0; v < N_VEC; v = v + 1) begin
            for (i = 0; i < DATA_W/8; i = i + 1)
                send(vec[v][i*8 +: 8]);
            send(label[v]);
        end
        repeat (CLKS_PER_BIT * 4) @(posedge clk);

        // ---- 3. run ----
        send("R");
        send(N_VEC[7:0]);
        send(8'h00);
        repeat (CLKS_PER_BIT * 4) @(posedge clk);
        wait (led[1] == 1'b1);          // done_sticky
        repeat (10) @(posedge clk);

        // ---- 4. read the result back over the wire ----
        reply_n = 0;
        send("S");
        repeat (CLKS_PER_BIT * 110) @(posedge clk);

        if (reply_n != 9) begin
            $display("  ERROR: status returned %0d bytes, expected 9", reply_n);
            errors = errors + 1;
        end else begin
            got_cycles  = {reply[3], reply[2], reply[1], reply[0]};
            got_correct = {reply[7], reply[6], reply[5], reply[4]};

            // Labels loaded were the golden model's own predictions, so anything less than a
            // perfect score is a hardware/software disagreement.
            if (got_correct !== N_VEC) begin
                $display("  ERROR: correct_count = %0d, expected %0d -- hardware disagrees with the golden model",
                         got_correct, N_VEC);
                errors = errors + 1;
            end
            // II=1 end to end: n issue cycles plus the pipeline depth.
            if (got_cycles !== N_VEC + `DWN_TOP_LATENCY + 1) begin
                $display("  ERROR: cycle_count = %0d, expected %0d (II=1 violated)",
                         got_cycles, N_VEC + `DWN_TOP_LATENCY + 1);
                errors = errors + 1;
            end
            $display("  vectors        : %0d", N_VEC);
            $display("  correct        : %0d / %0d", got_correct, N_VEC);
            $display("  cycles         : %0d (expected %0d, II=1)",
                     got_cycles, N_VEC + `DWN_TOP_LATENCY + 1);
        end

        // ---- 5. the display and LEDs are actually driven ----
        if (^seg === 1'bx || ^an === 1'bx) begin
            $display("  ERROR: seven-segment outputs are undriven/X");
            errors = errors + 1;
        end
        if (led[2] !== 1'b0) begin
            $display("  ERROR: framing errors were flagged during a clean run");
            errors = errors + 1;
        end

        $display("");
        $display("========================================");
        $display("BOARD INTEGRATION -- Gate 1b in simulation");
        $display("  path           : host UART -> loader -> BRAM -> dwn_top -> counters -> UART");
        $display("  mismatches     : %0d", errors);
        if (errors == 0) $display("  RESULT         : PASS");
        else             $display("  RESULT         : FAIL");
        $display("========================================");
        $display("");
        $finish;
    end

    initial begin
        #500_000_000;
        $display("  RESULT         : FAIL (timeout)");
        $finish;
    end

endmodule

`default_nettype wire
