// Unit test for uart_loader, driven through a REAL uart_tx/uart_rx pair in both directions.
//
// Driving rx_valid/rx_data directly would test the protocol logic in isolation and run faster,
// but it would not catch the thing most likely to go wrong on the board: the composition. A
// byte-order slip between the loader and uart_rx, or a reply that races tx_busy, only shows up
// when the real serial modules are in the loop. Brief §12 risk #7 is exactly this class of bug.
//
// What it checks:
//   ping round-trip        proves the link and reply path before anything else. This is the
//                          command you type by hand when the board goes quiet.
//   load -> BRAM contents  read back through vector_store's own read port, so the test
//                          verifies what the classifier will actually see -- including the
//                          little-endian byte packing, which is silently transposable.
//   run command            correct n decoded and start pulsed exactly once
//   status reply           9 bytes, little-endian, matching the counters
//   back-to-back commands  the parser must return to S_CMD cleanly, not need a resync

`timescale 1ns / 1ps
`default_nettype none

module loader_tb;

    localparam integer CLK_HZ       = 1_600_000;
    localparam integer BAUD         = 100_000;      // CLKS_PER_BIT = 16
    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;

    localparam integer DATA_W  = 256;
    localparam integer LABEL_W = 3;
    localparam integer ADDR_W  = 6;
    localparam integer DEPTH   = 64;
    localparam integer N_VEC   = 4;

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    // ---- host -> device ----
    reg        h_start = 1'b0;
    reg  [7:0] h_data  = 8'd0;
    wire       h_busy;
    wire       h2d;

    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_host_tx (
        .clk(clk), .rst(rst), .start(h_start), .data(h_data), .tx(h2d), .busy(h_busy));

    wire [7:0] d_rx_data;
    wire       d_rx_valid;
    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_dev_rx (
        .clk(clk), .rst(rst), .rx(h2d),
        .data(d_rx_data), .valid(d_rx_valid), .frame_err());

    // ---- device -> host ----
    wire [7:0] d_tx_data;
    wire       d_tx_start;
    wire       d_tx_busy;
    wire       d2h;

    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_dev_tx (
        .clk(clk), .rst(rst), .start(d_tx_start), .data(d_tx_data),
        .tx(d2h), .busy(d_tx_busy));

    wire [7:0] h_rx_data;
    wire       h_rx_valid;
    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_host_rx (
        .clk(clk), .rst(rst), .rx(d2h),
        .data(h_rx_data), .valid(h_rx_valid), .frame_err());

    // ---- device under test ----
    wire                wr_en;
    wire [ADDR_W-1:0]   wr_addr;
    wire [DATA_W-1:0]   wr_data;
    wire [LABEL_W-1:0]  wr_label;
    wire                run_start;
    wire [ADDR_W:0]     run_n;

    reg  [31:0]     cycle_count   = 32'hDEADBEEF;
    reg  [ADDR_W:0] correct_count = 7'd77;
    reg             bench_busy    = 1'b0;

    uart_loader #(.DATA_W(DATA_W), .LABEL_W(LABEL_W), .ADDR_W(ADDR_W)) dut (
        .clk(clk), .rst(rst),
        .rx_data(d_rx_data), .rx_valid(d_rx_valid),
        .tx_data(d_tx_data), .tx_start(d_tx_start), .tx_busy(d_tx_busy),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data), .wr_label(wr_label),
        .run_start(run_start), .run_n(run_n),
        .bench_busy(bench_busy), .cycle_count(cycle_count),
        .correct_count(correct_count));

    // ---- vector store, so loads can be read back the way the classifier will see them ----
    reg  [ADDR_W-1:0]  rd_addr = 0;
    wire [DATA_W-1:0]  rd_data;
    wire [LABEL_W-1:0] rd_label;

    vector_store #(.DATA_W(DATA_W), .LABEL_W(LABEL_W), .DEPTH(DEPTH), .ADDR_W(ADDR_W))
        u_store (.clk(clk), .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
                 .wr_label(wr_label), .rd_addr(rd_addr), .rd_data(rd_data),
                 .rd_label(rd_label));

    // ---- host-side reply capture ----
    reg [7:0]  reply [0:15];
    integer    reply_n = 0;
    always @(posedge clk) begin
        if (!rst && h_rx_valid) begin
            reply[reply_n] = h_rx_data;
            reply_n        = reply_n + 1;
        end
    end

    integer run_pulses = 0;
    always @(posedge clk) if (!rst && run_start) run_pulses = run_pulses + 1;

    integer errors = 0;
    integer i, v;
    reg [DATA_W-1:0] expect_data [0:N_VEC-1];
    reg [LABEL_W-1:0] expect_label [0:N_VEC-1];

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

    task check(input [199:0] name, input integer got, input integer want);
        begin
            if (got !== want) begin
                $display("  MISMATCH %0s: got %0d (0x%0h) expected %0d (0x%0h)",
                         name, got, got, want, want);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        // ---------- 1. ping ----------
        reply_n = 0;
        send("P");
        repeat (CLKS_PER_BIT * 14) @(posedge clk);
        check("ping reply count", reply_n, 1);
        if (reply_n > 0) check("ping reply byte", reply[0], 8'hA5);

        // ---------- 2. load N_VEC vectors ----------
        for (v = 0; v < N_VEC; v = v + 1) begin
            for (i = 0; i < DATA_W/8; i = i + 1)
                expect_data[v][i*8 +: 8] = (v * 8'h11) + i[7:0];
            expect_label[v] = v[LABEL_W-1:0];
        end

        send("L");
        send(N_VEC[7:0]);
        send(8'h00);
        for (v = 0; v < N_VEC; v = v + 1) begin
            for (i = 0; i < DATA_W/8; i = i + 1)
                send(expect_data[v][i*8 +: 8]);
            send({{(8-LABEL_W){1'b0}}, expect_label[v]});
        end
        repeat (CLKS_PER_BIT * 4) @(posedge clk);

        for (v = 0; v < N_VEC; v = v + 1) begin
            @(negedge clk);
            rd_addr = v[ADDR_W-1:0];
            @(negedge clk);
            @(negedge clk);
            if (rd_data !== expect_data[v]) begin
                $display("  MISMATCH vector %0d data", v);
                $display("    got      %h", rd_data);
                $display("    expected %h", expect_data[v]);
                errors = errors + 1;
            end
            if (rd_label !== expect_label[v]) begin
                $display("  MISMATCH vector %0d label: got %0d expected %0d",
                         v, rd_label, expect_label[v]);
                errors = errors + 1;
            end
        end

        // ---------- 3. run ----------
        run_pulses = 0;
        send("R");
        send(8'd3);
        send(8'd0);
        repeat (CLKS_PER_BIT * 4) @(posedge clk);
        check("run pulses", run_pulses, 1);
        check("run_n", run_n, 3);

        // ---------- 4. status ----------
        bench_busy = 1'b1;
        reply_n    = 0;
        send("S");
        repeat (CLKS_PER_BIT * 110) @(posedge clk);
        check("status reply count", reply_n, 9);
        if (reply_n == 9) begin
            check("cycle[7:0]",    reply[0], 8'hEF);
            check("cycle[15:8]",   reply[1], 8'hBE);
            check("cycle[23:16]",  reply[2], 8'hAD);
            check("cycle[31:24]",  reply[3], 8'hDE);
            check("correct[7:0]",  reply[4], 8'd77);
            check("correct[15:8]", reply[5], 8'd0);
            check("flags",         reply[8], 8'd1);
        end

        // ---------- 5. parser still in sync afterwards ----------
        reply_n = 0;
        send("P");
        repeat (CLKS_PER_BIT * 14) @(posedge clk);
        check("ping after status", reply_n, 1);
        if (reply_n > 0) check("ping byte after status", reply[0], 8'hA5);

        $display("");
        $display("========================================");
        $display("uart_loader unit test (through real UART)");
        $display("  vectors loaded : %0d x %0d bytes", N_VEC, DATA_W/8 + 1);
        $display("  mismatches     : %0d", errors);
        if (errors == 0) $display("  RESULT         : PASS");
        else             $display("  RESULT         : FAIL");
        $display("========================================");
        $display("");
        $finish;
    end

    initial begin
        #50_000_000;
        $display("  RESULT         : FAIL (timeout)");
        $finish;
    end

endmodule

`default_nettype wire
