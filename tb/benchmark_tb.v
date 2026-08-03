// Unit test for vector_store + benchmark_fsm.
//
// The classifier is a STUB here, not the real dwn_top: a fixed function of the input with the
// same LATENCY. That is deliberate. This test is about the things that go wrong in the
// harness, not in the datapath -- the datapath already has Gate 1. Using a stub means a
// failure here can only be an addressing, alignment, or counting bug.
//
// What it checks, and why each one is a real failure mode:
//   cycle count exact      II=1 is a claim about throughput. A design that stalls one cycle
//                          per vector still produces correct answers and a wrong headline
//                          number, so the count is asserted exactly, not as a bound.
//   correct count          the label delay line has to align predictions with the labels of
//                          the vectors that produced them. Misalignment yields a plausible
//                          but wrong accuracy -- the worst kind of bug for Gate 1b.
//   deliberate mismatches  a scorer that counted everything as correct would pass a test
//                          where everything happens to be correct, so some labels are wrong
//                          on purpose and the expected total is checked.
//   back-to-back batches   Gate 1b runs in batches because the test set does not fit on the
//                          device, so a second run must reset counters cleanly.

`timescale 1ns / 1ps
`default_nettype none

module benchmark_tb;

    localparam integer DATA_W  = 256;
    localparam integer LABEL_W = 3;
    localparam integer DEPTH   = 64;
    localparam integer ADDR_W  = 6;
    localparam integer LATENCY = 4;

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    // --- vector store ---
    reg                 wr_en    = 1'b0;
    reg  [ADDR_W-1:0]   wr_addr  = 0;
    reg  [DATA_W-1:0]   wr_data  = 0;
    reg  [LABEL_W-1:0]  wr_label = 0;

    wire [ADDR_W-1:0]   rd_addr;
    wire [DATA_W-1:0]   rd_data;
    wire [LABEL_W-1:0]  rd_label;

    vector_store #(.DATA_W(DATA_W), .LABEL_W(LABEL_W), .DEPTH(DEPTH), .ADDR_W(ADDR_W))
        u_store (.clk(clk), .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
                 .wr_label(wr_label), .rd_addr(rd_addr), .rd_data(rd_data),
                 .rd_label(rd_label));

    // --- stub classifier: class = low 3 bits of the vector, delayed by LATENCY ---
    reg [LABEL_W-1:0] stub [0:LATENCY-1];
    integer s;
    always @(posedge clk) begin
        stub[0] <= rd_data[LABEL_W-1:0];
        for (s = 1; s < LATENCY; s = s + 1)
            stub[s] <= stub[s-1];
    end
    wire [LABEL_W-1:0] class_idx = stub[LATENCY-1];

    // --- FSM ---
    reg              start = 1'b0;
    reg  [ADDR_W:0]  n_vectors = 0;
    wire             busy, done;
    wire [31:0]      cycle_count;
    wire [ADDR_W:0]  correct_count;

    wire [LABEL_W-1:0] last_class;

    benchmark_fsm #(.ADDR_W(ADDR_W), .LABEL_W(LABEL_W), .LATENCY(LATENCY))
        u_fsm (.clk(clk), .rst(rst), .start(start), .n_vectors(n_vectors),
               .rd_addr(rd_addr), .rd_label(rd_label), .class_idx(class_idx),
               .busy(busy), .done(done), .cycle_count(cycle_count),
               .correct_count(correct_count), .last_class(last_class));

    integer errors = 0;
    integer i;
    integer expect_correct;
    integer n_run;

    task load(input integer count, input integer n_wrong);
        integer j;
        begin
            expect_correct = 0;
            for (j = 0; j < count; j = j + 1) begin
                @(negedge clk);
                wr_en    = 1'b1;
                wr_addr  = j[ADDR_W-1:0];
                wr_data  = {{(DATA_W-LABEL_W){1'b0}}, j[LABEL_W-1:0]};
                // The first n_wrong entries get a label that does NOT match what the stub
                // will predict, so the scorer has something to get wrong.
                // XOR with all-ones always differs, so these are guaranteed misses.
                if (j < n_wrong) wr_label = j[LABEL_W-1:0] ^ 3'b111;
                else begin
                    wr_label = j[LABEL_W-1:0];
                    expect_correct = expect_correct + 1;
                end
            end
            @(negedge clk);
            wr_en = 1'b0;
        end
    endtask

    task run_batch(input integer count);
        begin
            @(negedge clk);
            n_vectors = count[ADDR_W:0];
            start     = 1'b1;
            @(negedge clk);
            start = 1'b0;
            wait (done == 1'b1);
            @(negedge clk);
        end
    endtask

    task check(input [127:0] name, input integer got, input integer want);
        begin
            if (got !== want) begin
                $display("  MISMATCH %0s: got %0d expected %0d", name, got, want);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        // ---- batch 1: 32 vectors, 5 labelled wrong ----
        n_run = 32;
        load(n_run, 5);
        run_batch(n_run);
        // One vector issued per cycle, then the pipe drains: LATENCY + 1 stages.
        check("cycle_count", cycle_count, n_run + LATENCY + 1);
        check("correct_count", correct_count, expect_correct);
        $display("  batch 1: %0d vectors, %0d cycles, %0d correct (expected %0d)",
                 n_run, cycle_count, correct_count, expect_correct);

        // ---- batch 2: different size, all labels correct. Counters must reset. ----
        n_run = 16;
        load(n_run, 0);
        run_batch(n_run);
        check("cycle_count (batch 2)", cycle_count, n_run + LATENCY + 1);
        check("correct_count (batch 2)", correct_count, expect_correct);
        $display("  batch 2: %0d vectors, %0d cycles, %0d correct (expected %0d)",
                 n_run, cycle_count, correct_count, expect_correct);

        // ---- batch 3: single vector, the degenerate case where the pipe never fills ----
        n_run = 1;
        load(n_run, 0);
        run_batch(n_run);
        check("cycle_count (single)", cycle_count, n_run + LATENCY + 1);
        check("correct_count (single)", correct_count, 1);
        // The stub predicts the low bits of the vector, and vector 0 holds 0.
        check("last_class held after run", last_class, 0);
        $display("  batch 3: %0d vector, %0d cycles, %0d correct",
                 n_run, cycle_count, correct_count);

        $display("");
        $display("========================================");
        $display("benchmark_fsm + vector_store unit test");
        $display("  mismatches     : %0d", errors);
        if (errors == 0) $display("  RESULT         : PASS");
        else             $display("  RESULT         : FAIL");
        $display("========================================");
        $display("");
        $finish;
    end

    initial begin
        #500_000;
        $display("  RESULT         : FAIL (timeout -- FSM never asserted done)");
        $finish;
    end

endmodule

`default_nettype wire
