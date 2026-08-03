// Benchmark mode: stream preloaded vectors through the classifier flat out, count cycles and
// correct predictions in hardware.
//
// This produces the two numbers Phase 1 exists to produce:
//   cycle_count    -> core throughput, measured rather than inferred. Comparable to the
//                     paper, because it excludes the serial link entirely (brief §9).
//   correct_count  -> Gate 1b's accuracy check, accumulated on-chip so only a running total
//                     crosses the UART instead of one prediction per sample.
//
// II=1 is the whole point: one vector issued per clock, no gaps. cycle_count should come back
// as exactly n_vectors + LATENCY + 1 -- the issue cycles plus the depth of the BRAM read and
// the classifier pipeline. Anything larger means something stalled and the throughput claim is
// wrong, so the testbench asserts the exact value rather than a bound.
//
// ALIGNMENT IS THE SUBTLE PART, and it is where this module was wrong on the first attempt.
// The BRAM read and the classifier are pipelines with different entry points:
//
//   posedge t      address A issued on rd_addr
//   posedge t+1    vector_store latches mem[A]; rd_data/rd_label valid during t+1..t+2
//   posedge t+2    the label can first be CAPTURED into a delay stage
//   posedge t+1+L  class_idx for A is valid  (L = classifier LATENCY)
//
// So a label captured at t+2 needs L-1 further stages, i.e. a delay line of depth L, and the
// "this stage holds a real vector" flag has to enter that line at the same cycle the label
// does -- not at issue time. The first version fed the valid flag in at issue and the label in
// two cycles later, so every prediction was scored against the wrong sample's label. That
// still produces a plausible accuracy number, which is precisely why the unit test asserts an
// exact expected count with some labels deliberately wrong.

`timescale 1ns / 1ps
`default_nettype none

module benchmark_fsm #(
    parameter integer ADDR_W  = 10,
    parameter integer LABEL_W = 3,
    parameter integer LATENCY = 4      // classifier latency; the BRAM read adds 1 more
)(
    input  wire                clk,
    input  wire                rst,

    input  wire                start,        // one-cycle pulse
    input  wire [ADDR_W:0]     n_vectors,    // how many to run this batch (1..DEPTH)

    // to vector_store
    output reg  [ADDR_W-1:0]   rd_addr,
    input  wire [LABEL_W-1:0]  rd_label,

    // from the classifier
    input  wire [LABEL_W-1:0]  class_idx,

    output reg                 busy,
    output reg                 done,         // one-cycle pulse when the batch completes
    output reg  [31:0]         cycle_count,  // issue of first vector -> last result
    output reg  [ADDR_W:0]     correct_count,
    output reg  [LABEL_W-1:0]  last_class    // most recent prediction, held for the display
);

    localparam [1:0] S_IDLE  = 2'd0,
                     S_ISSUE = 2'd1,   // issuing addresses, back to back
                     S_DRAIN = 2'd2;   // addresses exhausted, pipeline still emptying

    reg [1:0]      state;
    reg [ADDR_W:0] issued;
    reg [ADDR_W:0] retired;
    reg [ADDR_W:0] n_latched;

    // Two stages of "an address is in flight" ahead of the delay line, so the valid flag and
    // the label enter it on the same cycle. addr_valid marks rd_addr; label_valid marks
    // rd_label one cycle later.
    reg addr_valid;
    reg label_valid;

    reg [LABEL_W-1:0] lab [0:LATENCY-1];
    reg [LATENCY-1:0] vld;

    // The prediction leaving the pipeline this cycle, and whether it is real.
    wire            retire_now   = (state != S_IDLE) && vld[LATENCY-1];
    wire            hit          = (class_idx == lab[LATENCY-1]);

    integer k;

    always @(posedge clk) begin
        if (rst) begin
            state         <= S_IDLE;
            rd_addr       <= {ADDR_W{1'b0}};
            issued        <= {(ADDR_W+1){1'b0}};
            retired       <= {(ADDR_W+1){1'b0}};
            n_latched     <= {(ADDR_W+1){1'b0}};
            busy          <= 1'b0;
            done          <= 1'b0;
            cycle_count   <= 32'd0;
            correct_count <= {(ADDR_W+1){1'b0}};
            last_class    <= {LABEL_W{1'b0}};
            addr_valid    <= 1'b0;
            label_valid   <= 1'b0;
            vld           <= {LATENCY{1'b0}};
            for (k = 0; k < LATENCY; k = k + 1)
                lab[k] <= {LABEL_W{1'b0}};
        end else begin
            done <= 1'b0;

            // Delay line. Advances every cycle a run is active; the classifier has no enable
            // and never stalls, so this must not either.
            if (state != S_IDLE) begin
                for (k = LATENCY-1; k > 0; k = k - 1) begin
                    lab[k] <= lab[k-1];
                    vld[k] <= vld[k-1];
                end
                lab[0]      <= rd_label;      // valid now iff label_valid is set
                vld[0]      <= label_valid;
                label_valid <= addr_valid;
                addr_valid  <= 1'b0;          // overridden below when an address is issued
            end

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start && n_vectors != 0) begin
                        rd_addr       <= {ADDR_W{1'b0}};
                        issued        <= {{ADDR_W{1'b0}}, 1'b1};
                        retired       <= {(ADDR_W+1){1'b0}};
                        n_latched     <= n_vectors;
                        cycle_count   <= 32'd0;
                        correct_count <= {(ADDR_W+1){1'b0}};
                        addr_valid    <= 1'b1;          // address 0 is on the bus
                        label_valid   <= 1'b0;
                        vld           <= {LATENCY{1'b0}};
                        busy          <= 1'b1;
                        state         <= S_ISSUE;
                    end
                end

                S_ISSUE: begin
                    cycle_count <= cycle_count + 32'd1;
                    if (issued < n_latched) begin
                        rd_addr    <= rd_addr + {{(ADDR_W-1){1'b0}}, 1'b1};
                        issued     <= issued + {{ADDR_W{1'b0}}, 1'b1};
                        addr_valid <= 1'b1;
                    end else begin
                        state <= S_DRAIN;
                    end
                end

                S_DRAIN: begin
                    cycle_count <= cycle_count + 32'd1;
                    // Include this cycle's retirement in the completion test, so `done` lands
                    // on the cycle the last result actually emerges rather than one later.
                    if (retired + {{ADDR_W{1'b0}}, retire_now} >= n_latched) begin
                        busy  <= 1'b0;
                        done  <= 1'b1;
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase

            if (retire_now) begin
                retired    <= retired + {{ADDR_W{1'b0}}, 1'b1};
                last_class <= class_idx;    // held after the run for the 7-segment display
                if (hit)
                    correct_count <= correct_count + {{ADDR_W{1'b0}}, 1'b1};
            end
        end
    end

endmodule

`default_nettype wire
