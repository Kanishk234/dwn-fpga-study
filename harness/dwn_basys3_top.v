// Board top level: UART <-> vector store <-> DWN classifier <-> benchmark FSM.
//
// Everything below it has its own unit test; this is the wiring that turns them into something
// that can be programmed onto a Basys 3 and run Gate 1b.
//
// ---------------------------------------------------------------------------------------
// ON "TWO MODES" -- a deliberate simplification of brief §9, stated rather than smuggled in.
//
// The brief describes benchmark mode (vectors preloaded in BRAM, run flat out) and interactive
// mode (one sample over UART, prediction on the display) as two data paths. This builds ONE
// path, because the host can produce interactive behaviour with it exactly: `L 1 <record>`
// then `R 1` classifies a single sample and lights the display, and it is still visibly,
// honestly slow because the UART dominates. A second datapath would duplicate the classifier
// interface and add a second thing to verify for no behavioural difference.
//
// The switch therefore selects what the DISPLAY shows, not which datapath runs -- which is
// what brief §9 actually asks the 7-segment for ("class or measured throughput").
// ---------------------------------------------------------------------------------------
//
// The classifier's pipeline depth is taken from the GENERATED header, not written out here.
// benchmark_fsm aligns labels against predictions using that number, so a hand-copied value
// that drifts from the emitted pipeline would silently score every sample against the wrong
// answer -- see the ledger; that bug has already happened once.

`timescale 1ns / 1ps
`default_nettype none

`include "dwn_top_params.vh"

module dwn_basys3_top #(
    parameter integer CLK_HZ       = 100_000_000,
    // 5 Mbaud: 100_000_000 / 5_000_000 = 20 clocks per bit exactly, and the FT2232H divides
    // 120 MHz by 24 to reach it exactly too. Zero rounding error on either end.
    //
    // MEASURED CEILING, not a guess. 1 M, 2 M, 4 M and 5 M all work; 10 M does not respond at
    // all, even though both ends divide it exactly (100/10 and 120/12) -- so the limit is the
    // FTDI Windows VCP driver, not this design. Chasing higher would mean FTDI's D2XX API
    // instead of a COM port. See the sweep table in docs/phase1-ledger.md.
    //
    // Full 166k Gate 1b run: 480 s at 115200 -> 55 s at 1 M -> 11.2 s here.
    //
    // Override per build without editing this file:
    //     scripts/build_bitstream.py --baud 2000000
    // The host must match; a mismatch presents as a failed ping rather than corrupt data.
    parameter integer BAUD         = 5_000_000,
    parameter integer FEATURES     = 16,       // input features, from the model
    parameter integer WORD_BITS    = 16,      // bits per feature INSIDE the model
    parameter integer DATA_W       = 256,      // FEATURES * 8 * ceil(WORD_BITS/8), byte lanes
    parameter integer LABEL_W      = 3,
    parameter integer DEPTH        = 1024,
    parameter integer ADDR_W       = 10,
    parameter integer REFRESH_BITS = 17
)(
    input  wire        clk,        // 100 MHz, pin W5
    input  wire        btnC,       // reset, active high
    input  wire [1:0]  sw,         // sw[0]: 0 = show class, 1 = show cycle count
    input  wire        RsRx,
    output wire        RsTx,
    output wire [15:0] led,
    output wire [6:0]  seg,
    output wire [3:0]  an,
    output wire        dp
);

    localparam integer LATENCY = `DWN_TOP_LATENCY;

    // btnC is asynchronous to clk and mechanically bouncy. Two flops remove the metastability
    // risk; bounce is harmless for a reset that is held for milliseconds either way.
    reg rst_meta, rst_sync;
    always @(posedge clk) begin
        rst_meta <= btnC;
        rst_sync <= rst_meta;
    end
    wire rst = rst_sync;

    // ---- serial ----
    wire [7:0] rx_data;
    wire       rx_valid;
    wire       rx_frame_err;

    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_rx (
        .clk(clk), .rst(rst), .rx(RsRx),
        .data(rx_data), .valid(rx_valid), .frame_err(rx_frame_err));

    wire [7:0] tx_data;
    wire       tx_start;
    wire       tx_busy;

    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_tx (
        .clk(clk), .rst(rst), .start(tx_start), .data(tx_data),
        .tx(RsTx), .busy(tx_busy));

    // ---- host protocol ----
    wire                wr_en;
    wire [ADDR_W-1:0]   wr_addr;
    wire [DATA_W-1:0]   wr_data;
    wire [LABEL_W-1:0]  wr_label;
    wire                run_start;
    wire [ADDR_W:0]     run_n;

    wire                bench_busy;
    wire                bench_done;
    wire [31:0]         cycle_count;
    wire [ADDR_W:0]     correct_count;
    wire [LABEL_W-1:0]  last_class;

    uart_loader #(.DATA_W(DATA_W), .LABEL_W(LABEL_W), .ADDR_W(ADDR_W)) u_loader (
        .clk(clk), .rst(rst),
        .rx_data(rx_data), .rx_valid(rx_valid),
        .tx_data(tx_data), .tx_start(tx_start), .tx_busy(tx_busy),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data), .wr_label(wr_label),
        .run_start(run_start), .run_n(run_n),
        .bench_busy(bench_busy), .cycle_count(cycle_count),
        .correct_count(correct_count));

    // ---- vector store ----
    wire [ADDR_W-1:0]   rd_addr;
    wire [DATA_W-1:0]   rd_data;
    wire [LABEL_W-1:0]  rd_label;

    vector_store #(.DATA_W(DATA_W), .LABEL_W(LABEL_W), .DEPTH(DEPTH), .ADDR_W(ADDR_W))
        u_store (.clk(clk),
                 .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data), .wr_label(wr_label),
                 .rd_addr(rd_addr), .rd_data(rd_data), .rd_label(rd_label));

    // ---- the model ----
    wire [LABEL_W-1:0] class_idx;

    // UNPACK the store's byte-padded lanes into the packed word the model expects.
    //
    // A feature travels over UART in whole bytes, so a 9-bit word occupies 16 bits on the wire
    // and in the store: DATA_W = FEATURES * 8 * ceil(WORD_BITS/8). `dwn_top` packs at exactly
    // WORD_BITS, so its x_flat is FEATURES * WORD_BITS wide. For JSC the two coincide -- a
    // 16-bit word is exactly two bytes -- which is why this never existed before and why
    // wiring rd_data straight through elaborated for years.
    //
    // Taking the low WORD_BITS of each lane is correct because the host packs little-endian and
    // pads with the sign extension, so the discarded bits carry no information.
    localparam integer LANE_W = 8 * ((WORD_BITS + 7) / 8);   // wire lane, whole bytes
    wire [FEATURES*WORD_BITS-1:0] x_packed;
    genvar fi;
    generate
        for (fi = 0; fi < FEATURES; fi = fi + 1) begin : g_unpack
            assign x_packed[fi*WORD_BITS +: WORD_BITS] = rd_data[fi*LANE_W +: WORD_BITS];
        end
    endgenerate

    dwn_top u_dwn (.clk(clk), .x_flat(x_packed), .class_idx(class_idx));

    // ---- benchmark runner ----
    benchmark_fsm #(.ADDR_W(ADDR_W), .LABEL_W(LABEL_W), .LATENCY(LATENCY)) u_bench (
        .clk(clk), .rst(rst),
        .start(run_start), .n_vectors(run_n),
        .rd_addr(rd_addr), .rd_label(rd_label), .class_idx(class_idx),
        .busy(bench_busy), .done(bench_done),
        .cycle_count(cycle_count), .correct_count(correct_count),
        .last_class(last_class));

    // ---- display ----
    // sw[0] picks class vs throughput; the class is shown in the low digit so a single-sample
    // run reads directly off the board with no host involved.
    // sw[1:0] selects what the four hex digits show. correct_count gets a slot because it is
    // the number that actually matters during a run and the LED byte below truncates it: a
    // perfect 1024-sample batch is 0x400, whose low byte is 0x00, i.e. all LEDs dark.
    //   00  last predicted class   0=g 1=q 2=t 3=w 4=z (alphabetical, NOT physics order)
    //   01  cycle count, low 16    1029 = 0x0405 for a full 1024-sample batch
    //   10  correct count          0x0400 when a 1024-sample batch is perfect
    //   11  cycle count, high 16   nonzero only past 65535 cycles
    reg [15:0] disp_value;
    always @* begin
        case (sw)
            2'b00:   disp_value = {13'd0, last_class};
            2'b01:   disp_value = cycle_count[15:0];
            2'b10:   disp_value = {{(16-ADDR_W-1){1'b0}}, correct_count};
            default: disp_value = cycle_count[31:16];
        endcase
    end

    seg7 #(.REFRESH_BITS(REFRESH_BITS)) u_seg (
        .clk(clk), .rst(rst), .value(disp_value), .seg(seg), .an(an));

    assign dp = 1'b1;      // decimal point off (active low)

    // ---- status LEDs ----
    // Sticky flags, because the events they report are single-cycle and would otherwise be
    // invisible. A framing error that happened once is exactly what you need to see when the
    // link looks fine but the data is wrong.
    reg frame_err_sticky, done_sticky;
    always @(posedge clk) begin
        if (rst) begin
            frame_err_sticky <= 1'b0;
            done_sticky      <= 1'b0;
        end else begin
            if (rx_frame_err) frame_err_sticky <= 1'b1;
            if (bench_done)   done_sticky      <= 1'b1;
        end
    end

    assign led[0]    = bench_busy;        // a run is in progress
    assign led[1]    = done_sticky;       // at least one run has completed since reset
    assign led[2]    = frame_err_sticky;  // a UART framing error happened -- bytes are corrupt
    assign led[3]    = sw[0];             // display selector, echoed so the mode is visible
    assign led[4]    = sw[1];
    assign led[7:5]  = 3'd0;
    // Low bits only, so this wraps: a perfect full batch reads 0x00. Use sw=10 on the
    // seven-segment for the real value; this is a coarse "something is happening" indicator.
    //
    // Zero-extended rather than sliced [7:0]: correct_count is ADDR_W+1 wide, and a store small
    // enough to need only 6 address bits makes that 7 -- narrower than the 8 LEDs. JSC's
    // ADDR_W=10 hid this. A part-select past the end of a vector is an ERROR in synthesis, not
    // a truncation, so this failed the build outright rather than quietly.
    assign led[15:8] = (ADDR_W + 1 >= 8) ? correct_count[7:0]
                                         : {{(8 - (ADDR_W + 1)){1'b0}}, correct_count};

endmodule

`default_nettype wire
