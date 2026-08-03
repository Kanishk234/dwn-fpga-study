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
    parameter integer BAUD         = 115_200,
    parameter integer DATA_W       = 256,      // 16 features x 16-bit Q3.12
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

    dwn_top u_dwn (.clk(clk), .x_flat(rd_data), .class_idx(class_idx));

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
    wire [15:0] disp_value = sw[0] ? cycle_count[15:0]
                                   : {13'd0, last_class};

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

    assign led[0]    = bench_busy;
    assign led[1]    = done_sticky;
    assign led[2]    = frame_err_sticky;
    assign led[3]    = sw[0];
    assign led[7:4]  = 4'd0;
    assign led[15:8] = correct_count[7:0];

endmodule

`default_nettype wire
