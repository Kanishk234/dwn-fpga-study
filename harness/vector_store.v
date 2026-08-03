// Test-vector store: dual-port BRAM holding quantized feature vectors and their true labels.
//
// This is what makes benchmark mode possible. Streaming samples over UART is ~1000x slower
// than the core even at 12 Mbaud (brief §6), so measuring core throughput requires the vectors
// to already be on-chip. Preload here, then run the pipeline flat out and count cycles in
// hardware -- that is the number comparable to the paper.
//
// SIZING IS A HARD CONSTRAINT, not a tuning knob. The full JSC test set is 166,000 vectors x
// 256 bits = 42.5 Mbit. The Basys 3 has 1.8 Mbit. The test set does not fit and never will, so
// Gate 1b runs in BATCHES: load DEPTH vectors, classify, accumulate accuracy on-chip, repeat.
// Only the running totals cross the UART, not 166,000 predictions.
//
// DEPTH=1024 costs 1024 x 259 bits = 265 Kbit, about 15% of the device's block RAM. Raising it
// cuts the number of host round-trips per Gate 1b run; the ceiling is whatever leaves room for
// the rest of the harness.
//
// Labels live beside the data on purpose: comparing on-chip is what avoids shipping a
// prediction per sample back over the link.

`timescale 1ns / 1ps
`default_nettype none

module vector_store #(
    parameter integer DATA_W  = 256,   // 16 features x 16-bit Q3.12
    parameter integer LABEL_W = 3,     // 5 classes
    parameter integer DEPTH   = 1024,
    parameter integer ADDR_W  = 10     // must satisfy 2**ADDR_W >= DEPTH
)(
    input  wire                clk,

    // write port -- driven by the UART loader
    input  wire                wr_en,
    input  wire [ADDR_W-1:0]   wr_addr,
    input  wire [DATA_W-1:0]   wr_data,
    input  wire [LABEL_W-1:0]  wr_label,

    // read port -- driven by the benchmark FSM. One cycle of read latency; the FSM accounts
    // for it when aligning labels against predictions.
    input  wire [ADDR_W-1:0]   rd_addr,
    output reg  [DATA_W-1:0]   rd_data,
    output reg  [LABEL_W-1:0]  rd_label
);

    // Inferred block RAM. No initial value: contents are undefined until the host loads them,
    // and the FSM must not be started before that. Simple dual-port (one write, one read) is
    // what BRAM natively supports, so this maps without a bit of extra logic.
    reg [DATA_W-1:0]  mem_data  [0:DEPTH-1];
    reg [LABEL_W-1:0] mem_label [0:DEPTH-1];

    always @(posedge clk) begin
        if (wr_en) begin
            mem_data[wr_addr]  <= wr_data;
            mem_label[wr_addr] <= wr_label;
        end
        rd_data  <= mem_data[rd_addr];
        rd_label <= mem_label[rd_addr];
    end

endmodule

`default_nettype wire
