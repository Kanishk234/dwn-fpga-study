// UART transmitter, 8N1, no flow control.
//
// Counterpart to uart_rx.v; same BAUD parameter, and the two must be instantiated with the
// same value. See uart_rx.v for why BAUD is a parameter rather than a constant (the I/O wall
// is a result to measure, brief §14).
//
// Interface: assert `start` for one cycle with `data` valid. `busy` is high from that cycle
// until the stop bit completes. `start` while busy is IGNORED rather than queued -- there is
// no FIFO here on purpose, because the two callers do not need one:
//   interactive mode sends a single byte per classification
//   benchmark mode streams results only after the run has finished
// If a caller ever does need to stream continuously, add a FIFO around this rather than
// making the transmitter silently drop or corrupt bytes.

`timescale 1ns / 1ps
`default_nettype none

module uart_tx #(
    parameter integer CLK_HZ = 100_000_000,
    parameter integer BAUD   = 115_200
)(
    input  wire       clk,
    input  wire       rst,
    input  wire       start,
    input  wire [7:0] data,
    output reg        tx,
    output reg        busy
);

    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;

    localparam [1:0] S_IDLE  = 2'd0,
                     S_START = 2'd1,
                     S_DATA  = 2'd2,
                     S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] count;
    reg [2:0]  bit_idx;
    reg [7:0]  shifter;

    always @(posedge clk) begin
        if (rst) begin
            state   <= S_IDLE;
            count   <= 16'd0;
            bit_idx <= 3'd0;
            shifter <= 8'd0;
            tx      <= 1'b1;        // idle high -- a low line reads as a start bit
            busy    <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx    <= 1'b1;
                    count <= 16'd0;
                    if (start) begin
                        shifter <= data;    // latched, so `data` need not be held
                        busy    <= 1'b1;
                        tx      <= 1'b0;    // start bit
                        state   <= S_START;
                    end else begin
                        busy <= 1'b0;
                    end
                end

                S_START: begin
                    if (count == CLKS_PER_BIT[15:0] - 1) begin
                        count   <= 16'd0;
                        bit_idx <= 3'd0;
                        tx      <= shifter[0];
                        state   <= S_DATA;
                    end else begin
                        count <= count + 16'd1;
                    end
                end

                S_DATA: begin
                    if (count == CLKS_PER_BIT[15:0] - 1) begin
                        count <= 16'd0;
                        if (bit_idx == 3'd7) begin
                            tx    <= 1'b1;      // stop bit
                            state <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 3'd1;
                            tx      <= shifter[bit_idx + 3'd1];   // LSB first, per 8N1
                        end
                    end else begin
                        count <= count + 16'd1;
                    end
                end

                S_STOP: begin
                    if (count == CLKS_PER_BIT[15:0] - 1) begin
                        count <= 16'd0;
                        busy  <= 1'b0;
                        state <= S_IDLE;
                    end else begin
                        count <= count + 16'd1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
