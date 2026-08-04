// UART receiver, 8N1, no flow control.
//
// The Basys 3 talks to the host through an FT2232HQ USB-UART bridge -- there is no SPI to the
// host and no external DRAM (brief §6). So this is the only path in, which makes it the
// bottleneck the whole project has to design around: a 32-byte JSC sample takes ~320 us to
// arrive at 1 Mbaud while the core classifies in ~40 ns. That ~2,600x gap is not a flaw to fix
// but a result to measure (brief §14, "the I/O wall, quantified").
//
// BAUD is a parameter for exactly that reason. Sweeping it is how the I/O wall gets
// characterized, so it must not be hardcoded. 115200 is the default because first bring-up
// should fail for interesting reasons rather than marginal signal integrity; raise it once
// Gate 1b passes.
//
// Sampling: the start bit is detected on a falling edge, then every bit is sampled at its
// MIDPOINT (CLKS_PER_BIT/2 after the edge) rather than at a boundary, so a half-bit of
// accumulated clock drift is tolerated. With no oversampling filter, a glitch on rx during
// the start bit would still desync the byte -- acceptable on a short USB-bridge trace, and
// the framing check below catches the result.

`timescale 1ns / 1ps
`default_nettype none

module uart_rx #(
    parameter integer CLK_HZ = 100_000_000,
    parameter integer BAUD   = 115_200
)(
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid,      // one-cycle pulse when `data` is good
    output reg        frame_err   // one-cycle pulse: stop bit was not high
);

    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;

    // SYNCHRONIZER LATENCY IS SUBTRACTED, and it matters more the faster the link runs.
    //
    // rx_sync lags the pin by 2 clocks, so a start bit is not seen until 2 clocks after it
    // begins. Waiting a further CLKS_PER_BIT/2 therefore samples at (2 + half) into the bit,
    // not at the middle:
    //
    //   1 Mbaud   CLKS_PER_BIT=100 -> sample at 52/100 =  52%   harmless
    //   10 Mbaud  CLKS_PER_BIT=10  -> sample at  7/10  =  70%   near the trailing edge
    //
    // At 70% any jitter or ppm drift walks the sample point into the next bit, which is why
    // 10 Mbaud failed to receive at all while 1 Mbaud was fine. Subtracting the 2 clocks
    // re-centres it at ~50% for every rate.
    localparam integer SYNC_LATENCY = 2;
    localparam integer HALF_BIT     = (CLKS_PER_BIT / 2) > SYNC_LATENCY
                                      ? (CLKS_PER_BIT / 2) - SYNC_LATENCY : 0;

    localparam [1:0] S_IDLE  = 2'd0,
                     S_START = 2'd1,
                     S_DATA  = 2'd2,
                     S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] count;
    reg [2:0]  bit_idx;
    reg [7:0]  shifter;

    // Two-flop synchronizer: rx is asynchronous to clk and would otherwise risk metastability
    // on the capture flop. rx_sync is the only version anything else may look at.
    reg rx_meta, rx_sync;
    always @(posedge clk) begin
        rx_meta <= rx;
        rx_sync <= rx_meta;
    end

    always @(posedge clk) begin
        if (rst) begin
            state     <= S_IDLE;
            count     <= 16'd0;
            bit_idx   <= 3'd0;
            shifter   <= 8'd0;
            data      <= 8'd0;
            valid     <= 1'b0;
            frame_err <= 1'b0;
        end else begin
            valid     <= 1'b0;      // both outputs are single-cycle pulses
            frame_err <= 1'b0;

            case (state)
                S_IDLE: begin
                    count   <= 16'd0;
                    bit_idx <= 3'd0;
                    if (~rx_sync) state <= S_START;     // falling edge = start bit
                end

                S_START: begin
                    // Re-check at the middle of the start bit. A line that has gone high
                    // again was a glitch, not a start bit.
                    if (count == HALF_BIT[15:0]) begin
                        if (~rx_sync) begin
                            count <= 16'd0;
                            state <= S_DATA;
                        end else begin
                            state <= S_IDLE;
                        end
                    end else begin
                        count <= count + 16'd1;
                    end
                end

                S_DATA: begin
                    if (count == CLKS_PER_BIT[15:0] - 1) begin
                        count            <= 16'd0;
                        shifter[bit_idx] <= rx_sync;    // LSB first, per 8N1
                        if (bit_idx == 3'd7) state   <= S_STOP;
                        else                 bit_idx <= bit_idx + 3'd1;
                    end else begin
                        count <= count + 16'd1;
                    end
                end

                S_STOP: begin
                    if (count == CLKS_PER_BIT[15:0] - 1) begin
                        count <= 16'd0;
                        state <= S_IDLE;
                        data  <= shifter;
                        // Report the byte either way, but flag bad framing so the host-side
                        // protocol can resync rather than silently consuming garbage.
                        if (rx_sync) valid     <= 1'b1;
                        else         frame_err <= 1'b1;
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
