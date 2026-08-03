// Four-digit multiplexed seven-segment driver for the Basys 3.
//
// The board wires all four digits' segments together and selects one digit at a time with the
// anode lines, so the only way to show four different digits is to cycle through them faster
// than the eye can follow. Both `seg` and `an` are ACTIVE LOW on this board -- driving them
// active high gives an inverted display, which looks like a wiring fault but is not.
//
// REFRESH_BITS sets the per-digit dwell time: 2**REFRESH_BITS clocks. At 100 MHz and the
// default 17, that is ~1.3 ms per digit and ~190 Hz for a full pass -- above the flicker
// threshold, and slow enough that the digits are actually driven rather than smeared. It is a
// parameter because simulation needs it small; leaving it at the board value would make a
// testbench run for millions of cycles to see one refresh cycle.

`timescale 1ns / 1ps
`default_nettype none

module seg7 #(
    parameter integer REFRESH_BITS = 17
)(
    input  wire        clk,
    input  wire        rst,
    input  wire [15:0] value,        // displayed as four hex digits
    output reg  [6:0]  seg,          // {g,f,e,d,c,b,a}, active low
    output reg  [3:0]  an            // digit select, active low
);

    reg [REFRESH_BITS-1:0] refresh;
    always @(posedge clk) begin
        if (rst) refresh <= {REFRESH_BITS{1'b0}};
        else     refresh <= refresh + {{(REFRESH_BITS-1){1'b0}}, 1'b1};
    end

    wire [1:0] digit_sel = refresh[REFRESH_BITS-1:REFRESH_BITS-2];

    reg [3:0] nibble;
    always @* begin
        case (digit_sel)
            2'd0: nibble = value[3:0];
            2'd1: nibble = value[7:4];
            2'd2: nibble = value[11:8];
            default: nibble = value[15:12];
        endcase
    end

    always @* begin
        case (digit_sel)
            2'd0:    an = 4'b1110;
            2'd1:    an = 4'b1101;
            2'd2:    an = 4'b1011;
            default: an = 4'b0111;
        endcase
    end

    // Hex font. Active low, so a 0 bit lights that segment.
    always @* begin
        case (nibble)
            4'h0: seg = 7'b1000000;
            4'h1: seg = 7'b1111001;
            4'h2: seg = 7'b0100100;
            4'h3: seg = 7'b0110000;
            4'h4: seg = 7'b0011001;
            4'h5: seg = 7'b0010010;
            4'h6: seg = 7'b0000010;
            4'h7: seg = 7'b1111000;
            4'h8: seg = 7'b0000000;
            4'h9: seg = 7'b0010000;
            4'hA: seg = 7'b0001000;
            4'hB: seg = 7'b0000011;
            4'hC: seg = 7'b1000110;
            4'hD: seg = 7'b0100001;
            4'hE: seg = 7'b0000110;
            default: seg = 7'b0001110;   // F
        endcase
    end

endmodule

`default_nettype wire
