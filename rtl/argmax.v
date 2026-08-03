// Argmax over per-class popcounts.
//
// TIE-BREAKING IS PART OF THE SPEC, NOT AN IMPLEMENTATION DETAIL.
//
// With 10 nodes per class there are only 11 possible scores, so ties are common rather than
// exotic: 29 of the 1000 Gate 1 test vectors (2.9%) have two or more classes sharing the top
// score. numpy's argmax and torch's argmax both return the LOWEST index in that case, so the
// golden model does too, so this must as well.
//
// The implementation detail that enforces it is the strict `>` below. Using `>=` would keep
// the HIGHEST tied index instead, and the design would then disagree with the golden model on
// ~3% of vectors while looking completely correct on the other 97% -- exactly the kind of
// failure Gate 1 exists to catch, and exactly the kind that is invisible to spot-checking a
// handful of inputs.

`timescale 1ns / 1ps
`default_nettype none

module argmax #(
    parameter integer K = 5,            // classes
    parameter integer W = 4             // bits per score
)(
    input  wire [K*W-1:0]           scores_flat,   // class c occupies [c*W +: W]
    output reg  [$clog2(K)-1:0]     index
);

    reg [W-1:0] best;
    integer     c;

    always @* begin
        best  = scores_flat[0 +: W];
        index = 0;
        for (c = 1; c < K; c = c + 1) begin
            // strict > : a later class must BEAT the incumbent, not merely match it
            if (scores_flat[c*W +: W] > best) begin
                best  = scores_flat[c*W +: W];
                index = c;      // plain assignment; truncation to index's width is implicit.
                                // (Part-selecting the integer `c` is accepted by some tools
                                //  and rejected by others -- not worth the portability risk.)
            end
        end
    end

endmodule

`default_nettype wire
