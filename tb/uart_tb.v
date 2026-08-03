// UART loopback test: uart_tx -> wire -> uart_rx, checking every byte survives.
//
// Not Gate 1 -- that is the golden-model check on the DWN datapath. This is a unit test for
// the harness, and it exists because brief §12 risk #7 is exactly this: simulation-correct RTL
// says nothing about UART framing, and framing bugs on hardware present as silence or garbage
// with no clue which end is wrong. Finding them here costs seconds; finding them on the board
// costs an afternoon with a scope.
//
// CLKS_PER_BIT is deliberately small (CLK_HZ/BAUD = 16) so the whole byte space simulates
// quickly. The logic is identical at 868 (100 MHz / 115200) -- only the counter terminal value
// changes -- so this exercises the same state machine the board will run.
//
// What is covered:
//   all 256 byte values          catches bit-order and off-by-one shifter bugs
//   back-to-back transmission    catches busy/start handshake races
//   framing error detection      catches a receiver that accepts a bad stop bit

`timescale 1ns / 1ps
`default_nettype none

module uart_tb;

    localparam integer CLK_HZ = 1_600_000;
    localparam integer BAUD   = 100_000;      // -> CLKS_PER_BIT = 16
    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    reg        tx_start = 1'b0;
    reg  [7:0] tx_data  = 8'd0;
    wire       line;
    wire       tx_busy;

    wire [7:0] rx_data;
    wire       rx_valid;
    wire       rx_frame_err;

    reg  force_line = 1'b0;     // for the framing-error case
    reg  forced_val = 1'b1;
    wire rx_line = force_line ? forced_val : line;

    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_tx (
        .clk(clk), .rst(rst), .start(tx_start), .data(tx_data),
        .tx(line), .busy(tx_busy));

    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_rx (
        .clk(clk), .rst(rst), .rx(rx_line),
        .data(rx_data), .valid(rx_valid), .frame_err(rx_frame_err));

    integer errors = 0;
    integer received = 0;
    integer frame_errs = 0;
    integer i;
    reg [7:0] expect_q [0:511];
    integer   wr_ptr = 0;
    integer   rd_ptr = 0;

    // Collect everything the receiver produces, independently of the driving loop, so a byte
    // arriving late or early is still caught rather than silently missed.
    always @(posedge clk) begin
        if (!rst && rx_valid) begin
            received = received + 1;
            if (rx_data !== expect_q[rd_ptr]) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("  MISMATCH byte %0d: got %02h expected %02h",
                             rd_ptr, rx_data, expect_q[rd_ptr]);
            end
            rd_ptr = rd_ptr + 1;
        end
        if (!rst && rx_frame_err) frame_errs = frame_errs + 1;
    end

    task send_byte(input [7:0] b);
        begin
            @(posedge clk);
            while (tx_busy) @(posedge clk);
            expect_q[wr_ptr] = b;
            wr_ptr           = wr_ptr + 1;
            tx_data          <= b;
            tx_start         <= 1'b1;
            @(posedge clk);
            tx_start <= 1'b0;
            // Wait for the transmitter to actually finish before returning, so the caller can
            // issue the next byte immediately and exercise the back-to-back path.
            @(posedge clk);
            while (tx_busy) @(posedge clk);
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        // 1. every byte value, back to back
        for (i = 0; i < 256; i = i + 1)
            send_byte(i[7:0]);

        repeat (CLKS_PER_BIT * 4) @(posedge clk);

        if (received != 256) begin
            $display("  ERROR: received %0d bytes, expected 256", received);
            errors = errors + 1;
        end
        if (frame_errs != 0) begin
            $display("  ERROR: %0d spurious framing errors on clean traffic", frame_errs);
            errors = errors + 1;
        end

        // 2. framing error: drive a start bit and 8 data bits, then hold the line LOW where
        //    the stop bit belongs. A receiver that ignores the stop bit would report this as
        //    a good byte.
        frame_errs = 0;
        force_line = 1'b1;
        forced_val = 1'b1;
        repeat (CLKS_PER_BIT * 2) @(posedge clk);
        forced_val = 1'b0;                                  // start bit
        repeat (CLKS_PER_BIT * 9) @(posedge clk);           // start + 8 data bits, all low
        repeat (CLKS_PER_BIT) @(posedge clk);               // stop bit slot, still low
        forced_val = 1'b1;
        repeat (CLKS_PER_BIT * 3) @(posedge clk);
        force_line = 1'b0;

        if (frame_errs != 1) begin
            $display("  ERROR: bad stop bit produced %0d framing errors, expected 1",
                     frame_errs);
            errors = errors + 1;
        end

        $display("");
        $display("========================================");
        $display("UART loopback unit test");
        $display("  bytes sent     : 256 (all values, back to back)");
        $display("  bytes received : %0d", received);
        $display("  framing check  : %0d error(s) on a deliberately bad stop bit", frame_errs);
        $display("  mismatches     : %0d", errors);
        if (errors == 0) $display("  RESULT         : PASS");
        else             $display("  RESULT         : FAIL");
        $display("========================================");
        $display("");
        $finish;
    end

    // Safety net: if the handshake deadlocks, fail loudly instead of hanging the run.
    initial begin
        #10_000_000;
        $display("  RESULT         : FAIL (timeout -- tx/rx handshake deadlocked)");
        $finish;
    end

endmodule

`default_nettype wire
