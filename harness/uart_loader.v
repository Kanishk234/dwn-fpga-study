// Host protocol: parses UART commands, fills the vector store, starts benchmark runs, and
// reports results.
//
// This is the only way anything gets onto the board, so the protocol is deliberately boring
// and hand-debuggable. Command bytes are ASCII so bring-up can be done from a plain serial
// terminal before any host script exists -- when Gate 1b misbehaves, being able to type `P`
// and see 0xA5 come back separates "the link is dead" from "the design is wrong" in seconds.
//
//   'P'                                  ping -> replies 0xA5
//   'L' n_lo n_hi  then n x 33 bytes      load n vectors from address 0
//   'R' n_lo n_hi                        run a benchmark batch over n vectors
//   'S'                                  status -> 9 bytes back
//
// Each loaded vector is 33 bytes: 32 bytes of features then 1 byte of label. Feature bytes are
// LITTLE-ENDIAN and in order, so byte k lands at x_flat[k*8 +: 8]; feature f therefore occupies
// bytes 2f (low) and 2f+1 (high). This matches how tb/gen_vectors.py packs x_quant.hex, and
// getting it backwards would silently transpose every feature.
//
// Status reply is 9 bytes, all little-endian: cycle_count[31:0], correct_count[31:0], then a
// flags byte (bit 0 = benchmark busy). Accuracy comes back as a COUNT, never as per-sample
// predictions -- the full test set is 166k samples and does not fit on the device, so Gate 1b
// runs in batches and only totals cross the link.
//
// No CRC, no retries. The link is a 10 cm trace to an FT2232HQ; uart_rx already reports framing
// errors, and a corrupt load shows up immediately as a wrong accuracy. Adding a checksum is
// cheap if that ever proves optimistic.

`timescale 1ns / 1ps
`default_nettype none

module uart_loader #(
    parameter integer DATA_W  = 256,
    parameter integer LABEL_W = 3,
    parameter integer ADDR_W  = 10
)(
    input  wire                clk,
    input  wire                rst,

    // from uart_rx
    input  wire [7:0]          rx_data,
    input  wire                rx_valid,

    // to uart_tx
    output reg  [7:0]          tx_data,
    output reg                 tx_start,
    input  wire                tx_busy,

    // to vector_store
    output reg                 wr_en,
    output reg  [ADDR_W-1:0]   wr_addr,
    output reg  [DATA_W-1:0]   wr_data,
    output reg  [LABEL_W-1:0]  wr_label,

    // to benchmark_fsm
    output reg                 run_start,
    output reg  [ADDR_W:0]     run_n,

    // from benchmark_fsm
    input  wire                bench_busy,
    input  wire [31:0]         cycle_count,
    input  wire [ADDR_W:0]     correct_count
);

    localparam integer VEC_BYTES = DATA_W / 8;      // 32 feature bytes
    localparam integer REC_BYTES = VEC_BYTES + 1;   // + 1 label byte

    localparam [7:0] CMD_PING = "P",
                     CMD_LOAD = "L",
                     CMD_RUN  = "R",
                     CMD_STAT = "S";

    localparam [2:0] S_CMD    = 3'd0,
                     S_LOAD_N = 3'd1,
                     S_LOAD_D = 3'd2,
                     S_RUN_N  = 3'd3;

    reg [2:0]  state;
    reg        n_hi_next;                 // which half of the 16-bit count is next
    reg [15:0] n_expect;                  // vectors to load / run
    reg [15:0] n_seen;                    // vectors loaded so far
    reg [5:0]  byte_idx;                  // 0..REC_BYTES-1 within a record

    // Reply buffer. 9 bytes is the largest response (status).
    reg [7:0]  tx_buf [0:8];
    reg [3:0]  tx_len;
    reg [3:0]  tx_idx;

    reg [15:0] run_n_raw;

    // Zero-extended so the status reply can be sliced into bytes without the packing breaking
    // when ADDR_W changes. correct_count is only ADDR_W+1 bits wide.
    wire [31:0] correct32 = {{(31-ADDR_W){1'b0}}, correct_count};

    // Named because a part-select of a concatenation is not legal Verilog-2001.
    wire [15:0] run_n_full = {rx_data, run_n_raw[7:0]};

    integer b;

    always @(posedge clk) begin
        if (rst) begin
            state      <= S_CMD;
            n_hi_next  <= 1'b0;
            n_expect   <= 16'd0;
            n_seen     <= 16'd0;
            byte_idx   <= 6'd0;
            wr_en      <= 1'b0;
            wr_addr    <= {ADDR_W{1'b0}};
            wr_data    <= {DATA_W{1'b0}};
            wr_label   <= {LABEL_W{1'b0}};
            run_start  <= 1'b0;
            run_n      <= {(ADDR_W+1){1'b0}};
            tx_start   <= 1'b0;
            tx_data    <= 8'd0;
            tx_len     <= 4'd0;
            tx_idx     <= 4'd0;
            for (b = 0; b < 9; b = b + 1) tx_buf[b] <= 8'd0;
        end else begin
            wr_en     <= 1'b0;      // single-cycle pulses
            run_start <= 1'b0;
            tx_start  <= 1'b0;

            // Advance the write address only AFTER a write has been presented. wr_en is a
            // registered pulse, so it is high the cycle after it is set -- incrementing
            // wr_addr at the same time it is set would move the address out from under the
            // write and put every vector in the next slot. (The S_LOAD_N branch below resets
            // wr_addr to 0 and, being later in the block, correctly overrides this.)
            if (wr_en)
                wr_addr <= wr_addr + {{(ADDR_W-1){1'b0}}, 1'b1};

            // ---- reply engine: drain tx_buf one byte at a time ----
            if (tx_idx < tx_len && !tx_busy && !tx_start) begin
                tx_data  <= tx_buf[tx_idx];
                tx_start <= 1'b1;
                tx_idx   <= tx_idx + 4'd1;
            end

            // ---- command parser ----
            if (rx_valid) begin
                case (state)
                    S_CMD: begin
                        case (rx_data)
                            CMD_PING: begin
                                tx_buf[0] <= 8'hA5;
                                tx_len    <= 4'd1;
                                tx_idx    <= 4'd0;
                            end
                            CMD_LOAD: begin
                                n_hi_next <= 1'b0;
                                state     <= S_LOAD_N;
                            end
                            CMD_RUN: begin
                                n_hi_next <= 1'b0;
                                state     <= S_RUN_N;
                            end
                            CMD_STAT: begin
                                tx_buf[0] <= cycle_count[7:0];
                                tx_buf[1] <= cycle_count[15:8];
                                tx_buf[2] <= cycle_count[23:16];
                                tx_buf[3] <= cycle_count[31:24];
                                tx_buf[4] <= correct32[7:0];
                                tx_buf[5] <= correct32[15:8];
                                tx_buf[6] <= correct32[23:16];
                                tx_buf[7] <= correct32[31:24];
                                tx_buf[8] <= {7'd0, bench_busy};
                                tx_len    <= 4'd9;
                                tx_idx    <= 4'd0;
                            end
                            default: ;   // unknown byte: ignore, stay in sync on the next one
                        endcase
                    end

                    S_LOAD_N: begin
                        if (!n_hi_next) begin
                            n_expect[7:0] <= rx_data;
                            n_hi_next     <= 1'b1;
                        end else begin
                            n_expect[15:8] <= rx_data;
                            n_seen         <= 16'd0;
                            byte_idx       <= 6'd0;
                            wr_addr        <= {ADDR_W{1'b0}};
                            state          <= S_LOAD_D;
                        end
                    end

                    S_LOAD_D: begin
                        if (byte_idx < VEC_BYTES[5:0]) begin
                            // Direct indexed placement, not a shift: byte k -> [k*8 +: 8].
                            wr_data[byte_idx*8 +: 8] <= rx_data;
                            byte_idx                 <= byte_idx + 6'd1;
                        end else begin
                            // Final byte of the record is the label.
                            wr_label <= rx_data[LABEL_W-1:0];
                            wr_en    <= 1'b1;
                            byte_idx <= 6'd0;
                            n_seen   <= n_seen + 16'd1;
                            if (n_seen + 16'd1 >= n_expect)
                                state <= S_CMD;
                        end
                    end

                    S_RUN_N: begin
                        // Assembled at full 16 bits then truncated, so this does not depend on
                        // ADDR_W being >= 8.
                        if (!n_hi_next) begin
                            run_n_raw[7:0] <= rx_data;
                            n_hi_next      <= 1'b1;
                        end else begin
                            run_n     <= run_n_full[ADDR_W:0];
                            run_start <= 1'b1;
                            state     <= S_CMD;
                        end
                    end

                    default: state <= S_CMD;
                endcase
            end
        end
    end

endmodule

`default_nettype wire
