`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/16 17:06:53
// Design Name: 
// Module Name: acc_ctrlr
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module acc_ctrlr #(
    // SRAM address widths
    parameter ADDR_MAX_OFM      = 16,       // Address for one of the CONV_OC Output FM (OFM) memories
    parameter DATA_BW           = 32,       // float 32bit data
    parameter OUTPUT_BW         = 32
)(
    input                       clk,
    input                       resetn,
    // ------------------------------------------------------------------------
    // Start & Done Signals
    // ------------------------------------------------------------------------
    input  wire start,
    output wire done,
    // ------------------------------------------------------------------------
    // Configurable Data
    // ------------------------------------------------------------------------
    input  wire [13:0]  IC_ITER,              // MAX IC : 128 -> MAX IC_ITER : 128/8 = 16
    input  wire [13:0]  TOT_IC,
    input  wire [7:0]   TOT_OC,
    input  wire [4:0]   IMG_H,
    input  wire [4:0]   IMG_W,
    // ------------------------------------------------------------------------
    // input PSUM_MEM Data & output read address, en signals
    // ------------------------------------------------------------------------
    output wire                             psum_mem_rd_en,
    output wire [ADDR_MAX_OFM-1:0]          psum_mem_rd_addr,
    input wire signed [DATA_BW-1:0]         psum_mem_rd_dout,
    // ------------------------------------------------------------------------
    // input PSUM_MEM_ACC Address to read
    // ------------------------------------------------------------------------ 
//    output wire                             psum_mem_acc_rd_en,
//    output wire [ADDR_MAX_OFM-1:0]          psum_mem_acc_rd_addr,
    input wire signed [DATA_BW-1:0]         psum_mem_acc_rd_dout,
    // ------------------------------------------------------------------------
    // output PSUM_MEM_ACC Address to write
    // ------------------------------------------------------------------------ 
    output wire                            psum_mem_acc_wr_en,
    output wire                            psum_mem_acc_wr_we,
    output wire [ADDR_MAX_OFM-1:0]         psum_mem_acc_wr_addr,
    output wire signed [DATA_BW-1:0]       psum_mem_acc_wr_din
    // ------------------------------------------------------------------------
    // after Quantization PSUM Data
    // ------------------------------------------------------------------------
//    output wire signed [OUTPUT_BW-1:0]      out_quantized,
    // ------------------------------------------------------------------------
    // output OUT_MEM Address to write
    // ------------------------------------------------------------------------ 
//    output wire                            out_mem_en,
//    output wire                            out_mem_we,
//    output wire [ADDR_MAX_OFM-1:0]         out_mem_addr,
//    output wire signed [OUTPUT_BW-1:0]     out_mem_din
    );
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
//    reg  signed [OUTPUT_BW-1:0]      out_quantized;
//    wire signed [OUTPUT_BW-1:0]     out_quantized_after_reg;
//    reg  signed [OUTPUT_BW-1:0]      out_quantized_reg;
//    assign out_quantized_after_reg = out_quantized_reg;
    
//   always @(posedge clk or negedge resetn) begin
//        if(~resetn) begin
//            out_quantized       <= 0;
//            out_quantized_reg   <= 0;
//        end
//        else begin
//            out_quantized       <= psum_mem_acc_rd_dout_after_reg;
//            out_quantized_reg   <= out_quantized;
//        end
//    end
    ///////////////////////////////////////////////////////////////////////////////////////
//    assign out_mem_din = out_quantized_after_reg;
    
    localparam CONV_PE_NUM          =   8;
    
    localparam IDLE             = 3'd0;
    localparam READ             = 3'd1;
    localparam ACCUMULATE       = 3'd2;
    localparam WRITE            = 3'd3;
    localparam QUANTIZE_WAIT    = 3'd4;
//    localparam QUANTIZE         = 3'd5;
    localparam DONE             = 3'd6;

    wire [4:0] IMG_OC_H;
    wire [4:0] IMG_OC_W;
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    wire [ADDR_MAX_OFM-1:0] WRITE_ADDRESS_MAX;
    reg  [ADDR_MAX_OFM-1:0] WRITE_ADDRESS_MAX_reg;
    reg [4:0] IMG_OC_H_reg;
    reg [4:0] IMG_OC_W_reg;
   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            IMG_OC_H_reg <= 0;
            IMG_OC_W_reg <= 0;
            WRITE_ADDRESS_MAX_reg <= 0;   
        end
        else begin
            WRITE_ADDRESS_MAX_reg <= (IMG_OC_H)*(IMG_OC_W)*TOT_OC;
            if      ((IMG_H-2) >= 0)    IMG_OC_H_reg <= IMG_H - 2;
            else                        IMG_OC_H_reg <= 1;
            if      ((IMG_W-2) >= 0)    IMG_OC_W_reg <= IMG_W - 2;
            else                        IMG_OC_W_reg <= 1;
        end
    end
    assign IMG_OC_H = IMG_OC_H_reg;
    assign IMG_OC_W = IMG_OC_W_reg;
    assign WRITE_ADDRESS_MAX = WRITE_ADDRESS_MAX_reg; 
    /////////////////////////////////////////////////////////////////////////////////////// 
//    assign IMG_OC_H = ((IMG_H-2) >= 0) ? IMG_H-2 : 1;
//    assign IMG_OC_W = ((IMG_W-2) >= 0) ? IMG_W-2 : 1; 
    
    reg [2:0] state, n_state;
    
    reg [ADDR_MAX_OFM-1:0] psum_mem_rd_addr_reg;
    reg [ADDR_MAX_OFM-1:0] psum_mem_acc_rd_addr_reg;
    reg [ADDR_MAX_OFM-1:0] psum_mem_acc_wr_addr_reg;
//    reg [ADDR_MAX_OFM-1:0] psum_mem_acc_quantize_addr_reg;
    
    reg [ADDR_MAX_OFM-1:0] psum_mem_rd_addr_reg_delay;
    reg [ADDR_MAX_OFM-1:0] psum_mem_acc_rd_addr_reg_delay;
    reg [ADDR_MAX_OFM-1:0] psum_mem_acc_wr_addr_reg_delay;
//    reg [ADDR_MAX_OFM-1:0] psum_mem_acc_quantize_addr_reg_delay;
    
//    reg [ADDR_MAX_OFM-1:0] out_mem_addr_reg;
//    reg [ADDR_MAX_OFM-1:0] out_mem_addr_reg_delay, out_mem_addr_reg_delay2;
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    reg [4:0] quantize_wait_cnt;
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            quantize_wait_cnt <= 0;
        end
        else begin
            case (state)
                IDLE : begin
                    quantize_wait_cnt <= 0;
                end
                QUANTIZE_WAIT : begin
                    quantize_wait_cnt <= quantize_wait_cnt + 1;
                end
                default :  begin
                    quantize_wait_cnt <= 0;
                end
            endcase
        end
    end
    
    
    wire signed [DATA_BW-1:0]      psum_mem_rd_dout_after_reg;
    wire signed [DATA_BW-1:0]      psum_mem_acc_rd_dout_after_reg;
    
    reg signed [DATA_BW-1:0]      psum_mem_rd_dout_reg;
    reg signed [DATA_BW-1:0]      psum_mem_acc_rd_dout_reg;
    assign psum_mem_rd_dout_after_reg       = psum_mem_rd_dout_reg;
    assign psum_mem_acc_rd_dout_after_reg   = psum_mem_acc_rd_dout_reg;
    
    reg [ADDR_MAX_OFM-1:0]        psum_mem_acc_wr_addr_reg_delay2, psum_mem_acc_wr_addr_reg_delay3;
    reg [ADDR_MAX_OFM-1:0]        psum_mem_acc_wr_addr_reg_delay4, psum_mem_acc_wr_addr_reg_delay5;
    reg [ADDR_MAX_OFM-1:0]        psum_mem_acc_wr_addr_reg_delay6, psum_mem_acc_wr_addr_reg_delay7;
    
//    reg [ADDR_MAX_OFM-1:0]        out_mem_addr_reg_delay3, out_mem_addr_reg_delay4, out_mem_addr_reg_delay5;
    
    reg signed [DATA_BW-1:0]       psum_mem_acc_wr_din_just_write_reg;
    reg signed [DATA_BW-1:0]       psum_mem_acc_wr_din_just_write_reg_delay,    psum_mem_acc_wr_din_just_write_reg_delay2;
    reg signed [DATA_BW-1:0]       psum_mem_acc_wr_din_just_write_reg_delay3,   psum_mem_acc_wr_din_just_write_reg_delay4;
    reg signed [DATA_BW-1:0]       psum_mem_acc_wr_din_add_reg;
    
    wire signed [DATA_BW-1:0] psum_mem_acc_wr_din_add;
    wire signed [DATA_BW-1:0] psum_mem_acc_wr_din_just_write;
    assign psum_mem_acc_wr_din_just_write = psum_mem_rd_dout_reg;
    assign psum_mem_acc_wr_din = (IC_ITER == 1) ? psum_mem_acc_wr_din_just_write_reg_delay4 : psum_mem_acc_wr_din_add_reg; // for debug: TOT_IC == 1

   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            psum_mem_rd_dout_reg                        <= 0;
            psum_mem_acc_rd_dout_reg                    <= 0;
            
            psum_mem_acc_wr_addr_reg_delay2             <= 0; psum_mem_acc_wr_addr_reg_delay3 <= 0;
            psum_mem_acc_wr_addr_reg_delay4             <= 0; psum_mem_acc_wr_addr_reg_delay5 <= 0;
            psum_mem_acc_wr_addr_reg_delay6             <= 0; psum_mem_acc_wr_addr_reg_delay7 <= 0;
            
//            out_mem_addr_reg_delay3                     <= 0; out_mem_addr_reg_delay4         <= 0;
//            out_mem_addr_reg_delay5                     <= 0;
            
            psum_mem_acc_wr_din_just_write_reg          <= 0;
            psum_mem_acc_wr_din_just_write_reg_delay    <= 0; psum_mem_acc_wr_din_just_write_reg_delay2   <= 0;
            psum_mem_acc_wr_din_just_write_reg_delay3   <= 0; psum_mem_acc_wr_din_just_write_reg_delay4   <= 0;
            
            psum_mem_acc_wr_din_add_reg                 <= 0;
        end
        else begin
            psum_mem_rd_dout_reg                        <= psum_mem_rd_dout;
            psum_mem_acc_rd_dout_reg                    <= psum_mem_acc_rd_dout;
            
            psum_mem_acc_wr_addr_reg_delay2             <= psum_mem_acc_wr_addr_reg_delay;      psum_mem_acc_wr_addr_reg_delay3 <= psum_mem_acc_wr_addr_reg_delay2;
            psum_mem_acc_wr_addr_reg_delay4             <= psum_mem_acc_wr_addr_reg_delay3;     psum_mem_acc_wr_addr_reg_delay5 <= psum_mem_acc_wr_addr_reg_delay4;
            psum_mem_acc_wr_addr_reg_delay6             <= psum_mem_acc_wr_addr_reg_delay5;     psum_mem_acc_wr_addr_reg_delay7 <= psum_mem_acc_wr_addr_reg_delay6;
            
//            out_mem_addr_reg_delay3                     <= out_mem_addr_reg_delay2;             out_mem_addr_reg_delay4         <= out_mem_addr_reg_delay3;
//            out_mem_addr_reg_delay5                     <= out_mem_addr_reg_delay4;
            
            psum_mem_acc_wr_din_just_write_reg          <= psum_mem_acc_wr_din_just_write;
            psum_mem_acc_wr_din_just_write_reg_delay    <= psum_mem_acc_wr_din_just_write_reg;
            psum_mem_acc_wr_din_just_write_reg_delay2   <= psum_mem_acc_wr_din_just_write_reg_delay;
            psum_mem_acc_wr_din_just_write_reg_delay3   <= psum_mem_acc_wr_din_just_write_reg_delay2;
            psum_mem_acc_wr_din_just_write_reg_delay4   <= psum_mem_acc_wr_din_just_write_reg_delay3;
            
            psum_mem_acc_wr_din_add_reg                 <= psum_mem_acc_wr_din_add;
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////
    
    assign psum_mem_rd_addr     = psum_mem_rd_addr_reg_delay;
//    assign psum_mem_acc_rd_addr = psum_mem_acc_rd_addr_reg_delay;
//    assign psum_mem_acc_rd_addr = (state != QUANTIZE) ? psum_mem_acc_rd_addr_reg_delay : psum_mem_acc_quantize_addr_reg_delay;
    assign psum_mem_acc_wr_addr = ((state == ACCUMULATE) || (state == WRITE)) ? psum_mem_acc_rd_addr_reg_delay : psum_mem_acc_wr_addr_reg_delay7;  ///////////////////////////////////// 3
//    assign out_mem_addr         = out_mem_addr_reg_delay5;
    
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            psum_mem_rd_addr_reg_delay              <= 0;
            psum_mem_acc_rd_addr_reg_delay          <= 0;
            psum_mem_acc_wr_addr_reg_delay          <= 0;
//            psum_mem_acc_quantize_addr_reg_delay    <= 0;
            
        end
        else begin
            psum_mem_rd_addr_reg_delay              <= psum_mem_rd_addr_reg;
            psum_mem_acc_rd_addr_reg_delay          <= psum_mem_acc_rd_addr_reg;
            psum_mem_acc_wr_addr_reg_delay          <= psum_mem_acc_wr_addr_reg;
//            psum_mem_acc_quantize_addr_reg_delay    <= psum_mem_acc_quantize_addr_reg;
        end
    end
    
//    always @(posedge clk or negedge resetn) begin
//        if(~resetn) begin 
//            out_mem_addr_reg_delay  <= 0; 
//            out_mem_addr_reg_delay2 <= 0;
//        end
//        else begin        
//            out_mem_addr_reg_delay  <= out_mem_addr_reg;
//            out_mem_addr_reg_delay2 <= out_mem_addr_reg_delay;
//        end
//    end
    
    reg acc_done = 1'b1;
    
    assign done = (state == DONE); 
    
    /***** state transition *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) state <= IDLE;
        else state <= n_state;
    end
    
   /***** Defines when does the state change *****/ 
    always @(*) begin
        case (state)
        IDLE : begin 
            if(start) begin
                n_state = READ;
            end               
            else n_state = IDLE;
        end
        READ : begin                                    // Read PSUM_MEM & PSUM_MEM_ACC         (1 pixel read)
            if(IC_ITER == 1)    n_state = WRITE;
            else                n_state = ACCUMULATE;
        end
        ACCUMULATE : begin                              // Accumulate PSUM_MEM & PSUM_MEM_ACC   (1 pixel accumulation)
            if(acc_done)    n_state = WRITE;
            else            n_state = ACCUMULATE;
        end
        WRITE : begin                                   // Write to PSUM_MEM_ACC                (1 pixel write)
            if (IC_ITER < TOT_IC/CONV_PE_NUM)   begin
                if(psum_mem_acc_wr_addr == WRITE_ADDRESS_MAX)          n_state = DONE;
                else                                                                  n_state = READ;
            end
            else begin
//                if(psum_mem_acc_wr_addr_reg == WRITE_ADDRESS_MAX)          n_state = QUANTIZE;
                if(psum_mem_acc_wr_addr == WRITE_ADDRESS_MAX)          n_state = QUANTIZE_WAIT;
                else                                                       n_state = READ;
            end
        end
        QUANTIZE_WAIT : begin
            if(quantize_wait_cnt == 31)                     n_state = DONE;
            else                                            n_state = QUANTIZE_WAIT;
        end
//        QUANTIZE : begin                                // Quantize all the px in PSUM_MEM_ACC 
//            if(out_mem_addr_reg == WRITE_ADDRESS_MAX)      n_state = DONE;
//            else                                                    n_state = QUANTIZE;
//        end
        DONE : begin
            n_state = IDLE;
        end
        default :  n_state = IDLE;
        endcase
    end 
    
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            psum_mem_rd_addr_reg            <= 0;
            psum_mem_acc_rd_addr_reg        <= 0;
            psum_mem_acc_wr_addr_reg        <= 0;
//            out_mem_addr_reg                <= 0;
//            psum_mem_acc_quantize_addr_reg  <= 0;
        end
        else begin
            case (state)
                IDLE : begin
                    psum_mem_rd_addr_reg            <= 0;
                    psum_mem_acc_rd_addr_reg        <= 0;
                    psum_mem_acc_wr_addr_reg        <= 0;
//                    out_mem_addr_reg                <= 0;
//                    psum_mem_acc_quantize_addr_reg  <= 0;
                end
                READ : begin
                    psum_mem_rd_addr_reg <= psum_mem_rd_addr_reg + 1;
                    psum_mem_acc_rd_addr_reg <= psum_mem_acc_rd_addr_reg + 1;
                end
                WRITE : begin
                    psum_mem_acc_wr_addr_reg <= psum_mem_acc_wr_addr_reg + 1;
                end
//                QUANTIZE : begin
//                    psum_mem_acc_quantize_addr_reg  <= psum_mem_acc_quantize_addr_reg + 1;
//                    out_mem_addr_reg <= out_mem_addr_reg + 1;
//                end
                default :  begin
                    psum_mem_rd_addr_reg            <= psum_mem_rd_addr_reg;
                    psum_mem_acc_rd_addr_reg        <= psum_mem_acc_rd_addr_reg;
                    psum_mem_acc_wr_addr_reg        <= psum_mem_acc_wr_addr_reg;
//                    out_mem_addr_reg                <= out_mem_addr_reg;
//                    psum_mem_acc_quantize_addr_reg  <= psum_mem_acc_quantize_addr_reg;
                end
            endcase
        end
    end
    
    reg psum_mem_rd_en_reg;
    reg psum_mem_acc_rd_en_reg;
    reg psum_mem_acc_wr_en_reg;
    reg psum_mem_acc_wr_we_reg;
//    reg out_mem_en_reg;
//    reg out_mem_we_reg;
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    reg psum_mem_acc_wr_en_reg_delay,   psum_mem_acc_wr_en_reg_delay2;
    reg psum_mem_acc_wr_en_reg_delay3,  psum_mem_acc_wr_en_reg_delay4;
    reg psum_mem_acc_wr_en_reg_delay5,  psum_mem_acc_wr_en_reg_delay6;
    reg psum_mem_acc_wr_we_reg_delay,   psum_mem_acc_wr_we_reg_delay2;
    reg psum_mem_acc_wr_we_reg_delay3,  psum_mem_acc_wr_we_reg_delay4;
    reg psum_mem_acc_wr_we_reg_delay5,  psum_mem_acc_wr_we_reg_delay6;
//    reg out_mem_en_reg_delay,           out_mem_en_reg_delay2,           out_mem_en_reg_delay3;
//    reg out_mem_we_reg_delay,           out_mem_we_reg_delay2,           out_mem_we_reg_delay3;

   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            psum_mem_acc_wr_en_reg_delay <= 0; psum_mem_acc_wr_en_reg_delay2 <= 0;
            psum_mem_acc_wr_en_reg_delay3 <= 0;psum_mem_acc_wr_en_reg_delay4 <= 0;
            psum_mem_acc_wr_en_reg_delay5 <= 0;psum_mem_acc_wr_en_reg_delay6 <= 0;
            psum_mem_acc_wr_we_reg_delay <= 0; psum_mem_acc_wr_we_reg_delay2 <= 0;
            psum_mem_acc_wr_we_reg_delay3 <= 0;psum_mem_acc_wr_we_reg_delay4 <= 0;
            psum_mem_acc_wr_we_reg_delay5 <= 0;psum_mem_acc_wr_we_reg_delay6 <= 0;
//            out_mem_en_reg_delay         <= 0; out_mem_en_reg_delay2         <= 0;
//            out_mem_we_reg_delay         <= 0; out_mem_we_reg_delay2         <= 0;
//            out_mem_en_reg_delay3        <= 0; out_mem_we_reg_delay3         <= 0;
        end
        else begin
            psum_mem_acc_wr_en_reg_delay  <= psum_mem_acc_wr_en_reg;        psum_mem_acc_wr_en_reg_delay2 <= psum_mem_acc_wr_en_reg_delay;
            psum_mem_acc_wr_en_reg_delay3 <= psum_mem_acc_wr_en_reg_delay2; psum_mem_acc_wr_en_reg_delay4 <= psum_mem_acc_wr_en_reg_delay3;
            psum_mem_acc_wr_en_reg_delay5 <= psum_mem_acc_wr_en_reg_delay4; psum_mem_acc_wr_en_reg_delay6 <= psum_mem_acc_wr_en_reg_delay5;
            
            psum_mem_acc_wr_we_reg_delay  <= psum_mem_acc_wr_we_reg;        psum_mem_acc_wr_we_reg_delay2 <= psum_mem_acc_wr_we_reg_delay;
            psum_mem_acc_wr_we_reg_delay3 <= psum_mem_acc_wr_we_reg_delay2; psum_mem_acc_wr_we_reg_delay4 <= psum_mem_acc_wr_we_reg_delay3;
            psum_mem_acc_wr_we_reg_delay5 <= psum_mem_acc_wr_we_reg_delay4; psum_mem_acc_wr_we_reg_delay6 <= psum_mem_acc_wr_we_reg_delay5;
            
//            out_mem_en_reg_delay          <= out_mem_en_reg;          out_mem_en_reg_delay2         <= out_mem_en_reg_delay;
//            out_mem_we_reg_delay          <= out_mem_we_reg;          out_mem_we_reg_delay2         <= out_mem_we_reg_delay;
            
//            out_mem_en_reg_delay3         <= out_mem_en_reg_delay2;
//            out_mem_we_reg_delay3         <= out_mem_we_reg_delay2;
        end
    end
    
    ///////////////////////////////////////////////////////////////////////////////////////
    
    assign psum_mem_rd_en = psum_mem_rd_en_reg;
//    assign psum_mem_acc_rd_en = psum_mem_acc_rd_en_reg;
    assign psum_mem_acc_wr_en = ((state == ACCUMULATE) || (state == WRITE)) ? psum_mem_acc_rd_en_reg : psum_mem_acc_wr_en_reg_delay6;
//    assign psum_mem_acc_wr_en = psum_mem_acc_wr_en_reg_delay6;
    assign psum_mem_acc_wr_we = psum_mem_acc_wr_we_reg_delay6;
//    assign out_mem_en = out_mem_en_reg_delay3;
//    assign out_mem_we = out_mem_we_reg_delay3;
    
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            psum_mem_rd_en_reg        <= 0;
            psum_mem_acc_rd_en_reg    <= 0;
            psum_mem_acc_wr_en_reg    <= 0;
            psum_mem_acc_wr_we_reg    <= 0;
//            out_mem_en_reg            <= 0;
//            out_mem_we_reg            <= 0;
        end
        else begin
            case (state)
                IDLE : begin
                    psum_mem_rd_en_reg        <= 0;
                    psum_mem_acc_rd_en_reg    <= 0;
                    psum_mem_acc_wr_en_reg    <= 0;
                    psum_mem_acc_wr_we_reg    <= 0;
//                    out_mem_en_reg            <= 0;
//                    out_mem_we_reg            <= 0;
                end
                READ : begin
                    psum_mem_rd_en_reg        <= 1;
                    psum_mem_acc_rd_en_reg    <= 1;
                    psum_mem_acc_wr_en_reg    <= 0;
                    psum_mem_acc_wr_we_reg    <= 0;
//                    out_mem_en_reg            <= 0;
//                    out_mem_we_reg            <= 0;
                end
                WRITE : begin
                    psum_mem_rd_en_reg        <= 0;
                    psum_mem_acc_rd_en_reg    <= 0;
                    psum_mem_acc_wr_en_reg    <= 1;
                    psum_mem_acc_wr_we_reg    <= 1;
//                    out_mem_en_reg            <= 0;
//                    out_mem_we_reg            <= 0;
                end
//                QUANTIZE : begin 
//                    psum_mem_rd_en_reg        <= 0;
//                    psum_mem_acc_rd_en_reg    <= 1;
//                    psum_mem_acc_wr_en_reg    <= 0;
//                    psum_mem_acc_wr_we_reg    <= 0;
//                    out_mem_en_reg            <= 1;
//                    out_mem_we_reg            <= 1;
//                end
                default :  begin
                    psum_mem_rd_en_reg        <= 0;
                    psum_mem_acc_rd_en_reg    <= 0;
                    psum_mem_acc_wr_en_reg    <= 0;
                    psum_mem_acc_wr_we_reg    <= 0;
//                    out_mem_en_reg            <= 0;
//                    out_mem_we_reg            <= 0;
                end
            endcase
        end
    end
    
//    wire signed [DATA_BW-1:0] psum_mem_acc_wr_din_add;
//    wire signed [DATA_BW-1:0] psum_mem_acc_wr_din_just_write;
//    assign psum_mem_acc_wr_din_just_write = psum_mem_rd_dout;
//    assign psum_mem_acc_wr_din = (IC_ITER == 1) ? psum_mem_acc_wr_din_just_write : psum_mem_acc_wr_din_add; // for debug: TOT_IC == 1 
    
    
    // --- Adder ---
    float32_add float_add (
        .clk(clk), .resetn(resetn),
        .out_float(psum_mem_acc_wr_din_add),
        .inA_float(psum_mem_rd_dout_after_reg),
        .inB_float(psum_mem_acc_rd_dout_after_reg)
    );
    
//    // Quantization
//    quantize quant (
//        .clk(clk), .resetn(resetn),
//        .in_data(psum_mem_acc_rd_dout_after_reg),
//        .out_data(out_quantized)
//    );

endmodule
