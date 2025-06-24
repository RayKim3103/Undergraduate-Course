`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/05/25
// Design Name: MaxPool Layer
// Module Name: maxpool
// Description: 2x2 MaxPooling layer for streaming activation inputs
//              with CHANNEL-parallel outputs.
//
//////////////////////////////////////////////////////////////////////////////////

module max_pool #(
    parameter CHANNELS = 16,
    parameter D_IN_MAX_ADDR_LOG = 10,
    parameter D_IN_MAX_ADDR     = 576,
     parameter D_OUT_MAX_ADDR_LOG = 8,
    parameter IN_W = 24,
    parameter IN_H = 24
)(
    input  wire                             clk,
    input  wire                             resetn,
//    input  wire                             en,
    input  wire [D_IN_MAX_ADDR_LOG - 1:0]   din_addr,
    input  wire [4:0]                       din_col,
    input  wire signed [7:0]                din,        //from conv2 result 
    output wire  [7:0]                      dout,
    output wire [D_OUT_MAX_ADDR_LOG-1:0]    dout_addr
//    output reg                              vld_out
);
    
    localparam IDLE = 3'd0, ROW0 = 3'd1, ROW0_SKIP = 3'd2, ROW1 = 3'd3, ROW1_SKIP = 3'd4, DONE = 3'd5;
    localparam MAX_ROW_COUNT = IN_W + 2 - 1;
    localparam MAX_MAXPOOL_ADDRESS = 144;
    
    reg [2:0] state, n_state;
    
    // Internal 2x2 buffer for pooling
    reg signed [7:0] buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11;
    reg [7:0] maxpool_addr;     // 144 < 2^8 = 256
    
    ////////////////////////////////////////////////////// signals to know when does the Row Changes /////////////////////////////////////////////////////
    reg [4:0] din_col_delay;
    reg [4:0] din_col_delay2;
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            din_col_delay <= 0;
            din_col_delay2 <= 0;
        end
        else begin
            din_col_delay <= din_col;
            din_col_delay2 <= din_col_delay;
        end
    end
    
    wire toggle_row;
    
    ////////////////////////////////////////////////////// State Machine ////////////////////////////////////////////////////// 
    /***** state transition *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) state <= IDLE;
        else state <= n_state;
    end
    
   /***** Defines when does the state change *****/ 
   always @(*) begin
        case (state)
            IDLE : begin 
                if(din_col_delay2 == 1) begin
                    n_state = ROW0;
                end               
                else n_state = IDLE;
            end            
            ROW0 : begin
                if(din_addr == D_IN_MAX_ADDR-1)
                    n_state = IDLE;
                else if(din_col_delay2 == 25)  // NEEDED_WEIGHT_CYCLES - 2 = 18 / NEEDED_WEIGHT_CYCLES - 1 = 19
                    n_state = ROW0_SKIP;
                else n_state = ROW0;
            end
            ROW0_SKIP : begin
                if(din_col_delay2 == 1)
                    n_state = ROW1;
                else
                    n_state = ROW0_SKIP;
            end
            ROW1 : begin
                if(din_addr == D_IN_MAX_ADDR-1)
                    n_state = IDLE;
                else if(din_col_delay2 == 25)     // MAX_WEIGHT_CYCLES - 2 = 70 / MAX_WEIGHT_CYCLES - 1 = 71
                    n_state = ROW1_SKIP;
                else n_state = ROW1;
            end
            ROW1_SKIP : begin
                if(din_col_delay2 == 1)
                    n_state = ROW0;
                else
                    n_state = ROW1_SKIP;
            end
            default :  n_state = IDLE;
        endcase
    end
    
    ////////////////////////////////////////////////////// Use Buffers to calculate MaxPool & Also Caluculate Address in this code //////////////////////////////////////////////////////
    reg signed [7:0] dout_reg;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            maxpool_addr   <= 0;
//            vld_out     <= 0;
            dout_reg        <= 0;
            buf0 <= 8'd0; buf1 <= 8'd0; buf2 <= 8'd0;  buf3 <= 8'd0;
            buf4 <= 8'd0; buf5 <= 8'd0; buf6 <= 8'd0;  buf7 <= 8'd0;
            buf8 <= 8'd0; buf9 <= 8'd0; buf10 <= 8'd0; buf11 <= 8'd0;
        end 
        else begin
            case (state)
                IDLE : begin 
                    buf0 <= 8'd0; buf1 <= 8'd0; buf2 <= 8'd0;  buf3 <= 8'd0;
                    buf4 <= 8'd0; buf5 <= 8'd0; buf6 <= 8'd0;  buf7 <= 8'd0;
                    buf8 <= 8'd0; buf9 <= 8'd0; buf10 <= 8'd0; buf11 <= din;
                end            
                ROW0 : begin
                    if((din_addr & 1) == 0) begin
                            buf11 <= din;
                    end 
                    else begin
                        if(din_col_delay2 == 25) begin
                            if(din > buf11)
                                {buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11} <= {buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, din, buf0};
                            else
                                {buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11} <= {buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf0}; // buf0
                        end
                        else begin
                            if(din > buf11)
                                {buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11} <= {buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, din, 8'd0};
                            else
                                {buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11} <= {buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, 8'd0}; // buf0
                        end
                    end
                end
                ROW1 : begin
                    if((din_addr & 1) == 0) begin
                        if(din > buf11) begin
                            buf11<= din;
                        end
                    end 
                    else begin
                        if(maxpool_addr == MAX_MAXPOOL_ADDRESS-1)
                            maxpool_addr <= 0;
                        else
                            maxpool_addr <= maxpool_addr + 1;

                        if(din > buf11) begin// buf11 <= din;
                            dout_reg <= din;
                            {buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11} <= {buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, din, buf0};
                        end
                        else begin
                            dout_reg <= buf11;
                            {buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11} <= {buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf0};
                        end
                    end
                end
                default :  begin
                    buf0 <= buf0; buf1 <= buf1; buf2 <= buf2;  buf3 <= buf3;
                    buf4 <= buf4; buf5 <= buf5; buf6 <= buf6;  buf7 <= buf7;
                    buf8 <= buf8; buf9 <= buf9; buf10 <= buf10; buf11 <= buf11;
                end
            endcase
        end
    end
    
    reg [7:0] maxpool_addr_delay;     // 144 < 2^8 = 256
    reg [7:0] maxpool_addr_delay2;     // 144 < 2^8 = 256
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            maxpool_addr_delay <= 0;
            maxpool_addr_delay2 <= 0;
        end
        else begin
            maxpool_addr_delay <= maxpool_addr;
            maxpool_addr_delay2 <= maxpool_addr_delay;
        end
    end
    
    assign dout = dout_reg;
    assign dout_addr = maxpool_addr_delay2;

endmodule