`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/15 21:54:56
// Design Name: 
// Module Name: act_ctrlr
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


module act_ctrlr #(
    // SRAM address widths
    parameter ADDR_MAX_IFM      = 16,       // Address for one of the CONV_IC IFM memories (MAX: 20*20*128 = 51200 < 2^16 = 65536)
    parameter ACT_MEM_ADDR_MAX  = 10,
    parameter DATA_BW           = 32,       // float 32bit data
    parameter IMG_W_MAX         = 28 
)(
    input                       clk,
    input                       resetn,
    // ------------------------------------------------------------------------
    // Start & Done Signals
    // ------------------------------------------------------------------------
    input  wire start,          // Start signal for config_ctrlr operation
    output wire done,           // config_ctrlr operation finished
    // ------------------------------------------------------------------------
    // Configurable Data
    // ------------------------------------------------------------------------
    input  wire [13:0] TOT_IC,
    input  wire [4:0] IMG_H,
    input  wire [4:0] IMG_W,
    // ------------------------------------------------------------------------
    // IC Iteration Signals
    // ------------------------------------------------------------------------
    input wire  [13:0] IC_ITER_IN,       // MAX IC : 128 -> MAX IC_ITER : 128/8 = 16
//    output wire [3:0] IC_ITER_OUT,      // MAX IC : 128 -> MAX IC_ITER : 128/8 = 16
    // ------------------------------------------------------------------------
    // ifmap inputs & read
    // ------------------------------------------------------------------------
    input   [DATA_BW-1:0]       ifm_rd_data,        // Read data from IFM SRAMs
    output                      ifm_rd_en,          // Read enable for IFM SRAMs
    output  [ADDR_MAX_IFM-1:0]  ifm_rd_addr,        // Read address for IFM SRAMs
    // ------------------------------------------------------------------------
    // ifmap outputs & write
    // ------------------------------------------------------------------------
    output                           ifm_wr_en0,
    output                           ifm_wr_en1,
    output                           ifm_wr_en2,
    output                           ifm_wr_en3,
    output                           ifm_wr_en4,
    output                           ifm_wr_en5,
    output                           ifm_wr_en6,
    output                           ifm_wr_en7,
    output  [ACT_MEM_ADDR_MAX-1:0]   ifm_wr_addr,
    output   signed [DATA_BW-1:0]    ifm_out0,
    output   signed [DATA_BW-1:0]    ifm_out1,
    output   signed [DATA_BW-1:0]    ifm_out2,
    output   signed [DATA_BW-1:0]    ifm_out3,
    output   signed [DATA_BW-1:0]    ifm_out4,
    output   signed [DATA_BW-1:0]    ifm_out5,
    output   signed [DATA_BW-1:0]    ifm_out6,
    output   signed [DATA_BW-1:0]    ifm_out7
    );
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    reg  [4:0] IMG_H_reg;
    reg  [4:0] IMG_W_reg;
    wire [4:0] IMG_H_after_reg;
    wire [4:0] IMG_W_after_reg;
   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            IMG_H_reg   <= 0;
            IMG_W_reg   <= 0;
        end
        else begin
            IMG_H_reg   <= IMG_H;
            IMG_W_reg   <= IMG_W;
        end
    end
    assign IMG_H_after_reg = IMG_H_reg;
    assign IMG_W_after_reg = IMG_W_reg;
    
    wire [ADDR_MAX_IFM-1:0] CURRENT_IFM_MAX_ADDR_PER_ITER;      // IMG_H * IMG_W                                    (MAX: 28*28 Double Word = 784 Double Word)
    
    wire [ADDR_MAX_IFM-1:0] CURRENT_IFM_MAX_ADDR;               // IMG_H * IMG_W * CONV_PE_NUM                      (MAX: 28*28*8 Double Word = 6272 Double Word)
    wire [ADDR_MAX_IFM-1:0] CURRENT_IFM_OFFSET;                 // IC_ITER_IN * IMG_H * IMG_W * CONV_PE_NUM         (MAX: IC_ITER_IN*28*28*8 Double Word)
    
    reg  [ADDR_MAX_IFM-1:0] CURRENT_IFM_MAX_ADDR_reg;               // IMG_H * IMG_W * CONV_PE_NUM                      (MAX: 28*28*8 Double Word = 6272 Double Word)
    reg  [ADDR_MAX_IFM-1:0] CURRENT_IFM_OFFSET_reg;                 // IC_ITER_IN * IMG_H * IMG_W * CONV_PE_NUM         (MAX: IC_ITER_IN*28*28*8 Double Word)
    
    assign CURRENT_IFM_MAX_ADDR_PER_ITER = IMG_W_after_reg * IMG_H_after_reg;
    
//    assign CURRENT_IFM_MAX_ADDR          = IMG_W_after_reg * IMG_H_after_reg * 8;
//    assign CURRENT_IFM_OFFSET            = IC_ITER_IN * CURRENT_IFM_MAX_ADDR;
    
       always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            CURRENT_IFM_MAX_ADDR_reg   <= 0;
            CURRENT_IFM_OFFSET_reg     <= 0;
        end
        else begin
            CURRENT_IFM_MAX_ADDR_reg   <=  CURRENT_IFM_MAX_ADDR_PER_ITER * 8;
            CURRENT_IFM_OFFSET_reg     <= IC_ITER_IN * CURRENT_IFM_MAX_ADDR;
        end
    end
    
    assign CURRENT_IFM_MAX_ADDR          = CURRENT_IFM_MAX_ADDR_reg;
    assign CURRENT_IFM_OFFSET            = CURRENT_IFM_OFFSET_reg;
    /////////////////////////////////////////////////////////////////////////////////////// 
    

    
    localparam IDLE = 4'd0, 
               TRANSFER0 = 4'd1, TRANSFER1 = 4'd2, TRANSFER2 = 4'd3, TRANSFER3 = 4'd4,
               TRANSFER4 = 4'd5, TRANSFER5 = 4'd6, TRANSFER6 = 4'd7, TRANSFER7 = 4'd8, 
               DONE = 4'd9;
    
    reg [3:0] state, n_state;
    
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
                n_state = TRANSFER0;
            end
            else n_state = IDLE;
        end            
        TRANSFER0 : begin
            if(ifm_wr_addr == (CURRENT_IFM_MAX_ADDR_PER_ITER-2))                          //  + CURRENT_IFM_OFFSET
                n_state = TRANSFER1;
            else n_state = TRANSFER0;
        end
        TRANSFER1 : begin
            if(ifm_wr_addr == (CURRENT_IFM_MAX_ADDR_PER_ITER-2))                        // 2*   // + CURRENT_IFM_OFFSET
                n_state = TRANSFER2;
            else n_state = TRANSFER1;
        end
        TRANSFER2 : begin
            if(ifm_wr_addr == (CURRENT_IFM_MAX_ADDR_PER_ITER-2))                        // 3*
                n_state = TRANSFER3;
            else n_state = TRANSFER2;
        end
        TRANSFER3 : begin
            if(ifm_wr_addr == (CURRENT_IFM_MAX_ADDR_PER_ITER-2))                        // 4*
                n_state = TRANSFER4;
            else n_state = TRANSFER3;
        end
        TRANSFER4 : begin
            if(ifm_wr_addr == (CURRENT_IFM_MAX_ADDR_PER_ITER-2))                        // 5*
                n_state = TRANSFER5;
            else n_state = TRANSFER4;
        end
        TRANSFER5 : begin
            if(ifm_wr_addr == (CURRENT_IFM_MAX_ADDR_PER_ITER-2))                        // 6*
                n_state = TRANSFER6;
            else n_state = TRANSFER5;
        end
        TRANSFER6 : begin
            if(ifm_wr_addr == (CURRENT_IFM_MAX_ADDR_PER_ITER-2))                        // 7*
                n_state = TRANSFER7;
            else n_state = TRANSFER6;
        end
        TRANSFER7 : begin
            if(ifm_wr_addr == (CURRENT_IFM_MAX_ADDR_PER_ITER-2))                        // 8*
                n_state = DONE;
            else n_state = TRANSFER7;
        end
        DONE : begin
            n_state = IDLE;
        end
        default :  n_state = IDLE;
        endcase
    end 

    /////////////////////////////////////////// Calculate Read Address of IFM MEM ///////////////////////////////////////////
    reg  [ADDR_MAX_IFM-1:0] ifm_rd_addr_reg;
    reg  [ADDR_MAX_IFM-1:0] ifm_rd_addr_reg_delay;
    reg                     ifm_rd_en_reg;
    assign ifm_rd_addr = ifm_rd_addr_reg_delay;
    assign ifm_rd_en    = ifm_rd_en_reg;

    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin 
            ifm_rd_addr_reg_delay <= 0;
        end
        else begin
            ifm_rd_addr_reg_delay <= ifm_rd_addr_reg;
        end
    end
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            ifm_rd_addr_reg <= 0;
            ifm_rd_en_reg <= 0;
        end
        else begin
            case (state)
                IDLE : begin
                    ifm_rd_addr_reg <= CURRENT_IFM_OFFSET;
                    ifm_rd_en_reg <= 0;
                end
                TRANSFER0, TRANSFER1, TRANSFER2, TRANSFER3, TRANSFER4, TRANSFER5, TRANSFER6, TRANSFER7: begin
                    ifm_rd_addr_reg <= ifm_rd_addr_reg + 1;
                    ifm_rd_en_reg <= 1;
                end
                default :  begin
                    ifm_rd_addr_reg <= CURRENT_IFM_OFFSET;
                    ifm_rd_en_reg <= 0;
                end
            endcase
        end
    end
    
    /////////////////////////////////////////// Calculate Write Address of ACT MEM 0~7 ///////////////////////////////////////////
    reg  [ACT_MEM_ADDR_MAX-1:0]  ifm_wr_addr_reg;
    reg  [7:0]                   ifm_wr_en_array;
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            ifm_wr_addr_reg <= 0;
            ifm_wr_en_array <= 0;
        end
        else begin
            case (state)
                IDLE : begin
                    ifm_wr_addr_reg <= 0;
                    ifm_wr_en_array <= 0;
                end
                TRANSFER0 : begin
                    if(ifm_wr_addr_reg == CURRENT_IFM_MAX_ADDR_PER_ITER-1) begin
                        ifm_wr_addr_reg <= 0;
                    end
                    else begin
                        ifm_wr_addr_reg <= ifm_wr_addr_reg + 1;
                    end
                    if(state != n_state) begin                     // ifm_wr_addr_reg == (CURRENT_IFM_MAX_ADDR_PER_ITER-1 + CURRENT_IFM_OFFSET)
                        ifm_wr_en_array <= ifm_wr_en_array << 1; 
//                        ifm_wr_addr_reg <= 0;
                    end
                    else begin
                        ifm_wr_en_array <= 8'b0000_0001;
//                        ifm_wr_addr_reg <= ifm_wr_addr_reg + 1;
                    end
                end
                TRANSFER1, TRANSFER2, TRANSFER3, TRANSFER4, TRANSFER5, TRANSFER6, TRANSFER7: begin
                    if(ifm_wr_addr_reg == CURRENT_IFM_MAX_ADDR_PER_ITER-1) begin
                        ifm_wr_addr_reg <= 0;
                    end
                    else begin
                        ifm_wr_addr_reg <= ifm_wr_addr_reg + 1;
                    end
                    if((state != n_state)) begin                        // ifm_wr_addr_reg == (CURRENT_IFM_MAX_ADDR_PER_ITER-1 + CURRENT_IFM_OFFSET) 
                        ifm_wr_en_array <= ifm_wr_en_array << 1;
//                        ifm_wr_addr_reg <= 0;
                    end
                    else begin
                        ifm_wr_en_array <= ifm_wr_en_array;
//                        ifm_wr_addr_reg <= ifm_wr_addr_reg + 1;
                    end
                end
                default :  begin
                    ifm_wr_addr_reg <= 0;
                    ifm_wr_en_array <= 0;
                end
            endcase
        end
    end
    
    reg  [7:0]                  ifm_wr_en_array_delay;
    reg  [ACT_MEM_ADDR_MAX-1:0] ifm_wr_addr_reg_delay, ifm_wr_addr_reg_delay2;
   /***** 1 cycle delay Because of BRAM's 1 cycle read delay *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            ifm_wr_en_array_delay   <= 0;
            ifm_wr_addr_reg_delay   <= 0;
            ifm_wr_addr_reg_delay2  <= 0;
        end
        else begin
            ifm_wr_en_array_delay <= ifm_wr_en_array;
            ifm_wr_addr_reg_delay <= ifm_wr_addr_reg;
            ifm_wr_addr_reg_delay2 <= ifm_wr_addr_reg_delay;
        end
    end
    
    
    assign ifm_wr_addr = ifm_wr_addr_reg_delay2;
    assign {ifm_wr_en7,ifm_wr_en6,ifm_wr_en5,ifm_wr_en4,ifm_wr_en3,ifm_wr_en2,ifm_wr_en1,ifm_wr_en0} = ifm_wr_en_array_delay;
    assign ifm_out0 = ifm_rd_data; assign ifm_out1 = ifm_rd_data; assign ifm_out2 = ifm_rd_data; assign ifm_out3 = ifm_rd_data;
    assign ifm_out4 = ifm_rd_data; assign ifm_out5 = ifm_rd_data; assign ifm_out6 = ifm_rd_data; assign ifm_out7 = ifm_rd_data;

endmodule
