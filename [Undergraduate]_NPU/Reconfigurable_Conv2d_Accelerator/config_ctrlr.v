`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/15 00:06:51
// Design Name: 
// Module Name: config_ctrlr
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


module config_ctrlr #(
    // SRAM address widths
    parameter ADDR_MAX_IFM      = 10,       // Address for one of the CONV_IC(0~7) IFM memories (MAX: 28x28 for 1 Act MEM)
    parameter ADDR_MAX_WEIGHT   = 17,       // Address for MAX CONV_OC * CONV_IC * K_H * K_W weights (128*64*9 = 73728 < 2^17=131072)
    parameter ADDR_MAX_OFM      = 16,       // Address for one of the CONV_OC Output FM (OFM) memories
    parameter DATA_BW           = 32,       // float 32bit data
    parameter IMG_W_MAX         = 28 
)(
    input                       clk,
    input                       resetn,
    // ------------------------------------------------------------------------
    // Configurable Data
    // ------------------------------------------------------------------------
    input  wire [2:0] KH,
    input  wire [2:0] KW,
    input  wire [13:0] IC_ITER,      // MAX IC : 128 -> MAX IC_ITER : 128/8 = 16
    input  wire [13:0] TOT_IC,
    input  wire [4:0] IMG_H,
    input  wire [4:0] IMG_W,
    input  wire [7:0] OC,
//    input  wire stride,
    // ------------------------------------------------------------------------
    // Start & Done Signals
    // ------------------------------------------------------------------------
    input  wire start,          // Start signal for config_ctrlr operation
    output wire done,           // config_ctrlr operation finished
    // ------------------------------------------------------------------------
    // ifmap inputs & read
    // ------------------------------------------------------------------------
    output                      ifm_rd_en,      // Read enable for IFM SRAMs
    output  [ADDR_MAX_IFM-1:0]  ifm_rd_addr,    // Read address for IFM SRAMs

    input   signed [DATA_BW-1:0]    ifm_in0,        // Data from IFM SRAM
    input   signed [DATA_BW-1:0]    ifm_in1,
    input   signed [DATA_BW-1:0]    ifm_in2,
    input   signed [DATA_BW-1:0]    ifm_in3,
    input   signed [DATA_BW-1:0]    ifm_in4,
    input   signed [DATA_BW-1:0]    ifm_in5,
    input   signed [DATA_BW-1:0]    ifm_in6,
    input   signed [DATA_BW-1:0]    ifm_in7,
    // ------------------------------------------------------------------------
    // Weight read & cast to PE
    // ------------------------------------------------------------------------
    // Weights Read Interface
    output                          weight_rd_en,       // Read enable for Conv weight memory
    output  [ADDR_MAX_WEIGHT-1:0]   weight_rd_addr,     // Read address for Conv weight memory
    output wire                     weight_en_pe0, weight_en_pe1, weight_en_pe2,  weight_en_pe3,  weight_en_pe4,  weight_en_pe5,  weight_en_pe6,  weight_en_pe7,
    // ------------------------------------------------------------------------
    // Interface to 8 PEs (Processing Elements)
    // Output: 3x3 windows for each of the 8 input channels
    // ------------------------------------------------------------------------
    output reg signed [DATA_BW-1:0] window00_in0, window01_in0, window02_in0, window10_in0, window11_in0, window12_in0, window20_in0, window21_in0, window22_in0,
    output reg signed [DATA_BW-1:0] window00_in1, window01_in1, window02_in1, window10_in1, window11_in1, window12_in1, window20_in1, window21_in1, window22_in1,
    output reg signed [DATA_BW-1:0] window00_in2, window01_in2, window02_in2, window10_in2, window11_in2, window12_in2, window20_in2, window21_in2, window22_in2,
    output reg signed [DATA_BW-1:0] window00_in3, window01_in3, window02_in3, window10_in3, window11_in3, window12_in3, window20_in3, window21_in3, window22_in3,
    output reg signed [DATA_BW-1:0] window00_in4, window01_in4, window02_in4, window10_in4, window11_in4, window12_in4, window20_in4, window21_in4, window22_in4,
    output reg signed [DATA_BW-1:0] window00_in5, window01_in5, window02_in5, window10_in5, window11_in5, window12_in5, window20_in5, window21_in5, window22_in5,
    output reg signed [DATA_BW-1:0] window00_in6, window01_in6, window02_in6, window10_in6, window11_in6, window12_in6, window20_in6, window21_in6, window22_in6,
    output reg signed [DATA_BW-1:0] window00_in7, window01_in7, window02_in7, window10_in7, window11_in7, window12_in7, window20_in7, window21_in7, window22_in7,
    // ------------------------------------------------------------------------
    // Output PSUM Address or Output Pixel Address
    // ------------------------------------------------------------------------
    output wire [1:0]               out_addr_start_n,
    output wire [7:0]               CURRENT_OC_ITER,
    output wire [ADDR_MAX_OFM-1:0]  out_addr
    );
    // ------------------------------------------------------------------------
    // 
    // ------------------------------------------------------------------------
    localparam CONV_PE_H = 3;
    localparam CONV_PE_W = 3;
    localparam CONV_PE_NUM = 8;
    localparam CONV_PE_NUM_LOG = 3;
    
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    reg  [7:0] IMG_H_reg, IMG_H_reg_copy;
    reg  [7:0] IMG_W_reg, IMG_W_reg_copy, IMG_W_reg_copy2;
   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            IMG_H_reg       <= 0;
            IMG_H_reg_copy  <= 0;
            IMG_W_reg       <= 0;
            IMG_W_reg_copy  <= 0;
            IMG_W_reg_copy2 <= 0;
        end
        else begin
            IMG_H_reg       <= IMG_H;
            IMG_H_reg_copy  <= IMG_H;
            IMG_W_reg       <= IMG_W;
            IMG_W_reg_copy  <= IMG_W;
            IMG_W_reg_copy2 <= IMG_W;
        end
    end
    
    wire [4:0] IMG_OC_H;
    wire [4:0] IMG_OC_W;
    
    assign IMG_OC_H = ((IMG_H_reg_copy-2) >= 0) ? IMG_H_reg_copy-2 : 1;
    assign IMG_OC_W = ((IMG_W_reg_copy-2) >= 0) ? IMG_W_reg_copy-2 : 1;

//    assign IMG_OC_H = ((IMG_H-2) >= 0) ? IMG_H-2 : 1;
//    assign IMG_OC_W = ((IMG_W-2) >= 0) ? IMG_W-2 : 1; 
    
    
    wire [ADDR_MAX_OFM-1:0] CURRENT_OFM_MAX_ADDR;
    reg  [ADDR_MAX_OFM-1:0] CURRENT_OFM_MAX_ADDR_temp;
    reg  [ADDR_MAX_OFM-1:0] CURRENT_OFM_MAX_ADDR_reg;
    assign CURRENT_OFM_MAX_ADDR             = CURRENT_OFM_MAX_ADDR_reg;
//    assign CURRENT_OFM_MAX_ADDR             = OC * (IMG_OC_W) * (IMG_OC_H);
       always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            CURRENT_OFM_MAX_ADDR_reg <= 0;

        end
        else begin
            CURRENT_OFM_MAX_ADDR_temp <= OC * (IMG_OC_H);
            CURRENT_OFM_MAX_ADDR_reg  <= CURRENT_OFM_MAX_ADDR_temp * (IMG_OC_W);
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////
    
    wire [ADDR_MAX_IFM-1:0] CURRENT_IFM_MAX_ADDR_PER_ITER;      // IMG_H * IMG_W        (MAX: 28*28 Double Word = 784 Double Word)
    wire [9:0] CURRENT_TOTAL_WEIGHTS_PER_PE;
    assign CURRENT_IFM_MAX_ADDR_PER_ITER    = IMG_W_reg * IMG_H_reg;
    assign CURRENT_TOTAL_WEIGHTS_PER_PE     = KH * KW;
    
    wire [9:0] NEEDED_WEIGHT_CYCLES_PER_ITER;
    wire [9:0] MAX_WEIGHT_CYCLES_PER_ITER;
//    wire [9:0] NUM_ITER_IC_MAX;
    wire [9:0] NUM_ITER_OC_MAX;
    wire [19:0] NUM_ITER_MAX;
    assign NEEDED_WEIGHT_CYCLES_PER_ITER    = CONV_PE_NUM * CURRENT_TOTAL_WEIGHTS_PER_PE - IMG_W_reg * 2;    // 8 * KH * KW - IMG_Width * 2 (Because Line Buffer has 2 FIFO) / (MAX: 8 * 9 - 20 * 2)
    assign MAX_WEIGHT_CYCLES_PER_ITER       = CONV_PE_NUM * CURRENT_TOTAL_WEIGHTS_PER_PE;                // 8 * KH * KW (MAX: 72 )
//    assign MAX_WEIGHT_CYCLES_PER_ITER       = (CONV_PE_NUM_LOG << CURRENT_TOTAL_WEIGHTS_PER_PE);         // 8 * KH * KW (MAX: 72 )
//    assign NEEDED_WEIGHT_CYCLES_PER_ITER    = MAX_WEIGHT_CYCLES_PER_ITER - IMG_W * 2;                    // 8 * KH * KW - IMG_Width * 2 (Because Line Buffer has 2 FIFO) / (MAX: 8 * 9 - 20 * 2)
    assign NUM_ITER_OC_MAX                  = OC;                                                        // MAX: 128
    assign NUM_ITER_MAX                     = NUM_ITER_OC_MAX;                                           // MAX: 128
   
    
    /////////////////////////////// STATE Transition Logic /////////////////////////////// 
    localparam IDLE             = 4'd0;
    localparam LOAD_WEIGHTS     = 4'd1;
    localparam LOAD_LBF         = 4'd2;
    localparam RUN_CONV         = 4'd3;
    localparam IS_DONE_WAIT     = 4'd4;
    localparam IS_DONE_WAIT2    = 4'd5;
    localparam IS_DONE_WAIT3    = 4'd6;
    localparam IS_DONE_WAIT4    = 4'd7;
    localparam IS_DONE          = 4'd8;
    localparam DONE             = 4'd9;
    
    reg [3:0] state, n_state;
    reg [6:0] wait_cnt;                                     // TOTAL_WEIGHTS_TO_LOAD per itearation => 72(= 3*3*8) < 2^7 = 128
    reg [7:0] num_iteration, num_iteration_copy;            // Need to iterate IFM [OC] times                                   (MAX = 128 = 128)
    reg [ADDR_MAX_IFM-1:0] cnt_run;                         // Need to Read IFM Memory by input image size = IMG_H * IMG_W * IC (MAX = 22*22*64 = 30976)
    
    assign done = (state == DONE); 
    assign CURRENT_OC_ITER = num_iteration;
    
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
                n_state = LOAD_WEIGHTS;
            end               
            else n_state = IDLE;
        end            
        LOAD_WEIGHTS : begin
            if(wait_cnt == NEEDED_WEIGHT_CYCLES_PER_ITER-1)  // NEEDED_WEIGHT_CYCLES - 1 = 19 (ex. for CONV2)
                n_state = LOAD_LBF;
            else n_state = LOAD_WEIGHTS;
        end
        LOAD_LBF : begin
            if(wait_cnt == MAX_WEIGHT_CYCLES_PER_ITER-1)     // MAX_WEIGHT_CYCLES - 1 = 71 (ex. for CONV2)
                n_state = RUN_CONV;
            else n_state = LOAD_LBF;
        end
        RUN_CONV : begin
            if(cnt_run == CURRENT_IFM_MAX_ADDR_PER_ITER - 1) begin
                n_state = IS_DONE_WAIT;
            end
            else n_state = RUN_CONV;
        end
        IS_DONE_WAIT : begin
            n_state = IS_DONE_WAIT2;
        end
        IS_DONE_WAIT2 : begin
            n_state = IS_DONE_WAIT3;
        end
        IS_DONE_WAIT3 : begin
            n_state = IS_DONE_WAIT4;
        end
        IS_DONE_WAIT4 : begin
            n_state = IS_DONE;
        end
        IS_DONE : begin
            if(num_iteration == (NUM_ITER_MAX - 1))                    
                n_state = DONE;
            else
                n_state = LOAD_WEIGHTS;                
        end
        DONE : begin
            n_state = IDLE;
        end
        default :  n_state = IDLE;
        endcase
    end 
    
    /////////////////////////////// Iteration Logic /////////////////////////////// 
    /***** counter for LOAD_WEIGHTS *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) 
            wait_cnt <= 0;
        else begin
           case (state)     
                LOAD_WEIGHTS, LOAD_LBF : begin
                    wait_cnt <= wait_cnt + 1;
                end
                default : wait_cnt <= 0;
            endcase
        end
    end
    
    /***** count iterations for Multi Kernels *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            num_iteration       <= 0;
            num_iteration_copy  <= 0;
        end
        else begin
            case (state)
                IDLE : begin
                    num_iteration       <= 0;
                    num_iteration_copy  <= 0;
                end
                IS_DONE : begin
                    num_iteration       <= num_iteration + 1;
                    num_iteration_copy  <= num_iteration_copy + 1;
                end
                default :  begin
                    num_iteration       <= num_iteration;
                    num_iteration_copy  <= num_iteration_copy;
                end
            endcase
        end
    end
 
     /////////////////////////////// Load Weights to each PE ///////////////////////////////
    reg [3:0]                   weight_en_trigger_cnt;
    reg [7:0]                   which_weight_en;
    reg                         weight_rd_en_reg;
    reg [ADDR_MAX_WEIGHT-1:0]   weight_addr_reg, weight_addr_reg_delay;
    
     /***** trigger signal for weight_en; trigger every 9 cycles (since: there are 9 MACS in PE) *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            weight_en_trigger_cnt <= 0;
        end
        else begin
            case (state)         
                LOAD_WEIGHTS, LOAD_LBF: begin
                    if(weight_en_trigger_cnt == 9 - 1) begin // TOTAL_WEIGHTS_PER_PE - 1 = 8
                        weight_en_trigger_cnt <= 0;
                    end
                    else begin
                        weight_en_trigger_cnt <= weight_en_trigger_cnt + 1;
                    end
                end
                default :  begin
                    weight_en_trigger_cnt <= 0; 
                end
            endcase
        end
    end
    
    /**** shift which_weight_en to figure out which weight load needed *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            which_weight_en <= 8'd0;
        end
        else begin
            case (state)         
                LOAD_WEIGHTS, LOAD_LBF: begin
                    if(weight_en_trigger_cnt == 0) begin
                        if(which_weight_en == 8'd0 || which_weight_en == 8'd1)
                            which_weight_en <= 8'b10000000;
                        else
                            which_weight_en <= (which_weight_en >> 1);
                    end
                end
                default :  begin
                    which_weight_en <= 8'd0;
                end
            endcase
        end
    end
    
   /***** Increment Address to read Weight MEM & make weight read enable High (FLOW: Filter0 IC_0~7 ----> Filter1 IC_0~7 -----> ...) *****/
   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            weight_addr_reg <= 0;
            weight_rd_en_reg <= 0;
        end
        else begin
            case (state)
                    IDLE: begin
//                        weight_addr_reg <= CONV_PE_NUM_LOG << ((IC_ITER-1) * CURRENT_TOTAL_WEIGHTS_PER_PE); // offset for input channel
                        weight_addr_reg <= (IC_ITER-1) * CONV_PE_NUM * CURRENT_TOTAL_WEIGHTS_PER_PE; // offset for input channel
                    end
                    LOAD_WEIGHTS, LOAD_LBF: begin
                        weight_addr_reg <= weight_addr_reg + 1;
                        weight_rd_en_reg <= 1;
                    end
                    IS_DONE_WAIT : begin
                            weight_addr_reg <= ((num_iteration_copy+1) * TOT_IC); // offset for input channel & output channel
                    end
                    IS_DONE_WAIT2 : begin
                            weight_addr_reg <= weight_addr_reg * CURRENT_TOTAL_WEIGHTS_PER_PE; // offset for input channel & output channel
                    end
                    IS_DONE : begin
                        if(TOT_IC < CONV_PE_NUM) begin
                            weight_addr_reg <= (num_iteration_copy + 1) * CURRENT_TOTAL_WEIGHTS_PER_PE;
                            weight_rd_en_reg <= 0;
                        end
                        else begin
                            weight_addr_reg <= weight_addr_reg + (IC_ITER-1) * CONV_PE_NUM * CURRENT_TOTAL_WEIGHTS_PER_PE;
//                            weight_addr_reg <= ((num_iteration_copy+1) * TOT_IC + (IC_ITER-1) * CONV_PE_NUM) * CURRENT_TOTAL_WEIGHTS_PER_PE; // offset for input channel & output channel
                        end
                    end
                    DONE : begin
                        weight_addr_reg <= 0;
                        weight_rd_en_reg <= 0;
                    end
                    default :  begin
                        weight_addr_reg <= weight_addr_reg;
                        weight_rd_en_reg <= 0;
                    end
            endcase
        end
    end
    
   /***** 1cycle delay Address to read Weight MEM *****/
   always @(posedge clk or negedge resetn) begin
        if(~resetn)
            weight_addr_reg_delay <= 0;
        else begin
            weight_addr_reg_delay <= weight_addr_reg;
        end
    end
    
    assign {weight_en_pe0, weight_en_pe1, weight_en_pe2,  weight_en_pe3,  weight_en_pe4,  weight_en_pe5,  weight_en_pe6,  weight_en_pe7} = (weight_rd_en_reg) ? which_weight_en : 0;
    
    assign weight_rd_en = weight_rd_en_reg; //(state == LOAD_WEIGHTS | state == LOAD_LBF);
    assign weight_rd_addr = weight_addr_reg_delay;
    
    /////////////////////////////// Read Activations from ACT_MEM_0 ~ 7 ///////////////////////////////
    reg isRunning;
    reg ifm_rd_en_delay;
    
    assign ifm_rd_en = isRunning;
    assign ifm_rd_we = 1'b0;
    assign ifm_rd_addr = (isRunning) ? cnt_run: {ADDR_MAX_IFM{1'b0}};
    
    /***** Defines what address to read *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) cnt_run <= 0;
        else begin
            if(isRunning)
                if(cnt_run == CURRENT_IFM_MAX_ADDR_PER_ITER - 1)
                    cnt_run <= 0;
                else
                    cnt_run <= cnt_run + 1;
            else
                cnt_run <= 0;
        end
    end
    /***** Define whether Computing is Activated *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) 
            isRunning <= 0;
        else begin
            case (state)     
                LOAD_LBF, RUN_CONV : begin
                    isRunning <= 1;  
                end         
                default :  begin
                    isRunning <= 0;
                end
            endcase
        end
    end
   /***** 1 cycle delay Because of BRAM's 1 cycle read delay *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            ifm_rd_en_delay <= 0;
        end
        else begin
            ifm_rd_en_delay <= ifm_rd_en;
        end
    end
    
    /////////////////////////////// Control Line Buffer & Make Window of Pixels ///////////////////////////////
    localparam Window_Row_Bit_Width = DATA_BW * CONV_PE_W;
    
    /***** Line Buffer Variables *****/
    reg [4:0] addr_counter;                 // fifo address counter (count for IMG_W, MAX: 28)

    wire lb_resetn = (resetn & ifm_rd_en);

    always @(posedge clk or negedge lb_resetn) begin
        if (!lb_resetn) begin
            addr_counter <= 0;
        end
        else if (ifm_rd_en) begin
            if (addr_counter == (IMG_W_reg - 1)) begin
                addr_counter <= 0;
            end
            else begin
                addr_counter <= addr_counter + 1;
            end
        end
    end
    
    /***** Line buffer 0 *****/
    wire [Window_Row_Bit_Width-1:0] lb_data_0;
    wire lb_ready_0;
    line_buffer #(.DATA_WIDTH(DATA_BW), .FIFO_DEPTH(IMG_W_MAX), .NUM_FIFO(3), .IMG_W(IMG_W_MAX)) LBUF1 (
        .clk(clk), .resetn(lb_resetn),
        .CURRENT_IMG_W(IMG_W_reg_copy2),
        .ready(lb_ready_0),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_in0), .data_out(lb_data_0),
        .addr_counter(addr_counter),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 1 *****/
    wire [Window_Row_Bit_Width-1:0] lb_data_1;
    wire lb_ready_1;
    line_buffer #(.DATA_WIDTH(DATA_BW), .FIFO_DEPTH(IMG_W_MAX), .NUM_FIFO(3), .IMG_W(IMG_W_MAX)) LBUF2 (
        .clk(clk), .resetn(lb_resetn),
        .CURRENT_IMG_W(IMG_W_reg_copy2),
        .ready(lb_ready_1),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_in1), .data_out(lb_data_1),
        .addr_counter(addr_counter),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 2 *****/
    wire [Window_Row_Bit_Width-1:0] lb_data_2;
    wire lb_ready_2;
    line_buffer #(.DATA_WIDTH(DATA_BW), .FIFO_DEPTH(IMG_W_MAX), .NUM_FIFO(3), .IMG_W(IMG_W_MAX)) LBUF3 (
        .clk(clk), .resetn(lb_resetn),
        .CURRENT_IMG_W(IMG_W_reg_copy2),
        .ready(lb_ready_2),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_in2), .data_out(lb_data_2),
        .addr_counter(addr_counter),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 3 *****/
    wire [Window_Row_Bit_Width-1:0] lb_data_3;
    wire lb_ready_3;  
    line_buffer #(.DATA_WIDTH(DATA_BW), .FIFO_DEPTH(IMG_W_MAX), .NUM_FIFO(3), .IMG_W(IMG_W_MAX)) LBUF4 (
        .clk(clk), .resetn(lb_resetn),
        .CURRENT_IMG_W(IMG_W_reg_copy2),
        .ready(lb_ready_3),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_in3), .data_out(lb_data_3),
        .addr_counter(addr_counter),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 4 *****/
    wire [Window_Row_Bit_Width-1:0] lb_data_4;
    wire lb_ready_4;
    line_buffer #(.DATA_WIDTH(DATA_BW), .FIFO_DEPTH(IMG_W_MAX), .NUM_FIFO(3), .IMG_W(IMG_W_MAX)) LBUF5 (
        .clk(clk), .resetn(lb_resetn),
        .CURRENT_IMG_W(IMG_W_reg_copy2),
        .ready(lb_ready_4),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_in4), .data_out(lb_data_4),
        .addr_counter(addr_counter),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 5 *****/
    wire [Window_Row_Bit_Width-1:0] lb_data_5;
    wire lb_ready_5;
    line_buffer #(.DATA_WIDTH(DATA_BW), .FIFO_DEPTH(IMG_W_MAX), .NUM_FIFO(3), .IMG_W(IMG_W_MAX)) LBUF6 (
        .clk(clk), .resetn(lb_resetn),
        .CURRENT_IMG_W(IMG_W_reg_copy2),
        .ready(lb_ready_5),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_in5), .data_out(lb_data_5),
        .addr_counter(addr_counter),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 6 *****/
    wire [Window_Row_Bit_Width-1:0] lb_data_6;
    wire lb_ready_6;
    line_buffer #(.DATA_WIDTH(DATA_BW), .FIFO_DEPTH(IMG_W_MAX), .NUM_FIFO(3), .IMG_W(IMG_W_MAX)) LBUF7 (
        .clk(clk), .resetn(lb_resetn),
        .CURRENT_IMG_W(IMG_W_reg_copy2),
        .ready(lb_ready_6),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_in6), .data_out(lb_data_6),
        .addr_counter(addr_counter),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 7 *****/
    wire [Window_Row_Bit_Width-1:0] lb_data_7;
    wire lb_ready_7;
    line_buffer #(.DATA_WIDTH(DATA_BW), .FIFO_DEPTH(IMG_W_MAX), .NUM_FIFO(3), .IMG_W(IMG_W_MAX)) LBUF8 (
        .clk(clk), .resetn(lb_resetn),
        .CURRENT_IMG_W(IMG_W_reg_copy2),
        .ready(lb_ready_7),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_in7), .data_out(lb_data_7),
        .addr_counter(addr_counter),
        .wr_sel(wr_sel)
    );

    /***** assign datas to get each pixel data *****/
    // should consider 1cycle read delay from FIFO Bram -> every edge when lb_ready is high the first element is 0
    reg lb_ready_0_delay;
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            lb_ready_0_delay <= 0;
        end
        else begin
            lb_ready_0_delay <= lb_ready_0;
        end
    end
    
    wire signed [Window_Row_Bit_Width-1:0] row_ch0, row_ch1, row_ch2, row_ch3, row_ch4, row_ch5, row_ch6, row_ch7;

    assign row_ch0 = lb_ready_0_delay ? lb_data_0[Window_Row_Bit_Width-1:0] : 96'b0;
    assign row_ch1 = lb_ready_0_delay ? lb_data_1[Window_Row_Bit_Width-1:0] : 96'b0;
    assign row_ch2 = lb_ready_0_delay ? lb_data_2[Window_Row_Bit_Width-1:0] : 96'b0;
    assign row_ch3 = lb_ready_0_delay ? lb_data_3[Window_Row_Bit_Width-1:0] : 96'b0;
    assign row_ch4 = lb_ready_0_delay ? lb_data_4[Window_Row_Bit_Width-1:0] : 96'b0;
    assign row_ch5 = lb_ready_0_delay ? lb_data_5[Window_Row_Bit_Width-1:0] : 96'b0;
    assign row_ch6 = lb_ready_0_delay ? lb_data_6[Window_Row_Bit_Width-1:0] : 96'b0;
    assign row_ch7 = lb_ready_0_delay ? lb_data_7[Window_Row_Bit_Width-1:0] : 96'b0;
    
    /***** make 3x3 window *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            window00_in0<=0; window01_in0<=0; window02_in0<=0; window10_in0<=0; window11_in0<=0; window12_in0<=0; window20_in0<=0; window21_in0<=0; window22_in0<=0;
            window00_in1<=0; window01_in1<=0; window02_in1<=0; window10_in1<=0; window11_in1<=0; window12_in1<=0; window20_in1<=0; window21_in1<=0; window22_in1<=0;
            window00_in2<=0; window01_in2<=0; window02_in2<=0; window10_in2<=0; window11_in2<=0; window12_in2<=0; window20_in2<=0; window21_in2<=0; window22_in2<=0;
            window00_in3<=0; window01_in3<=0; window02_in3<=0; window10_in3<=0; window11_in3<=0; window12_in3<=0; window20_in3<=0; window21_in3<=0; window22_in3<=0;
            window00_in4<=0; window01_in4<=0; window02_in4<=0; window10_in4<=0; window11_in4<=0; window12_in4<=0; window20_in4<=0; window21_in4<=0; window22_in4<=0;
            window00_in5<=0; window01_in5<=0; window02_in5<=0; window10_in5<=0; window11_in5<=0; window12_in5<=0; window20_in5<=0; window21_in5<=0; window22_in5<=0;
            window00_in6<=0; window01_in6<=0; window02_in6<=0; window10_in6<=0; window11_in6<=0; window12_in6<=0; window20_in6<=0; window21_in6<=0; window22_in6<=0;
            window00_in7<=0; window01_in7<=0; window02_in7<=0; window10_in7<=0; window11_in7<=0; window12_in7<=0; window20_in7<=0; window21_in7<=0; window22_in7<=0;
        end
        else begin
            // when line buffer pops the data out
            if(lb_ready_0) begin
                // window shift
                {window00_in0,window10_in0,window20_in0} <= {window01_in0,window11_in0,window21_in0};     // col 3, shifted
                {window01_in0,window11_in0,window21_in0} <= {window02_in0,window12_in0,window22_in0};     // col 2, shifted
                {window02_in0,window12_in0,window22_in0} <= {row_ch0};                                    // col 1
            end
            if(lb_ready_1) begin
                {window00_in1,window10_in1,window20_in1} <= {window01_in1,window11_in1,window21_in1};     // col 3, shifted
                {window01_in1,window11_in1,window21_in1} <= {window02_in1,window12_in1,window22_in1};     // col 2, shifted
                {window02_in1,window12_in1,window22_in1} <= {row_ch1};                                     // col 1
            end
            if(lb_ready_2) begin
                {window00_in2,window10_in2,window20_in2} <= {window01_in2,window11_in2,window21_in2};     // col 3, shifted
                {window01_in2,window11_in2,window21_in2} <= {window02_in2,window12_in2,window22_in2};     // col 2, shifted
                {window02_in2,window12_in2,window22_in2} <= {row_ch2};                                    // col 1
            end
            if(lb_ready_3) begin
                {window00_in3,window10_in3,window20_in3} <= {window01_in3,window11_in3,window21_in3};     // col 3, shifted
                {window01_in3,window11_in3,window21_in3} <= {window02_in3,window12_in3,window22_in3};     // col 2, shifted
                {window02_in3,window12_in3,window22_in3} <= {row_ch3};                                    // col 1
            end
            if(lb_ready_4) begin
                {window00_in4,window10_in4,window20_in4} <= {window01_in4,window11_in4,window21_in4};     // col 3, shifted
                {window01_in4,window11_in4,window21_in4} <= {window02_in4,window12_in4,window22_in4};     // col 2, shifted
                {window02_in4,window12_in4,window22_in4} <= {row_ch4};                                    // col 1
            end
            if(lb_ready_5) begin
                {window00_in5,window10_in5,window20_in5} <= {window01_in5,window11_in5,window21_in5};     // col 3, shifted
                {window01_in5,window11_in5,window21_in5} <= {window02_in5,window12_in5,window22_in5};     // col 2, shifted
                {window02_in5,window12_in5,window22_in5} <= {row_ch5};                                    // col 1
            end
            if(lb_ready_6) begin
                {window00_in6,window10_in6,window20_in6} <= {window01_in6,window11_in6,window21_in6};     // col 3, shifted
                {window01_in6,window11_in6,window21_in6} <= {window02_in6,window12_in6,window22_in6};     // col 2, shifted
                {window02_in6,window12_in6,window22_in6} <= {row_ch6};                                    // col 1
            end
            if(lb_ready_7) begin
                {window00_in7,window10_in7,window20_in7} <= {window01_in7,window11_in7,window21_in7};     // col 3, shifted
                {window01_in7,window11_in7,window21_in7} <= {window02_in7,window12_in7,window22_in7};     // col 2, shifted
                {window02_in7,window12_in7,window22_in7} <= {row_ch7};                                    // col 1
            end
        end
    end
    
    /////////////////////////////// Control Address of output piexels ///////////////////////////////
    localparam IFM_W_LOG        = 5;    // MAX: 28 < 2^5 = 32
    localparam IFM_H_LOG        = 5;    // MAX: 28 < 2^5 = 32
    
    reg [ADDR_MAX_OFM-1 : 0]    out_addr_reg;
    reg [ADDR_MAX_OFM-1 : 0]    out_addr_reg_delay;
//    reg [ADDR_MAX_OFM-1 : 0]    out_addr_reg_delay2;
    
    reg [IFM_W_LOG-1 : 0]           row;
    reg [IFM_H_LOG-1 : 0]           col;
    reg [1:0]                       local_we;
    reg [1:0]                       local_we_delay, local_we_delay2, local_we_delay3, local_we_delay4;   // delay2, delay3, delay4 is for 8channel accumulation start signal
    
    assign out_addr_start_n = local_we_delay4;
    
    // wait for line buffet to be ready (local_we == 0 when, lb_ready0 goes to HIgh)
    always @(posedge clk or negedge lb_resetn) begin
        if(!lb_resetn)                                          local_we <= 2;
        else if ((addr_counter == IMG_W_reg - 1) && local_we > 0)   local_we <= local_we - 1;
    end 
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin 
            local_we_delay <= 0; local_we_delay2 <= 0; local_we_delay3 <= 0; local_we_delay4 <= 0;
        end
        else        
            begin 
                local_we_delay <= local_we; local_we_delay2 <= local_we_delay; local_we_delay3 <= local_we_delay2; local_we_delay4 <= local_we_delay3; 
            end
    end

    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            row <= 0; 
            col <= 0;
        end
        else begin
            case (state)
                IDLE : begin
                    row <= 0; 
                    col <= 0;
                end
                default :  begin
                    if(!isRunning) begin
                        col <= 0;
                        row <= row;
                    end
                    else if(row == IMG_H - 2) begin
                        row <= 0;
                    end
                    else if(col == IMG_W_reg - 1) begin
                        col <= 0; 
                        row <= row + 1'b1;
                    end 
                    else if (!local_we_delay) begin
                        col <= col + 1'b1;
                    end
                end
            endcase
        end
    end
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            out_addr_reg <= 0;
        end
        else begin
            case (state)
                IDLE : begin
                    out_addr_reg <= 0;
                end
                IS_DONE_WAIT3 : begin
//                    out_addr_reg <= (num_iteration + 1) * (IMG_OC_W) * (IMG_OC_H);
                    out_addr_reg <= (num_iteration + 1);
                end
                IS_DONE_WAIT4 : begin
//                    out_addr_reg <= (num_iteration + 1) * (IMG_OC_W) * (IMG_OC_H);
                    out_addr_reg <= out_addr_reg * (IMG_OC_W);
                end
                IS_DONE : begin
//                    out_addr_reg <= (num_iteration + 1) * (IMG_OC_W) * (IMG_OC_H);
                    out_addr_reg <= out_addr_reg * (IMG_OC_H);
                end
                default :  begin
                    if(out_addr_reg == CURRENT_OFM_MAX_ADDR) begin  /////////////////// 25.08.19 FC layer 중 수정 ////////////////////////////////// CURRENT_OFM_MAX_ADDR - 1
                        out_addr_reg <= out_addr_reg;               /////////////////// 25.08.19 FC layer 중 수정 ////////////////////////////////// out_addr_reg <= 0;
                    end
                    else if(col >= 2) begin
                        out_addr_reg <= out_addr_reg + 1'b1;
                    end
                end
            endcase
        end
    end
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            out_addr_reg_delay <= 0;
        end
        else begin
            out_addr_reg_delay <= out_addr_reg;
        end
    end
    
    assign out_addr = out_addr_reg_delay;
    
//   ///////////////////////////////// FOR TIMING CONSTRAINTS ///////////////////////////////////
//    reg [IFM_H_LOG-1:0] col_delay, col_delay2, col_delay3, col_delay4, col_delay5, col_delay6, col_delay7;
//    always @(posedge clk or negedge resetn) begin
//        if(!resetn) begin
//            col_delay <= 0;
//            col_delay2 <= 0;
//            col_delay3 <= 0;
//            col_delay4 <= 0;
//            col_delay5 <= 0;
//            col_delay6 <= 0;
//            col_delay7 <= 0;
//        end
//        else begin
//            col_delay <= col;
//            col_delay2 <= col_delay;
//            col_delay3 <= col_delay2;
//            col_delay4 <= col_delay3;
//            col_delay5 <= col_delay4;
//            col_delay6 <= col_delay5;
//            col_delay7 <= col_delay6;
//        end
//    end
//    //////////////////////////////////////////////////////////////////////////////////////////////
    
//    always @(posedge clk or negedge resetn) begin
//        if(!resetn) begin
//            out_addr_reg <= 0;
//        end
//        else begin
//            if(out_addr_reg >= CURRENT_OFM_MAX_ADDR - 1) begin
//                out_addr_reg <= 0;
//            end
//            else if(col_delay7 >= 2) begin
//                out_addr_reg <= out_addr_reg + 1'b1;
//            end
//        end
//    end
    
//    always @(posedge clk or negedge resetn) begin
//        if(!resetn) begin
//            out_addr_reg_delay <= 0;
//            out_addr_reg_delay2 <= 0;
//        end
//        else begin
//            out_addr_reg_delay <= out_addr_reg;
//            out_addr_reg_delay2 <= out_addr_reg_delay;
//        end
//    end
    
//    assign out_addr = out_addr_reg_delay2;
//    assign col_out = col_delay7;
  
    
endmodule
