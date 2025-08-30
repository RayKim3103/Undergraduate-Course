
`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/27 15:03:20
// Design Name: 
// Module Name: top_memory_ctrlr
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


module single_core#(
    // SRAM address widths (enough to cover depth)
    parameter ADDR_TOTALW   = 17,           // 2^17 = 131,072 > 128000 Double Word
    parameter ADDR_IN       = 16,           // ASSUME Batch Size = 2; 2^16 = 65,536 > 51200 Double Word
    parameter ADDR_OUT      = 16,            // Psum out => 2^17 = 131,072 > 128000 Byte / output logits 2^4=16 > 10 Byte
    parameter INPUT_BW      = 32,           // 32bit Data comes from AXI interface
    parameter OUTPUT_BW     = 32            // 8bit Data goes to AXI interface (after Quantization)
    )(
    input wire clk,
    input wire resetn,
    // ------------------------------------------------------------------------
    // Configurable Data
    // ------------------------------------------------------------------------
    input  wire start,
    input  wire [2:0] KH,
    input  wire [2:0] KW,
    input  wire [13:0] IC,
    input  wire [4:0] IMG_H,
    input  wire [4:0] IMG_W,
    input  wire [7:0] OC,
//    input  wire stride,
//    output wire ready,
    output wire done,
    // ------------------------------------------------------------------------
    // input image memory [port A (PS writes image in advance), port B (PL reads)]
    // ------------------------------------------------------------------------
    input wire                          input_clka,
    input wire                          input_ena,
    input wire                          input_wea,
    input wire [ADDR_IN-1:0]            input_addra, 
    input wire signed [INPUT_BW-1:0]    input_dina,
    output wire signed [INPUT_BW-1:0]   input_douta,
    // ------------------------------------------------------------------------
    // Store Total Weights
    // ------------------------------------------------------------------------
    input  wire                             weight_clka,
    input  wire                             weight_ena,
    input  wire                             weight_wea,
    input  wire [ADDR_TOTALW-1:0]           weight_addra,
    input  wire signed [INPUT_BW-1:0]       weight_dina,
    output wire signed [INPUT_BW-1:0]       weight_douta,
    // ------------------------------------------------------------------------
    // Store Output Logits
    // ------------------------------------------------------------------------
    input  wire                             out_mem_clkb,
    input  wire                             out_mem_enb,
    input  wire                             out_mem_web,
    input  wire [ADDR_OUT-1:0]              out_mem_addrb,
    input  wire signed [OUTPUT_BW-1:0]      out_mem_dinb,
    output wire signed [OUTPUT_BW-1:0]      out_mem_doutb
    );
    // ------------------------------------------------------------------------
    // IF MAP & TOTAL WEIGHT MEM B-ports (READ), OUT MEM A-ports (WRITE)
    // ------------------------------------------------------------------------
    wire                            input_enb;
    wire                            input_web;
    wire [ADDR_IN-1:0]              input_addrb;
    wire signed [INPUT_BW-1:0]      input_dinb;
    wire signed [INPUT_BW-1:0]      input_doutb;
    
    wire                            weight_enb;
    wire                            weight_web;
    wire [ADDR_TOTALW-1:0]          weight_addrb;
    wire signed [INPUT_BW-1:0]      weight_dinb;
    wire signed [INPUT_BW-1:0]      weight_doutb;
    
    wire                            out_mem_ena;
    wire                            out_mem_wea;
    wire [ADDR_OUT-1:0]             out_mem_addra;
    wire signed [OUTPUT_BW-1:0]     out_mem_dina;
    wire signed [OUTPUT_BW-1:0]     out_mem_douta;
    // ------------------------------------------------------------------------
    // ACT MEM Ports A-ports (WRITE), B-ports (READ)
    // ------------------------------------------------------------------------
    localparam ACT_MEM_ADDR_MAX = 10; 
    wire                            act_mem_0_ena, act_mem_1_ena, act_mem_2_ena, act_mem_3_ena, act_mem_4_ena, act_mem_5_ena, act_mem_6_ena, act_mem_7_ena;
    wire                            act_mem_0_wea, act_mem_1_wea, act_mem_2_wea, act_mem_3_wea, act_mem_4_wea, act_mem_5_wea, act_mem_6_wea, act_mem_7_wea;
    wire [ACT_MEM_ADDR_MAX-1:0]     act_mem_0_addra, act_mem_1_addra, act_mem_2_addra, act_mem_3_addra, act_mem_4_addra, act_mem_5_addra, act_mem_6_addra, act_mem_7_addra;
    wire signed [INPUT_BW-1:0]      act_mem_0_dina, act_mem_1_dina, act_mem_2_dina, act_mem_3_dina, act_mem_4_dina, act_mem_5_dina, act_mem_6_dina, act_mem_7_dina;
 
    wire                            act_mem_0_enb, act_mem_1_enb, act_mem_2_enb, act_mem_3_enb, act_mem_4_enb, act_mem_5_enb, act_mem_6_enb, act_mem_7_enb;
    wire [ACT_MEM_ADDR_MAX-1:0]     act_mem_0_addrb, act_mem_1_addrb, act_mem_2_addrb, act_mem_3_addrb, act_mem_4_addrb, act_mem_5_addrb, act_mem_6_addrb, act_mem_7_addrb;
    wire signed [INPUT_BW-1:0]      act_mem_0_doutb, act_mem_1_doutb, act_mem_2_doutb, act_mem_3_doutb, act_mem_4_doutb, act_mem_5_doutb, act_mem_6_doutb, act_mem_7_doutb;
    // ------------------------------------------------------------------------
    // PSUM MEM & PSUM_MEM_ACC Ports A-ports (WRITE), B-ports (READ)
    // ------------------------------------------------------------------------
    wire                            psum_mem_ena;
    wire                            psum_mem_wea;
    wire [ADDR_OUT-1:0]             psum_mem_addra;
    wire signed [INPUT_BW-1:0]      psum_mem_dina;
 
    wire                            psum_mem_enb;
    wire [ADDR_OUT-1:0]             psum_mem_addrb;
    wire signed [INPUT_BW-1:0]      psum_mem_doutb;
    
    wire                            psum_mem_acc_ena;
    wire                            psum_mem_acc_wea;
    wire [ADDR_OUT-1:0]             psum_mem_acc_addra;
    wire signed [INPUT_BW-1:0]      psum_mem_acc_dina;
 
    wire                            psum_mem_acc_enb;
    wire [ADDR_OUT-1:0]             psum_mem_acc_addrb;
    wire signed [INPUT_BW-1:0]      psum_mem_acc_doutb;
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    wire [13:0] IC_act,     IC_acc,     IC_config;
    wire [4:0]  IMG_H_act,  IMG_H_acc,  IMG_H_config;
    wire [4:0]  IMG_W_act,  IMG_W_acc,  IMG_W_config;
    wire [7:0]  OC_act,     OC_acc,     OC_config;
    
    reg [13:0] IC_act_reg,     IC_acc_reg,      IC_config_reg;
    reg [4:0]  IMG_H_act_reg,  IMG_H_acc_reg,   IMG_H_config_reg;
    reg [4:0]  IMG_W_act_reg,  IMG_W_acc_reg,   IMG_W_config_reg;
    reg [7:0]  OC_act_reg,     OC_acc_reg,      OC_config_reg;
    
    assign IC_act = IC_act_reg;         assign IC_acc = IC_acc_reg;         assign IC_config    = IC_config_reg;
    assign IMG_H_act = IMG_H_act_reg;   assign IMG_H_acc = IMG_H_acc_reg;   assign IMG_H_config = IMG_H_config_reg;
    assign IMG_W_act = IMG_W_act_reg;   assign IMG_W_acc = IMG_W_acc_reg;   assign IMG_W_config = IMG_W_config_reg;
    assign OC_act = OC_act_reg;         assign OC_acc = OC_acc_reg;         assign OC_config    = OC_config_reg;
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            IC_act_reg <= 0;     IC_acc_reg <= 0;       IC_config_reg <= 0;
            IMG_H_act_reg <= 0;  IMG_H_acc_reg <= 0;    IMG_H_config_reg <= 0;
            IMG_W_act_reg <= 0;  IMG_W_acc_reg <= 0;    IMG_W_config_reg <= 0;
            OC_act_reg <= 0;     OC_acc_reg <= 0;       OC_config_reg <= 0;
        end
        else begin
            IC_act_reg <= IC;           IC_acc_reg <= IC;           IC_config_reg <= IC;
            IMG_H_act_reg <= IMG_H;     IMG_H_acc_reg <= IMG_H;     IMG_H_config_reg <= IMG_H;
            IMG_W_act_reg <= IMG_W;     IMG_W_acc_reg <= IMG_W;     IMG_W_config_reg <= IMG_W;
            OC_act_reg <= OC;           OC_acc_reg <= OC;           OC_config_reg <= OC;
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////

    // ------------------------------------------------------------------------
    // Main Control FSM
    // ------------------------------------------------------------------------
    localparam CONV_PE_NUM = 8;
    wire [13:0] IC_ITER_IN;
    reg  [13:0] IC_ITER_IN_reg;
    reg  [13:0] main_iteration_num;  // (MAX: IC/CONV_PE_NUM = 128/8 = 16)
    
    reg        act_ctrlr_start_reg, config_ctrlr_start_reg, acc_ctrlr_start_reg;    // fc_ctrlr_start_reg, 
    wire       act_ctrlr_start, config_ctrlr_start, acc_ctrlr_start;                // fc_ctrlr_start, 
    wire       act_ctrlr_done, config_ctrlr_done, acc_ctrlr_done;                   // fc_ctrlr_done, 
    
    reg config_ctrlr_done_delay,   config_ctrlr_done_delay2,  config_ctrlr_done_delay3,  config_ctrlr_done_delay4,  config_ctrlr_done_delay5,  config_ctrlr_done_delay6,  config_ctrlr_done_delay7;
    reg config_ctrlr_done_delay8,  config_ctrlr_done_delay9,  config_ctrlr_done_delay10, config_ctrlr_done_delay11, config_ctrlr_done_delay12, config_ctrlr_done_delay13, config_ctrlr_done_delay14;
    reg config_ctrlr_done_delay15, config_ctrlr_done_delay16, config_ctrlr_done_delay17, config_ctrlr_done_delay18, config_ctrlr_done_delay19;
    reg config_ctrlr_done_delay20, config_ctrlr_done_delay21, config_ctrlr_done_delay22, config_ctrlr_done_delay23;
    reg config_ctrlr_done_delay24, config_ctrlr_done_delay25, config_ctrlr_done_delay26, config_ctrlr_done_delay27;
    reg config_ctrlr_done_delay28, config_ctrlr_done_delay29, config_ctrlr_done_delay30, config_ctrlr_done_delay31;
    reg config_ctrlr_done_delay32, config_ctrlr_done_delay33, config_ctrlr_done_delay34, config_ctrlr_done_delay35;
    reg config_ctrlr_done_delay36;
    
    localparam  IDLE                = 4'd0,
                ACT_CTRLR_START     = 4'd1,
                ACT_CTRLR           = 4'd2,
                CONFIG_CTRLR_START  = 4'd3,
                CONFIG_CTRLR        = 4'd4,
                ACC_CTRLR_START     = 4'd5,
                ACC_CTRLR           = 4'd6,
                IS_DONE             = 4'd7,
                DONE                = 4'd8;
    reg [3:0] state, n_state;
    
    
    ////////////////////////// CSR DONE Signal Logic //////////////////////////
    wire [ADDR_OUT-1:0] WRITE_ADDRESS_MAX_single_core;
    reg  [ADDR_OUT-1:0] WRITE_ADDRESS_MAX_single_core_reg;
    wire [4:0] IMG_OC_H_single_core;
    wire [4:0] IMG_OC_W_single_core;
    reg  [4:0] IMG_OC_H_single_core_reg;
    reg  [4:0] IMG_OC_W_single_core_reg;
   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            IMG_OC_H_single_core_reg <= 0;
            IMG_OC_W_single_core_reg <= 0;
            WRITE_ADDRESS_MAX_single_core_reg <= 0;   
        end
        else begin
//            WRITE_ADDRESS_MAX_single_core_reg <= 16;
            WRITE_ADDRESS_MAX_single_core_reg <=    (IMG_OC_H_single_core)*(IMG_OC_W_single_core)*OC;
            if      ((IMG_H-2) >= 0)    IMG_OC_H_single_core_reg <= IMG_H - 2;
            else                        IMG_OC_H_single_core_reg <= 1;
            if      ((IMG_W-2) >= 0)    IMG_OC_W_single_core_reg <= IMG_W - 2;
            else                        IMG_OC_W_single_core_reg <= 1;
        end
    end
    assign IMG_OC_H_single_core = IMG_OC_H_single_core_reg;
    assign IMG_OC_W_single_core = IMG_OC_W_single_core_reg;
    assign WRITE_ADDRESS_MAX_single_core = WRITE_ADDRESS_MAX_single_core_reg;
    
    reg out_mem_write_done;
    always@(posedge clk or negedge resetn) begin
        if(!resetn) begin
            out_mem_write_done          <= 0;
        end
        else begin
            if(out_mem_addra >= WRITE_ADDRESS_MAX_single_core && state == DONE)  // -1
                out_mem_write_done <= 1;
            else if(start) begin
                out_mem_write_done <= 0;
            end
            else begin
                out_mem_write_done <= out_mem_write_done;
            end
        end
    end
    // from out_mem_write done ~ start => done signal is High
    assign done = (start == 1) ?  0 : out_mem_write_done;
    //////////////////////////////////////////////////////////////////////////
//    assign done = (state == DONE);
    //////////////////////////////////////////////////////////////////////////

    assign IC_ITER_IN = IC_ITER_IN_reg;
    assign act_ctrlr_start = act_ctrlr_start_reg; 
    assign config_ctrlr_start = config_ctrlr_start_reg; 
    assign acc_ctrlr_start = acc_ctrlr_start_reg;
    
    reg start_delay;
    always @(posedge clk or negedge resetn) begin
        if(~resetn) start_delay <= 0;
        else        start_delay <= start;
    end
    
    /***** state transition *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) state <= IDLE;
        else state <= n_state;
    end

    ////////////////////////// CHECK CSR DONE Signal //////////////////////////
//    // output OUT_MEM Address to write
//    wire                            out_mem_en;
//    wire                            out_mem_we;
//    wire [ADDR_OUT-1:0]             out_mem_addr;
//    wire signed [OUTPUT_BW-1:0]     out_mem_din;
//    wire signed [OUTPUT_BW-1:0]     out_mem_dout;
    
//    reg [3:0]   csr_cnt;
    
//    /********** You should change the note in below which is real write signals operation **********/
    
//    reg                 psum_mem_wr_en_csr;
//    reg                 psum_mem_wr_we_csr;
//    reg [3:0]           psum_mem_wr_addr_csr;
//    reg signed [13:0]   psum_mem_wr_data_csr;
    
//    always @(posedge clk or negedge resetn) begin
//        if(~resetn) begin
//            psum_mem_wr_en_csr      <= 0;
//            psum_mem_wr_we_csr      <= 0;
//            psum_mem_wr_addr_csr    <= 0;
//            psum_mem_wr_data_csr    <= 0;
//        end
//        else begin
//            case (state)
//                IS_DONE : begin
//                    psum_mem_wr_en_csr      <= 1;
//                    psum_mem_wr_we_csr      <= 1;
//                    psum_mem_wr_addr_csr    <= csr_cnt;
//                    if     (csr_cnt == 1 && KW != 0)        psum_mem_wr_data_csr    <= KW;
//                    else if(csr_cnt == 2 && KH != 0)        psum_mem_wr_data_csr    <= KH;
//                    else if(csr_cnt == 3 && IC != 0)        psum_mem_wr_data_csr    <= IC;
//                    else if(csr_cnt == 4 && IMG_H != 0)     psum_mem_wr_data_csr    <= IMG_H;
//                    else if(csr_cnt == 5 && IMG_W != 0)     psum_mem_wr_data_csr    <= IMG_W;
//                    else if(csr_cnt == 6 && OC != 0)        psum_mem_wr_data_csr    <= OC;
//                    else                                    psum_mem_wr_data_csr    <= 123;
//                end
//                default : begin
//                    psum_mem_wr_en_csr      <= 0;
//                    psum_mem_wr_we_csr      <= 0;
//                    psum_mem_wr_addr_csr    <= 0;
//                    psum_mem_wr_data_csr    <= 0;
//                end
//            endcase
//        end
//    end
    
//    assign out_mem_en      = psum_mem_wr_en_csr;
//    assign out_mem_we      = psum_mem_wr_we_csr;
//    assign out_mem_addr    = psum_mem_wr_addr_csr;
//    assign out_mem_din     = psum_mem_wr_data_csr;
    
    
//    always @(posedge clk or negedge resetn) begin
//        if(~resetn) csr_cnt <= 0;
//        else begin
//            case (state)
//                IS_DONE : begin
//                    csr_cnt <= csr_cnt + 1; 
//                end
//                default : begin
//                    csr_cnt <= 0;
//                end
//            endcase
//        end
//    end
    //////////////////////////////////////////////////////////////////////////
    
   /***** Defines when does the state change *****/ 
    always @(*) begin
        case (state)
        IDLE : begin
            if(start || start_delay) begin
            ////////////////////////// CHECK CSR DONE Signal //////////////////////////
//                n_state = IS_DONE;
            //////////////////////////////////////////////////////////////////////////
                n_state = ACT_CTRLR_START;
            end
            else n_state = IDLE;
        end
        ACT_CTRLR_START : begin
            n_state = ACT_CTRLR;
        end
        ACT_CTRLR : begin
            if(act_ctrlr_done)
                n_state = CONFIG_CTRLR_START;
            else n_state = ACT_CTRLR;
        end
        CONFIG_CTRLR_START : begin
            n_state = CONFIG_CTRLR;
        end
        CONFIG_CTRLR : begin
            if(config_ctrlr_done_delay36)
                n_state = ACC_CTRLR_START;
            else n_state = CONFIG_CTRLR;
        end
        ACC_CTRLR_START : begin
            n_state = ACC_CTRLR;
        end
        ACC_CTRLR : begin
            if(acc_ctrlr_done)
                n_state = IS_DONE;
            else n_state = ACC_CTRLR;
        end
        IS_DONE : begin
            ////////////////////////// CHECK CSR DONE Signal //////////////////////////
//            if (csr_cnt == 15) begin
//                n_state = DONE;
//            end
            //////////////////////////////////////////////////////////////////////////
            if(main_iteration_num == (IC/CONV_PE_NUM) || IC == 1)                    
                n_state = DONE;
            else
                n_state = ACT_CTRLR_START;                
        end
        DONE : begin
//            if(start) begin
                n_state     = IDLE;
//            end
//            else begin
//                n_state     = DONE;
//            end 
        end
        default :  n_state = IDLE;
        endcase
    end 
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            IC_ITER_IN_reg      <= 0;
            main_iteration_num  <= 0;
            act_ctrlr_start_reg <= 0;   config_ctrlr_start_reg <= 0;    acc_ctrlr_start_reg <= 0;
        end
        else begin
            case (state)
                IDLE : begin
                    IC_ITER_IN_reg      <= 0;
                    main_iteration_num  <= 0;
                    act_ctrlr_start_reg <= 0; config_ctrlr_start_reg <= 0; acc_ctrlr_start_reg <= 0;
                end
                ACT_CTRLR_START: begin
                    main_iteration_num  <= main_iteration_num + 1;
                    act_ctrlr_start_reg <= 1;
                end
                ACT_CTRLR: begin
                    act_ctrlr_start_reg <= 0;
                end
                CONFIG_CTRLR_START: begin
                    IC_ITER_IN_reg         <= IC_ITER_IN_reg + 1;
                    config_ctrlr_start_reg <= 1;
                end
                CONFIG_CTRLR: begin
                    config_ctrlr_start_reg <= 0;
                end
                ACC_CTRLR_START: begin
                    acc_ctrlr_start_reg <= 1;
                end
                ACC_CTRLR: begin
                    acc_ctrlr_start_reg <= 0;
                end
                default :  begin
                    IC_ITER_IN_reg      <= IC_ITER_IN_reg;
                    main_iteration_num  <= main_iteration_num;
                    act_ctrlr_start_reg <= act_ctrlr_start_reg; config_ctrlr_start_reg <= config_ctrlr_start_reg;   acc_ctrlr_start_reg <= acc_ctrlr_start_reg;
                end
            endcase
        end
    end
    
    // ------------------------------------------------------------------------
    // Input Activation Controller
    // ------------------------------------------------------------------------
    wire [ACT_MEM_ADDR_MAX-1:0]  ifm_wr_addr;
    
    assign act_mem_0_ena = act_mem_0_wea; assign act_mem_1_ena = act_mem_1_wea; assign act_mem_2_ena = act_mem_2_wea; assign act_mem_3_ena = act_mem_3_wea;
    assign act_mem_4_ena = act_mem_4_wea; assign act_mem_5_ena = act_mem_5_wea; assign act_mem_6_ena = act_mem_6_wea; assign act_mem_7_ena = act_mem_7_wea;
    
    assign act_mem_0_addra = ifm_wr_addr; assign act_mem_1_addra = ifm_wr_addr; assign act_mem_2_addra = ifm_wr_addr; assign act_mem_3_addra = ifm_wr_addr;
    assign act_mem_4_addra = ifm_wr_addr; assign act_mem_5_addra = ifm_wr_addr; assign act_mem_6_addra = ifm_wr_addr; assign act_mem_7_addra = ifm_wr_addr;
    
    act_ctrlr act_ctrlr (   .clk(clk), .resetn(resetn),
                            
                            .start(act_ctrlr_start),
                            .done(act_ctrlr_done),
                            
                            .TOT_IC(IC_act), .IMG_H(IMG_H_act), .IMG_W(IMG_W_act),
                            .IC_ITER_IN(IC_ITER_IN),
                            
                            // ifmap inputs & read
                            .ifm_rd_data(input_doutb), .ifm_rd_en(input_enb), .ifm_rd_addr(input_addrb),
                            
                            // ifmap outputs & write
                            .ifm_wr_en0(act_mem_0_wea), .ifm_wr_en1(act_mem_1_wea), .ifm_wr_en2(act_mem_2_wea), .ifm_wr_en3(act_mem_3_wea), 
                            .ifm_wr_en4(act_mem_4_wea), .ifm_wr_en5(act_mem_5_wea), .ifm_wr_en6(act_mem_6_wea), .ifm_wr_en7(act_mem_7_wea),
                            .ifm_wr_addr(ifm_wr_addr),
                            .ifm_out0(act_mem_0_dina), .ifm_out1(act_mem_1_dina), .ifm_out2(act_mem_2_dina), .ifm_out3(act_mem_3_dina), 
                            .ifm_out4(act_mem_4_dina), .ifm_out5(act_mem_5_dina), .ifm_out6(act_mem_6_dina), .ifm_out7(act_mem_7_dina)
                        );
    // ------------------------------------------------------------------------
    // Configurable Controller
    // ------------------------------------------------------------------------
    assign input_web = 0;
        
    wire                        ifm_rd_en;
    wire [ACT_MEM_ADDR_MAX-1:0] ifm_rd_addr;
    
    wire                        weight_rd_en;
    wire [ADDR_TOTALW-1:0]      weight_rd_addr;
    wire                        weight_en_0, weight_en_1, weight_en_2, weight_en_3, weight_en_4, weight_en_5, weight_en_6, weight_en_7;
    
    wire signed [INPUT_BW-1:0]  window00_in0, window01_in0, window02_in0, window10_in0, window11_in0, window12_in0, window20_in0, window21_in0, window22_in0,
                                window00_in1, window01_in1, window02_in1, window10_in1, window11_in1, window12_in1, window20_in1, window21_in1, window22_in1,
                                window00_in2, window01_in2, window02_in2, window10_in2, window11_in2, window12_in2, window20_in2, window21_in2, window22_in2,
                                window00_in3, window01_in3, window02_in3, window10_in3, window11_in3, window12_in3, window20_in3, window21_in3, window22_in3,
                                window00_in4, window01_in4, window02_in4, window10_in4, window11_in4, window12_in4, window20_in4, window21_in4, window22_in4,
                                window00_in5, window01_in5, window02_in5, window10_in5, window11_in5, window12_in5, window20_in5, window21_in5, window22_in5,
                                window00_in6, window01_in6, window02_in6, window10_in6, window11_in6, window12_in6, window20_in6, window21_in6, window22_in6,
                                window00_in7, window01_in7, window02_in7, window10_in7, window11_in7, window12_in7, window20_in7, window21_in7, window22_in7;
                                
    wire [7:0]               CURRENT_OC_ITER;
    wire [1:0]               out_addr_start_n;
    wire [ADDR_OUT-1:0]      out_addr;

    assign act_mem_0_enb = ifm_rd_en; assign act_mem_1_enb = ifm_rd_en; assign act_mem_2_enb = ifm_rd_en; assign act_mem_3_enb = ifm_rd_en;
    assign act_mem_4_enb = ifm_rd_en; assign act_mem_5_enb = ifm_rd_en; assign act_mem_6_enb = ifm_rd_en; assign act_mem_7_enb = ifm_rd_en;
    
    assign act_mem_0_addrb = ifm_rd_addr; assign act_mem_1_addrb = ifm_rd_addr; assign act_mem_2_addrb = ifm_rd_addr; assign act_mem_3_addrb = ifm_rd_addr;
    assign act_mem_4_addrb = ifm_rd_addr; assign act_mem_5_addrb = ifm_rd_addr; assign act_mem_6_addrb = ifm_rd_addr; assign act_mem_7_addrb = ifm_rd_addr;

    assign weight_web   = 1'b0;
    assign weight_enb   = weight_rd_en;
    assign weight_addrb = weight_rd_addr;

    config_ctrlr config_ctrlr ( .clk(clk), .resetn(resetn),
                                .KH(KH),.KW(KW),.TOT_IC(IC_config), .IC_ITER(IC_ITER_IN), .IMG_H(IMG_H_config), .IMG_W(IMG_W_config), .OC(OC_config),
                                .start(config_ctrlr_start),
                                .done(config_ctrlr_done),
                                
                                // ifmap inputs & read
                                .ifm_rd_en(ifm_rd_en),
                                .ifm_rd_addr(ifm_rd_addr),
                                .ifm_in0(act_mem_0_doutb), .ifm_in1(act_mem_1_doutb), .ifm_in2(act_mem_2_doutb), .ifm_in3(act_mem_3_doutb), 
                                .ifm_in4(act_mem_4_doutb), .ifm_in5(act_mem_5_doutb), .ifm_in6(act_mem_6_doutb), .ifm_in7(act_mem_7_doutb),
                                
                                // Weights Read Interface
                                .weight_rd_en(weight_rd_en),
                                .weight_rd_addr(weight_rd_addr),
                                .weight_en_pe0(weight_en_0), .weight_en_pe1(weight_en_1), .weight_en_pe2(weight_en_2), .weight_en_pe3(weight_en_3), 
                                .weight_en_pe4(weight_en_4), .weight_en_pe5(weight_en_5), .weight_en_pe6(weight_en_6), .weight_en_pe7(weight_en_7),
                                
                                // Interface to 8 PEs (Processing Elements)
                                .window00_in0(window00_in0), .window01_in0(window01_in0), .window02_in0(window02_in0), .window10_in0(window10_in0), .window11_in0(window11_in0), .window12_in0(window12_in0), .window20_in0(window20_in0), .window21_in0(window21_in0), .window22_in0(window22_in0),
                                .window00_in1(window00_in1), .window01_in1(window01_in1), .window02_in1(window02_in1), .window10_in1(window10_in1), .window11_in1(window11_in1), .window12_in1(window12_in1), .window20_in1(window20_in1), .window21_in1(window21_in1), .window22_in1(window22_in1),
                                .window00_in2(window00_in2), .window01_in2(window01_in2), .window02_in2(window02_in2), .window10_in2(window10_in2), .window11_in2(window11_in2), .window12_in2(window12_in2), .window20_in2(window20_in2), .window21_in2(window21_in2), .window22_in2(window22_in2),
                                .window00_in3(window00_in3), .window01_in3(window01_in3), .window02_in3(window02_in3), .window10_in3(window10_in3), .window11_in3(window11_in3), .window12_in3(window12_in3), .window20_in3(window20_in3), .window21_in3(window21_in3), .window22_in3(window22_in3),
                                .window00_in4(window00_in4), .window01_in4(window01_in4), .window02_in4(window02_in4), .window10_in4(window10_in4), .window11_in4(window11_in4), .window12_in4(window12_in4), .window20_in4(window20_in4), .window21_in4(window21_in4), .window22_in4(window22_in4),
                                .window00_in5(window00_in5), .window01_in5(window01_in5), .window02_in5(window02_in5), .window10_in5(window10_in5), .window11_in5(window11_in5), .window12_in5(window12_in5), .window20_in5(window20_in5), .window21_in5(window21_in5), .window22_in5(window22_in5),
                                .window00_in6(window00_in6), .window01_in6(window01_in6), .window02_in6(window02_in6), .window10_in6(window10_in6), .window11_in6(window11_in6), .window12_in6(window12_in6), .window20_in6(window20_in6), .window21_in6(window21_in6), .window22_in6(window22_in6),
                                .window00_in7(window00_in7), .window01_in7(window01_in7), .window02_in7(window02_in7), .window10_in7(window10_in7), .window11_in7(window11_in7), .window12_in7(window12_in7), .window20_in7(window20_in7), .window21_in7(window21_in7), .window22_in7(window22_in7),
                                
                                // Output PSUM Address or Output Pixel Address
                                .out_addr_start_n(out_addr_start_n),
                                .CURRENT_OC_ITER(CURRENT_OC_ITER),
                                .out_addr(out_addr)
                                );
    // ------------------------------------------------------------------------
    // Start Computation depending on Configuration Data
    // ------------------------------------------------------------------------
    wire signed [INPUT_BW-1:0]  window00_pe0, window01_pe0, window02_pe0, window10_pe0, window11_pe0, window12_pe0, window20_pe0, window21_pe0, window22_pe0,
                                window00_pe1, window01_pe1, window02_pe1, window10_pe1, window11_pe1, window12_pe1, window20_pe1, window21_pe1, window22_pe1,
                                window00_pe2, window01_pe2, window02_pe2, window10_pe2, window11_pe2, window12_pe2, window20_pe2, window21_pe2, window22_pe2,
                                window00_pe3, window01_pe3, window02_pe3, window10_pe3, window11_pe3, window12_pe3, window20_pe3, window21_pe3, window22_pe3,
                                window00_pe4, window01_pe4, window02_pe4, window10_pe4, window11_pe4, window12_pe4, window20_pe4, window21_pe4, window22_pe4,
                                window00_pe5, window01_pe5, window02_pe5, window10_pe5, window11_pe5, window12_pe5, window20_pe5, window21_pe5, window22_pe5,
                                window00_pe6, window01_pe6, window02_pe6, window10_pe6, window11_pe6, window12_pe6, window20_pe6, window21_pe6, window22_pe6,
                                window00_pe7, window01_pe7, window02_pe7, window10_pe7, window11_pe7, window12_pe7, window20_pe7, window21_pe7, window22_pe7;

//    wire signed [INPUT_BW-1:0]  weight_doutb;
    
    assign window00_pe0 = window00_in0; assign window01_pe0 = window01_in0; assign window02_pe0 = window02_in0; 
    assign window00_pe1 = window00_in1; assign window01_pe1 = window01_in1; assign window02_pe1 = window02_in1; 
    assign window00_pe2 = window00_in2; assign window01_pe2 = window01_in2; assign window02_pe2 = window02_in2; 
    assign window00_pe3 = window00_in3; assign window01_pe3 = window01_in3; assign window02_pe3 = window02_in3; 
    assign window00_pe4 = window00_in4; assign window01_pe4 = window01_in4; assign window02_pe4 = window02_in4; 
    assign window00_pe5 = window00_in5; assign window01_pe5 = window01_in5; assign window02_pe5 = window02_in5; 
    assign window00_pe6 = window00_in6; assign window01_pe6 = window01_in6; assign window02_pe6 = window02_in6; 
    assign window00_pe7 = window00_in7; assign window01_pe7 = window01_in7; assign window02_pe7 = window02_in7; 
    
    assign window10_pe0 = window10_in0; assign window11_pe0 = window11_in0; assign window12_pe0 = window12_in0;
    assign window10_pe1 = window10_in1; assign window11_pe1 = window11_in1; assign window12_pe1 = window12_in1;
    assign window10_pe2 = window10_in2; assign window11_pe2 = window11_in2; assign window12_pe2 = window12_in2;
    assign window10_pe3 = window10_in3; assign window11_pe3 = window11_in3; assign window12_pe3 = window12_in3;
    assign window10_pe4 = window10_in4; assign window11_pe4 = window11_in4; assign window12_pe4 = window12_in4;
    assign window10_pe5 = window10_in5; assign window11_pe5 = window11_in5; assign window12_pe5 = window12_in5;
    assign window10_pe6 = window10_in6; assign window11_pe6 = window11_in6; assign window12_pe6 = window12_in6;
    assign window10_pe7 = window10_in7; assign window11_pe7 = window11_in7; assign window12_pe7 = window12_in7;
    
    assign window20_pe0 = window20_in0; assign window21_pe0 = window21_in0; assign window22_pe0 = window22_in0;
    assign window20_pe1 = window20_in1; assign window21_pe1 = window21_in1; assign window22_pe1 = window22_in1;
    assign window20_pe2 = window20_in2; assign window21_pe2 = window21_in2; assign window22_pe2 = window22_in2;
    assign window20_pe3 = window20_in3; assign window21_pe3 = window21_in3; assign window22_pe3 = window22_in3;
    assign window20_pe4 = window20_in4; assign window21_pe4 = window21_in4; assign window22_pe4 = window22_in4;
    assign window20_pe5 = window20_in5; assign window21_pe5 = window21_in5; assign window22_pe5 = window22_in5;
    assign window20_pe6 = window20_in6; assign window21_pe6 = window21_in6; assign window22_pe6 = window22_in6;
    assign window20_pe7 = window20_in7; assign window21_pe7 = window21_in7; assign window22_pe7 = window22_in7;
    // ------------------------------------------------------------------------
    // ACCUMULATION of 8 input channel PSUM in PSUM_MEM
    // ------------------------------------------------------------------------
    wire [INPUT_BW-1:0] out_px0,out_px1, out_px2, out_px3, out_px4, out_px5, out_px6, out_px7;
    
    wire signed [INPUT_BW-1:0] s0;
    wire signed [INPUT_BW-1:0] s1;
    wire signed [INPUT_BW-1:0] s2;
    wire signed [INPUT_BW-1:0] s3;
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    reg signed [INPUT_BW-1:0] s0_delay, s1_delay, s2_delay, s3_delay;

   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            s0_delay <= 0; s1_delay <= 0; s2_delay <= 0; s3_delay <= 0;
        end
        else begin
            s0_delay <= s0; s1_delay <= s1; 
            s2_delay <= s2; s3_delay <= s3;
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////
    
    wire signed [INPUT_BW-1:0] t0;
    wire signed [INPUT_BW-1:0] t1;
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    reg signed [INPUT_BW-1:0] t0_delay, t1_delay;

   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            t0_delay <= 0; t1_delay <= 0;
        end
        else begin
            t0_delay <= t0; t1_delay <= t1; 
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////
    
    wire signed [INPUT_BW-1:0] psum_8channel;
    
    // --- Binary Adder Tree ---
    float32_add float_add_s0 (
        .clk(clk), .resetn(resetn),
        .out_float(s0),
        .inA_float(out_px0),
        .inB_float(out_px1)
    );
    float32_add float_add_s1 (
        .clk(clk), .resetn(resetn),
        .out_float(s1),
        .inA_float(out_px2),
        .inB_float(out_px3)
    );
    float32_add float_add_s2 (
        .clk(clk), .resetn(resetn),
        .out_float(s2),
        .inA_float(out_px4),
        .inB_float(out_px5)
    );
    float32_add float_add_s3 (
        .clk(clk), .resetn(resetn),
        .out_float(s3),
        .inA_float(out_px6),
        .inB_float(out_px7)
    );
      
    float32_add float_add_t0 (
        .clk(clk), .resetn(resetn),
        .out_float(t0),
        .inA_float(s0_delay),
        .inB_float(s1_delay)
    );
    float32_add float_add_t1 (
        .clk(clk), .resetn(resetn),
        .out_float(t1),
        .inA_float(s2_delay),
        .inB_float(s3_delay)
    );
    float32_add float_add_u0 (
        .clk(clk), .resetn(resetn),
        .out_float(psum_8channel),
        .inA_float(t0_delay),
        .inB_float(t1_delay)
    );
    
    wire psum_mem_wr_en; 
    wire psum_mem_wr_we;
    wire [ADDR_OUT-1:0]        psum_mem_wr_addr;
    wire signed [INPUT_BW-1:0] psum_mem_wr_data;
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    reg [ADDR_OUT-1:0]        psum_mem_wr_addr_delay,   psum_mem_wr_addr_delay2, psum_mem_wr_addr_delay3;
    reg [ADDR_OUT-1:0]        psum_mem_wr_addr_delay4,  psum_mem_wr_addr_delay5, psum_mem_wr_addr_delay6, psum_mem_wr_addr_delay7;

   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            psum_mem_wr_addr_delay <= 0;    psum_mem_wr_addr_delay2 <= 0; psum_mem_wr_addr_delay3 <= 0;
            psum_mem_wr_addr_delay4 <= 0;   psum_mem_wr_addr_delay5 <= 0; psum_mem_wr_addr_delay6 <= 0;
            psum_mem_wr_addr_delay7 <= 0;
        end
        else begin
            psum_mem_wr_addr_delay <= psum_mem_wr_addr;         psum_mem_wr_addr_delay2 <= psum_mem_wr_addr_delay;  psum_mem_wr_addr_delay3 <= psum_mem_wr_addr_delay2;
            psum_mem_wr_addr_delay4 <= psum_mem_wr_addr_delay3; psum_mem_wr_addr_delay5 <= psum_mem_wr_addr_delay4; psum_mem_wr_addr_delay6 <= psum_mem_wr_addr_delay5;
            psum_mem_wr_addr_delay7 <= psum_mem_wr_addr_delay6;
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////
    
    assign psum_mem_wr_en   = (out_addr_start_n == 0);
    assign psum_mem_wr_we   = (out_addr_start_n == 0);
    assign psum_mem_wr_addr = out_addr;
    assign psum_mem_wr_data = psum_8channel;
    
    //////////////////////////////// FOR WRITE ENABLE CONSTRAINTS(= TIMING CONSTRAINTS) /////////////////////////////////    
    reg psum_mem_wr_en_delay,   psum_mem_wr_en_delay2,  psum_mem_wr_en_delay3,  psum_mem_wr_en_delay4,  psum_mem_wr_en_delay5,  psum_mem_wr_en_delay6,  psum_mem_wr_en_delay7;
    reg psum_mem_wr_en_delay8,  psum_mem_wr_en_delay9,  psum_mem_wr_en_delay10, psum_mem_wr_en_delay11, psum_mem_wr_en_delay12, psum_mem_wr_en_delay13, psum_mem_wr_en_delay14;
    reg psum_mem_wr_en_delay15, psum_mem_wr_en_delay16, psum_mem_wr_en_delay17, psum_mem_wr_en_delay18, psum_mem_wr_en_delay19;
    reg psum_mem_wr_en_delay20, psum_mem_wr_en_delay21, psum_mem_wr_en_delay22, psum_mem_wr_en_delay23;
    reg psum_mem_wr_en_delay24, psum_mem_wr_en_delay25, psum_mem_wr_en_delay26, psum_mem_wr_en_delay27;
    reg psum_mem_wr_en_delay28, psum_mem_wr_en_delay29, psum_mem_wr_en_delay30, psum_mem_wr_en_delay31;
    reg psum_mem_wr_en_delay32, psum_mem_wr_en_delay33, psum_mem_wr_en_delay34, psum_mem_wr_en_delay35;
    reg psum_mem_wr_en_delay36;
    
    reg psum_mem_wr_we_delay,   psum_mem_wr_we_delay2,  psum_mem_wr_we_delay3,  psum_mem_wr_we_delay4,  psum_mem_wr_we_delay5,  psum_mem_wr_we_delay6,  psum_mem_wr_we_delay7;
    reg psum_mem_wr_we_delay8,  psum_mem_wr_we_delay9,  psum_mem_wr_we_delay10, psum_mem_wr_we_delay11, psum_mem_wr_we_delay12, psum_mem_wr_we_delay13, psum_mem_wr_we_delay14;
    reg psum_mem_wr_we_delay15, psum_mem_wr_we_delay16, psum_mem_wr_we_delay17, psum_mem_wr_we_delay18, psum_mem_wr_we_delay19;
    reg psum_mem_wr_we_delay20, psum_mem_wr_we_delay21, psum_mem_wr_we_delay22, psum_mem_wr_we_delay23;
    reg psum_mem_wr_we_delay24, psum_mem_wr_we_delay25, psum_mem_wr_we_delay26, psum_mem_wr_we_delay27;
    reg psum_mem_wr_we_delay28, psum_mem_wr_we_delay29, psum_mem_wr_we_delay30, psum_mem_wr_we_delay31;
    reg psum_mem_wr_we_delay32, psum_mem_wr_we_delay33, psum_mem_wr_we_delay34, psum_mem_wr_we_delay35;
    reg psum_mem_wr_we_delay36;
    
    reg [ADDR_OUT-1:0]        psum_mem_wr_addr_delay8;
    reg [ADDR_OUT-1:0]        psum_mem_wr_addr_delay9,  psum_mem_wr_addr_delay10, psum_mem_wr_addr_delay11, psum_mem_wr_addr_delay12;
    reg [ADDR_OUT-1:0]        psum_mem_wr_addr_delay13, psum_mem_wr_addr_delay14, psum_mem_wr_addr_delay15, psum_mem_wr_addr_delay16;
    reg [ADDR_OUT-1:0]        psum_mem_wr_addr_delay17, psum_mem_wr_addr_delay18, psum_mem_wr_addr_delay19, psum_mem_wr_addr_delay20;
    reg [ADDR_OUT-1:0]        psum_mem_wr_addr_delay21, psum_mem_wr_addr_delay22, psum_mem_wr_addr_delay23, psum_mem_wr_addr_delay24;
    reg [ADDR_OUT-1:0]        psum_mem_wr_addr_delay25, psum_mem_wr_addr_delay26, psum_mem_wr_addr_delay27, psum_mem_wr_addr_delay28;
    reg [ADDR_OUT-1:0]        psum_mem_wr_addr_delay29, psum_mem_wr_addr_delay30, psum_mem_wr_addr_delay31, psum_mem_wr_addr_delay32;
    reg [ADDR_OUT-1:0]        psum_mem_wr_addr_delay33, psum_mem_wr_addr_delay34, psum_mem_wr_addr_delay35, psum_mem_wr_addr_delay36;
    reg [ADDR_OUT-1:0]        psum_mem_wr_addr_delay37;
    
    reg signed [INPUT_BW-1:0] psum_mem_wr_data_delay;
    
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            config_ctrlr_done_delay    <= 0;  config_ctrlr_done_delay2    <= 0; 
            config_ctrlr_done_delay3   <= 0;  config_ctrlr_done_delay4    <= 0;
            config_ctrlr_done_delay5   <= 0;  config_ctrlr_done_delay6    <= 0; 
            config_ctrlr_done_delay7   <= 0;  config_ctrlr_done_delay8    <= 0;
            config_ctrlr_done_delay9   <= 0;  config_ctrlr_done_delay10   <= 0; 
            config_ctrlr_done_delay11  <= 0;  config_ctrlr_done_delay12   <= 0;
            config_ctrlr_done_delay13  <= 0;  config_ctrlr_done_delay14   <= 0; 
            config_ctrlr_done_delay15  <= 0;  config_ctrlr_done_delay16   <= 0;
            config_ctrlr_done_delay17  <= 0;  config_ctrlr_done_delay18   <= 0; 
            config_ctrlr_done_delay19  <= 0;  config_ctrlr_done_delay20   <= 0;
            config_ctrlr_done_delay21  <= 0;  config_ctrlr_done_delay22   <= 0;
            config_ctrlr_done_delay23  <= 0;  config_ctrlr_done_delay24   <= 0; 
            config_ctrlr_done_delay25  <= 0;  config_ctrlr_done_delay26   <= 0;
            config_ctrlr_done_delay27  <= 0;  config_ctrlr_done_delay28   <= 0; 
            config_ctrlr_done_delay29  <= 0;  config_ctrlr_done_delay30   <= 0;
            config_ctrlr_done_delay31  <= 0;  config_ctrlr_done_delay32   <= 0;
            config_ctrlr_done_delay33  <= 0;  config_ctrlr_done_delay34   <= 0; 
            config_ctrlr_done_delay35  <= 0;
            config_ctrlr_done_delay36  <= 0;
        
            psum_mem_wr_en_delay    <= 0;  psum_mem_wr_en_delay2    <= 0; 
            psum_mem_wr_en_delay3   <= 0;  psum_mem_wr_en_delay4    <= 0;
            psum_mem_wr_en_delay5   <= 0;  psum_mem_wr_en_delay6    <= 0; 
            psum_mem_wr_en_delay7   <= 0;  psum_mem_wr_en_delay8    <= 0;
            psum_mem_wr_en_delay9   <= 0;  psum_mem_wr_en_delay10   <= 0; 
            psum_mem_wr_en_delay11  <= 0;  psum_mem_wr_en_delay12   <= 0;
            psum_mem_wr_en_delay13  <= 0;  psum_mem_wr_en_delay14   <= 0; 
            psum_mem_wr_en_delay15  <= 0;  psum_mem_wr_en_delay16   <= 0;
            psum_mem_wr_en_delay17  <= 0;  psum_mem_wr_en_delay18   <= 0; 
            psum_mem_wr_en_delay19  <= 0;  psum_mem_wr_en_delay20   <= 0;
            psum_mem_wr_en_delay21  <= 0;  psum_mem_wr_en_delay22   <= 0;
            psum_mem_wr_en_delay23  <= 0;  psum_mem_wr_en_delay24   <= 0; 
            psum_mem_wr_en_delay25  <= 0;  psum_mem_wr_en_delay26   <= 0;
            psum_mem_wr_en_delay27  <= 0;  psum_mem_wr_en_delay28   <= 0; 
            psum_mem_wr_en_delay29  <= 0;  psum_mem_wr_en_delay30   <= 0;
            psum_mem_wr_en_delay31  <= 0;  psum_mem_wr_en_delay32   <= 0;
            psum_mem_wr_en_delay33  <= 0;  psum_mem_wr_en_delay34   <= 0; 
            psum_mem_wr_en_delay35  <= 0;
            psum_mem_wr_en_delay36  <= 0;
            
            psum_mem_wr_we_delay    <= 0;  psum_mem_wr_we_delay2    <= 0; 
            psum_mem_wr_we_delay3   <= 0;  psum_mem_wr_we_delay4    <= 0;
            psum_mem_wr_we_delay5   <= 0;  psum_mem_wr_we_delay6    <= 0; 
            psum_mem_wr_we_delay7   <= 0;  psum_mem_wr_we_delay8    <= 0;
            psum_mem_wr_we_delay9   <= 0;  psum_mem_wr_we_delay10   <= 0; 
            psum_mem_wr_we_delay11  <= 0;  psum_mem_wr_we_delay12   <= 0;
            psum_mem_wr_we_delay13  <= 0;  psum_mem_wr_we_delay14   <= 0; 
            psum_mem_wr_we_delay15  <= 0;  psum_mem_wr_we_delay16   <= 0;
            psum_mem_wr_we_delay17  <= 0;  psum_mem_wr_we_delay18   <= 0; 
            psum_mem_wr_we_delay19  <= 0;  psum_mem_wr_we_delay20   <= 0;
            psum_mem_wr_we_delay21  <= 0;  psum_mem_wr_we_delay22   <= 0;
            psum_mem_wr_we_delay23  <= 0;  psum_mem_wr_we_delay24   <= 0; 
            psum_mem_wr_we_delay25  <= 0;  psum_mem_wr_we_delay26   <= 0;
            psum_mem_wr_we_delay27  <= 0;  psum_mem_wr_we_delay28   <= 0; 
            psum_mem_wr_we_delay29  <= 0;  psum_mem_wr_we_delay30   <= 0;
            psum_mem_wr_we_delay31  <= 0;  psum_mem_wr_we_delay32   <= 0;
            psum_mem_wr_we_delay33  <= 0;  psum_mem_wr_we_delay34   <= 0; 
            psum_mem_wr_we_delay35  <= 0;
            psum_mem_wr_we_delay36  <= 0;
            
            psum_mem_wr_addr_delay8 <= 0;
            psum_mem_wr_addr_delay9  <= 0; psum_mem_wr_addr_delay10 <= 0; psum_mem_wr_addr_delay11 <= 0; psum_mem_wr_addr_delay12 <= 0;
            psum_mem_wr_addr_delay13 <= 0; psum_mem_wr_addr_delay14 <= 0; psum_mem_wr_addr_delay15 <= 0; psum_mem_wr_addr_delay16 <= 0;
            psum_mem_wr_addr_delay17 <= 0; psum_mem_wr_addr_delay18 <= 0; psum_mem_wr_addr_delay19 <= 0; psum_mem_wr_addr_delay20 <= 0;
            psum_mem_wr_addr_delay21 <= 0; psum_mem_wr_addr_delay22 <= 0; psum_mem_wr_addr_delay23 <= 0; psum_mem_wr_addr_delay24 <= 0;
            psum_mem_wr_addr_delay25 <= 0; psum_mem_wr_addr_delay26 <= 0; psum_mem_wr_addr_delay27 <= 0; psum_mem_wr_addr_delay28 <= 0;
            psum_mem_wr_addr_delay29 <= 0; psum_mem_wr_addr_delay30 <= 0; psum_mem_wr_addr_delay31 <= 0; psum_mem_wr_addr_delay32 <= 0;
            psum_mem_wr_addr_delay33 <= 0; psum_mem_wr_addr_delay34 <= 0; psum_mem_wr_addr_delay35 <= 0; psum_mem_wr_addr_delay36 <= 0;
            psum_mem_wr_addr_delay37 <= 0;
            
            psum_mem_wr_data_delay  <= 0;
        end
        else begin
            config_ctrlr_done_delay    <= config_ctrlr_done;          config_ctrlr_done_delay2    <= config_ctrlr_done_delay; 
            config_ctrlr_done_delay3   <= config_ctrlr_done_delay2;   config_ctrlr_done_delay4    <= config_ctrlr_done_delay3;
            config_ctrlr_done_delay5   <= config_ctrlr_done_delay4;   config_ctrlr_done_delay6    <= config_ctrlr_done_delay5; 
            config_ctrlr_done_delay7   <= config_ctrlr_done_delay6;   config_ctrlr_done_delay8    <= config_ctrlr_done_delay7;
            config_ctrlr_done_delay9   <= config_ctrlr_done_delay8;   config_ctrlr_done_delay10   <= config_ctrlr_done_delay9;
            config_ctrlr_done_delay11  <= config_ctrlr_done_delay10;  config_ctrlr_done_delay12   <= config_ctrlr_done_delay11;
            config_ctrlr_done_delay13  <= config_ctrlr_done_delay12;  config_ctrlr_done_delay14   <= config_ctrlr_done_delay13;
            config_ctrlr_done_delay15  <= config_ctrlr_done_delay14;  config_ctrlr_done_delay16   <= config_ctrlr_done_delay15;
            config_ctrlr_done_delay17  <= config_ctrlr_done_delay16;  config_ctrlr_done_delay18   <= config_ctrlr_done_delay17;
            config_ctrlr_done_delay19  <= config_ctrlr_done_delay18;  config_ctrlr_done_delay20   <= config_ctrlr_done_delay19;
            config_ctrlr_done_delay21  <= config_ctrlr_done_delay20;  config_ctrlr_done_delay22   <= config_ctrlr_done_delay21;
            config_ctrlr_done_delay23  <= config_ctrlr_done_delay22;  config_ctrlr_done_delay24   <= config_ctrlr_done_delay23;
            config_ctrlr_done_delay25  <= config_ctrlr_done_delay24;  config_ctrlr_done_delay26   <= config_ctrlr_done_delay25;
            config_ctrlr_done_delay27  <= config_ctrlr_done_delay26;  config_ctrlr_done_delay28   <= config_ctrlr_done_delay27;
            config_ctrlr_done_delay29  <= config_ctrlr_done_delay28;  config_ctrlr_done_delay30   <= config_ctrlr_done_delay29;
            config_ctrlr_done_delay31  <= config_ctrlr_done_delay30;  config_ctrlr_done_delay32   <= config_ctrlr_done_delay31;
            config_ctrlr_done_delay33  <= config_ctrlr_done_delay32;  config_ctrlr_done_delay34   <= config_ctrlr_done_delay33;
            config_ctrlr_done_delay35  <= config_ctrlr_done_delay34;
            config_ctrlr_done_delay36  <= config_ctrlr_done_delay35;
        
            psum_mem_wr_en_delay    <= psum_mem_wr_en;          psum_mem_wr_en_delay2    <= psum_mem_wr_en_delay; 
            psum_mem_wr_en_delay3   <= psum_mem_wr_en_delay2;   psum_mem_wr_en_delay4    <= psum_mem_wr_en_delay3;
            psum_mem_wr_en_delay5   <= psum_mem_wr_en_delay4;   psum_mem_wr_en_delay6    <= psum_mem_wr_en_delay5; 
            psum_mem_wr_en_delay7   <= psum_mem_wr_en_delay6;   psum_mem_wr_en_delay8    <= psum_mem_wr_en_delay7;
            psum_mem_wr_en_delay9   <= psum_mem_wr_en_delay8;   psum_mem_wr_en_delay10   <= psum_mem_wr_en_delay9;
            psum_mem_wr_en_delay11  <= psum_mem_wr_en_delay10;  psum_mem_wr_en_delay12   <= psum_mem_wr_en_delay11;
            psum_mem_wr_en_delay13  <= psum_mem_wr_en_delay12;  psum_mem_wr_en_delay14   <= psum_mem_wr_en_delay13;
            psum_mem_wr_en_delay15  <= psum_mem_wr_en_delay14;  psum_mem_wr_en_delay16   <= psum_mem_wr_en_delay15;
            psum_mem_wr_en_delay17  <= psum_mem_wr_en_delay16;  psum_mem_wr_en_delay18   <= psum_mem_wr_en_delay17;
            psum_mem_wr_en_delay19  <= psum_mem_wr_en_delay18;  psum_mem_wr_en_delay20   <= psum_mem_wr_en_delay19;
            psum_mem_wr_en_delay21  <= psum_mem_wr_en_delay20;  psum_mem_wr_en_delay22   <= psum_mem_wr_en_delay21;
            psum_mem_wr_en_delay23  <= psum_mem_wr_en_delay22;  psum_mem_wr_en_delay24   <= psum_mem_wr_en_delay23;
            psum_mem_wr_en_delay25  <= psum_mem_wr_en_delay24;  psum_mem_wr_en_delay26   <= psum_mem_wr_en_delay25;
            psum_mem_wr_en_delay27  <= psum_mem_wr_en_delay26;  psum_mem_wr_en_delay28   <= psum_mem_wr_en_delay27;
            psum_mem_wr_en_delay29  <= psum_mem_wr_en_delay28;  psum_mem_wr_en_delay30   <= psum_mem_wr_en_delay29;
            psum_mem_wr_en_delay31  <= psum_mem_wr_en_delay30;  psum_mem_wr_en_delay32   <= psum_mem_wr_en_delay31;
            psum_mem_wr_en_delay33  <= psum_mem_wr_en_delay32;  psum_mem_wr_en_delay34   <= psum_mem_wr_en_delay33;
            psum_mem_wr_en_delay35  <= psum_mem_wr_en_delay34;
            psum_mem_wr_en_delay36  <= psum_mem_wr_en_delay35;
            
            psum_mem_wr_we_delay    <= psum_mem_wr_we;          psum_mem_wr_we_delay2    <= psum_mem_wr_we_delay; 
            psum_mem_wr_we_delay3   <= psum_mem_wr_we_delay2;   psum_mem_wr_we_delay4    <= psum_mem_wr_we_delay3;
            psum_mem_wr_we_delay5   <= psum_mem_wr_we_delay4;   psum_mem_wr_we_delay6    <= psum_mem_wr_we_delay5; 
            psum_mem_wr_we_delay7   <= psum_mem_wr_we_delay6;   psum_mem_wr_we_delay8    <= psum_mem_wr_we_delay7;
            psum_mem_wr_we_delay9   <= psum_mem_wr_we_delay8;   psum_mem_wr_we_delay10   <= psum_mem_wr_we_delay9;
            psum_mem_wr_we_delay11  <= psum_mem_wr_we_delay10;  psum_mem_wr_we_delay12   <= psum_mem_wr_we_delay11;
            psum_mem_wr_we_delay13  <= psum_mem_wr_we_delay12;  psum_mem_wr_we_delay14   <= psum_mem_wr_we_delay13;
            psum_mem_wr_we_delay15  <= psum_mem_wr_we_delay14;  psum_mem_wr_we_delay16   <= psum_mem_wr_we_delay15;
            psum_mem_wr_we_delay17  <= psum_mem_wr_we_delay16;  psum_mem_wr_we_delay18   <= psum_mem_wr_we_delay17;
            psum_mem_wr_we_delay19  <= psum_mem_wr_we_delay18;  psum_mem_wr_we_delay20   <= psum_mem_wr_we_delay19;
            psum_mem_wr_we_delay21  <= psum_mem_wr_we_delay20;  psum_mem_wr_we_delay22   <= psum_mem_wr_we_delay21;
            psum_mem_wr_we_delay23  <= psum_mem_wr_we_delay22;  psum_mem_wr_we_delay24   <= psum_mem_wr_we_delay23;
            psum_mem_wr_we_delay25  <= psum_mem_wr_we_delay24;  psum_mem_wr_we_delay26   <= psum_mem_wr_we_delay25;
            psum_mem_wr_we_delay27  <= psum_mem_wr_we_delay26;  psum_mem_wr_we_delay28   <= psum_mem_wr_we_delay27;
            psum_mem_wr_we_delay29  <= psum_mem_wr_we_delay28;  psum_mem_wr_we_delay30   <= psum_mem_wr_we_delay29;
            psum_mem_wr_we_delay31  <= psum_mem_wr_we_delay30;  psum_mem_wr_we_delay32   <= psum_mem_wr_we_delay31;
            psum_mem_wr_we_delay33  <= psum_mem_wr_we_delay32;  psum_mem_wr_we_delay34   <= psum_mem_wr_we_delay33;
            psum_mem_wr_we_delay35  <= psum_mem_wr_we_delay34;
            psum_mem_wr_we_delay36  <= psum_mem_wr_we_delay35;
            
            psum_mem_wr_addr_delay8  <= psum_mem_wr_addr_delay7;
            psum_mem_wr_addr_delay9  <= psum_mem_wr_addr_delay8;  psum_mem_wr_addr_delay10 <= psum_mem_wr_addr_delay9; 
            psum_mem_wr_addr_delay11 <= psum_mem_wr_addr_delay10; psum_mem_wr_addr_delay12 <= psum_mem_wr_addr_delay11;
            psum_mem_wr_addr_delay13 <= psum_mem_wr_addr_delay12; psum_mem_wr_addr_delay14 <= psum_mem_wr_addr_delay13;
            psum_mem_wr_addr_delay15 <= psum_mem_wr_addr_delay14; psum_mem_wr_addr_delay16 <= psum_mem_wr_addr_delay15;
            psum_mem_wr_addr_delay17 <= psum_mem_wr_addr_delay16; psum_mem_wr_addr_delay18 <= psum_mem_wr_addr_delay17;
            psum_mem_wr_addr_delay19 <= psum_mem_wr_addr_delay18; psum_mem_wr_addr_delay20 <= psum_mem_wr_addr_delay19;
            psum_mem_wr_addr_delay21 <= psum_mem_wr_addr_delay20; psum_mem_wr_addr_delay22 <= psum_mem_wr_addr_delay21;
            psum_mem_wr_addr_delay23 <= psum_mem_wr_addr_delay22; psum_mem_wr_addr_delay24 <= psum_mem_wr_addr_delay23;
            psum_mem_wr_addr_delay25 <= psum_mem_wr_addr_delay24; psum_mem_wr_addr_delay26 <= psum_mem_wr_addr_delay25;
            psum_mem_wr_addr_delay27 <= psum_mem_wr_addr_delay26; psum_mem_wr_addr_delay28 <= psum_mem_wr_addr_delay27;
            psum_mem_wr_addr_delay29 <= psum_mem_wr_addr_delay28; psum_mem_wr_addr_delay30 <= psum_mem_wr_addr_delay29;
            psum_mem_wr_addr_delay31 <= psum_mem_wr_addr_delay30; psum_mem_wr_addr_delay32 <= psum_mem_wr_addr_delay31;
            psum_mem_wr_addr_delay33 <= psum_mem_wr_addr_delay32; psum_mem_wr_addr_delay34 <= psum_mem_wr_addr_delay33;
            psum_mem_wr_addr_delay35 <= psum_mem_wr_addr_delay34; psum_mem_wr_addr_delay36 <= psum_mem_wr_addr_delay35;
            psum_mem_wr_addr_delay37 <= psum_mem_wr_addr_delay36;
            
            psum_mem_wr_data_delay  <= psum_mem_wr_data;
        
        end
    end
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    assign psum_mem_ena     = psum_mem_wr_en_delay36;     ///////////////////////////////////////////////////////////////////////////////////////////// 7
    assign psum_mem_wea     = psum_mem_wr_we_delay36;     ///////////////////////////////////////////////////////////////////////////////////////////// 7
    assign psum_mem_addra   = psum_mem_wr_addr_delay37;   ///////////////////////////////////////////////////////////////////////////////////////////// 8
    assign psum_mem_dina    = psum_mem_wr_data_delay;
    // ------------------------------------------------------------------------
    // ACC Controller (Control PSUM_MEM_ACC)
    // ------------------------------------------------------------------------
    // input PSUM_MEM Data & output read address, en signals
    wire                            psum_mem_rd_en;
    wire [ADDR_OUT-1:0]             psum_mem_rd_addr;
    wire signed [INPUT_BW-1:0]      psum_mem_rd_dout;
    
    // input PSUM_MEM_ACC Address to read
    wire                            psum_mem_acc_rd_en;
    wire [ADDR_OUT-1:0]             psum_mem_acc_rd_addr;
    wire signed [INPUT_BW-1:0]      psum_mem_acc_rd_dout;
    
    // output PSUM_MEM_ACC Address to write
    wire                            psum_mem_acc_wr_en;
    wire                            psum_mem_acc_wr_we;
    wire [ADDR_OUT-1:0]             psum_mem_acc_wr_addr;
    wire signed [INPUT_BW-1:0]      psum_mem_acc_wr_din;
    
    // after Quantization PSUM Data
    wire signed [OUTPUT_BW-1:0]     out_quantized;
    
    // output OUT_MEM Address to write
    wire                            out_mem_en;
    wire                            out_mem_we;
    wire [ADDR_OUT-1:0]             out_mem_addr;
    wire signed [OUTPUT_BW-1:0]     out_mem_din;
    wire signed [OUTPUT_BW-1:0]     out_mem_dout;
    
    assign psum_mem_enb     = psum_mem_rd_en;
    assign psum_mem_addrb   = psum_mem_rd_addr;
    assign psum_mem_rd_dout = psum_mem_doutb;
    
//    assign psum_mem_acc_ena     = psum_mem_acc_wr_en;
//    assign psum_mem_acc_wea     = psum_mem_acc_wr_we;
//    assign psum_mem_acc_addra   = psum_mem_acc_wr_addr;
//    assign psum_mem_acc_dina    = psum_mem_acc_wr_din;
    
//    assign psum_mem_acc_enb     = psum_mem_acc_rd_en;
//    assign psum_mem_acc_addrb   = psum_mem_acc_rd_addr;
//    assign psum_mem_acc_rd_dout = psum_mem_acc_doutb;
    
    // ------------------------------------------------------------------------
    // Signals to put in the out_mem
    // ------------------------------------------------------------------------ 
//    assign out_mem_ena          = out_mem_en;
//    assign out_mem_wea          = out_mem_we;
//    assign out_mem_addra        = out_mem_addr;
//    assign out_mem_dina         = out_mem_din;
    assign out_mem_ena          = psum_mem_acc_wr_en;
    assign out_mem_wea          = psum_mem_acc_wr_we;
    assign out_mem_addra        = psum_mem_acc_wr_addr;
    assign out_mem_dina         = psum_mem_acc_wr_din;
    assign psum_mem_acc_rd_dout = out_mem_douta;
     
    acc_ctrlr acc_ctrlr ( 
    .clk(clk), .resetn(resetn),
    // Start & Done Signals
    .start(acc_ctrlr_start), .done(acc_ctrlr_done),
    // Configurable Data
    .IC_ITER(IC_ITER_IN), .TOT_IC(IC_acc), .TOT_OC(OC_acc), .IMG_H(IMG_H_acc), .IMG_W(IMG_W_acc),
    // input PSUM_MEM Data & output read address, en signals
    .psum_mem_rd_en(psum_mem_rd_en), .psum_mem_rd_addr(psum_mem_rd_addr), .psum_mem_rd_dout(psum_mem_rd_dout),
    // input PSUM_MEM_ACC Address to read 
//    .psum_mem_acc_rd_en(psum_mem_acc_rd_en), .psum_mem_acc_rd_addr(psum_mem_acc_rd_addr), 
    .psum_mem_acc_rd_dout(psum_mem_acc_rd_dout),
    // output PSUM_MEM_ACC Address to write
    .psum_mem_acc_wr_en(psum_mem_acc_wr_en), .psum_mem_acc_wr_we(psum_mem_acc_wr_we), .psum_mem_acc_wr_addr(psum_mem_acc_wr_addr), .psum_mem_acc_wr_din(psum_mem_acc_wr_din)
//    // after Quantization PSUM Data
//    .out_quantized(out_quantized),
//    // output OUT_MEM Address to write 
//    .out_mem_en(out_mem_en), .out_mem_we(out_mem_we), .out_mem_addr(out_mem_addr), .out_mem_din(out_mem_din)
    );

    // ------------------------------------------------------------------------
    // 8 PE
    // ------------------------------------------------------------------------
    /***** 3x3 MAC computer instance *****/
    PE_CONV_3x3 CONV1_PE0 ( .clk(clk), .resetn(resetn),
                            .act00(window00_pe0),.act01(window01_pe0),.act02(window02_pe0),
                            .act10(window10_pe0),.act11(window11_pe0),.act12(window12_pe0),
                            .act20(window20_pe0),.act21(window21_pe0),.act22(window22_pe0),
                            .weight_en(weight_en_0), .weight_in(weight_doutb),
                            .out(out_px0));
    
    PE_CONV_3x3 CONV1_PE1 ( .clk(clk), .resetn(resetn),
                            .act00(window00_pe1),.act01(window01_pe1),.act02(window02_pe1),
                            .act10(window10_pe1),.act11(window11_pe1),.act12(window12_pe1),
                            .act20(window20_pe1),.act21(window21_pe1),.act22(window22_pe1),
                            .weight_en(weight_en_1), .weight_in(weight_doutb),
                            .out(out_px1));
    
    PE_CONV_3x3 CONV1_PE2 ( .clk(clk), .resetn(resetn),
                            .act00(window00_pe2),.act01(window01_pe2),.act02(window02_pe2),
                            .act10(window10_pe2),.act11(window11_pe2),.act12(window12_pe2),
                            .act20(window20_pe2),.act21(window21_pe2),.act22(window22_pe2),
                            .weight_en(weight_en_2), .weight_in(weight_doutb),
                            .out(out_px2));
                        
    PE_CONV_3x3 CONV1_PE3 ( .clk(clk), .resetn(resetn),
                            .act00(window00_pe3),.act01(window01_pe3),.act02(window02_pe3),
                            .act10(window10_pe3),.act11(window11_pe3),.act12(window12_pe3),
                            .act20(window20_pe3),.act21(window21_pe3),.act22(window22_pe3),
                            .weight_en(weight_en_3), .weight_in(weight_doutb),
                            .out(out_px3));                    
    PE_CONV_3x3 CONV1_PE4 ( .clk(clk), .resetn(resetn),
                            .act00(window00_pe4),.act01(window01_pe4),.act02(window02_pe4),
                            .act10(window10_pe4),.act11(window11_pe4),.act12(window12_pe4),
                            .act20(window20_pe4),.act21(window21_pe4),.act22(window22_pe4),
                            .weight_en(weight_en_4), .weight_in(weight_doutb),
                            .out(out_px4));
    PE_CONV_3x3 CONV1_PE5 ( .clk(clk), .resetn(resetn),
                            .act00(window00_pe5),.act01(window01_pe5),.act02(window02_pe5),
                            .act10(window10_pe5),.act11(window11_pe5),.act12(window12_pe5),
                            .act20(window20_pe5),.act21(window21_pe5),.act22(window22_pe5),
                            .weight_en(weight_en_5), .weight_in(weight_doutb),
                            .out(out_px5));
    PE_CONV_3x3 CONV1_PE6 ( .clk(clk), .resetn(resetn),
                            .act00(window00_pe6),.act01(window01_pe6),.act02(window02_pe6),
                            .act10(window10_pe6),.act11(window11_pe6),.act12(window12_pe6),
                            .act20(window20_pe6),.act21(window21_pe6),.act22(window22_pe6),
                            .weight_en(weight_en_6), .weight_in(weight_doutb),
                            .out(out_px6));                        
    PE_CONV_3x3 CONV1_PE7 ( .clk(clk), .resetn(resetn),
                            .act00(window00_pe7),.act01(window01_pe7),.act02(window02_pe7),
                            .act10(window10_pe7),.act11(window11_pe7),.act12(window12_pe7),
                            .act20(window20_pe7),.act21(window21_pe7),.act22(window22_pe7),
                            .weight_en(weight_en_7), .weight_in(weight_doutb),
                            .out(out_px7));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    INPUT_MEM INPUT_MEM (
      .clka (input_clka),       // input wire clka
      .ena  (input_ena),         // input wire ena
      .wea  (input_wea),         // input wire [1 : 0] wea
      .addra(input_addra),     // input wire [13 : 0] addra
      .dina (input_dina),       // input wire [7 : 0] dina
      .douta(input_douta),     // output wire [7 : 0] douta
      .clkb (clk),           // input wire clkb
      .enb  (input_enb),          // input wire enb
      .web  (input_web),         // input wire [1 : 0] web
      .addrb(input_addrb),      // input wire [13 : 0] addrb
      .dinb (input_dinb),        // input wire [7 : 0] dinb
      .doutb(input_doutb)       // output wire [7 : 0] doutb
    );
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    WEIGHT_MEM WEIGHT_MEM (
      .clka (weight_clka),
      .ena  (weight_ena),
      .wea  (weight_wea),
      .addra(weight_addra), 
      .dina (weight_dina),
      .douta(weight_douta), 
      .clkb (clk),
      .enb  (weight_enb),
      .web  (weight_web),         
      .addrb(weight_addrb),     
      .dinb (weight_dinb),        
      .doutb(weight_doutb)   
    );
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    OUT_MEM OUT_MEM (
      .clka (clk),
      .ena  (out_mem_ena),
      .wea  (out_mem_wea),
      .addra(out_mem_addra), 
      .dina (out_mem_dina),
      .douta(out_mem_douta), 
      .clkb (clk),
      .enb  (out_mem_enb),
      .web  (out_mem_web),         
      .addrb(out_mem_addrb),     
      .dinb (out_mem_dinb),        
      .doutb(out_mem_doutb)   
    ); 
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ACT_MEM ACT_MEM_0 (
      .clka (clk),
      .ena  (act_mem_0_ena),
      .wea  (act_mem_0_wea),
      .addra(act_mem_0_addra),
      .dina (act_mem_0_dina),
 
      .clkb (clk),
      .enb  (act_mem_0_enb),
      .addrb(act_mem_0_addrb),
      .doutb(act_mem_0_doutb)
    );
    ACT_MEM ACT_MEM_1 (
      .clka (clk),
      .ena  (act_mem_1_ena),
      .wea  (act_mem_1_wea),
      .addra(act_mem_1_addra),
      .dina (act_mem_1_dina),
 
      .clkb (clk),
      .enb  (act_mem_1_enb),
      .addrb(act_mem_1_addrb),
      .doutb(act_mem_1_doutb)
    );
    ACT_MEM ACT_MEM_2 (
      .clka (clk),
      .ena  (act_mem_2_ena),
      .wea  (act_mem_2_wea),
      .addra(act_mem_2_addra),
      .dina (act_mem_2_dina),
 
      .clkb (clk),
      .enb  (act_mem_2_enb),
      .addrb(act_mem_2_addrb),
      .doutb(act_mem_2_doutb)
    );
    ACT_MEM ACT_MEM_3 (
      .clka (clk),
      .ena  (act_mem_3_ena),
      .wea  (act_mem_3_wea),
      .addra(act_mem_3_addra),
      .dina (act_mem_3_dina),
 
      .clkb (clk),
      .enb  (act_mem_3_enb),
      .addrb(act_mem_3_addrb),
      .doutb(act_mem_3_doutb)
    );
    ACT_MEM ACT_MEM_4 (
      .clka (clk),
      .ena  (act_mem_4_ena),
      .wea  (act_mem_4_wea),
      .addra(act_mem_4_addra),
      .dina (act_mem_4_dina),
 
      .clkb (clk),
      .enb  (act_mem_4_enb),
      .addrb(act_mem_4_addrb),
      .doutb(act_mem_4_doutb)
    );
    ACT_MEM ACT_MEM_5 (
      .clka (clk),
      .ena  (act_mem_5_ena),
      .wea  (act_mem_5_wea),
      .addra(act_mem_5_addra),
      .dina (act_mem_5_dina),
 
      .clkb (clk),
      .enb  (act_mem_5_enb),
      .addrb(act_mem_5_addrb),
      .doutb(act_mem_5_doutb)
    );
    ACT_MEM ACT_MEM_6 (
      .clka (clk),
      .ena  (act_mem_6_ena),
      .wea  (act_mem_6_wea),
      .addra(act_mem_6_addra),
      .dina (act_mem_6_dina),
 
      .clkb (clk),
      .enb  (act_mem_6_enb),
      .addrb(act_mem_6_addrb),
      .doutb(act_mem_6_doutb)
    );
    ACT_MEM ACT_MEM_7 (
      .clka (clk),
      .ena  (act_mem_7_ena),
      .wea  (act_mem_7_wea),
      .addra(act_mem_7_addra),
      .dina (act_mem_7_dina),
 
      .clkb (clk),
      .enb  (act_mem_7_enb),
      .addrb(act_mem_7_addrb),
      .doutb(act_mem_7_doutb)
    );
    PSUM_MEM PSUM_MEM (
      .clka (clk),
      .ena  (psum_mem_ena),
      .wea  (psum_mem_wea),
      .addra(psum_mem_addra),
      .dina (psum_mem_dina),
 
      .clkb (clk),
      .enb  (psum_mem_enb),
      .addrb(psum_mem_addrb),
      .doutb(psum_mem_doutb)
    );
//    PSUM_MEM PSUM_MEM_ACC (
//      .clka (clk),
//      .ena  (psum_mem_acc_ena),
//      .wea  (psum_mem_acc_wea),
//      .addra(psum_mem_acc_addra),
//      .dina (psum_mem_acc_dina),
 
//      .clkb (clk),
//      .enb  (psum_mem_acc_enb),
//      .addrb(psum_mem_acc_addrb),
//      .doutb(psum_mem_acc_doutb)
//    );
 
endmodule