
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
    // image dims
    parameter IMG_W    = 28,
    parameter IMG_H    = 28,
    // conv dims
    parameter CONV1_IC = 1,
    parameter CONV1_OC = 8,
    parameter CONV2_IC = 8,
    parameter CONV2_OC = 16,
    // fc dims
    parameter FC_IN    = 16*12*12,
    parameter FC_OUT   = 10,
    // SRAM address widths (enough to cover depth)
    parameter ADDR_TOTALW = 15,    // 2^15 = 32768 > 24264
    parameter ADDR_IMG  = 10,      // ASSUME Batch Size = 2; 2^10 = 1024 > 28*28 = 784
    parameter ADDR_W1   = 7,       // 2^7 = 128 > 1*8*9 = 72
    parameter ADDR_W2   = 11,      // 2^11 = 2048 > 8*16*9 = 1152
    parameter ADDR_WFC  = 12,      // 2^12=4096>23040               -> parallelize to 10 diff mems(23040/10 = 2304)
    parameter ADDR_F1   = 10,      // conv1 output 8*26*26 = 5408   -> parallelize to 8 diff mems (5408/8 = 676)
    parameter ADDR_F2   = 12,      // conv2 output 16*24*24 = 9216  -> parallelize to 4 diff mems (9216/4 = 2304)
    parameter ADDR_F3   = 12,      // maxpool output 16*24*24/4 = 2304
    parameter ADDR_LOG  = 11       // output logits 2^4=16>10 -> For Batch Size = 100, need to be 10bits
)(
    input wire clk,
    input wire resetn,
    
    input  wire start,
    output wire done,
    
    // read weights from top
    output wire                 fc_weight_0_enb,   fc_weight_1_enb,   fc_weight_2_enb,   fc_weight_3_enb,   fc_weight_4_enb,
                                fc_weight_5_enb,   fc_weight_6_enb,   fc_weight_7_enb,   fc_weight_8_enb,   fc_weight_9_enb,
    output wire [ADDR_WFC-1:0]  fc_weight_0_addrb, fc_weight_1_addrb, fc_weight_2_addrb, fc_weight_3_addrb, fc_weight_4_addrb, 
                                 fc_weight_5_addrb, fc_weight_6_addrb, fc_weight_7_addrb, fc_weight_8_addrb, fc_weight_9_addrb,     
    input  wire signed [7:0]    fc_weight_0_doutb, fc_weight_1_doutb, fc_weight_2_doutb, fc_weight_3_doutb, fc_weight_4_doutb, 
                                 fc_weight_5_doutb, fc_weight_6_doutb, fc_weight_7_doutb, fc_weight_8_doutb, fc_weight_9_doutb,
                                 
    output wire                  conv1_weight_enb, conv2_weight_enb,
    output wire [ADDR_W2-1:0]    conv1_weight_addrb, conv2_weight_addrb,
    input wire signed [7:0]      conv1_weight_doutb, conv2_weight_doutb,


//    input wire [ADDR_WFC-1:0] rx_weight_addr,
//    input wire [7:0]          rx_weight_din,
//    input wire [3:0]          weight_transfer_state,
            
    input wire                  rx_input_en,
    input wire                  rx_input_we,
    input wire [ADDR_IMG-1 : 0] rx_input_addr, 
    input wire [7:0]            rx_input_din,

//    output wire done_led

    // ------------------------------------------------------------------------
    // Store Output Logits
    // ------------------------------------------------------------------------
    // port A (PL writes output logits in advance), port B (PS reads)
    output wire                  out_mem_ena,
    output wire                  out_mem_wea,
    output wire [ADDR_LOG-1:0]   out_mem_addra,
    output wire signed [7:0]     out_mem_dina
    
//    input  wire [7:0]             out_mem_dinb,
//    output wire                    out_mem_write_done
//    output wire                    out_done
    );
    
    reg done_reg;
    assign done = done_reg;

    // ------------------------------------------------------------------------
    // IF MAP & TOTAL WEIGHT MEM B-ports
    // ------------------------------------------------------------------------
//    wire [7:0]          input_douta;
//    wire [7:0]          tot_weight_douta;
    
//    wire                    input_clkb;
//    wire                    input_enb;
//    wire                    input_web;
//    wire [ADDR_IMG-1:0]     input_addrb;
//    wire signed [7:0]       input_dinb;
//    wire signed [7:0]       input_doutb;
    
//    wire                    tot_weight_clkb;
//    wire                    tot_weight_enb;
//    wire                    tot_weight_web;
//    wire [ADDR_TOTALW-1:0]  tot_weight_addrb;
//    wire signed [7:0]       tot_weight_dinb;
//    wire signed [7:0]       tot_weight_doutb;
    
    // ------------------------------------------------------------------------
    // Conv1 weights
    // ------------------------------------------------------------------------
////    wire                  conv1_weight_clka;
//    wire                  conv1_weight_ena;
//    wire                  conv1_weight_wea;
//    wire [ADDR_W1-1:0]    conv1_weight_addra;
//    wire signed [7:0]     conv1_weight_dina;
//    wire signed [7:0]     conv1_weight_douta;

////    wire                  conv1_weight_clkb;
//    wire                  conv1_weight_enb;
//    wire                  conv1_weight_web;
//    wire [ADDR_W1-1:0]    conv1_weight_addrb;
//    wire signed [7:0]     conv1_weight_dinb;
//    wire signed [7:0]     conv1_weight_doutb;
    
    // ------------------------------------------------------------------------
    // Conv2 weights
    // ------------------------------------------------------------------------
////    wire                  conv2_weight_clka;
//    wire                  conv2_weight_ena;
//    wire                  conv2_weight_wea;
//    wire [ADDR_W2-1:0]    conv2_weight_addra;
//    wire signed [7:0]     conv2_weight_dina;
//    wire signed [7:0]     conv2_weight_douta;

////    wire                  conv2_weight_clkb;
//    wire                  conv2_weight_enb;
//    wire                  conv2_weight_web;
//    wire [ADDR_W2-1:0]    conv2_weight_addrb;
//    wire signed [7:0]     conv2_weight_dinb;
//    wire signed [7:0]     conv2_weight_doutb;
    
    // ------------------------------------------------------------------------
    // FC weights (10 different memorys)
    // ------------------------------------------------------------------------
//    wire                  fc_weight_0_clka,  fc_weight_1_clka,  fc_weight_2_clka,  fc_weight_3_clka,  fc_weight_4_clka,  fc_weight_5_clka,  fc_weight_6_clka,  fc_weight_7_clka,  fc_weight_8_clka,  fc_weight_9_clka;
//    wire                  fc_weight_0_ena,   fc_weight_1_ena,   fc_weight_2_ena,   fc_weight_3_ena,   fc_weight_4_ena,   fc_weight_5_ena,   fc_weight_6_ena,   fc_weight_7_ena,   fc_weight_8_ena,   fc_weight_9_ena;
//    wire                  fc_weight_0_wea,   fc_weight_1_wea,   fc_weight_2_wea,   fc_weight_3_wea,   fc_weight_4_wea,   fc_weight_5_wea,   fc_weight_6_wea,   fc_weight_7_wea,   fc_weight_8_wea,   fc_weight_9_wea; 
//    wire [ADDR_WFC-1:0]   fc_weight_0_addra, fc_weight_1_addra, fc_weight_2_addra, fc_weight_3_addra, fc_weight_4_addra, fc_weight_5_addra, fc_weight_6_addra, fc_weight_7_addra, fc_weight_8_addra, fc_weight_9_addra; 
//    wire signed [7:0]     fc_weight_0_dina,  fc_weight_1_dina,  fc_weight_2_dina,  fc_weight_3_dina,  fc_weight_4_dina,  fc_weight_5_dina,  fc_weight_6_dina,  fc_weight_7_dina,  fc_weight_8_dina,   fc_weight_9_dina;
//    wire signed [7:0]     fc_weight_0_douta, fc_weight_1_douta, fc_weight_2_douta, fc_weight_3_douta, fc_weight_4_douta, fc_weight_5_douta, fc_weight_6_douta, fc_weight_7_douta, fc_weight_8_douta, fc_weight_9_douta;

//    wire                  fc_weight_0_clkb,  fc_weight_1_clkb,  fc_weight_2_clkb,  fc_weight_3_clkb,  fc_weight_4_clkb,  fc_weight_5_clkb,  fc_weight_6_clkb,  fc_weight_7_clkb,  fc_weight_8_clkb,  fc_weight_9_clkb;
//    wire                  fc_weight_0_enb,   fc_weight_1_enb,   fc_weight_2_enb,   fc_weight_3_enb,   fc_weight_4_enb,   fc_weight_5_enb,   fc_weight_6_enb,   fc_weight_7_enb,   fc_weight_8_enb,   fc_weight_9_enb;
//    wire                  fc_weight_0_web,   fc_weight_1_web,   fc_weight_2_web,   fc_weight_3_web,   fc_weight_4_web,   fc_weight_5_web,   fc_weight_6_web,   fc_weight_7_web,   fc_weight_8_web,   fc_weight_9_web;
//    wire [ADDR_WFC-1:0]   fc_weight_0_addrb, fc_weight_1_addrb, fc_weight_2_addrb, fc_weight_3_addrb, fc_weight_4_addrb, fc_weight_5_addrb, fc_weight_6_addrb, fc_weight_7_addrb, fc_weight_8_addrb, fc_weight_9_addrb; 
//    wire signed [7:0]     fc_weight_0_dinb,  fc_weight_1_dinb,  fc_weight_2_dinb,  fc_weight_3_dinb,  fc_weight_4_dinb,  fc_weight_5_dinb,  fc_weight_6_dinb,  fc_weight_7_dinb,  fc_weight_8_dinb,  fc_weight_9_dinb;
//    wire signed [7:0]     fc_weight_0_doutb, fc_weight_1_doutb, fc_weight_2_doutb, fc_weight_3_doutb, fc_weight_4_doutb, fc_weight_5_doutb, fc_weight_6_doutb, fc_weight_7_doutb, fc_weight_8_doutb, fc_weight_9_doutb;
    
    // ------------------------------------------------------------------------
    // Conv1 output activations + Conv2 output activations 
    // ------------------------------------------------------------------------
//    wire                  act_mem_0_clka,  act_mem_1_clka,  act_mem_2_clka,  act_mem_3_clka,  act_mem_4_clka,  act_mem_5_clka,  act_mem_6_clka,  act_mem_7_clka;
    wire                  act_mem_0_ena,   act_mem_1_ena,   act_mem_2_ena,   act_mem_3_ena,   act_mem_4_ena,   act_mem_5_ena,   act_mem_6_ena,   act_mem_7_ena;
    wire                  act_mem_0_wea,   act_mem_1_wea,   act_mem_2_wea,   act_mem_3_wea,   act_mem_4_wea,   act_mem_5_wea,   act_mem_6_wea,   act_mem_7_wea;
    wire [ADDR_F1-1:0]    act_mem_0_addra, act_mem_1_addra, act_mem_2_addra, act_mem_3_addra, act_mem_4_addra, act_mem_5_addra, act_mem_6_addra, act_mem_7_addra;
    wire signed [7:0]     act_mem_0_dina,  act_mem_1_dina,  act_mem_2_dina,  act_mem_3_dina,  act_mem_4_dina,  act_mem_5_dina,  act_mem_6_dina,  act_mem_7_dina;
    wire signed [7:0]     act_mem_0_douta, act_mem_1_douta, act_mem_2_douta, act_mem_3_douta, act_mem_4_douta, act_mem_5_douta, act_mem_6_douta, act_mem_7_douta;

//    wire                  act_mem_0_clkb,  act_mem_1_clkb,  act_mem_2_clkb,  act_mem_3_clkb,  act_mem_4_clkb,  act_mem_5_clkb,  act_mem_6_clkb,  act_mem_7_clkb;
    wire                  act_mem_0_enb,   act_mem_1_enb,   act_mem_2_enb,   act_mem_3_enb,   act_mem_4_enb,   act_mem_5_enb,   act_mem_6_enb,   act_mem_7_enb;
    wire                  act_mem_0_web,   act_mem_1_web,   act_mem_2_web,   act_mem_3_web,   act_mem_4_web,   act_mem_5_web,   act_mem_6_web,   act_mem_7_web;
    wire [ADDR_F1-1:0]    act_mem_0_addrb, act_mem_1_addrb, act_mem_2_addrb, act_mem_3_addrb, act_mem_4_addrb, act_mem_5_addrb, act_mem_6_addrb, act_mem_7_addrb;
    wire signed [7:0]     act_mem_0_dinb,  act_mem_1_dinb,  act_mem_2_dinb,  act_mem_3_dinb,  act_mem_4_dinb,  act_mem_5_dinb,  act_mem_6_dinb,  act_mem_7_dinb;
    wire signed [7:0]     act_mem_0_doutb, act_mem_1_doutb, act_mem_2_doutb, act_mem_3_doutb, act_mem_4_doutb, act_mem_5_doutb, act_mem_6_doutb, act_mem_7_doutb;
    
    // ------------------------------------------------------------------------
    // out_mem: final logits
    // ------------------------------------------------------------------------
//    wire                  out_mem_clka;
//    wire                  out_mem_ena;
//    wire                  out_mem_wea;
//    wire [ADDR_LOG-1:0]   out_mem_addra;
//    wire signed [7:0]     out_mem_dina;
//    wire signed [7:0]     out_mem_douta;

//    wire                  out_mem_clkb;
//    wire                  out_mem_enb;
//    wire                  out_mem_web;
//    wire [ADDR_LOG-1:0]   out_mem_addrb;
//    wire signed [7:0]     out_mem_dinb;
//    wire signed [7:0]     out_mem_doutb;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    localparam CONV_IDLE = 2'b00, CONV1 = 2'b01, CONV2 = 2'b10, CONV_DONE = 2'b11; 
    reg [1:0] conv_state, n_conv_state;
    reg conv1_start, conv2_start;
    wire conv1_done, conv2_done;
   /***** Defines when does the state change *****/ 
   always @(*) begin
        n_conv_state = conv_state;
        conv1_start = 0;
        conv2_start = 0;
        done_reg = 0;
        
        case (conv_state)
            CONV_IDLE: begin
                if (start) begin
                    n_conv_state = CONV1;
                    conv1_start = 1;
                end
            end
            CONV1: begin
                if (conv1_done) begin
                    n_conv_state = CONV2;
                    conv2_start = 1;
                end
            end
            CONV2: begin
                if (conv2_done) begin
                    n_conv_state = CONV_DONE;
                end
            end
            CONV_DONE: begin
                done_reg = 1;
                n_conv_state = CONV_IDLE;
            end
            default: begin
                n_conv_state = CONV_IDLE;
            end
        endcase
    end
    
    /***** state transition *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) conv_state <= CONV_IDLE;
        else conv_state <= n_conv_state;
    end
    
    //////////////////////////////////////////////////////////// PE computed output ////////////////////////////////////////////////////////////
    wire signed [20:0] out_px0;
    wire signed [20:0] out_px1;
    wire signed [20:0] out_px2;
    wire signed [20:0] out_px3;
    wire signed [20:0] out_px4;
    wire signed [20:0] out_px5;
    wire signed [20:0] out_px6;
    wire signed [20:0] out_px7;
    //////////////////////////////////////////////////////////// Wiring for act_mem_0 ////////////////////////////////////////////////////////////
    wire conv1_act_mem_0_enb,   conv2_act_mem_0_enb;
    wire conv1_act_mem_0_web,   conv2_act_mem_0_web;
    wire [ADDR_F1-1:0]  conv1_act_mem_0_addrb, conv2_act_mem_0_addrb;
    
    
    /////////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////////////////
//    reg act_mem_0_enb_reg;
//    reg act_mem_0_web_reg;
//    reg [ADDR_F1-1:0] act_mem_0_addrb_reg;
    
//    assign act_mem_0_enb   = act_mem_0_enb_reg;
//    assign act_mem_0_web   = act_mem_0_web_reg;
//    assign act_mem_0_addrb = act_mem_0_addrb_reg;
    
//    always @(posedge clk or negedge resetn) begin
//        if (!resetn) begin
//            act_mem_0_enb_reg   <= 0;
//            act_mem_0_web_reg   <= 0;
//            act_mem_0_addrb_reg <= 0;
//        end else begin
//            case (conv_state)
//                CONV1: begin
//                    act_mem_0_enb_reg   <= conv1_act_mem_0_enb;
//                    act_mem_0_web_reg   <= conv1_act_mem_0_web;
//                    act_mem_0_addrb_reg <= conv1_act_mem_0_addrb;
//                end
//                default: begin
//                    act_mem_0_enb_reg   <= conv2_act_mem_0_enb;
//                    act_mem_0_web_reg   <= conv2_act_mem_0_web;
//                    act_mem_0_addrb_reg <= conv2_act_mem_0_addrb;
//                end
//            endcase
//        end
//    end
    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    assign act_mem_0_enb    =  (conv_state == CONV1) ? conv1_act_mem_0_enb : conv2_act_mem_0_enb; 
    assign act_mem_0_web    =  (conv_state == CONV1) ? conv1_act_mem_0_web : conv2_act_mem_0_web; 
    assign act_mem_0_addrb  =  (conv_state == CONV1) ? conv1_act_mem_0_addrb : conv2_act_mem_0_addrb;
    
    //////////////////////////////////////////////////////////// CONV1 Layer Instantiating ////////////////////////////////////////////////////////////
    wire signed [7:0]   act_mem_din;
    wire [ADDR_F1-1:0]  act_mem_addr;
    wire                act_mem_en;
    wire                act_mem_we;
    /*********************************** CONV1 signal ****************************************/
    wire signed [7:0] conv1_w00, conv1_w01, conv1_w02, conv1_w10, conv1_w11, conv1_w12, conv1_w20, conv1_w21, conv1_w22;                // window
    
    // PE interface signals
//    wire signed [20:0]  conv1_out_px0, conv1_out_px1, conv1_out_px2,  conv1_out_px3,  conv1_out_px4,  conv1_out_px5,  conv1_out_px6,  conv1_out_px7;
    wire                conv1_w_en_pe0, conv1_w_en_pe1, conv1_w_en_pe2, conv1_w_en_pe3, conv1_w_en_pe4, conv1_w_en_pe5, conv1_w_en_pe6, conv1_w_en_pe7;

    
    wire signed [7:0]  conv1_act_mem_0_dina;
    
    wire signed [7:0]  if_dout = (conv_state == CONV1) ? act_mem_0_doutb : 0;

    /*********************************** conv1_ctrlr***************************************/
    conv1_ctrlr #(
        .IMG_W(28),.IMG_H(28),.CONV1_IC(1),.CONV1_OC(8),.ADDR_IMG(10),.ADDR_W1(7),.ADDR_F1(10)
    ) u_conv1_ctrlr (
        .clk(clk),.resetn(resetn),.start(conv1_start),.done(conv1_done),
        
        .if_en(conv1_act_mem_0_enb),.if_we(conv1_act_mem_0_web),.if_addr(conv1_act_mem_0_addrb),.if_dout(if_dout),

        .w1_en(conv1_weight_enb),.w1_we(conv1_weight_web),.w1_addr(conv1_weight_addrb), //.w1_dout(conv1_weight_doutb),

        .act_mem_2_0_en(act_mem_en),

        .act_mem_2_0_we(act_mem_we),
        
        .act_mem_2_addr(act_mem_addr),                                                                                                          //changed
        .act_mem_2_0_din(conv1_act_mem_0_dina),.act_mem_2_1_din(act_mem_1_dina), .act_mem_2_2_din(act_mem_2_dina), .act_mem_2_3_din(act_mem_3_dina), 
        .act_mem_2_4_din(act_mem_4_dina), .act_mem_2_5_din(act_mem_5_dina), .act_mem_2_6_din(act_mem_6_dina), .act_mem_2_7_din(act_mem_7_dina),
        
        .w00(conv1_w00), .w01(conv1_w01), .w02(conv1_w02), .w10(conv1_w10),.w11(conv1_w11),.w12(conv1_w12), .w20(conv1_w20), .w21(conv1_w21),.w22(conv1_w22),                                         // pixel window
        
        .w_en_0(conv1_w_en_pe0), .w_en_1(conv1_w_en_pe1), .w_en_2(conv1_w_en_pe2), .w_en_3(conv1_w_en_pe3),
        .w_en_4(conv1_w_en_pe4), .w_en_5(conv1_w_en_pe5), .w_en_6(conv1_w_en_pe6), .w_en_7(conv1_w_en_pe7),    //weight enable
        
        .out_px0(out_px0),.out_px1(out_px1),.out_px2(out_px2),.out_px3(out_px3),
        .out_px4(out_px4),.out_px5(out_px5),.out_px6(out_px6),.out_px7(out_px7) 
         
    );
    
    ///////////////////////// LOGIC for transfering output data of CONV1 or transfering data from INPUT MEM to Each Processor's ACT_MEM_0 ///////////////////////////
    assign act_mem_0_dina = (rx_input_en == 1) ? rx_input_din: conv1_act_mem_0_dina;
    
    assign act_mem_0_addra = (rx_input_en == 1) ? rx_input_addr: act_mem_addr;
    assign act_mem_1_addra = act_mem_addr;
    assign act_mem_2_addra = act_mem_addr;
    assign act_mem_3_addra = act_mem_addr;
    assign act_mem_4_addra = act_mem_addr;
    assign act_mem_5_addra = act_mem_addr;
    assign act_mem_6_addra = act_mem_addr;
    assign act_mem_7_addra = act_mem_addr;
    
    assign act_mem_0_ena = (rx_input_en == 1) ? 1 : act_mem_en;
    assign act_mem_1_ena = act_mem_en;
    assign act_mem_2_ena = act_mem_en;
    assign act_mem_3_ena = act_mem_en;
    assign act_mem_4_ena = act_mem_en;
    assign act_mem_5_ena = act_mem_en;
    assign act_mem_6_ena = act_mem_en;
    assign act_mem_7_ena = act_mem_en;
    
    assign act_mem_0_wea = (rx_input_we == 1) ? 1 : act_mem_we;
    assign act_mem_1_wea = act_mem_we;
    assign act_mem_2_wea = act_mem_we;
    assign act_mem_3_wea = act_mem_we;
    assign act_mem_4_wea = act_mem_we;
    assign act_mem_5_wea = act_mem_we;
    assign act_mem_6_wea = act_mem_we;
    assign act_mem_7_wea = act_mem_we;
    
    //////////////////////////////////////////////////////////// CONV2 Layer Instantiating ////////////////////////////////////////////////////////////
    // Instantiate IFM read signals
    wire                   ifm_rd_en;
    wire [9:0]             ifm_rd_addr;
    
    assign conv2_act_mem_0_enb  = ifm_rd_en;
    assign act_mem_1_enb        = ifm_rd_en;
    assign act_mem_2_enb        = ifm_rd_en;
    assign act_mem_3_enb        = ifm_rd_en;
    assign act_mem_4_enb        = ifm_rd_en;
    assign act_mem_5_enb        = ifm_rd_en;
    assign act_mem_6_enb        = ifm_rd_en;
    assign act_mem_7_enb        = ifm_rd_en;
    
    assign conv2_act_mem_0_addrb  = ifm_rd_addr;
    assign act_mem_1_addrb        = ifm_rd_addr;
    assign act_mem_2_addrb        = ifm_rd_addr;
    assign act_mem_3_addrb        = ifm_rd_addr;
    assign act_mem_4_addrb        = ifm_rd_addr;
    assign act_mem_5_addrb        = ifm_rd_addr;
    assign act_mem_6_addrb        = ifm_rd_addr;
    assign act_mem_7_addrb        = ifm_rd_addr;
    
    // Data from IFM SRAM channel 0 ~ 7
    wire   signed [7:0]    ifm_dout_ch0 = (conv_state == CONV2) ? act_mem_0_doutb : 0;
    wire   signed [7:0]    ifm_dout_ch1 = act_mem_1_doutb;
    wire   signed [7:0]    ifm_dout_ch2 = act_mem_2_doutb;
    wire   signed [7:0]    ifm_dout_ch3 = act_mem_3_doutb;
    wire   signed [7:0]    ifm_dout_ch4 = act_mem_4_doutb;
    wire   signed [7:0]    ifm_dout_ch5 = act_mem_5_doutb;
    wire   signed [7:0]    ifm_dout_ch6 = act_mem_6_doutb;
    wire   signed [7:0]    ifm_dout_ch7 = act_mem_7_doutb;

    // Instantiate weight read signals
    wire        conv2_w2_rd_en;
    wire [10:0] conv2_w2_rd_addr;
    
    assign conv2_weight_enb     = conv2_w2_rd_en;
    assign conv2_weight_addrb   = conv2_w2_rd_addr;

    wire signed [7:0] w00_ic0, w01_ic0, w02_ic0, w10_ic0, w11_ic0, w12_ic0, w20_ic0, w21_ic0, w22_ic0;
    wire signed [7:0] w00_ic1, w01_ic1, w02_ic1, w10_ic1, w11_ic1, w12_ic1, w20_ic1, w21_ic1, w22_ic1;
    wire signed [7:0] w00_ic2, w01_ic2, w02_ic2, w10_ic2, w11_ic2, w12_ic2, w20_ic2, w21_ic2, w22_ic2;
    wire signed [7:0] w00_ic3, w01_ic3, w02_ic3, w10_ic3, w11_ic3, w12_ic3, w20_ic3, w21_ic3, w22_ic3;
    wire signed [7:0] w00_ic4, w01_ic4, w02_ic4, w10_ic4, w11_ic4, w12_ic4, w20_ic4, w21_ic4, w22_ic4;
    wire signed [7:0] w00_ic5, w01_ic5, w02_ic5, w10_ic5, w11_ic5, w12_ic5, w20_ic5, w21_ic5, w22_ic5;
    wire signed [7:0] w00_ic6, w01_ic6, w02_ic6, w10_ic6, w11_ic6, w12_ic6, w20_ic6, w21_ic6, w22_ic6;
    wire signed [7:0] w00_ic7, w01_ic7, w02_ic7, w10_ic7, w11_ic7, w12_ic7, w20_ic7, w21_ic7, w22_ic7;

//    // Instantiate OFM write signals
//    wire                ofm_wr_en;
//    wire [9:0]          ofm_wr_addr;
//    wire signed [7:0]   ofm_dout;

    // PE interface signals
//    wire signed [23:0]  conv2_out_px = out_px0 + out_px1 + out_px2 + out_px3 + out_px4 + out_px5 + out_px6 + out_px7;
       /////////////////////////////////// FOR TIMING CONSTRAINTS ///////////////////////////////////
    reg signed [20:0] s0_delay, s1_delay, s2_delay, s3_delay;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            s0_delay <= 0; 
            s1_delay <= 0; 
            s2_delay <= 0;
            s3_delay <= 0;
        end
        else begin
            s0_delay <= out_px0 + out_px1; 
            s1_delay <= out_px2 + out_px3; 
            s2_delay <= out_px4 + out_px5;
            s3_delay <= out_px6 + out_px7;
        end
    end
    
    reg signed [21:0] t0_delay2, t1_delay2;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            t0_delay2 <= 0;
            t1_delay2 <= 0;
        end 
        else begin
            t0_delay2 <= s0_delay + s1_delay;
            t1_delay2 <= s2_delay + s3_delay;
        end
    end
    
    reg signed [22:0] u0_delay3;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            u0_delay3 <= 0;
        end
        else begin
            u0_delay3 <= t0_delay2 + t1_delay2;
        end
    end

    wire signed [23:0] conv2_out_px = u0_delay3;
    ////////////////////////////////////////////////////////////////////////////////////////////////
    
    wire signed [7:0]   conv2_out_px_quantized, conv2_out_relu;
    wire                conv2_w_en_pe0, conv2_w_en_pe1, conv2_w_en_pe2, conv2_w_en_pe3, conv2_w_en_pe4, conv2_w_en_pe5, conv2_w_en_pe6, conv2_w_en_pe7;
    
    wire [9:0] conv2_out_addr;  // CONV2 -> OFM_MAX_ADDR_LOG = 10
    wire [4:0] col_out;
    /*********************************** conv2_ctrlr***************************************/
    conv2_ctrlr u_conv2_ctrlr (
        .clk(clk),
        .resetn(resetn),
        .start(conv2_start),
        .done(conv2_done),
//        .conv1_done(conv1_done),

        // IFM Read Interface
        .ifm_rd_en      (ifm_rd_en),
        .ifm_rd_addr    (ifm_rd_addr),
        
        .ifm_dout_ch0   (ifm_dout_ch0),
        .ifm_dout_ch1   (ifm_dout_ch1),
        .ifm_dout_ch2   (ifm_dout_ch2),
        .ifm_dout_ch3   (ifm_dout_ch3),
        .ifm_dout_ch4   (ifm_dout_ch4),
        .ifm_dout_ch5   (ifm_dout_ch5),
        .ifm_dout_ch6   (ifm_dout_ch6),
        .ifm_dout_ch7   (ifm_dout_ch7),

        // Weight Read Interface
        .w2_rd_en   (conv2_w2_rd_en),
        .w2_rd_addr (conv2_w2_rd_addr),
        
        // windows for pixels
        .w00_ic0(w00_ic0), .w01_ic0(w01_ic0), .w02_ic0(w02_ic0), .w10_ic0(w10_ic0), .w11_ic0(w11_ic0), .w12_ic0(w12_ic0), .w20_ic0(w20_ic0), .w21_ic0(w21_ic0), .w22_ic0(w22_ic0),
        .w00_ic1(w00_ic1), .w01_ic1(w01_ic1), .w02_ic1(w02_ic1), .w10_ic1(w10_ic1), .w11_ic1(w11_ic1), .w12_ic1(w12_ic1), .w20_ic1(w20_ic1), .w21_ic1(w21_ic1), .w22_ic1(w22_ic1),
        .w00_ic2(w00_ic2), .w01_ic2(w01_ic2), .w02_ic2(w02_ic2), .w10_ic2(w10_ic2), .w11_ic2(w11_ic2), .w12_ic2(w12_ic2), .w20_ic2(w20_ic2), .w21_ic2(w21_ic2), .w22_ic2(w22_ic2),
        .w00_ic3(w00_ic3), .w01_ic3(w01_ic3), .w02_ic3(w02_ic3), .w10_ic3(w10_ic3), .w11_ic3(w11_ic3), .w12_ic3(w12_ic3), .w20_ic3(w20_ic3), .w21_ic3(w21_ic3), .w22_ic3(w22_ic3),
        .w00_ic4(w00_ic4), .w01_ic4(w01_ic4), .w02_ic4(w02_ic4), .w10_ic4(w10_ic4), .w11_ic4(w11_ic4), .w12_ic4(w12_ic4), .w20_ic4(w20_ic4), .w21_ic4(w21_ic4), .w22_ic4(w22_ic4),
        .w00_ic5(w00_ic5), .w01_ic5(w01_ic5), .w02_ic5(w02_ic5), .w10_ic5(w10_ic5), .w11_ic5(w11_ic5), .w12_ic5(w12_ic5), .w20_ic5(w20_ic5), .w21_ic5(w21_ic5), .w22_ic5(w22_ic5),
        .w00_ic6(w00_ic6), .w01_ic6(w01_ic6), .w02_ic6(w02_ic6), .w10_ic6(w10_ic6), .w11_ic6(w11_ic6), .w12_ic6(w12_ic6), .w20_ic6(w20_ic6), .w21_ic6(w21_ic6), .w22_ic6(w22_ic6),
        .w00_ic7(w00_ic7), .w01_ic7(w01_ic7), .w02_ic7(w02_ic7), .w10_ic7(w10_ic7), .w11_ic7(w11_ic7), .w12_ic7(w12_ic7), .w20_ic7(w20_ic7), .w21_ic7(w21_ic7), .w22_ic7(w22_ic7),

//        // OFM Write Interface
//        .ofm_wr_en  (ofm_wr_en),
//        .ofm_wr_addr(ofm_wr_addr),
//        .ofm_dout   (ofm_dout),

        // PE Interface
        .out_px0(out_px0),
        .out_px1(out_px1),
        .out_px2(out_px2),
        .out_px3(out_px3),
        .out_px4(out_px4),
        .out_px5(out_px5),
        .out_px6(out_px6),
        .out_px7(out_px7),
        
        .w_en_pe0(conv2_w_en_pe0),
        .w_en_pe1(conv2_w_en_pe1),
        .w_en_pe2(conv2_w_en_pe2),
        .w_en_pe3(conv2_w_en_pe3),
        .w_en_pe4(conv2_w_en_pe4),
        .w_en_pe5(conv2_w_en_pe5),
        .w_en_pe6(conv2_w_en_pe6),
        .w_en_pe7(conv2_w_en_pe7),
        
        .col_out(col_out),
        .out_addr(conv2_out_addr)
    );
    
    quantize #( .INPUT_DATA_WIDTH(24), .OUTPUT_DATA_WIDTH(8) )  i_quantize_0 ( .in_data(conv2_out_px), .out_data(conv2_out_px_quantized));
    relu #( .DATA_WIDTH(8) )                                    i_relu_0 ( .in_data(conv2_out_px_quantized), .out_data(conv2_out_relu) );
    
    /////////////////////////////////////////////////////////////// MAXPOOL LOGIC /////////////////////////////////////////////////////////////////////////////////
    wire signed [7:0]   maxpool_out;
    reg signed [7:0] maxpool_out_reg;
    reg signed [7:0] maxpool_out_reg2;
    wire [7:0]          maxpool_out_addr;
    reg [7:0] maxpool_out_addr_reg;
    reg [7:0] maxpool_out_addr_reg2;
    reg [3:0]           maxpool_out_addr_iter;
    wire addr_bounce;
    assign addr_bounce = (maxpool_out_addr[0] != maxpool_out_addr_reg[0]);     //have to change?
    reg [ADDR_WFC-1:0] fc_addr_tot;
    
    max_pool #(
        .CHANNELS(16),
        .D_IN_MAX_ADDR_LOG(10),
        .D_IN_MAX_ADDR(576), 
        .IN_W(24),
        .IN_H(24)
    ) u_max_pool (
        .clk(clk),
        .resetn(resetn),
//        .en(),
        .din_addr(conv2_out_addr),
        .din(conv2_out_relu),
        .dout(maxpool_out),
        
        .din_col(col_out),
        .dout_addr(maxpool_out_addr)
    );
    always@(posedge clk or negedge resetn) begin
        if(!resetn) begin
            maxpool_out_addr_reg <= 0;
            maxpool_out_addr_reg2 <= 0;
        end
        else begin
            maxpool_out_addr_reg <= maxpool_out_addr;
            maxpool_out_addr_reg2 <= maxpool_out_addr_reg;
        end
    end
    always@(posedge clk or negedge resetn) begin
        if(!resetn) begin
            maxpool_out_reg <= 0;
            maxpool_out_reg2 <= 0;
        end
        else 
            maxpool_out_reg <= maxpool_out;
            maxpool_out_reg2 <= maxpool_out_reg;
    end
    always@(posedge clk or negedge resetn) begin
        if(!resetn) begin
            maxpool_out_addr_iter <= 0;
        end
        else if(maxpool_out_addr == 8'd143 && (addr_bounce == 0)) begin
            maxpool_out_addr_iter <= maxpool_out_addr_iter + 1;
        end
    end
    always@(posedge clk or negedge resetn) begin
        if(!resetn) begin
            fc_addr_tot <= 0;
        end
        else 
            fc_addr_tot <=  maxpool_out_addr + (maxpool_out_addr_iter<<7) + (maxpool_out_addr_iter<<4);
    end
    
    /////////////////////////////////////////////////////////////// PE FC LOGIC /////////////////////////////////////////////////////////////////////////////////
    // FC Layer Control Wires
    reg  [11:0] fc_addr;
    reg         fc_start;
    wire signed [7:0]  fc_result_0;
    wire signed [7:0]  fc_result_1;
    wire signed [7:0]  fc_result_2;
    wire signed [7:0]  fc_result_3;
    wire signed [7:0]  fc_result_4;
    wire signed [7:0]  fc_result_5;
    wire signed [7:0]  fc_result_6;
    wire signed [7:0]  fc_result_7;
    wire signed [7:0]  fc_result_8;
    wire signed [7:0]  fc_result_9;
    wire        fc_done;
    
    pe_fc_multi fc_layer (
    .clk(clk), .resetn(resetn),
    .act_in(maxpool_out_reg2), .valid_in(addr_bounce),
    .w0_in(fc_weight_0_doutb), .w1_in(fc_weight_1_doutb), .w2_in(fc_weight_2_doutb),
    .w3_in(fc_weight_3_doutb), .w4_in(fc_weight_4_doutb), .w5_in(fc_weight_5_doutb),
    .w6_in(fc_weight_6_doutb), .w7_in(fc_weight_7_doutb),
    .w8_in(fc_weight_8_doutb), .w9_in(fc_weight_9_doutb),
    .out_0(fc_result_0), .out_1(fc_result_1), .out_2(fc_result_2),
    .out_3(fc_result_3), .out_4(fc_result_4), .out_5(fc_result_5),
    .out_6(fc_result_6), .out_7(fc_result_7),
    .out_8(fc_result_8), .out_9(fc_result_9),
    .reg_done(fc_done)
);
    
    // Weight reads from fc_weight_0 ~ 9
    assign fc_weight_0_enb = 1'b1;  assign fc_weight_0_addrb = fc_addr_tot;        //144*iter 
    assign fc_weight_1_enb = 1'b1;  assign fc_weight_1_addrb = fc_addr_tot;    //have to chage?
    assign fc_weight_2_enb = 1'b1;  assign fc_weight_2_addrb = fc_addr_tot;
    assign fc_weight_3_enb = 1'b1;  assign fc_weight_3_addrb = fc_addr_tot;
    assign fc_weight_4_enb = 1'b1;  assign fc_weight_4_addrb = fc_addr_tot;
    assign fc_weight_5_enb = 1'b1;  assign fc_weight_5_addrb = fc_addr_tot;
    assign fc_weight_6_enb = 1'b1;  assign fc_weight_6_addrb = fc_addr_tot;
    assign fc_weight_7_enb = 1'b1;  assign fc_weight_7_addrb = fc_addr_tot;
    assign fc_weight_8_enb = 1'b1;  assign fc_weight_8_addrb = fc_addr_tot;
    assign fc_weight_9_enb = 1'b1;  assign fc_weight_9_addrb = fc_addr_tot;
    
    assign out_done = fc_done;
    /////////////////////////////////////////////////////////////// FC Result Transfer LOGIC /////////////////////////////////////////////////////////////////////////////////
    
    reg signed [7:0]  fc_result_0_reg;
    reg signed [7:0]  fc_result_1_reg;
    reg signed [7:0]  fc_result_2_reg;
    reg signed [7:0]  fc_result_3_reg;
    reg signed [7:0]  fc_result_4_reg;
    reg signed [7:0]  fc_result_5_reg;
    reg signed [7:0]  fc_result_6_reg;
    reg signed [7:0]  fc_result_7_reg;
    reg signed [7:0]  fc_result_8_reg;
    reg signed [7:0]  fc_result_9_reg;

//    /***** Storing FC Output in Global Buffer *****/
//    always @(posedge fc_done or negedge resetn) begin
//        if(!resetn) begin
//            fc_result_0_reg <= 0; fc_result_1_reg <= 0; fc_result_2_reg <= 0; fc_result_3_reg <= 0; fc_result_4_reg <= 0;
//            fc_result_5_reg <= 0; fc_result_6_reg <= 0; fc_result_7_reg <= 0; fc_result_8_reg <= 0; fc_result_9_reg <= 0;
//        end
//        else begin
//            fc_result_0_reg <= fc_result_0; fc_result_1_reg <= fc_result_1; fc_result_2_reg <= fc_result_2; fc_result_3_reg <= fc_result_3; fc_result_4_reg <= fc_result_4;
//            fc_result_5_reg <= fc_result_5; fc_result_6_reg <= fc_result_6; fc_result_7_reg <= fc_result_7; fc_result_8_reg <= fc_result_8; fc_result_9_reg <= fc_result_9;
//        end
//    end
    
    reg [3:0] fc_transfer_counter;
    reg fc_to_out_mem_start;
//    reg fc_to_out_mem_done_reg;
    reg fc_to_out_mem_ena_reg;
    reg fc_to_out_mem_wea_reg;

//    reg fc_to_out_mem_done;
    /***** COUNTER for FC out Addressing *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            fc_transfer_counter <= 0;
            
            fc_to_out_mem_ena_reg <= 0;
            fc_to_out_mem_wea_reg <= 0;
            fc_to_out_mem_start <= 0;
//            fc_to_out_mem_done_reg <= 0;
        end
        else if(fc_done) begin
            fc_to_out_mem_ena_reg <= 1;
            fc_to_out_mem_wea_reg <= 1;
            
            fc_to_out_mem_start <= 1;
        end
        else if (fc_transfer_counter == 9) begin
            fc_to_out_mem_ena_reg <= 0;
            fc_to_out_mem_wea_reg <= 0;
            fc_to_out_mem_start <= 0;
            fc_transfer_counter <= 0;
//            fc_to_out_mem_done_reg <= 1;
        end
        else if (fc_to_out_mem_start) begin
            fc_transfer_counter <= fc_transfer_counter + 1;
        end
        else begin
//            fc_to_out_mem_done_reg <= 0;
            fc_to_out_mem_ena_reg <= 0;
            fc_to_out_mem_wea_reg <= 0;
            fc_to_out_mem_start <= 0;
            fc_transfer_counter <= 0;
        end
    end

    reg signed [7:0] fc_to_out_mem_dina_reg;
    /***** FC out Aligning corresponding to Address *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            fc_to_out_mem_dina_reg <= 0;
            fc_result_0_reg <= 0; fc_result_1_reg <= 0; fc_result_2_reg <= 0; fc_result_3_reg <= 0; fc_result_4_reg <= 0;
            fc_result_5_reg <= 0; fc_result_6_reg <= 0; fc_result_7_reg <= 0; fc_result_8_reg <= 0; fc_result_9_reg <= 0;
        end
        else if(fc_done) begin
            fc_to_out_mem_dina_reg <= fc_result_0;
            
            {fc_result_0_reg, fc_result_1_reg, fc_result_2_reg, fc_result_3_reg, fc_result_4_reg, 
            fc_result_5_reg, fc_result_6_reg, fc_result_7_reg, fc_result_8_reg, fc_result_9_reg}
            <=  {fc_result_1, fc_result_2, fc_result_3, fc_result_4, fc_result_5, 
                fc_result_6, fc_result_7, fc_result_8, fc_result_9, fc_result_0};
        end
        else if (fc_to_out_mem_start) begin
            fc_to_out_mem_dina_reg <= fc_result_0_reg;
            
            {fc_result_0_reg, fc_result_1_reg, fc_result_2_reg, fc_result_3_reg, fc_result_4_reg, 
            fc_result_5_reg, fc_result_6_reg, fc_result_7_reg, fc_result_8_reg, fc_result_9_reg}
            <=  {fc_result_1_reg, fc_result_2_reg, fc_result_3_reg, fc_result_4_reg, fc_result_5_reg, 
                fc_result_6_reg, fc_result_7_reg, fc_result_8_reg, fc_result_9_reg, fc_result_0_reg};
        end
        else begin
            fc_to_out_mem_dina_reg <= 0;
            fc_result_0_reg <= 0; fc_result_1_reg <= 0; fc_result_2_reg <= 0; fc_result_3_reg <= 0; fc_result_4_reg <= 0;
            fc_result_5_reg <= 0; fc_result_6_reg <= 0; fc_result_7_reg <= 0; fc_result_8_reg <= 0; fc_result_9_reg <= 0;
        end
    end

    assign out_mem_ena          = fc_to_out_mem_ena_reg;
    assign out_mem_wea          = fc_to_out_mem_wea_reg;
    assign out_mem_addra        = fc_transfer_counter;
    assign out_mem_dina         = fc_to_out_mem_dina_reg;
    
//    assign out_mem_write_done   = fc_to_out_mem_done_reg;

    //////////////////////////////////////////////////////////// PE Instantiating //////////////////////////////////////////////////////////////////////////////
    // window 00 for pe0~8
//    wire signed [7:0] w00_pe0 = (conv_state == CONV1) ? conv1_w00 : w00_ic0;
//    wire signed [7:0] w00_pe1 = (conv_state == CONV1) ? conv1_w00 : w00_ic1;
//    wire signed [7:0] w00_pe2 = (conv_state == CONV1) ? conv1_w00 : w00_ic2;
//    wire signed [7:0] w00_pe3 = (conv_state == CONV1) ? conv1_w00 : w00_ic3;
//    wire signed [7:0] w00_pe4 = (conv_state == CONV1) ? conv1_w00 : w00_ic4;
//    wire signed [7:0] w00_pe5 = (conv_state == CONV1) ? conv1_w00 : w00_ic5;
//    wire signed [7:0] w00_pe6 = (conv_state == CONV1) ? conv1_w00 : w00_ic6;
//    wire signed [7:0] w00_pe7 = (conv_state == CONV1) ? conv1_w00 : w00_ic7;

    /////////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////////////////
    reg signed [7:0] w00_pe0, w00_pe1, w00_pe2, w00_pe3;
    reg signed [7:0] w00_pe4, w00_pe5, w00_pe6, w00_pe7;
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            w00_pe0 <= 0; w00_pe1 <= 0; w00_pe2 <= 0; w00_pe3 <= 0;
            w00_pe4 <= 0; w00_pe5 <= 0; w00_pe6 <= 0; w00_pe7 <= 0;
        end else begin
            if (conv_state == CONV1) begin
                w00_pe0 <= conv1_w00; w00_pe1 <= conv1_w00; w00_pe2 <= conv1_w00; w00_pe3 <= conv1_w00;
                w00_pe4 <= conv1_w00; w00_pe5 <= conv1_w00; w00_pe6 <= conv1_w00; w00_pe7 <= conv1_w00;
            end else begin
                w00_pe0 <= w00_ic0; w00_pe1 <= w00_ic1; w00_pe2 <= w00_ic2; w00_pe3 <= w00_ic3;
                w00_pe4 <= w00_ic4; w00_pe5 <= w00_ic5; w00_pe6 <= w00_ic6; w00_pe7 <= w00_ic7;
            end
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // window 01 for pe0~8
//    wire signed [7:0] w01_pe0 = (conv_state == CONV1) ? conv1_w01 : w01_ic0;
//    wire signed [7:0] w01_pe1 = (conv_state == CONV1) ? conv1_w01 : w01_ic1;
//    wire signed [7:0] w01_pe2 = (conv_state == CONV1) ? conv1_w01 : w01_ic2;
//    wire signed [7:0] w01_pe3 = (conv_state == CONV1) ? conv1_w01 : w01_ic3;
//    wire signed [7:0] w01_pe4 = (conv_state == CONV1) ? conv1_w01 : w01_ic4;
//    wire signed [7:0] w01_pe5 = (conv_state == CONV1) ? conv1_w01 : w01_ic5;
//    wire signed [7:0] w01_pe6 = (conv_state == CONV1) ? conv1_w01 : w01_ic6;
//    wire signed [7:0] w01_pe7 = (conv_state == CONV1) ? conv1_w01 : w01_ic7;

    /////////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////////////////
    reg signed [7:0] w01_pe0, w01_pe1, w01_pe2, w01_pe3;
    reg signed [7:0] w01_pe4, w01_pe5, w01_pe6, w01_pe7;
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            w01_pe0 <= 0; w01_pe1 <= 0; w01_pe2 <= 0; w01_pe3 <= 0;
            w01_pe4 <= 0; w01_pe5 <= 0; w01_pe6 <= 0; w01_pe7 <= 0;
        end else begin
            if (conv_state == CONV1) begin
                w01_pe0 <= conv1_w01; w01_pe1 <= conv1_w01; w01_pe2 <= conv1_w01; w01_pe3 <= conv1_w01;
                w01_pe4 <= conv1_w01; w01_pe5 <= conv1_w01; w01_pe6 <= conv1_w01; w01_pe7 <= conv1_w01;
            end else begin
                w01_pe0 <= w01_ic0; w01_pe1 <= w01_ic1; w01_pe2 <= w01_ic2; w01_pe3 <= w01_ic3;
                w01_pe4 <= w01_ic4; w01_pe5 <= w01_ic5; w01_pe6 <= w01_ic6; w01_pe7 <= w01_ic7;
            end
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // window 02 for pe0~8
//    wire signed [7:0] w02_pe0 = (conv_state == CONV1) ? conv1_w02 : w02_ic0;
//    wire signed [7:0] w02_pe1 = (conv_state == CONV1) ? conv1_w02 : w02_ic1;
//    wire signed [7:0] w02_pe2 = (conv_state == CONV1) ? conv1_w02 : w02_ic2;
//    wire signed [7:0] w02_pe3 = (conv_state == CONV1) ? conv1_w02 : w02_ic3;
//    wire signed [7:0] w02_pe4 = (conv_state == CONV1) ? conv1_w02 : w02_ic4;
//    wire signed [7:0] w02_pe5 = (conv_state == CONV1) ? conv1_w02 : w02_ic5;
//    wire signed [7:0] w02_pe6 = (conv_state == CONV1) ? conv1_w02 : w02_ic6;
//    wire signed [7:0] w02_pe7 = (conv_state == CONV1) ? conv1_w02 : w02_ic7;

    /////////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////////////////
    reg signed [7:0] w02_pe0, w02_pe1, w02_pe2, w02_pe3;
    reg signed [7:0] w02_pe4, w02_pe5, w02_pe6, w02_pe7;
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            w02_pe0 <= 0; w02_pe1 <= 0; w02_pe2 <= 0; w02_pe3 <= 0;
            w02_pe4 <= 0; w02_pe5 <= 0; w02_pe6 <= 0; w02_pe7 <= 0;
        end else begin
            if (conv_state == CONV1) begin
                w02_pe0 <= conv1_w02; w02_pe1 <= conv1_w02; w02_pe2 <= conv1_w02; w02_pe3 <= conv1_w02;
                w02_pe4 <= conv1_w02; w02_pe5 <= conv1_w02; w02_pe6 <= conv1_w02; w02_pe7 <= conv1_w02;
            end else begin
                w02_pe0 <= w02_ic0; w02_pe1 <= w02_ic1; w02_pe2 <= w02_ic2; w02_pe3 <= w02_ic3;
                w02_pe4 <= w02_ic4; w02_pe5 <= w02_ic5; w02_pe6 <= w02_ic6; w02_pe7 <= w02_ic7;
            end
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // window 10 for pe0~8
//    wire signed [7:0] w10_pe0 = (conv_state == CONV1) ? conv1_w10 : w10_ic0;
//    wire signed [7:0] w10_pe1 = (conv_state == CONV1) ? conv1_w10 : w10_ic1;
//    wire signed [7:0] w10_pe2 = (conv_state == CONV1) ? conv1_w10 : w10_ic2;
//    wire signed [7:0] w10_pe3 = (conv_state == CONV1) ? conv1_w10 : w10_ic3;
//    wire signed [7:0] w10_pe4 = (conv_state == CONV1) ? conv1_w10 : w10_ic4;
//    wire signed [7:0] w10_pe5 = (conv_state == CONV1) ? conv1_w10 : w10_ic5;
//    wire signed [7:0] w10_pe6 = (conv_state == CONV1) ? conv1_w10 : w10_ic6;
//    wire signed [7:0] w10_pe7 = (conv_state == CONV1) ? conv1_w10 : w10_ic7;

    /////////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////////////////
    reg signed [7:0] w10_pe0, w10_pe1, w10_pe2, w10_pe3;
    reg signed [7:0] w10_pe4, w10_pe5, w10_pe6, w10_pe7;
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            w10_pe0 <= 0; w10_pe1 <= 0; w10_pe2 <= 0; w10_pe3 <= 0;
            w10_pe4 <= 0; w10_pe5 <= 0; w10_pe6 <= 0; w10_pe7 <= 0;
        end else begin
            if (conv_state == CONV1) begin
                w10_pe0 <= conv1_w10; w10_pe1 <= conv1_w10; w10_pe2 <= conv1_w10; w10_pe3 <= conv1_w10;
                w10_pe4 <= conv1_w10; w10_pe5 <= conv1_w10; w10_pe6 <= conv1_w10; w10_pe7 <= conv1_w10;
            end else begin
                w10_pe0 <= w10_ic0; w10_pe1 <= w10_ic1; w10_pe2 <= w10_ic2; w10_pe3 <= w10_ic3;
                w10_pe4 <= w10_ic4; w10_pe5 <= w10_ic5; w10_pe6 <= w10_ic6; w10_pe7 <= w10_ic7;
            end
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // window 11 for pe0~8
//    wire signed [7:0] w11_pe0 = (conv_state == CONV1) ? conv1_w11 : w11_ic0;
//    wire signed [7:0] w11_pe1 = (conv_state == CONV1) ? conv1_w11 : w11_ic1;
//    wire signed [7:0] w11_pe2 = (conv_state == CONV1) ? conv1_w11 : w11_ic2;
//    wire signed [7:0] w11_pe3 = (conv_state == CONV1) ? conv1_w11 : w11_ic3;
//    wire signed [7:0] w11_pe4 = (conv_state == CONV1) ? conv1_w11 : w11_ic4;
//    wire signed [7:0] w11_pe5 = (conv_state == CONV1) ? conv1_w11 : w11_ic5;
//    wire signed [7:0] w11_pe6 = (conv_state == CONV1) ? conv1_w11 : w11_ic6;
//    wire signed [7:0] w11_pe7 = (conv_state == CONV1) ? conv1_w11 : w11_ic7;

    /////////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////////////////
    reg signed [7:0] w11_pe0, w11_pe1, w11_pe2, w11_pe3;
    reg signed [7:0] w11_pe4, w11_pe5, w11_pe6, w11_pe7;
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            w11_pe0 <= 0; w11_pe1 <= 0; w11_pe2 <= 0; w11_pe3 <= 0;
            w11_pe4 <= 0; w11_pe5 <= 0; w11_pe6 <= 0; w11_pe7 <= 0;
        end else begin
            if (conv_state == CONV1) begin
                w11_pe0 <= conv1_w11; w11_pe1 <= conv1_w11; w11_pe2 <= conv1_w11; w11_pe3 <= conv1_w11;
                w11_pe4 <= conv1_w11; w11_pe5 <= conv1_w11; w11_pe6 <= conv1_w11; w11_pe7 <= conv1_w11;
            end else begin
                w11_pe0 <= w11_ic0; w11_pe1 <= w11_ic1; w11_pe2 <= w11_ic2; w11_pe3 <= w11_ic3;
                w11_pe4 <= w11_ic4; w11_pe5 <= w11_ic5; w11_pe6 <= w11_ic6; w11_pe7 <= w11_ic7;
            end
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // window 12 for pe0~8
//    wire signed [7:0] w12_pe0 = (conv_state == CONV1) ? conv1_w12 : w12_ic0;
//    wire signed [7:0] w12_pe1 = (conv_state == CONV1) ? conv1_w12 : w12_ic1;
//    wire signed [7:0] w12_pe2 = (conv_state == CONV1) ? conv1_w12 : w12_ic2;
//    wire signed [7:0] w12_pe3 = (conv_state == CONV1) ? conv1_w12 : w12_ic3;
//    wire signed [7:0] w12_pe4 = (conv_state == CONV1) ? conv1_w12 : w12_ic4;
//    wire signed [7:0] w12_pe5 = (conv_state == CONV1) ? conv1_w12 : w12_ic5;
//    wire signed [7:0] w12_pe6 = (conv_state == CONV1) ? conv1_w12 : w12_ic6;
//    wire signed [7:0] w12_pe7 = (conv_state == CONV1) ? conv1_w12 : w12_ic7;

    /////////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////////////////
    reg signed [7:0] w12_pe0, w12_pe1, w12_pe2, w12_pe3;
    reg signed [7:0] w12_pe4, w12_pe5, w12_pe6, w12_pe7;
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            w12_pe0 <= 0; w12_pe1 <= 0; w12_pe2 <= 0; w12_pe3 <= 0;
            w12_pe4 <= 0; w12_pe5 <= 0; w12_pe6 <= 0; w12_pe7 <= 0;
        end else begin
            if (conv_state == CONV1) begin
                w12_pe0 <= conv1_w12; w12_pe1 <= conv1_w12; w12_pe2 <= conv1_w12; w12_pe3 <= conv1_w12;
                w12_pe4 <= conv1_w12; w12_pe5 <= conv1_w12; w12_pe6 <= conv1_w12; w12_pe7 <= conv1_w12;
            end else begin
                w12_pe0 <= w12_ic0; w12_pe1 <= w12_ic1; w12_pe2 <= w12_ic2; w12_pe3 <= w12_ic3;
                w12_pe4 <= w12_ic4; w12_pe5 <= w12_ic5; w12_pe6 <= w12_ic6; w12_pe7 <= w12_ic7;
            end
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // window 20 for pe0~8
//    wire signed [7:0] w20_pe0 = (conv_state == CONV1) ? conv1_w20 : w20_ic0;
//    wire signed [7:0] w20_pe1 = (conv_state == CONV1) ? conv1_w20 : w20_ic1;
//    wire signed [7:0] w20_pe2 = (conv_state == CONV1) ? conv1_w20 : w20_ic2;
//    wire signed [7:0] w20_pe3 = (conv_state == CONV1) ? conv1_w20 : w20_ic3;
//    wire signed [7:0] w20_pe4 = (conv_state == CONV1) ? conv1_w20 : w20_ic4;
//    wire signed [7:0] w20_pe5 = (conv_state == CONV1) ? conv1_w20 : w20_ic5;
//    wire signed [7:0] w20_pe6 = (conv_state == CONV1) ? conv1_w20 : w20_ic6;
//    wire signed [7:0] w20_pe7 = (conv_state == CONV1) ? conv1_w20 : w20_ic7;

    /////////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////////////////
    reg signed [7:0] w20_pe0, w20_pe1, w20_pe2, w20_pe3;
    reg signed [7:0] w20_pe4, w20_pe5, w20_pe6, w20_pe7;
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            w20_pe0 <= 0; w20_pe1 <= 0; w20_pe2 <= 0; w20_pe3 <= 0;
            w20_pe4 <= 0; w20_pe5 <= 0; w20_pe6 <= 0; w20_pe7 <= 0;
        end else begin
            if (conv_state == CONV1) begin
                w20_pe0 <= conv1_w20; w20_pe1 <= conv1_w20; w20_pe2 <= conv1_w20; w20_pe3 <= conv1_w20;
                w20_pe4 <= conv1_w20; w20_pe5 <= conv1_w20; w20_pe6 <= conv1_w20; w20_pe7 <= conv1_w20;
            end else begin
                w20_pe0 <= w20_ic0; w20_pe1 <= w20_ic1; w20_pe2 <= w20_ic2; w20_pe3 <= w20_ic3;
                w20_pe4 <= w20_ic4; w20_pe5 <= w20_ic5; w20_pe6 <= w20_ic6; w20_pe7 <= w20_ic7;
            end
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // window 21 for pe0~8
//    wire signed [7:0] w21_pe0 = (conv_state == CONV1) ? conv1_w21 : w21_ic0;
//    wire signed [7:0] w21_pe1 = (conv_state == CONV1) ? conv1_w21 : w21_ic1;
//    wire signed [7:0] w21_pe2 = (conv_state == CONV1) ? conv1_w21 : w21_ic2;
//    wire signed [7:0] w21_pe3 = (conv_state == CONV1) ? conv1_w21 : w21_ic3;
//    wire signed [7:0] w21_pe4 = (conv_state == CONV1) ? conv1_w21 : w21_ic4;
//    wire signed [7:0] w21_pe5 = (conv_state == CONV1) ? conv1_w21 : w21_ic5;
//    wire signed [7:0] w21_pe6 = (conv_state == CONV1) ? conv1_w21 : w21_ic6;
//    wire signed [7:0] w21_pe7 = (conv_state == CONV1) ? conv1_w21 : w21_ic7;

    /////////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////////////////
    reg signed [7:0] w21_pe0, w21_pe1, w21_pe2, w21_pe3;
    reg signed [7:0] w21_pe4, w21_pe5, w21_pe6, w21_pe7;
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            w21_pe0 <= 0; w21_pe1 <= 0; w21_pe2 <= 0; w21_pe3 <= 0;
            w21_pe4 <= 0; w21_pe5 <= 0; w21_pe6 <= 0; w21_pe7 <= 0;
        end else begin
            if (conv_state == CONV1) begin
                w21_pe0 <= conv1_w21; w21_pe1 <= conv1_w21; w21_pe2 <= conv1_w21; w21_pe3 <= conv1_w21;
                w21_pe4 <= conv1_w21; w21_pe5 <= conv1_w21; w21_pe6 <= conv1_w21; w21_pe7 <= conv1_w21;
            end else begin
                w21_pe0 <= w21_ic0; w21_pe1 <= w21_ic1; w21_pe2 <= w21_ic2; w21_pe3 <= w21_ic3;
                w21_pe4 <= w21_ic4; w21_pe5 <= w21_ic5; w21_pe6 <= w21_ic6; w21_pe7 <= w21_ic7;
            end
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // window 22 for pe0~8
//    wire signed [7:0] w22_pe0 = (conv_state == CONV1) ? conv1_w22 : w22_ic0;
//    wire signed [7:0] w22_pe1 = (conv_state == CONV1) ? conv1_w22 : w22_ic1;
//    wire signed [7:0] w22_pe2 = (conv_state == CONV1) ? conv1_w22 : w22_ic2;
//    wire signed [7:0] w22_pe3 = (conv_state == CONV1) ? conv1_w22 : w22_ic3;
//    wire signed [7:0] w22_pe4 = (conv_state == CONV1) ? conv1_w22 : w22_ic4;
//    wire signed [7:0] w22_pe5 = (conv_state == CONV1) ? conv1_w22 : w22_ic5;
//    wire signed [7:0] w22_pe6 = (conv_state == CONV1) ? conv1_w22 : w22_ic6;
//    wire signed [7:0] w22_pe7 = (conv_state == CONV1) ? conv1_w22 : w22_ic7;

    /////////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////////////////
    reg signed [7:0] w22_pe0, w22_pe1, w22_pe2, w22_pe3;
    reg signed [7:0] w22_pe4, w22_pe5, w22_pe6, w22_pe7;
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            w22_pe0 <= 0; w22_pe1 <= 0; w22_pe2 <= 0; w22_pe3 <= 0;
            w22_pe4 <= 0; w22_pe5 <= 0; w22_pe6 <= 0; w22_pe7 <= 0;
        end else begin
            if (conv_state == CONV1) begin
                w22_pe0 <= conv1_w22; w22_pe1 <= conv1_w22; w22_pe2 <= conv1_w22; w22_pe3 <= conv1_w22;
                w22_pe4 <= conv1_w22; w22_pe5 <= conv1_w22; w22_pe6 <= conv1_w22; w22_pe7 <= conv1_w22;
            end else begin
                w22_pe0 <= w22_ic0; w22_pe1 <= w22_ic1; w22_pe2 <= w22_ic2; w22_pe3 <= w22_ic3;
                w22_pe4 <= w22_ic4; w22_pe5 <= w22_ic5; w22_pe6 <= w22_ic6; w22_pe7 <= w22_ic7;
            end
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // weight enable signals for each PE
//    wire weight_en_0 = (conv_state == CONV1) ? conv1_w_en_pe0 : conv2_w_en_pe0;
//    wire weight_en_1 = (conv_state == CONV1) ? conv1_w_en_pe1 : conv2_w_en_pe1;
//    wire weight_en_2 = (conv_state == CONV1) ? conv1_w_en_pe2 : conv2_w_en_pe2;
//    wire weight_en_3 = (conv_state == CONV1) ? conv1_w_en_pe3 : conv2_w_en_pe3;
//    wire weight_en_4 = (conv_state == CONV1) ? conv1_w_en_pe4 : conv2_w_en_pe4;
//    wire weight_en_5 = (conv_state == CONV1) ? conv1_w_en_pe5 : conv2_w_en_pe5;
//    wire weight_en_6 = (conv_state == CONV1) ? conv1_w_en_pe6 : conv2_w_en_pe6;
//    wire weight_en_7 = (conv_state == CONV1) ? conv1_w_en_pe7 : conv2_w_en_pe7;

    /////////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////////////////
    reg weight_en_0, weight_en_1, weight_en_2, weight_en_3;
    reg weight_en_4, weight_en_5, weight_en_6, weight_en_7;
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            weight_en_0 <= 0; weight_en_1 <= 0; weight_en_2 <= 0; weight_en_3 <= 0;
            weight_en_4 <= 0; weight_en_5 <= 0; weight_en_6 <= 0; weight_en_7 <= 0;
        end else begin
            if (conv_state == CONV1) begin
                weight_en_0 <= conv1_w_en_pe0; weight_en_1 <= conv1_w_en_pe1;
                weight_en_2 <= conv1_w_en_pe2; weight_en_3 <= conv1_w_en_pe3;
                weight_en_4 <= conv1_w_en_pe4; weight_en_5 <= conv1_w_en_pe5;
                weight_en_6 <= conv1_w_en_pe6; weight_en_7 <= conv1_w_en_pe7;
            end else begin
                weight_en_0 <= conv2_w_en_pe0; weight_en_1 <= conv2_w_en_pe1;
                weight_en_2 <= conv2_w_en_pe2; weight_en_3 <= conv2_w_en_pe3;
                weight_en_4 <= conv2_w_en_pe4; weight_en_5 <= conv2_w_en_pe5;
                weight_en_6 <= conv2_w_en_pe6; weight_en_7 <= conv2_w_en_pe7;
            end
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
   
//    wire signed [7:0] weight_doutb = (conv_state == CONV1) ? conv1_weight_doutb : conv2_weight_doutb;

    /////////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////////////////
    reg signed [7:0] weight_doutb;
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            weight_doutb <= 0;
        end else begin
            if (conv_state == CONV1)
                weight_doutb <= conv1_weight_doutb;
            else
                weight_doutb <= conv2_weight_doutb;
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
   
    /***** 3x3 MAC computer instance *****/
    pe_conv CONV1_PE0 ( .clk(clk), .resetn(resetn),
                        .act00(w00_pe0),.act01(w01_pe0),.act02(w02_pe0),
                        .act10(w10_pe0),.act11(w11_pe0),.act12(w12_pe0),
                        .act20(w20_pe0),.act21(w21_pe0),.act22(w22_pe0),
                        .w1_en(weight_en_0), .weight_in(weight_doutb),
                        .out(out_px0));
    
    pe_conv CONV1_PE1 ( .clk(clk), .resetn(resetn),
                        .act00(w00_pe1),.act01(w01_pe1),.act02(w02_pe1),
                        .act10(w10_pe1),.act11(w11_pe1),.act12(w12_pe1),
                        .act20(w20_pe1),.act21(w21_pe1),.act22(w22_pe1),
                        .w1_en(weight_en_1), .weight_in(weight_doutb),
                        .out(out_px1));
    
    pe_conv CONV1_PE2 ( .clk(clk), .resetn(resetn),
                        .act00(w00_pe2),.act01(w01_pe2),.act02(w02_pe2),
                        .act10(w10_pe2),.act11(w11_pe2),.act12(w12_pe2),
                        .act20(w20_pe2),.act21(w21_pe2),.act22(w22_pe2),
                        .w1_en(weight_en_2), .weight_in(weight_doutb),
                        .out(out_px2));
                        
    pe_conv CONV1_PE3 ( .clk(clk), .resetn(resetn),
                        .act00(w00_pe3),.act01(w01_pe3),.act02(w02_pe3),
                        .act10(w10_pe3),.act11(w11_pe3),.act12(w12_pe3),
                        .act20(w20_pe3),.act21(w21_pe3),.act22(w22_pe3),
                        .w1_en(weight_en_3), .weight_in(weight_doutb),
                        .out(out_px3));                    
    pe_conv CONV1_PE4 ( .clk(clk), .resetn(resetn),
                        .act00(w00_pe4),.act01(w01_pe4),.act02(w02_pe4),
                        .act10(w10_pe4),.act11(w11_pe4),.act12(w12_pe4),
                        .act20(w20_pe4),.act21(w21_pe4),.act22(w22_pe4),
                        .w1_en(weight_en_4), .weight_in(weight_doutb),
                        .out(out_px4));
    pe_conv CONV1_PE5 ( .clk(clk), .resetn(resetn),
                        .act00(w00_pe5),.act01(w01_pe5),.act02(w02_pe5),
                        .act10(w10_pe5),.act11(w11_pe5),.act12(w12_pe5),
                        .act20(w20_pe5),.act21(w21_pe5),.act22(w22_pe5),
                        .w1_en(weight_en_5), .weight_in(weight_doutb),
                        .out(out_px5));
    pe_conv CONV1_PE6 ( .clk(clk), .resetn(resetn),
                        .act00(w00_pe6),.act01(w01_pe6),.act02(w02_pe6),
                        .act10(w10_pe6),.act11(w11_pe6),.act12(w12_pe6),
                        .act20(w20_pe6),.act21(w21_pe6),.act22(w22_pe6),
                        .w1_en(weight_en_6), .weight_in(weight_doutb),
                        .out(out_px6));                        
    pe_conv CONV1_PE7 ( .clk(clk), .resetn(resetn),
                        .act00(w00_pe7),.act01(w01_pe7),.act02(w02_pe7),
                        .act10(w10_pe7),.act11(w11_pe7),.act12(w12_pe7),
                        .act20(w20_pe7),.act21(w21_pe7),.act22(w22_pe7),
                        .w1_en(weight_en_7), .weight_in(weight_doutb),
                        .out(out_px7));

//    pe_fc #(.IN_SIZE(2304)) pe_fc0 (
//        .clk(clk), .resetn(resetn), .start(fc_start),
//        .act_in(act_in), .w_in(fc_weight_0_doutb), .valid_in(fc_start),
//        .psum_out(fc_result[0]), .done(fc_done[0])
//    );
//    pe_fc #(.IN_SIZE(2304)) pe_fc1 (
//        .clk(clk), .resetn(resetn), .start(fc_start),
//        .act_in(act_in), .w_in(fc_weight_1_doutb), .valid_in(fc_start),
//        .psum_out(fc_result[1]), .done(fc_done[1])
//    );
//    pe_fc #(.IN_SIZE(2304)) pe_fc2 (
//        .clk(clk), .resetn(resetn), .start(fc_start),
//        .act_in(act_in), .w_in(fc_weight_2_doutb), .valid_in(fc_start),
//        .psum_out(fc_result[2]), .done(fc_done[2])
//    );
//    pe_fc #(.IN_SIZE(2304)) pe_fc3 (
//        .clk(clk), .resetn(resetn), .start(fc_start),
//        .act_in(act_in), .w_in(fc_weight_3_doutb), .valid_in(fc_start),
//        .psum_out(fc_result[3]), .done(fc_done[3])
//    );
//    pe_fc #(.IN_SIZE(2304)) pe_fc4 (
//        .clk(clk), .resetn(resetn), .start(fc_start),
//        .act_in(act_in), .w_in(fc_weight_4_doutb), .valid_in(fc_start),
//        .psum_out(fc_result[4]), .done(fc_done[4])
//    );
//    pe_fc #(.IN_SIZE(2304)) pe_fc5 (
//        .clk(clk), .resetn(resetn), .start(fc_start),
//        .act_in(act_in), .w_in(fc_weight_5_doutb), .valid_in(fc_start),
//        .psum_out(fc_result[5]), .done(fc_done[5])
//    );
//    pe_fc #(.IN_SIZE(2304)) pe_fc6 (
//        .clk(clk), .resetn(resetn), .start(fc_start),
//        .act_in(act_in), .w_in(fc_weight_6_doutb), .valid_in(fc_start),
//        .psum_out(fc_result[6]), .done(fc_done[6])
//    );
//    pe_fc #(.IN_SIZE(2304)) pe_fc7 (
//        .clk(clk), .resetn(resetn), .start(fc_start),
//        .act_in(act_in), .w_in(s15_doutb), .valid_in(fc_start),
//        .psum_out(fc_result[7]), .done(fc_done[7])
//    );
//    pe_fc #(.IN_SIZE(2304)) pe_fc8 (
//        .clk(clk), .resetn(resetn), .start(fc_start),
//        .act_in(act_in), .w_in(s16_doutb), .valid_in(fc_start),
//        .psum_out(fc_result[8]), .done(fc_done[8])
//    );
//    pe_fc #(.IN_SIZE(2304)) pe_fc9 (
//        .clk(clk), .resetn(resetn), .start(fc_start),
//        .act_in(act_in), .w_in(s17_doutb), .valid_in(fc_start),
//        .psum_out(fc_result[9]), .done(fc_done[9])
//    );
    
    /////////////////////////////////////////////////

    /////////////////////////////////////////////////
    
    
    ////////////////////////////// LOGIC for transfering data from TOTAL WEIGHT RAM to Corresponding Weight RAMS //////////////////////////////    
//    localparam  IDLE = 4'd0, TO_CONV1 = 4'd1, TO_CONV2 = 4'd2, 
//                TO_FC_0 = 4'd3, TO_FC_1 = 4'd4, TO_FC_2 = 4'd5, TO_FC_3 = 4'd6, TO_FC_4 = 4'd7,
//                TO_FC_5 = 4'd8, TO_FC_6 = 4'd9, TO_FC_7 = 4'd10, TO_FC_8 = 4'd11, TO_FC_9 = 4'd12,
//                TRANSFER_DONE = 4'd13;
    
//    assign conv1_weight_ena     = (weight_transfer_state == TO_CONV1);
//    assign conv1_weight_wea     = (weight_transfer_state == TO_CONV1);
//    assign conv1_weight_addra   = rx_weight_addr;
//    assign conv1_weight_dina    = rx_weight_din;
//    assign conv2_weight_ena     = (weight_transfer_state == TO_CONV2);
//    assign conv2_weight_wea     = (weight_transfer_state == TO_CONV2);
//    assign conv2_weight_addra   = rx_weight_addr;
//    assign conv2_weight_dina    = rx_weight_din;
    
//    assign fc_weight_0_ena      = (weight_transfer_state == TO_FC_0);
//    assign fc_weight_0_wea      = (weight_transfer_state == TO_FC_0);
//    assign fc_weight_0_addra    = rx_weight_addr;
//    assign fc_weight_0_dina     = rx_weight_din;
//    assign fc_weight_1_ena      = (weight_transfer_state == TO_FC_1);
//    assign fc_weight_1_wea      = (weight_transfer_state == TO_FC_1);
//    assign fc_weight_1_addra    = rx_weight_addr;
//    assign fc_weight_1_dina     = rx_weight_din;
//    assign fc_weight_2_ena      = (weight_transfer_state == TO_FC_2);
//    assign fc_weight_2_wea      = (weight_transfer_state == TO_FC_2);
//    assign fc_weight_2_addra    = rx_weight_addr;
//    assign fc_weight_2_dina     = rx_weight_din;
//    assign fc_weight_3_ena      = (weight_transfer_state == TO_FC_3);
//    assign fc_weight_3_wea      = (weight_transfer_state == TO_FC_3);
//    assign fc_weight_3_addra    = rx_weight_addr;
//    assign fc_weight_3_dina     = rx_weight_din;
//    assign fc_weight_4_ena      = (weight_transfer_state == TO_FC_4);
//    assign fc_weight_4_wea      = (weight_transfer_state == TO_FC_4);
//    assign fc_weight_4_addra    = rx_weight_addr;
//    assign fc_weight_4_dina     = rx_weight_din;
//    assign fc_weight_5_ena      = (weight_transfer_state == TO_FC_5);
//    assign fc_weight_5_wea      = (weight_transfer_state == TO_FC_5);
//    assign fc_weight_5_addra    = rx_weight_addr;
//    assign fc_weight_5_dina     = rx_weight_din;
//    assign fc_weight_6_ena      = (weight_transfer_state == TO_FC_6);
//    assign fc_weight_6_wea      = (weight_transfer_state == TO_FC_6);
//    assign fc_weight_6_addra    = rx_weight_addr;
//    assign fc_weight_6_dina     = rx_weight_din;
//    assign fc_weight_7_ena      = (weight_transfer_state == TO_FC_7);
//    assign fc_weight_7_wea      = (weight_transfer_state == TO_FC_7);
//    assign fc_weight_7_addra    = rx_weight_addr;
//    assign fc_weight_7_dina     = rx_weight_din;
//    assign fc_weight_8_ena      = (weight_transfer_state == TO_FC_8);
//    assign fc_weight_8_wea      = (weight_transfer_state == TO_FC_8);
//    assign fc_weight_8_addra    = rx_weight_addr;
//    assign fc_weight_8_dina     = rx_weight_din;
//    assign fc_weight_9_ena      = (weight_transfer_state == TO_FC_9);
//    assign fc_weight_9_wea      = (weight_transfer_state == TO_FC_9);
//    assign fc_weight_9_addra    = rx_weight_addr;
//    assign fc_weight_9_dina     = rx_weight_din;
 
    ////////////////////////////////////////////////////////// MEMORIES ////////////////////////////////////////////////////////////////////////////////
//    CONV1_WEIGHT_RAM CONV1_WEIGHT_RAM (
//      .clka (clk),       // input wire clka
//      .ena  (conv1_weight_ena),         // input wire ena
//      .wea  (conv1_weight_wea),         // input wire [3 : 0] wea
//      .addra(conv1_weight_addra),     // input wire [13 : 0] addra
//      .dina (conv1_weight_dina),       // input wire [7 : 0] dina
////      .douta(conv1_weight_douta),     // output wire [7 : 0] douta
//      .clkb (clk),                       // input wire clkb
//      .enb  (conv1_weight_enb),          // input wire enb
////      .web  (conv1_weight_we),         // input wire [3 : 0] web
//      .addrb(conv1_weight_addrb),      // input wire [13 : 0] addrb
////      .dinb (conv1_weight_din),        // input wire [7 : 0] dinb
//      .doutb(conv1_weight_doutb)       // output wire [7 : 0] doutb
//    );
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    CONV2_WEIGHT_RAM CONV2_WEIGHT_RAM (
//      .clka (clk),       // input wire clka
//      .ena  (conv2_weight_ena),         // input wire ena
//      .wea  (conv2_weight_wea),         // input wire [3 : 0] wea
//      .addra(conv2_weight_addra),     // input wire [13 : 0] addra
//      .dina (conv2_weight_dina),       // input wire [7 : 0] dina
////      .douta(conv2_weight_douta),     // output wire [7 : 0] douta
//      .clkb (clk),           // input wire clkb
//      .enb  (conv2_weight_enb),          // input wire enb
////      .web  (conv2_weight_web),         // input wire [3 : 0] web
//      .addrb(conv2_weight_addrb),      // input wire [13 : 0] addrb
////      .dinb (conv2_weight_dinb),        // input wire [7 : 0] dinb
//      .doutb(conv2_weight_doutb)       // output wire [7 : 0] doutb
//    );
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    FC_WEIGHT_RAM FC_WEIGHT_RAM_0 (
//      .clka (clk),       // input wire clka
//      .ena  (fc_weight_0_ena),         // input wire ena
//      .wea  (fc_weight_0_wea),         // input wire [3 : 0] wea
//      .addra(fc_weight_0_addra),     // input wire [13 : 0] addra
//      .dina (fc_weight_0_dina),       // input wire [7 : 0] dina
////      .douta(fc_weight_0_douta),     // output wire [7 : 0] douta
//      .clkb (clk),           // input wire clkb
//      .enb  (fc_weight_0_enb),          // input wire enb
////      .web  (fc_weight_0_web),         // input wire [3 : 0] web
//      .addrb(fc_weight_0_addrb),      // input wire [13 : 0] addrb
////      .dinb (fc_weight_0_dinb),        // input wire [7 : 0] dinb
//      .doutb(fc_weight_0_doutb)       // output wire [7 : 0] doutb
//    );
//    FC_WEIGHT_RAM FC_WEIGHT_RAM_1 (
//      .clka (clk),       // input wire clka
//      .ena  (fc_weight_1_ena),         // input wire ena
//      .wea  (fc_weight_1_wea),         // input wire [3 : 0] wea
//      .addra(fc_weight_1_addra),     // input wire [13 : 0] addra
//      .dina (fc_weight_1_dina),       // input wire [7 : 0] dina
////      .douta(fc_weight_1_douta),     // output wire [7 : 0] douta
//      .clkb (clk),           // input wire clkb
//      .enb  (fc_weight_1_enb),          // input wire enb
////      .web  (fc_weight_1_web),         // input wire [3 : 0] web
//      .addrb(fc_weight_1_addrb),      // input wire [13 : 0] addrb
////      .dinb (fc_weight_1_dinb),        // input wire [7 : 0] dinb
//      .doutb(fc_weight_1_doutb)       // output wire [7 : 0] doutb
//    );
//    FC_WEIGHT_RAM FC_WEIGHT_RAM_2 (
//      .clka (clk),       // input wire clka
//      .ena  (fc_weight_2_ena),         // input wire ena
//      .wea  (fc_weight_2_wea),         // input wire [3 : 0] wea
//      .addra(fc_weight_2_addra),     // input wire [13 : 0] addra
//      .dina (fc_weight_2_dina),       // input wire [7 : 0] dina
////      .douta(fc_weight_2_douta),     // output wire [7 : 0] douta
//      .clkb (clk),           // input wire clkb
//      .enb  (fc_weight_2_enb),          // input wire enb
////      .web  (fc_weight_2_web),         // input wire [3 : 0] web
//      .addrb(fc_weight_2_addrb),      // input wire [13 : 0] addrb
////      .dinb (fc_weight_2_dinb),        // input wire [7 : 0] dinb
//      .doutb(fc_weight_2_doutb)       // output wire [7 : 0] doutb
//    );
//    FC_WEIGHT_RAM FC_WEIGHT_RAM_3 (
//      .clka (clk),       // input wire clka
//      .ena  (fc_weight_3_ena),         // input wire ena
//      .wea  (fc_weight_3_wea),         // input wire [3 : 0] wea
//      .addra(fc_weight_3_addra),     // input wire [13 : 0] addra
//      .dina (fc_weight_3_dina),       // input wire [7 : 0] dina
////      .douta(fc_weight_3_douta),     // output wire [7 : 0] douta
//      .clkb (clk),           // input wire clkb
//      .enb  (fc_weight_3_enb),          // input wire enb
////      .web  (fc_weight_3_web),         // input wire [3 : 0] web
//      .addrb(fc_weight_3_addrb),      // input wire [13 : 0] addrb
////      .dinb (fc_weight_3_dinb),        // input wire [7 : 0] dinb
//      .doutb(fc_weight_3_doutb)       // output wire [7 : 0] doutb
//    );
//    FC_WEIGHT_RAM FC_WEIGHT_RAM_4 (
//      .clka (clk),       // input wire clka
//      .ena  (fc_weight_4_ena),         // input wire ena
//      .wea  (fc_weight_4_wea),         // input wire [3 : 0] wea
//      .addra(fc_weight_4_addra),     // input wire [13 : 0] addra
//      .dina (fc_weight_4_dina),       // input wire [7 : 0] dina
////      .douta(fc_weight_4_douta),     // output wire [7 : 0] douta
//      .clkb (clk),           // input wire clkb
//      .enb  (fc_weight_4_enb),          // input wire enb
////      .web  (fc_weight_4_web),         // input wire [3 : 0] web
//      .addrb(fc_weight_4_addrb),      // input wire [13 : 0] addrb
////      .dinb (fc_weight_4_dinb),        // input wire [7 : 0] dinb
//      .doutb(fc_weight_4_doutb)       // output wire [7 : 0] doutb
//    );
//    FC_WEIGHT_RAM FC_WEIGHT_RAM_5 (
//      .clka (clk),       // input wire clka
//      .ena  (fc_weight_5_ena),         // input wire ena
//      .wea  (fc_weight_5_wea),         // input wire [3 : 0] wea
//      .addra(fc_weight_5_addra),     // input wire [13 : 0] addra
//      .dina (fc_weight_5_dina),       // input wire [7 : 0] dina
////      .douta(fc_weight_5_douta),     // output wire [7 : 0] douta
//      .clkb (clk),           // input wire clkb
//      .enb  (fc_weight_5_enb),          // input wire enb
////      .web  (fc_weight_5_web),         // input wire [3 : 0] web
//      .addrb(fc_weight_5_addrb),      // input wire [13 : 0] addrb
////      .dinb (fc_weight_5_dinb),        // input wire [7 : 0] dinb
//      .doutb(fc_weight_5_doutb)       // output wire [7 : 0] doutb
//    );
//    FC_WEIGHT_RAM FC_WEIGHT_RAM_6 (
//      .clka (clk),       // input wire clka
//      .ena  (fc_weight_6_ena),         // input wire ena
//      .wea  (fc_weight_6_wea),         // input wire [3 : 0] wea
//      .addra(fc_weight_6_addra),     // input wire [13 : 0] addra
//      .dina (fc_weight_6_dina),       // input wire [7 : 0] dina
////      .douta(fc_weight_6_douta),     // output wire [7 : 0] douta
//      .clkb (clk),           // input wire clkb
//      .enb  (fc_weight_6_enb),          // input wire enb
////      .web  (fc_weight_6_web),         // input wire [3 : 0] web
//      .addrb(fc_weight_6_addrb),      // input wire [13 : 0] addrb
////      .dinb (fc_weight_6_dinb),        // input wire [7 : 0] dinb
//      .doutb(fc_weight_6_doutb)       // output wire [7 : 0] doutb
//    );
//    FC_WEIGHT_RAM FC_WEIGHT_RAM_7 (
//      .clka (clk),       // input wire clka
//      .ena  (fc_weight_7_ena),         // input wire ena
//      .wea  (fc_weight_7_wea),         // input wire [3 : 0] wea
//      .addra(fc_weight_7_addra),     // input wire [13 : 0] addra
//      .dina (fc_weight_7_dina),       // input wire [7 : 0] dina
////      .douta(fc_weight_7_douta),     // output wire [7 : 0] douta
//      .clkb (clk),           // input wire clkb
//      .enb  (fc_weight_7_enb),          // input wire enb
////      .web  (fc_weight_7_web),         // input wire [3 : 0] web
//      .addrb(fc_weight_7_addrb),      // input wire [13 : 0] addrb
////      .dinb (fc_weight_7_dinb),        // input wire [7 : 0] dinb
//      .doutb(fc_weight_7_doutb)       // output wire [7 : 0] doutb
//    );
//    FC_WEIGHT_RAM FC_WEIGHT_RAM_8 (
//      .clka (clk),       // input wire clka
//      .ena  (fc_weight_8_ena),         // input wire ena
//      .wea  (fc_weight_8_wea),         // input wire [3 : 0] wea
//      .addra(fc_weight_8_addra),     // input wire [13 : 0] addra
//      .dina (fc_weight_8_dina),       // input wire [7 : 0] dina
////      .douta(fc_weight_8_douta),     // output wire [7 : 0] douta
//      .clkb (clk),           // input wire clkb
//      .enb  (fc_weight_8_enb),          // input wire enb
////      .web  (fc_weight_8_web),         // input wire [3 : 0] web
//      .addrb(fc_weight_8_addrb),      // input wire [13 : 0] addrb
////      .dinb (fc_weight_8_dinb),        // input wire [7 : 0] dinb
//      .doutb(fc_weight_8_doutb)       // output wire [7 : 0] doutb
//    );
//    FC_WEIGHT_RAM FC_WEIGHT_RAM_9 (
//      .clka (clk),       // input wire clka
//      .ena  (fc_weight_9_ena),         // input wire ena
//      .wea  (fc_weight_9_wea),         // input wire [3 : 0] wea
//      .addra(fc_weight_9_addra),     // input wire [13 : 0] addra
//      .dina (fc_weight_9_dina),       // input wire [7 : 0] dina
////      .douta(fc_weight_9_douta),     // output wire [7 : 0] douta
//      .clkb (clk),           // input wire clkb
//      .enb  (fc_weight_9_enb),          // input wire enb
////      .web  (fc_weight_9_web),         // input wire [3 : 0] web
//      .addrb(fc_weight_9_addrb),      // input wire [13 : 0] addrb
////      .dinb (fc_weight_9_dinb),        // input wire [7 : 0] dinb
//      .doutb(fc_weight_9_doutb)       // output wire [7 : 0] doutb
//    );
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ACT_MEM_CONV1 ACT_MEM_0 (
      .clka (clk),
      .ena  (act_mem_0_ena),
      .wea  (act_mem_0_wea),
      .addra(act_mem_0_addra), 
      .dina (act_mem_0_dina),
//      .douta(act_mem_0_douta), 
      .clkb (clk),
      .enb  (act_mem_0_enb),
//      .web  (act_mem_0_web),         
      .addrb(act_mem_0_addrb),     
//      .dinb (act_mem_0_dinb),        
      .doutb(act_mem_0_doutb)   
    );
    ACT_MEM_CONV2 ACT_MEM_1 (
      .clka (clk),
      .ena  (act_mem_1_ena),
      .wea  (act_mem_1_wea),
      .addra(act_mem_1_addra), 
      .dina (act_mem_1_dina),
//      .douta(act_mem_1_douta), 
      .clkb (clk),
      .enb  (act_mem_1_enb),
//      .web  (act_mem_1_web),         
      .addrb(act_mem_1_addrb),     
//      .dinb (act_mem_1_dinb),        
      .doutb(act_mem_1_doutb)   
    );  
    ACT_MEM_CONV2 ACT_MEM_2 (
      .clka (clk),
      .ena  (act_mem_2_ena),
      .wea  (act_mem_2_wea),
      .addra(act_mem_2_addra), 
      .dina (act_mem_2_dina),
//      .douta(act_mem_2_douta), 
      .clkb (clk),
      .enb  (act_mem_2_enb),
//      .web  (act_mem_2_web),         
      .addrb(act_mem_2_addrb),     
//      .dinb (act_mem_2_dinb),        
      .doutb(act_mem_2_doutb)   
    );  
    ACT_MEM_CONV2 ACT_MEM_3 (
      .clka (clk),
      .ena  (act_mem_3_ena),
      .wea  (act_mem_3_wea),
      .addra(act_mem_3_addra), 
      .dina (act_mem_3_dina),
//      .douta(act_mem_3_douta), 
      .clkb (clk),
      .enb  (act_mem_3_enb),
//      .web  (act_mem_3_web),         
      .addrb(act_mem_3_addrb),     
//      .dinb (act_mem_3_dinb),        
      .doutb(act_mem_3_doutb)   
    );  
    ACT_MEM_CONV2 ACT_MEM_4 (
      .clka (clk),
      .ena  (act_mem_4_ena),
      .wea  (act_mem_4_wea),
      .addra(act_mem_4_addra), 
      .dina (act_mem_4_dina),
//      .douta(act_mem_4_douta), 
      .clkb (clk),
      .enb  (act_mem_4_enb),
//      .web  (act_mem_4_web),         
      .addrb(act_mem_4_addrb),     
//      .dinb (act_mem_4_dinb),        
      .doutb(act_mem_4_doutb)   
    );  
    ACT_MEM_CONV2 ACT_MEM_5 (
      .clka (clk),
      .ena  (act_mem_5_ena),
      .wea  (act_mem_5_wea),
      .addra(act_mem_5_addra), 
      .dina (act_mem_5_dina),
//      .douta(act_mem_5_douta), 
      .clkb (clk),
      .enb  (act_mem_5_enb),
//      .web  (act_mem_5_web),         
      .addrb(act_mem_5_addrb),     
//      .dinb (act_mem_5_dinb),        
      .doutb(act_mem_5_doutb)   
    );
    ACT_MEM_CONV2 ACT_MEM_6 (
      .clka (clk),
      .ena  (act_mem_6_ena),
      .wea  (act_mem_6_wea),
      .addra(act_mem_6_addra), 
      .dina (act_mem_6_dina),
//      .douta(act_mem_6_douta), 
      .clkb (clk),
      .enb  (act_mem_6_enb),
//      .web  (act_mem_6_web),         
      .addrb(act_mem_6_addrb),     
//      .dinb (act_mem_6_dinb),        
      .doutb(act_mem_6_doutb)   
    );  
    ACT_MEM_CONV2 ACT_MEM_7 (
      .clka (clk),
      .ena  (act_mem_7_ena),
      .wea  (act_mem_7_wea),
      .addra(act_mem_7_addra), 
      .dina (act_mem_7_dina),
//      .douta(act_mem_7_douta), 
      .clkb (clk),
      .enb  (act_mem_7_enb),
//      .web  (act_mem_7_web),         
      .addrb(act_mem_7_addrb),     
//      .dinb (act_mem_7_dinb),        
      .doutb(act_mem_7_doutb)   
    );

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FC output mem
//    OUT_MEM OUT_MEM (.clka(clk),.ena(out_mem_ena),.wea(out_mem_wea),.addra(out_mem_addra),.dina(out_mem_dina),.douta(out_mem_douta),
//                     .clkb(out_mem_clkb),.enb(out_mem_enb),.web(out_mem_web),.addrb(out_mem_addrb),.dinb(out_mem_dinb),.doutb(out_mem_doutb));
endmodule