`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/05/28 22:43:57
// Design Name: 
// Module Name: top
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


module top#(
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
    parameter ADDR_IMG  = 17,      // ASSUME Batch Size = 2; 2^10 = 1024 > 28*28 = 784 => x4 since Batch Size = 4 / => x100 since Batch Size = 100 -> 78400 < 2^17 = 131072
    parameter ADDR_W1   = 7,       // 2^7 = 128 > 1*8*9 = 72
    parameter ADDR_W2   = 11,      // 2^11 = 2048 > 8*16*9 = 1152
    parameter ADDR_WFC  = 12,      // 2^12=4096>23040               -> parallelize to 10 diff mems(23040/10 = 2304)
    parameter ADDR_F1   = 10,      // conv1 output 8*26*26 = 5408   -> parallelize to 8 diff mems (5408/8 = 676)
    parameter ADDR_F2   = 12,      // conv2 output 16*24*24 = 9216  -> parallelize to 4 diff mems (9216/4 = 2304)
    parameter ADDR_F3   = 12,      // maxpool output 16*24*24/4 = 2304
    parameter ADDR_LOG  = 11,        // output logits 2^4=16>10 => x125 since Batch Size = 125 -> 2^11 = 2048 > 1250
    
    parameter INPUT_BW = 32
    )(
    input wire clk,
    input wire resetn,
    
    input  wire start,
    output wire done,
    
    // ------------------------------------------------------------------------
    // input image memory
    // ------------------------------------------------------------------------
    // port A (PS writes image in advance), port B (PL reads)
    input wire                 input_clka,
    input wire                 input_ena,
    input wire                 input_wea,
    input wire [ADDR_IMG-1:0]  input_addra, 
    input wire [INPUT_BW-1:0]  input_dina,
    output wire [INPUT_BW-1:0] input_douta,

    // ------------------------------------------------------------------------
    // Store Total Weights
    // ------------------------------------------------------------------------
    // port A (PS writes weights in advance), port B (PL reads)
    input  wire                   tot_weight_clka,
    input  wire                   tot_weight_ena,
    input  wire                   tot_weight_wea,
    input  wire [ADDR_TOTALW-1:0] tot_weight_addra,
    input  wire [7:0]             tot_weight_dina,
    output wire [7:0]             tot_weight_douta,

    output wire done_led,

    // ------------------------------------------------------------------------
    // Store Output Logits
    // ------------------------------------------------------------------------
    // port A (PL writes output logits in advance), port B (PS reads)
    input  wire                   out_mem_0_clkb,
    input  wire                   out_mem_0_enb,
    input  wire                   out_mem_0_web,
    input  wire [ADDR_LOG-1:0]    out_mem_0_addrb,
    input  wire [INPUT_BW-1:0]    out_mem_0_dinb,
    output wire [INPUT_BW-1:0]    out_mem_0_doutb
    
    );
    
    // ------------------------------------------------------------------------
    // IF MAP & TOTAL WEIGHT MEM B-ports
    // ------------------------------------------------------------------------
    
//    wire                    input_clkb;
    wire                        input_enb;
    wire                        input_web;
    wire [ADDR_IMG-1:0]         input_addrb;
    wire signed [INPUT_BW-1:0]  input_dinb;
    wire signed [INPUT_BW-1:0]  input_doutb;
    
//    wire                    tot_weight_clkb;
    wire                    tot_weight_enb;
    wire                    tot_weight_web;
    wire [ADDR_TOTALW-1:0]  tot_weight_addrb;
    wire signed [7:0]       tot_weight_dinb;
    wire signed [7:0]       tot_weight_doutb;
    
    wire                        out_mem_0_ena;
    wire                        out_mem_0_wea;
    wire [ADDR_LOG-1:0]         out_mem_0_addra;
    wire signed [INPUT_BW-1:0]  out_mem_0_dina;
    wire signed [INPUT_BW-1:0]  out_mem_0_douta;
    
    // ------------------------------------------------------------------------
    // Multiple Done signals from Cores
    // ------------------------------------------------------------------------
    wire core0_done;
    wire core1_done;
    wire core2_done;
    wire core3_done;

    /////////////////////////////////// LOGIC for transfering data from Each Processor's OUTPUT to INPUT MEM ///////////////////////////////////
    
    /***** signals coming frome CORE0 & 1 *****/
    wire                    out_mem_0_ena_core0;
    wire                    out_mem_0_wea_core0;
    wire [ADDR_LOG-1:0]     out_mem_0_addra_core0;
    wire signed [7:0]       out_mem_0_dina_core0;

    wire                    out_mem_1_ena_core1;
    wire                    out_mem_1_wea_core1;
    wire [ADDR_LOG-1:0]     out_mem_1_addra_core1;
    wire signed [7:0]       out_mem_1_dina_core1;
    
    wire                    out_mem_2_ena_core2;
    wire                    out_mem_2_wea_core2;
    wire [ADDR_LOG-1:0]     out_mem_2_addra_core2;
    wire signed [7:0]       out_mem_2_dina_core2;

    wire                    out_mem_3_ena_core3;
    wire                    out_mem_3_wea_core3;
    wire [ADDR_LOG-1:0]     out_mem_3_addra_core3;
    wire signed [7:0]       out_mem_3_dina_core3;

    /***** Signals to put in the out_mem *****/
    
    reg [ADDR_LOG-1:0] global_out_mem_offset;
                                
    assign out_mem_0_ena = (out_mem_0_ena_core0);
//    assign out_mem_1_ena = (out_mem_1_ena_core1);
//    assign out_mem_2_ena = (out_mem_2_ena_core2);
//    assign out_mem_3_ena = (out_mem_3_ena_core3);
    
    assign out_mem_0_wea = (out_mem_0_wea_core0);
//    assign out_mem_1_wea = (out_mem_1_wea_core1);
//    assign out_mem_2_wea = (out_mem_2_wea_core2);
//    assign out_mem_3_wea = (out_mem_3_wea_core3);
    
    assign out_mem_0_addra =    out_mem_0_addra_core0 + global_out_mem_offset; 
//    assign out_mem_1_addra =    out_mem_1_addra_core1 + global_out_mem_offset; 
//    assign out_mem_2_addra =    out_mem_2_addra_core2 + global_out_mem_offset; 
//    assign out_mem_3_addra =    out_mem_3_addra_core3 + global_out_mem_offset;
                                
    assign out_mem_0_dina =     {out_mem_0_dina_core0, out_mem_1_dina_core1, out_mem_2_dina_core2, out_mem_3_dina_core3};
//    assign out_mem_1_dina =     out_mem_1_dina_core1; 
//    assign out_mem_2_dina =     out_mem_2_dina_core2; 
//    assign out_mem_3_dina =     out_mem_3_dina_core3;

    reg fc_write_done;
    
    reg [ADDR_IMG-1:0]      input_ram_addr;              // 78400 < 2^17 = 131072
    localparam              MAX_ADDR_ACT = 3136;         // batch_16: 12544 but 3136 -> for BW_32bit / batch_125: 98000
    localparam              MAX_OUT_ADDR = 40;           // batch_125:1250 
    localparam              MAX_OUT_ADDR_DIVIDED = 40;   // batch_125:1250 
    
    always@(posedge clk or negedge resetn) begin
        if(!resetn) begin
            global_out_mem_offset <= 0;
        end
        else if(start) begin
            global_out_mem_offset <= 0;
        end
        else begin
            if(out_mem_3_addra_core3 == 9) begin
                global_out_mem_offset <= global_out_mem_offset + 11'd10;
//                global_out_mem_offset <= global_out_mem_offset + 11'd40;
            end
        end
    end
    
    always@(posedge clk or negedge resetn) begin
        if(!resetn) begin
            fc_write_done <= 0;
        end
        else begin
            if(out_mem_0_addra == MAX_OUT_ADDR_DIVIDED-1)
                fc_write_done <= 1;
            else if(start) begin
                fc_write_done <= 0;
            end
            else begin
                fc_write_done <= fc_write_done;
            end
        end
    end
    
    // from fc_write done ~ start => done signal is High ---> we need to change when we want to utilize all PEs~~~~~
    assign done = (start == 1) ?  0 : fc_write_done;
    assign done_led = done;
    
    /////////////////////////////////// LOGIC for transfering data from INPUT MEM to Each Processor's ACT_MEM_0 ///////////////////////////////////
    localparam CORE_IDLE = 3'd0, TRANSFER_CORE0 = 3'd1, TRANSFER_CORE1 = 3'd2, TRANSFER_CORE2 = 3'd3, TRANSFER_CORE3 = 3'd4, IS_DONE = 3'd5, CORE_DONE = 3'd6;
    localparam              MAX_ADDR_PER_IMAGE = 784;

    reg [2:0]               core_state, n_core_state;

    reg [9:0]               act_mem_0_addr;     
    wire                    start_core0;
    wire                    start_core1;
    wire                    start_core2;
    wire                    start_core3;
    reg                     input_transfer_start;
    wire                    input_transfer_done;
    
    wire                    weight_transfer_done;
    reg                     weight_transfer_done_delay;
    
    reg [9:0] act_mem_0_addr_delay;
    reg [1:0] bw_count_delay;
    
    /***** state transition *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) core_state <= CORE_IDLE;
        else core_state <= n_core_state;
    end
    
   /***** Defines when does the state change *****/ 
   always @(*) begin
        case (core_state)
            CORE_IDLE : begin 
                if((start && (weight_transfer_done == 1)) || (weight_transfer_done & !weight_transfer_done_delay)) begin
                    n_core_state = TRANSFER_CORE0;
                end
                else n_core_state = CORE_IDLE;
            end            
            TRANSFER_CORE0 : begin
                if(act_mem_0_addr == (MAX_ADDR_PER_IMAGE-1))        // NEEDED_WEIGHT_CYCLES - 2 = 18 / NEEDED_WEIGHT_CYCLES - 1 = 19
                    n_core_state = TRANSFER_CORE1;
                else n_core_state = TRANSFER_CORE0;
            end
            TRANSFER_CORE1 : begin
                if(act_mem_0_addr == (MAX_ADDR_PER_IMAGE-1))        // MAX_WEIGHT_CYCLES - 2 = 70 / MAX_WEIGHT_CYCLES - 1 = 71
                    n_core_state = TRANSFER_CORE2;
                else n_core_state = TRANSFER_CORE1;
            end
            TRANSFER_CORE2 : begin
                if(act_mem_0_addr == (MAX_ADDR_PER_IMAGE-1))        // MAX_WEIGHT_CYCLES - 2 = 70 / MAX_WEIGHT_CYCLES - 1 = 71
                    n_core_state = TRANSFER_CORE3;
                else n_core_state = TRANSFER_CORE2;
            end
            TRANSFER_CORE3 : begin
                if(act_mem_0_addr == (MAX_ADDR_PER_IMAGE-1))        // MAX_WEIGHT_CYCLES - 2 = 70 / MAX_WEIGHT_CYCLES - 1 = 71
                    n_core_state = IS_DONE;
                else n_core_state = TRANSFER_CORE3;
            end
            IS_DONE : begin
                if(input_ram_addr >= (MAX_ADDR_ACT-1))
                    n_core_state = CORE_DONE;
                else if (out_mem_3_addra_core3 == 9) begin 
                    n_core_state = TRANSFER_CORE0;
                end
                else begin
                    n_core_state = IS_DONE;
                end
            end
            CORE_DONE : begin
                n_core_state = CORE_IDLE;
            end
            default :  n_core_state = CORE_IDLE;
        endcase
    end
    
    /***** BW Addressing Count *****/
    reg [1:0] bw_count; 
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            bw_count <= 0;
        end
        else begin
            case (core_state)
                CORE_IDLE : begin 
                    if (start) begin
//                        bw_count <= bw_count + 1;
                    end
                end            
                TRANSFER_CORE0, TRANSFER_CORE1, TRANSFER_CORE2, TRANSFER_CORE3: begin
                    bw_count <= bw_count + 1;
                end
                IS_DONE : begin
                    bw_count <= bw_count;
                end
                CORE_DONE : begin
                    bw_count <= 0;
                end
                default :  begin 
                    bw_count <= 0;
                end
            endcase
        end
    end
    
    /***** Addressing *****/ 
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            input_transfer_start <= 0;
            input_ram_addr <= 0;
            act_mem_0_addr <= 0;
        end
        else begin
            case (core_state)
                CORE_IDLE : begin 
//                    if (start) begin
//                        input_transfer_start <= 1;
//                        input_ram_addr <= input_ram_addr + 1;
//                    end
                end            
                TRANSFER_CORE0, TRANSFER_CORE1, TRANSFER_CORE2, TRANSFER_CORE3: begin
                    input_transfer_start <= 1;
                    if(bw_count == 3)
                        input_ram_addr <= input_ram_addr + 1;

                    if(act_mem_0_addr == MAX_ADDR_PER_IMAGE-1)
                        act_mem_0_addr <= 0;
                    else
                        act_mem_0_addr <= act_mem_0_addr + 1;
                end
                IS_DONE : begin
                    input_transfer_start <= 0;
                    act_mem_0_addr <= 0;
                    input_ram_addr <= input_ram_addr;
                end
                CORE_DONE : begin
                    input_transfer_start <= 0;
                    input_ram_addr <= 0;
                    act_mem_0_addr <= 0;
                end
                default :  begin 
                    input_transfer_start <= 0;
                    input_ram_addr <= 0;
                    act_mem_0_addr <= 0;
                end
            endcase
        end
    end
    
    /***** Slicing the Input to make it 8bit *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            act_mem_0_addr_delay <= 0;
            bw_count_delay <= 0;
        end
        else begin
            act_mem_0_addr_delay <= act_mem_0_addr;
            bw_count_delay <= bw_count;
        end
    end
    wire [7:0] input_slice;
    reg [7:0] input_slice_reg;
    assign input_slice =     (bw_count_delay == 0) ? input_doutb[31:24] :
                             (bw_count_delay == 1) ? input_doutb[23:16] :
                             (bw_count_delay == 2) ? input_doutb[15:8] :
                                                     input_doutb[7:0];
        
    /***** START CORE Operation when Input Feaures are loaded to CORE'S Memory *****/
    assign input_transfer_done  = (core_state == CORE_DONE);
    
    /***** Need to add code when number_p increases *****/
//    assign core0_input_en    = (core_state == TRANSFER_CORE0);
//    assign core0_input_we    = (core_state == TRANSFER_CORE0);
//    assign core1_input_en    = (core_state == TRANSFER_CORE1);
//    assign core1_input_we    = (core_state == TRANSFER_CORE1);
//    assign core2_input_en    = (core_state == TRANSFER_CORE2);
//    assign core2_input_we    = (core_state == TRANSFER_CORE2);
//    assign core3_input_en    = (core_state == TRANSFER_CORE3);
//    assign core3_input_we    = (core_state == TRANSFER_CORE3);
    
    reg core0_input_en_reg;
    reg core0_input_we_reg;
    reg core1_input_en_reg;
    reg core1_input_we_reg;
    reg core2_input_en_reg;
    reg core2_input_we_reg;
    reg core3_input_en_reg;
    reg core3_input_we_reg;
    
    assign core0_input_en = core0_input_en_reg;
    assign core0_input_we = core0_input_we_reg;
    assign core1_input_en = core1_input_en_reg;
    assign core1_input_we = core1_input_we_reg;
    assign core2_input_en = core2_input_en_reg;
    assign core2_input_we = core2_input_we_reg;
    assign core3_input_en = core3_input_en_reg;
    assign core3_input_we = core3_input_we_reg;
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            core0_input_en_reg <= 0;
            core0_input_we_reg <= 0;
            core1_input_en_reg <= 0;
            core1_input_we_reg <= 0;
            core2_input_en_reg <= 0;
            core2_input_we_reg <= 0;
            core3_input_en_reg <= 0;
            core3_input_we_reg <= 0;
        end else begin
            // inititialize to default value
            core0_input_en_reg <= 0;
            core0_input_we_reg <= 0;
            core1_input_en_reg <= 0;
            core1_input_we_reg <= 0;
            core2_input_en_reg <= 0;
            core2_input_we_reg <= 0;
            core3_input_en_reg <= 0;
            core3_input_we_reg <= 0;
    
            case (core_state)
                TRANSFER_CORE0: begin
                    core0_input_en_reg <= 1;
                    core0_input_we_reg <= 1;
                end
                TRANSFER_CORE1: begin
                    core1_input_en_reg <= 1;
                    core1_input_we_reg <= 1;
                end
                TRANSFER_CORE2: begin
                    core2_input_en_reg <= 1;
                    core2_input_we_reg <= 1;
                end
                TRANSFER_CORE3: begin
                    core3_input_en_reg <= 1;
                    core3_input_we_reg <= 1;
                end
            endcase
        end
    end
    
    assign input_enb    = (start | input_transfer_start) ? 1 : 0;
    assign input_addrb  = (start | input_transfer_start) ? input_ram_addr : 0;
    
    ////////////////////////////// LOGIC for transfering data from TOTAL WEIGHT RAM to Corresponding Weight RAMS //////////////////////////////    
    reg                     weight_transfer_start;
    reg [ADDR_TOTALW-1:0]   total_weight_mem_addr;
    reg [3:0]               weight_transfer_state, n_weight_transfer_state;
    reg [ADDR_WFC-1:0]      rx_mem_addr;

//    reg isFC;
    wire fc_trigger;
    
    localparam to_ADDR_CONV1_W = 72, to_ADDR_CONV2_W = 1224, to_ADDR_FC_W = 2304; // 1224 = 72 + 1152 
    
    localparam  IDLE = 4'd0, TO_CONV1 = 4'd1, TO_CONV2 = 4'd2, 
                TO_FC_0 = 4'd3, TO_FC_1 = 4'd4, TO_FC_2 = 4'd5, TO_FC_3 = 4'd6, TO_FC_4 = 4'd7,
                TO_FC_5 = 4'd8, TO_FC_6 = 4'd9, TO_FC_7 = 4'd10, TO_FC_8 = 4'd11, TO_FC_9 = 4'd12,
                TRANSFER_DONE = 4'd13;
                
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    always @(posedge clk or negedge resetn) begin
        if(!resetn) weight_transfer_done_delay <= 0;
        else weight_transfer_done_delay <= weight_transfer_done;
    end
    
    reg [2:0] core_state_delay;
    always @(posedge clk or negedge resetn) begin
        if(!resetn) core_state_delay <= 0;
        else core_state_delay <= core_state;
    end

    assign start_core0          = (core_state == IS_DONE) && (core_state_delay != IS_DONE); // (core_state == TRANSFER_CORE1);
    assign start_core1          = (core_state == IS_DONE) && (core_state_delay != IS_DONE); // (core_state == TRANSFER_CORE2);
    assign start_core2          = (core_state == IS_DONE) && (core_state_delay != IS_DONE); // (core_state == TRANSFER_CORE3);
    assign start_core3          = (core_state == IS_DONE) && (core_state_delay != IS_DONE);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    assign fc_trigger = (rx_mem_addr == (to_ADDR_FC_W-1));
    
    assign tot_weight_enb      = (start | weight_transfer_start);
    assign tot_weight_addrb    = total_weight_mem_addr;
    
    assign weight_transfer_done = (weight_transfer_state == TRANSFER_DONE);
    
    /***** weight_transfer_state transition *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) weight_transfer_state <= IDLE;
        else weight_transfer_state <= n_weight_transfer_state;
    end
    
    /***** transfer weights from Total Weight mem to each corresponding mems *****/    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            total_weight_mem_addr <= 0;
            weight_transfer_start <= 0;
        end
        else if (start == 1) begin
            weight_transfer_start <= 1;
            total_weight_mem_addr <= total_weight_mem_addr +1;
        end
        else if(weight_transfer_state == TRANSFER_DONE) begin
            weight_transfer_start <= 0;
        end
        else if(weight_transfer_start == 1) begin
            total_weight_mem_addr <= total_weight_mem_addr +1;
        end
        else begin
            total_weight_mem_addr <= 0;
        end
    end
    
    /***** CONV1&2 Addressing or Counter for counting 2304 to put FC weight in corresponding mem & addressing *****/
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            rx_mem_addr <= 0;
        end else begin
            if (weight_transfer_done == 1) begin
                rx_mem_addr <= 0;
            end else begin
                case (weight_transfer_state)
                    TO_FC_0, TO_FC_1, TO_FC_2, TO_FC_3, TO_FC_4, TO_FC_5, TO_FC_6, TO_FC_7, TO_FC_8, TO_FC_9: begin
                        if (rx_mem_addr == (to_ADDR_FC_W - 1))
                            rx_mem_addr <= 0;
                        else
                            rx_mem_addr <= rx_mem_addr + 1;
                    end
                    TO_CONV1: begin
                        if (rx_mem_addr == (to_ADDR_CONV1_W - 1))
                            rx_mem_addr <= 0;
                        else
                            rx_mem_addr <= rx_mem_addr + 1;
                    end
                    TO_CONV2: begin
                        if (rx_mem_addr == (to_ADDR_CONV2_W - to_ADDR_CONV1_W - 1))
                            rx_mem_addr <= 0;
                        else
                            rx_mem_addr <= rx_mem_addr + 1;
                    end
                    default: begin
                            rx_mem_addr <= 0;
                    end
                endcase
            end
        end
    end
    
   /***** Defines when does the weight_transfer_state change *****/ 
   always @(*) begin
        case (weight_transfer_state)
            IDLE : begin 
                if(start) n_weight_transfer_state = TO_CONV1;
                else n_weight_transfer_state = IDLE;
            end            
            TO_CONV1 : begin
                if(total_weight_mem_addr >= to_ADDR_CONV1_W) n_weight_transfer_state = TO_CONV2;
                else n_weight_transfer_state = TO_CONV1;
            end
            TO_CONV2 : begin
                if(total_weight_mem_addr >= to_ADDR_CONV2_W) n_weight_transfer_state = TO_FC_0;
                else n_weight_transfer_state = TO_CONV2;
            end
            TO_FC_0 : begin
                if(fc_trigger) n_weight_transfer_state = TO_FC_1;
                else n_weight_transfer_state = TO_FC_0;
            end
            TO_FC_1 : begin
                if(fc_trigger) n_weight_transfer_state = TO_FC_2;
                else n_weight_transfer_state = TO_FC_1;
            end
            TO_FC_2 : begin
                if(fc_trigger) n_weight_transfer_state = TO_FC_3;
                else n_weight_transfer_state = TO_FC_2;
            end
            TO_FC_3 : begin
                if(fc_trigger) n_weight_transfer_state = TO_FC_4;
                else n_weight_transfer_state = TO_FC_3;
            end
            TO_FC_4 : begin
                if(fc_trigger) n_weight_transfer_state = TO_FC_5;
                else n_weight_transfer_state = TO_FC_4;
            end
            TO_FC_5 : begin
                if(fc_trigger) n_weight_transfer_state = TO_FC_6;
                else n_weight_transfer_state = TO_FC_5;
            end
            TO_FC_6 : begin
                if(fc_trigger) n_weight_transfer_state = TO_FC_7;
                else n_weight_transfer_state = TO_FC_6;
            end
            TO_FC_7 : begin
                if(fc_trigger) n_weight_transfer_state = TO_FC_8;
                else n_weight_transfer_state = TO_FC_7;
            end
            TO_FC_8 : begin
                if(fc_trigger) n_weight_transfer_state = TO_FC_9;
                else n_weight_transfer_state = TO_FC_8;
            end
            TO_FC_9 : begin
                if(fc_trigger) n_weight_transfer_state = TRANSFER_DONE;
                else n_weight_transfer_state = TO_FC_9;
            end
            TRANSFER_DONE : begin
                if(~resetn) n_weight_transfer_state = IDLE;
                else n_weight_transfer_state = TRANSFER_DONE;
            end
            default :  n_weight_transfer_state = IDLE;
        endcase
    end
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ------------------------------------------------------------------------
    // Conv1 weights
    // ------------------------------------------------------------------------
//    wire                  conv1_weight_clka;
    wire                  conv1_weight_ena;
    wire                  conv1_weight_wea;
    wire [ADDR_W1-1:0]    conv1_weight_addra;
    wire signed [7:0]     conv1_weight_dina;
    wire signed [7:0]     conv1_weight_douta;

//    wire                  conv1_weight_clkb;
    wire                  conv1_weight_enb;
    wire                  conv1_weight_web;
    wire [ADDR_W1-1:0]    conv1_weight_addrb;
    wire signed [7:0]     conv1_weight_dinb;
    wire signed [7:0]     conv1_weight_doutb;
    
    // ------------------------------------------------------------------------
    // Conv2 weights
    // ------------------------------------------------------------------------
//    wire                  conv2_weight_clka;
    wire                  conv2_weight_ena;
    wire                  conv2_weight_wea;
    wire [ADDR_W2-1:0]    conv2_weight_addra;
    wire signed [7:0]     conv2_weight_dina;
    wire signed [7:0]     conv2_weight_douta;

//    wire                  conv2_weight_clkb;
    wire                  conv2_weight_enb;
    wire                  conv2_weight_web;
    wire [ADDR_W2-1:0]    conv2_weight_addrb;
    wire signed [7:0]     conv2_weight_dinb;
    wire signed [7:0]     conv2_weight_doutb;
    // ------------------------------------------------------------------------
    // FC weights (10 different memorys)
    // ------------------------------------------------------------------------
    wire                  fc_weight_0_clka,  fc_weight_1_clka,  fc_weight_2_clka,  fc_weight_3_clka,  fc_weight_4_clka,  fc_weight_5_clka,  fc_weight_6_clka,  fc_weight_7_clka,  fc_weight_8_clka,  fc_weight_9_clka;
    wire                  fc_weight_0_ena,   fc_weight_1_ena,   fc_weight_2_ena,   fc_weight_3_ena,   fc_weight_4_ena,   fc_weight_5_ena,   fc_weight_6_ena,   fc_weight_7_ena,   fc_weight_8_ena,   fc_weight_9_ena;
    wire                  fc_weight_0_wea,   fc_weight_1_wea,   fc_weight_2_wea,   fc_weight_3_wea,   fc_weight_4_wea,   fc_weight_5_wea,   fc_weight_6_wea,   fc_weight_7_wea,   fc_weight_8_wea,   fc_weight_9_wea; 
    wire [ADDR_WFC-1:0]   fc_weight_0_addra, fc_weight_1_addra, fc_weight_2_addra, fc_weight_3_addra, fc_weight_4_addra, fc_weight_5_addra, fc_weight_6_addra, fc_weight_7_addra, fc_weight_8_addra, fc_weight_9_addra; 
    wire signed [7:0]     fc_weight_0_dina,  fc_weight_1_dina,  fc_weight_2_dina,  fc_weight_3_dina,  fc_weight_4_dina,  fc_weight_5_dina,  fc_weight_6_dina,  fc_weight_7_dina,  fc_weight_8_dina,   fc_weight_9_dina;
    wire signed [7:0]     fc_weight_0_douta, fc_weight_1_douta, fc_weight_2_douta, fc_weight_3_douta, fc_weight_4_douta, fc_weight_5_douta, fc_weight_6_douta, fc_weight_7_douta, fc_weight_8_douta, fc_weight_9_douta;

    wire                  fc_weight_0_clkb,  fc_weight_1_clkb,  fc_weight_2_clkb,  fc_weight_3_clkb,  fc_weight_4_clkb,  fc_weight_5_clkb,  fc_weight_6_clkb,  fc_weight_7_clkb,  fc_weight_8_clkb,  fc_weight_9_clkb;
    wire                  fc_weight_0_enb,   fc_weight_1_enb,   fc_weight_2_enb,   fc_weight_3_enb,   fc_weight_4_enb,   fc_weight_5_enb,   fc_weight_6_enb,   fc_weight_7_enb,   fc_weight_8_enb,   fc_weight_9_enb;
    wire                  fc_weight_0_web,   fc_weight_1_web,   fc_weight_2_web,   fc_weight_3_web,   fc_weight_4_web,   fc_weight_5_web,   fc_weight_6_web,   fc_weight_7_web,   fc_weight_8_web,   fc_weight_9_web;
    wire [ADDR_WFC-1:0]   fc_weight_0_addrb, fc_weight_1_addrb, fc_weight_2_addrb, fc_weight_3_addrb, fc_weight_4_addrb, fc_weight_5_addrb, fc_weight_6_addrb, fc_weight_7_addrb, fc_weight_8_addrb, fc_weight_9_addrb; 
    wire signed [7:0]     fc_weight_0_dinb,  fc_weight_1_dinb,  fc_weight_2_dinb,  fc_weight_3_dinb,  fc_weight_4_dinb,  fc_weight_5_dinb,  fc_weight_6_dinb,  fc_weight_7_dinb,  fc_weight_8_dinb,  fc_weight_9_dinb;
    wire signed [7:0]     fc_weight_0_doutb, fc_weight_1_doutb, fc_weight_2_doutb, fc_weight_3_doutb, fc_weight_4_doutb, fc_weight_5_doutb, fc_weight_6_doutb, fc_weight_7_doutb, fc_weight_8_doutb, fc_weight_9_doutb;
    
    
    assign conv1_weight_ena     = (weight_transfer_state == TO_CONV1);
    assign conv1_weight_wea     = (weight_transfer_state == TO_CONV1);
    assign conv1_weight_addra   = rx_mem_addr;
    assign conv1_weight_dina    = tot_weight_doutb;
    assign conv2_weight_ena     = (weight_transfer_state == TO_CONV2);
    assign conv2_weight_wea     = (weight_transfer_state == TO_CONV2);
    assign conv2_weight_addra   = rx_mem_addr;
    assign conv2_weight_dina    = tot_weight_doutb;
    
    assign fc_weight_0_ena      = (weight_transfer_state == TO_FC_0);
    assign fc_weight_0_wea      = (weight_transfer_state == TO_FC_0);
    assign fc_weight_0_addra    = rx_mem_addr;
    assign fc_weight_0_dina     = tot_weight_doutb;
    assign fc_weight_1_ena      = (weight_transfer_state == TO_FC_1);
    assign fc_weight_1_wea      = (weight_transfer_state == TO_FC_1);
    assign fc_weight_1_addra    = rx_mem_addr;
    assign fc_weight_1_dina     = tot_weight_doutb;
    assign fc_weight_2_ena      = (weight_transfer_state == TO_FC_2);
    assign fc_weight_2_wea      = (weight_transfer_state == TO_FC_2);
    assign fc_weight_2_addra    = rx_mem_addr;
    assign fc_weight_2_dina     = tot_weight_doutb;
    assign fc_weight_3_ena      = (weight_transfer_state == TO_FC_3);
    assign fc_weight_3_wea      = (weight_transfer_state == TO_FC_3);
    assign fc_weight_3_addra    = rx_mem_addr;
    assign fc_weight_3_dina     = tot_weight_doutb;
    assign fc_weight_4_ena      = (weight_transfer_state == TO_FC_4);
    assign fc_weight_4_wea      = (weight_transfer_state == TO_FC_4);
    assign fc_weight_4_addra    = rx_mem_addr;
    assign fc_weight_4_dina     = tot_weight_doutb;
    assign fc_weight_5_ena      = (weight_transfer_state == TO_FC_5);
    assign fc_weight_5_wea      = (weight_transfer_state == TO_FC_5);
    assign fc_weight_5_addra    = rx_mem_addr;
    assign fc_weight_5_dina     = tot_weight_doutb;
    assign fc_weight_6_ena      = (weight_transfer_state == TO_FC_6);
    assign fc_weight_6_wea      = (weight_transfer_state == TO_FC_6);
    assign fc_weight_6_addra    = rx_mem_addr;
    assign fc_weight_6_dina     = tot_weight_doutb;
    assign fc_weight_7_ena      = (weight_transfer_state == TO_FC_7);
    assign fc_weight_7_wea      = (weight_transfer_state == TO_FC_7);
    assign fc_weight_7_addra    = rx_mem_addr;
    assign fc_weight_7_dina     = tot_weight_doutb;
    assign fc_weight_8_ena      = (weight_transfer_state == TO_FC_8);
    assign fc_weight_8_wea      = (weight_transfer_state == TO_FC_8);
    assign fc_weight_8_addra    = rx_mem_addr;
    assign fc_weight_8_dina     = tot_weight_doutb;
    assign fc_weight_9_ena      = (weight_transfer_state == TO_FC_9);
    assign fc_weight_9_wea      = (weight_transfer_state == TO_FC_9);
    assign fc_weight_9_addra    = rx_mem_addr;
    assign fc_weight_9_dina     = tot_weight_doutb;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    single_core CORE0(
        .clk(clk), .resetn(resetn), .start(start_core0),.done(core0_done),
        
//        .rx_weight_ena  (rx_mem_en),         // input wire ena
//        .rx_weight_wea  (rx_mem_we),         // input wire [3 : 0] wea

        .fc_weight_0_enb(fc_weight_0_enb),   .fc_weight_1_enb(fc_weight_1_enb),   .fc_weight_2_enb(fc_weight_2_enb),   .fc_weight_3_enb(fc_weight_3_enb),   .fc_weight_4_enb(fc_weight_4_enb),
        .fc_weight_5_enb(fc_weight_5_enb),   .fc_weight_6_enb(fc_weight_6_enb),   .fc_weight_7_enb(fc_weight_7_enb),   .fc_weight_8_enb(fc_weight_8_enb),   .fc_weight_9_enb(fc_weight_9_enb),
                 
        .fc_weight_0_addrb(fc_weight_0_addrb), .fc_weight_1_addrb(fc_weight_1_addrb), .fc_weight_2_addrb(fc_weight_2_addrb), .fc_weight_3_addrb(fc_weight_3_addrb), .fc_weight_4_addrb(fc_weight_4_addrb), 
        .fc_weight_5_addrb(fc_weight_5_addrb), .fc_weight_6_addrb(fc_weight_6_addrb), .fc_weight_7_addrb(fc_weight_7_addrb), .fc_weight_8_addrb(fc_weight_8_addrb), .fc_weight_9_addrb(fc_weight_9_addrb),
                             
        .fc_weight_0_doutb(fc_weight_0_doutb), .fc_weight_1_doutb(fc_weight_1_doutb), .fc_weight_2_doutb(fc_weight_2_doutb), .fc_weight_3_doutb(fc_weight_3_doutb), .fc_weight_4_doutb(fc_weight_4_doutb), 
        .fc_weight_5_doutb(fc_weight_5_doutb), .fc_weight_6_doutb(fc_weight_6_doutb), .fc_weight_7_doutb(fc_weight_7_doutb), .fc_weight_8_doutb(fc_weight_8_doutb), .fc_weight_9_doutb(fc_weight_9_doutb),
        
        .conv1_weight_enb(conv1_weight_enb), .conv2_weight_enb(conv2_weight_enb),
        .conv1_weight_addrb(conv1_weight_addrb), .conv2_weight_addrb(conv2_weight_addrb),
        .conv1_weight_doutb(conv1_weight_doutb), .conv2_weight_doutb(conv2_weight_doutb),

//        .rx_weight_addr(rx_mem_addr),               // input wire [13 : 0] addra
//        .rx_weight_din (tot_weight_doutb),          // input wire [7 : 0] dina
        
        .rx_input_en  (core0_input_en),
        .rx_input_we  (core0_input_we),
        .rx_input_addr(act_mem_0_addr_delay), 
        .rx_input_din (input_slice), // input_doutb
        
//        .weight_transfer_state(weight_transfer_state),
        
        .out_mem_ena(out_mem_0_ena_core0),
        .out_mem_wea(out_mem_0_wea_core0),
        .out_mem_addra(out_mem_0_addra_core0),
        .out_mem_dina(out_mem_0_dina_core0)
    );
    
    single_core CORE1(
        .clk(clk), .resetn(resetn), .start(start_core1),.done(core1_done),
        
//        .rx_weight_ena  (rx_mem_en),              // input wire ena
//        .rx_weight_wea  (rx_mem_we),              // input wire [3 : 0] wea

        /***** Don't Have to be connected since it is synchronized with CORE0 *****/
        .fc_weight_0_enb(),   .fc_weight_1_enb(),   .fc_weight_2_enb(),   .fc_weight_3_enb(),   .fc_weight_4_enb(),
        .fc_weight_5_enb(),   .fc_weight_6_enb(),   .fc_weight_7_enb(),   .fc_weight_8_enb(),   .fc_weight_9_enb(),
                 
        .fc_weight_0_addrb(), .fc_weight_1_addrb(), .fc_weight_2_addrb(), .fc_weight_3_addrb(), .fc_weight_4_addrb(), 
        .fc_weight_5_addrb(), .fc_weight_6_addrb(), .fc_weight_7_addrb(), .fc_weight_8_addrb(), .fc_weight_9_addrb(),
                             
        .fc_weight_0_doutb(fc_weight_0_doutb), .fc_weight_1_doutb(fc_weight_1_doutb), .fc_weight_2_doutb(fc_weight_2_doutb), .fc_weight_3_doutb(fc_weight_3_doutb), .fc_weight_4_doutb(fc_weight_4_doutb), 
        .fc_weight_5_doutb(fc_weight_5_doutb), .fc_weight_6_doutb(fc_weight_6_doutb), .fc_weight_7_doutb(fc_weight_7_doutb), .fc_weight_8_doutb(fc_weight_8_doutb), .fc_weight_9_doutb(fc_weight_9_doutb),
        
        .conv1_weight_enb(), .conv2_weight_enb(),
        .conv1_weight_addrb(), .conv2_weight_addrb(),
        .conv1_weight_doutb(conv1_weight_doutb), .conv2_weight_doutb(conv2_weight_doutb),

//        .rx_weight_addr(rx_mem_addr),               // input wire [13 : 0] addra
//        .rx_weight_din (tot_weight_doutb),          // input wire [7 : 0] dina
        
        .rx_input_en  (core1_input_en),
        .rx_input_we  (core1_input_we),
        .rx_input_addr(act_mem_0_addr_delay), 
        .rx_input_din (input_slice), // input_doutb
        
//        .weight_transfer_state(weight_transfer_state),
        
        .out_mem_ena(out_mem_1_ena_core1),
        .out_mem_wea(out_mem_1_wea_core1),
        .out_mem_addra(out_mem_1_addra_core1),
        .out_mem_dina(out_mem_1_dina_core1)
    );
    
    single_core CORE2(
        .clk(clk), .resetn(resetn), .start(start_core2),.done(core2_done),
        
//        .rx_weight_ena  (rx_mem_en),              // input wire ena
//        .rx_weight_wea  (rx_mem_we),              // input wire [3 : 0] wea

        /***** Don't Have to be connected since it is synchronized with CORE0 *****/
        .fc_weight_0_enb(),   .fc_weight_1_enb(),   .fc_weight_2_enb(),   .fc_weight_3_enb(),   .fc_weight_4_enb(),
        .fc_weight_5_enb(),   .fc_weight_6_enb(),   .fc_weight_7_enb(),   .fc_weight_8_enb(),   .fc_weight_9_enb(),
                 
        .fc_weight_0_addrb(), .fc_weight_1_addrb(), .fc_weight_2_addrb(), .fc_weight_3_addrb(), .fc_weight_4_addrb(), 
        .fc_weight_5_addrb(), .fc_weight_6_addrb(), .fc_weight_7_addrb(), .fc_weight_8_addrb(), .fc_weight_9_addrb(),
                             
        .fc_weight_0_doutb(fc_weight_0_doutb), .fc_weight_1_doutb(fc_weight_1_doutb), .fc_weight_2_doutb(fc_weight_2_doutb), .fc_weight_3_doutb(fc_weight_3_doutb), .fc_weight_4_doutb(fc_weight_4_doutb), 
        .fc_weight_5_doutb(fc_weight_5_doutb), .fc_weight_6_doutb(fc_weight_6_doutb), .fc_weight_7_doutb(fc_weight_7_doutb), .fc_weight_8_doutb(fc_weight_8_doutb), .fc_weight_9_doutb(fc_weight_9_doutb),
        
        .conv1_weight_enb(), .conv2_weight_enb(),
        .conv1_weight_addrb(), .conv2_weight_addrb(),
        .conv1_weight_doutb(conv1_weight_doutb), .conv2_weight_doutb(conv2_weight_doutb),

//        .rx_weight_addr(rx_mem_addr),               // input wire [13 : 0] addra
//        .rx_weight_din (tot_weight_doutb),          // input wire [7 : 0] dina
        
        .rx_input_en  (core2_input_en),
        .rx_input_we  (core2_input_we),
        .rx_input_addr(act_mem_0_addr_delay), 
        .rx_input_din (input_slice), // input_doutb
        
//        .weight_transfer_state(weight_transfer_state),
        
        .out_mem_ena(out_mem_2_ena_core2),
        .out_mem_wea(out_mem_2_wea_core2),
        .out_mem_addra(out_mem_2_addra_core2),
        .out_mem_dina(out_mem_2_dina_core2)
    );
    
    single_core CORE3(
        .clk(clk), .resetn(resetn), .start(start_core3),.done(core3_done),
        
//        .rx_weight_ena  (rx_mem_en),              // input wire ena
//        .rx_weight_wea  (rx_mem_we),              // input wire [3 : 0] wea

        /***** Don't Have to be connected since it is synchronized with CORE0 *****/
        .fc_weight_0_enb(),   .fc_weight_1_enb(),   .fc_weight_2_enb(),   .fc_weight_3_enb(),   .fc_weight_4_enb(),
        .fc_weight_5_enb(),   .fc_weight_6_enb(),   .fc_weight_7_enb(),   .fc_weight_8_enb(),   .fc_weight_9_enb(),
                 
        .fc_weight_0_addrb(), .fc_weight_1_addrb(), .fc_weight_2_addrb(), .fc_weight_3_addrb(), .fc_weight_4_addrb(), 
        .fc_weight_5_addrb(), .fc_weight_6_addrb(), .fc_weight_7_addrb(), .fc_weight_8_addrb(), .fc_weight_9_addrb(),
                             
        .fc_weight_0_doutb(fc_weight_0_doutb), .fc_weight_1_doutb(fc_weight_1_doutb), .fc_weight_2_doutb(fc_weight_2_doutb), .fc_weight_3_doutb(fc_weight_3_doutb), .fc_weight_4_doutb(fc_weight_4_doutb), 
        .fc_weight_5_doutb(fc_weight_5_doutb), .fc_weight_6_doutb(fc_weight_6_doutb), .fc_weight_7_doutb(fc_weight_7_doutb), .fc_weight_8_doutb(fc_weight_8_doutb), .fc_weight_9_doutb(fc_weight_9_doutb),
        
        .conv1_weight_enb(), .conv2_weight_enb(),
        .conv1_weight_addrb(), .conv2_weight_addrb(),
        .conv1_weight_doutb(conv1_weight_doutb), .conv2_weight_doutb(conv2_weight_doutb),

//        .rx_weight_addr(rx_mem_addr),               // input wire [13 : 0] addra
//        .rx_weight_din (tot_weight_doutb),          // input wire [7 : 0] dina
        
        .rx_input_en  (core3_input_en),
        .rx_input_we  (core3_input_we),
        .rx_input_addr(act_mem_0_addr_delay), 
        .rx_input_din (input_slice), // input_doutb

//        .weight_transfer_state(weight_transfer_state),

        .out_mem_ena(out_mem_3_ena_core3),
        .out_mem_wea(out_mem_3_wea_core3),
        .out_mem_addra(out_mem_3_addra_core3),
        .out_mem_dina(out_mem_3_dina_core3)
    );
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    INPUT_RAM INPUT_RAM (
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
    TOT_WEIGHT_MEM TOT_WEIGHT_MEM (
      .clka (tot_weight_clka),
      .ena  (tot_weight_ena),
      .wea  (tot_weight_wea),
      .addra(tot_weight_addra), 
      .dina (tot_weight_dina),
      .douta(tot_weight_douta), 
      .clkb (clk),
      .enb  (tot_weight_enb),
      .web  (tot_weight_web),         
      .addrb(tot_weight_addrb),     
      .dinb (tot_weight_dinb),        
      .doutb(tot_weight_doutb)   
    );
    ///////////////////////////////////////////////// DIVIDED WEIGHT MEMORIES //////////////////////////////////////////////////////////////////////
    CONV1_WEIGHT_RAM CONV1_WEIGHT_RAM (
      .clka (clk),       // input wire clka
      .ena  (conv1_weight_ena),         // input wire ena
      .wea  (conv1_weight_wea),         // input wire [3 : 0] wea
      .addra(conv1_weight_addra),     // input wire [13 : 0] addra
      .dina (conv1_weight_dina),       // input wire [7 : 0] dina
//      .douta(conv1_weight_douta),     // output wire [7 : 0] douta
      .clkb (clk),                       // input wire clkb
      .enb  (conv1_weight_enb),          // input wire enb
//      .web  (conv1_weight_we),         // input wire [3 : 0] web
      .addrb(conv1_weight_addrb),      // input wire [13 : 0] addrb
//      .dinb (conv1_weight_din),        // input wire [7 : 0] dinb
      .doutb(conv1_weight_doutb)       // output wire [7 : 0] doutb
    );
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    CONV2_WEIGHT_RAM CONV2_WEIGHT_RAM (
      .clka (clk),       // input wire clka
      .ena  (conv2_weight_ena),         // input wire ena
      .wea  (conv2_weight_wea),         // input wire [3 : 0] wea
      .addra(conv2_weight_addra),     // input wire [13 : 0] addra
      .dina (conv2_weight_dina),       // input wire [7 : 0] dina
//      .douta(conv2_weight_douta),     // output wire [7 : 0] douta
      .clkb (clk),           // input wire clkb
      .enb  (conv2_weight_enb),          // input wire enb
//      .web  (conv2_weight_web),         // input wire [3 : 0] web
      .addrb(conv2_weight_addrb),      // input wire [13 : 0] addrb
//      .dinb (conv2_weight_dinb),        // input wire [7 : 0] dinb
      .doutb(conv2_weight_doutb)       // output wire [7 : 0] doutb
    );
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    FC_WEIGHT_RAM FC_WEIGHT_RAM_0 (
      .clka (clk),       // input wire clka
      .ena  (fc_weight_0_ena),         // input wire ena
      .wea  (fc_weight_0_wea),         // input wire [3 : 0] wea
      .addra(fc_weight_0_addra),     // input wire [13 : 0] addra
      .dina (fc_weight_0_dina),       // input wire [7 : 0] dina
//      .douta(fc_weight_0_douta),     // output wire [7 : 0] douta
      .clkb (clk),           // input wire clkb
      .enb  (fc_weight_0_enb),          // input wire enb
//      .web  (fc_weight_0_web),         // input wire [3 : 0] web
      .addrb(fc_weight_0_addrb),      // input wire [13 : 0] addrb
//      .dinb (fc_weight_0_dinb),        // input wire [7 : 0] dinb
      .doutb(fc_weight_0_doutb)       // output wire [7 : 0] doutb
    );
    FC_WEIGHT_RAM FC_WEIGHT_RAM_1 (
      .clka (clk),       // input wire clka
      .ena  (fc_weight_1_ena),         // input wire ena
      .wea  (fc_weight_1_wea),         // input wire [3 : 0] wea
      .addra(fc_weight_1_addra),     // input wire [13 : 0] addra
      .dina (fc_weight_1_dina),       // input wire [7 : 0] dina
//      .douta(fc_weight_1_douta),     // output wire [7 : 0] douta
      .clkb (clk),           // input wire clkb
      .enb  (fc_weight_1_enb),          // input wire enb
//      .web  (fc_weight_1_web),         // input wire [3 : 0] web
      .addrb(fc_weight_1_addrb),      // input wire [13 : 0] addrb
//      .dinb (fc_weight_1_dinb),        // input wire [7 : 0] dinb
      .doutb(fc_weight_1_doutb)       // output wire [7 : 0] doutb
    );
    FC_WEIGHT_RAM FC_WEIGHT_RAM_2 (
      .clka (clk),       // input wire clka
      .ena  (fc_weight_2_ena),         // input wire ena
      .wea  (fc_weight_2_wea),         // input wire [3 : 0] wea
      .addra(fc_weight_2_addra),     // input wire [13 : 0] addra
      .dina (fc_weight_2_dina),       // input wire [7 : 0] dina
//      .douta(fc_weight_2_douta),     // output wire [7 : 0] douta
      .clkb (clk),           // input wire clkb
      .enb  (fc_weight_2_enb),          // input wire enb
//      .web  (fc_weight_2_web),         // input wire [3 : 0] web
      .addrb(fc_weight_2_addrb),      // input wire [13 : 0] addrb
//      .dinb (fc_weight_2_dinb),        // input wire [7 : 0] dinb
      .doutb(fc_weight_2_doutb)       // output wire [7 : 0] doutb
    );
    FC_WEIGHT_RAM FC_WEIGHT_RAM_3 (
      .clka (clk),       // input wire clka
      .ena  (fc_weight_3_ena),         // input wire ena
      .wea  (fc_weight_3_wea),         // input wire [3 : 0] wea
      .addra(fc_weight_3_addra),     // input wire [13 : 0] addra
      .dina (fc_weight_3_dina),       // input wire [7 : 0] dina
//      .douta(fc_weight_3_douta),     // output wire [7 : 0] douta
      .clkb (clk),           // input wire clkb
      .enb  (fc_weight_3_enb),          // input wire enb
//      .web  (fc_weight_3_web),         // input wire [3 : 0] web
      .addrb(fc_weight_3_addrb),      // input wire [13 : 0] addrb
//      .dinb (fc_weight_3_dinb),        // input wire [7 : 0] dinb
      .doutb(fc_weight_3_doutb)       // output wire [7 : 0] doutb
    );
    FC_WEIGHT_RAM FC_WEIGHT_RAM_4 (
      .clka (clk),       // input wire clka
      .ena  (fc_weight_4_ena),         // input wire ena
      .wea  (fc_weight_4_wea),         // input wire [3 : 0] wea
      .addra(fc_weight_4_addra),     // input wire [13 : 0] addra
      .dina (fc_weight_4_dina),       // input wire [7 : 0] dina
//      .douta(fc_weight_4_douta),     // output wire [7 : 0] douta
      .clkb (clk),           // input wire clkb
      .enb  (fc_weight_4_enb),          // input wire enb
//      .web  (fc_weight_4_web),         // input wire [3 : 0] web
      .addrb(fc_weight_4_addrb),      // input wire [13 : 0] addrb
//      .dinb (fc_weight_4_dinb),        // input wire [7 : 0] dinb
      .doutb(fc_weight_4_doutb)       // output wire [7 : 0] doutb
    );
    FC_WEIGHT_RAM FC_WEIGHT_RAM_5 (
      .clka (clk),       // input wire clka
      .ena  (fc_weight_5_ena),         // input wire ena
      .wea  (fc_weight_5_wea),         // input wire [3 : 0] wea
      .addra(fc_weight_5_addra),     // input wire [13 : 0] addra
      .dina (fc_weight_5_dina),       // input wire [7 : 0] dina
//      .douta(fc_weight_5_douta),     // output wire [7 : 0] douta
      .clkb (clk),           // input wire clkb
      .enb  (fc_weight_5_enb),          // input wire enb
//      .web  (fc_weight_5_web),         // input wire [3 : 0] web
      .addrb(fc_weight_5_addrb),      // input wire [13 : 0] addrb
//      .dinb (fc_weight_5_dinb),        // input wire [7 : 0] dinb
      .doutb(fc_weight_5_doutb)       // output wire [7 : 0] doutb
    );
    FC_WEIGHT_RAM FC_WEIGHT_RAM_6 (
      .clka (clk),       // input wire clka
      .ena  (fc_weight_6_ena),         // input wire ena
      .wea  (fc_weight_6_wea),         // input wire [3 : 0] wea
      .addra(fc_weight_6_addra),     // input wire [13 : 0] addra
      .dina (fc_weight_6_dina),       // input wire [7 : 0] dina
//      .douta(fc_weight_6_douta),     // output wire [7 : 0] douta
      .clkb (clk),           // input wire clkb
      .enb  (fc_weight_6_enb),          // input wire enb
//      .web  (fc_weight_6_web),         // input wire [3 : 0] web
      .addrb(fc_weight_6_addrb),      // input wire [13 : 0] addrb
//      .dinb (fc_weight_6_dinb),        // input wire [7 : 0] dinb
      .doutb(fc_weight_6_doutb)       // output wire [7 : 0] doutb
    );
    FC_WEIGHT_RAM FC_WEIGHT_RAM_7 (
      .clka (clk),       // input wire clka
      .ena  (fc_weight_7_ena),         // input wire ena
      .wea  (fc_weight_7_wea),         // input wire [3 : 0] wea
      .addra(fc_weight_7_addra),     // input wire [13 : 0] addra
      .dina (fc_weight_7_dina),       // input wire [7 : 0] dina
//      .douta(fc_weight_7_douta),     // output wire [7 : 0] douta
      .clkb (clk),           // input wire clkb
      .enb  (fc_weight_7_enb),          // input wire enb
//      .web  (fc_weight_7_web),         // input wire [3 : 0] web
      .addrb(fc_weight_7_addrb),      // input wire [13 : 0] addrb
//      .dinb (fc_weight_7_dinb),        // input wire [7 : 0] dinb
      .doutb(fc_weight_7_doutb)       // output wire [7 : 0] doutb
    );
    FC_WEIGHT_RAM FC_WEIGHT_RAM_8 (
      .clka (clk),       // input wire clka
      .ena  (fc_weight_8_ena),         // input wire ena
      .wea  (fc_weight_8_wea),         // input wire [3 : 0] wea
      .addra(fc_weight_8_addra),     // input wire [13 : 0] addra
      .dina (fc_weight_8_dina),       // input wire [7 : 0] dina
//      .douta(fc_weight_8_douta),     // output wire [7 : 0] douta
      .clkb (clk),           // input wire clkb
      .enb  (fc_weight_8_enb),          // input wire enb
//      .web  (fc_weight_8_web),         // input wire [3 : 0] web
      .addrb(fc_weight_8_addrb),      // input wire [13 : 0] addrb
//      .dinb (fc_weight_8_dinb),        // input wire [7 : 0] dinb
      .doutb(fc_weight_8_doutb)       // output wire [7 : 0] doutb
    );
    FC_WEIGHT_RAM FC_WEIGHT_RAM_9 (
      .clka (clk),       // input wire clka
      .ena  (fc_weight_9_ena),         // input wire ena
      .wea  (fc_weight_9_wea),         // input wire [3 : 0] wea
      .addra(fc_weight_9_addra),     // input wire [13 : 0] addra
      .dina (fc_weight_9_dina),       // input wire [7 : 0] dina
//      .douta(fc_weight_9_douta),     // output wire [7 : 0] douta
      .clkb (clk),           // input wire clkb
      .enb  (fc_weight_9_enb),          // input wire enb
//      .web  (fc_weight_9_web),         // input wire [3 : 0] web
      .addrb(fc_weight_9_addrb),      // input wire [13 : 0] addrb
//      .dinb (fc_weight_9_dinb),        // input wire [7 : 0] dinb
      .doutb(fc_weight_9_doutb)       // output wire [7 : 0] doutb
    );
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    OUT_MEM OUT_MEM_0 (
      .clka (clk),
      .ena  (out_mem_0_ena),
      .wea  (out_mem_0_wea),
      .addra(out_mem_0_addra), 
      .dina (out_mem_0_dina),
      .douta(out_mem_0_douta), 
      .clkb (clk),
      .enb  (out_mem_0_enb),
      .web  (out_mem_0_web),         
      .addrb(out_mem_0_addrb),     
      .dinb (out_mem_0_dinb),        
      .doutb(out_mem_0_doutb)   
    ); 
//    OUT_MEM OUT_MEM_1 (
//      .clka (clk),
//      .ena  (out_mem_1_ena),
//      .wea  (out_mem_1_wea),
//      .addra(out_mem_1_addra), 
//      .dina (out_mem_1_dina),
//      .douta(out_mem_1_douta), 
//      .clkb (clk),
//      .enb  (out_mem_1_enb),
//      .web  (out_mem_1_web),         
//      .addrb(out_mem_1_addrb),     
//      .dinb (out_mem_1_dinb),        
//      .doutb(out_mem_1_doutb)   
//    );  
//    OUT_MEM OUT_MEM_2 (
//      .clka (clk),
//      .ena  (out_mem_2_ena),
//      .wea  (out_mem_2_wea),
//      .addra(out_mem_2_addra), 
//      .dina (out_mem_2_dina),
//      .douta(out_mem_2_douta), 
//      .clkb (clk),
//      .enb  (out_mem_2_enb),
//      .web  (out_mem_2_web),         
//      .addrb(out_mem_2_addrb),     
//      .dinb (out_mem_2_dinb),        
//      .doutb(out_mem_2_doutb)   
//    );  
//    OUT_MEM OUT_MEM_3 (
//      .clka (clk),
//      .ena  (out_mem_3_ena),
//      .wea  (out_mem_3_wea),
//      .addra(out_mem_3_addra), 
//      .dina (out_mem_3_dina),
//      .douta(out_mem_3_douta), 
//      .clkb (clk),
//      .enb  (out_mem_3_enb),
//      .web  (out_mem_3_web),         
//      .addrb(out_mem_3_addrb),     
//      .dinb (out_mem_3_dinb),        
//      .doutb(out_mem_3_doutb)   
//    );  
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
endmodule
