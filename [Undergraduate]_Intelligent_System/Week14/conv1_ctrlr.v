`timescale 1ns / 1ps

module conv1_ctrlr #(
    // image dims
    parameter IMG_W    = 28,
    parameter IMG_H    = 28,
    // conv dims
    parameter CONV1_IC = 1,
    parameter CONV1_OC = 8,
    // SRAM address widths (enough to cover depth)
    parameter ADDR_IMG  = 10,      // 2^10 = 1024 > 28*28 = 784
    parameter ADDR_W1   = 7,       // 2^7 = 128 > 1*8*9 = 72
    parameter ADDR_F1   = 10       // conv1 output 8*26*26 = 5408   -> parallelize to 8 diff mems (5408/8 = 676)
)(
    input                       clk,
    input                       resetn,

    input                       start,
    output                      done,
    //----------------------------------------------------------
    // SRAM7  (Port-B : PL side, Read Only)
    output                  if_en,
    output                  if_we,
    output  [ADDR_IMG-1:0]  if_addr,
    input   signed [7:0]    if_dout,
    //----------------------------------------------------------
    // SRAM1  (Port-B : PL side, Read Only)
    output                  w1_en,
    output                  w1_we,
    output  [ADDR_W1-1:0]   w1_addr,
    //input   signed [7:0]    w1_dout,  change to top signal
    //----------------------------------------------------------
    output                  act_mem_2_0_en,
    output                  act_mem_2_0_we,
    output [ADDR_F1-1:0]    act_mem_2_addr,
    output signed [7:0]     act_mem_2_0_din, act_mem_2_1_din, act_mem_2_2_din, act_mem_2_3_din, 
                            act_mem_2_4_din, act_mem_2_5_din, act_mem_2_6_din, act_mem_2_7_din,
    
    /***** for pe *****/
    output reg signed [7:0] w00,    w01,    w02,    w10,    w11,    w12,    w20,   w21,    w22,         // pixel
    output wire             w_en_0, w_en_1, w_en_2, w_en_3, w_en_4, w_en_5, w_en_6, w_en_7,             // weight enable
    input wire signed [20:0] out_px0,                                                                   // changed
    input wire signed [20:0] out_px1,
    input wire signed [20:0] out_px2,
    input wire signed [20:0] out_px3,
    input wire signed [20:0] out_px4,
    input wire signed [20:0] out_px5,
    input wire signed [20:0] out_px6,
    input wire signed [20:0] out_px7
);

    localparam IDLE         = 3'd0;
    localparam LOAD_WEIGHTS = 3'd1;
    localparam LOAD_LBF     = 3'd2;
    localparam RUN_CONV     = 3'd3;
    localparam DONE         = 3'd4;
    
    localparam IN_MAXADDR  = 10'd784;
    localparam OUT_MAXADDR = 10'd676;

    localparam NEEDED_WEIGHT_CYCLES = 5'd16;    //72-28*2=16
    localparam MAX_WEIGHT_CYCLES    = 72;       // 8*9 = 72
    /////////////////////////////// STATE Transition Logic /////////////////////////////// 
    reg [3:0] state, n_state;
    reg [6:0] wait_cnt;             // TOTAL_WEIGHTS_TO_LOAD = 72 > 2^7 = 128
    reg [ADDR_IMG-1:0] cnt_run;
    
    reg [ADDR_F1-1:0] out_addr_reg;
    
    /***** state transition *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) state <= IDLE;
        else state <= n_state;
    end
    
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
                if(wait_cnt == NEEDED_WEIGHT_CYCLES-1)  // NEEDED_WEIGHT_CYCLES - 1 = 19
                    n_state = LOAD_LBF;
                else n_state = LOAD_WEIGHTS;
            end
            LOAD_LBF : begin
                if(wait_cnt == MAX_WEIGHT_CYCLES-1)     //  MAX_WEIGHT_CYCLES-1 - 1 = 71
                    n_state = RUN_CONV;
                else n_state = LOAD_LBF;
            end
            RUN_CONV : begin
                if(out_addr_reg >= OUT_MAXADDR-1) n_state <= DONE;        //changed
                else n_state = RUN_CONV;
            end
            DONE : begin
                n_state = IDLE;
//                if(~resetn) n_state = IDLE;
//                else n_state = DONE;
            end
            default :  n_state = IDLE;
        endcase
    end
    
    /////////////////////////////// Load Weights to each PE ///////////////////////////////
    reg [6:0]           weight_en_cnt = 0;
    reg [ADDR_W1-1:0]   weight_addr_reg, weight_addr_reg_delay;
    reg [7:0]           which_weight_en;
    reg [3:0]           weight_en_trigger_cnt;
//    reg                 weight_en_trigger;
    reg                 w1_rd_en_reg;
    
    /***** trigger signal for w_en; trigger every 9 cycles *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
//            weight_en_trigger <= 0;
            weight_en_trigger_cnt <= 0;
        end
        else begin
            case (state)         
                LOAD_WEIGHTS, LOAD_LBF: begin
                    if(weight_en_trigger_cnt == 9 - 1) begin // TOTAL_WEIGHTS_PER_PE - 1 = 8
                        weight_en_trigger_cnt <= 0;
                    end
                    else begin
//                        weight_en_trigger <= 0;
                        weight_en_trigger_cnt <= weight_en_trigger_cnt + 1;
                    end
                    
                    if(weight_en_trigger_cnt == 0) begin
//                        weight_en_trigger <= 1;
                    end
                end
                default :  begin
//                    weight_en_trigger <= 0;
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
//                    weight_en_trigger <= 0;
//                    weight_en_trigger_cnt <= 0; 
                end
            endcase
        end
    end
    
//    always @(posedge weight_en_trigger or negedge resetn) begin
//        if (!resetn) begin
//            which_weight_en <= 8'd0;
//        end
//        else begin
//            if(which_weight_en == 8'd0 || which_weight_en == 8'd1)
//                which_weight_en <= 8'b10000000;
//            else
//                which_weight_en <= (which_weight_en >> 1);
//        end
//    end
    
   /***** Increment Address to read Weight MEM & make weight read enable High *****/
   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            weight_addr_reg <= 0;
            w1_rd_en_reg <= 0;
        end
        else begin
            case (state)         
                    LOAD_WEIGHTS, LOAD_LBF: begin
                        weight_addr_reg <= weight_addr_reg + 1;
                        w1_rd_en_reg <= 1;
                    end
                    default :  begin
                        weight_addr_reg <= 0;
                        w1_rd_en_reg <= 0;
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
    
    assign {w_en_0, w_en_1, w_en_2, w_en_3, w_en_4, w_en_5, w_en_6, w_en_7} = (w1_rd_en_reg) ? which_weight_en : 0;
    
    assign w1_en = w1_rd_en_reg; //(state == LOAD_WEIGHTS | state == LOAD_LBF);
    assign w1_addr = weight_addr_reg_delay;


    /////////////////////////////// Read Activations from ACT_MEM_0 ///////////////////////////////
    reg isRunning;
    reg if_en_delay;
    
    assign if_en = isRunning;
    assign if_we = 1'b0;
    assign if_addr = (isRunning) ? cnt_run: {ADDR_IMG{1'b0}};
    
    /***** Defines what address to read *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) cnt_run <= 0;
        else begin
            if(isRunning)
                if(cnt_run == IN_MAXADDR - 1)
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
            if(wait_cnt == NEEDED_WEIGHT_CYCLES)  
                isRunning <= 1;           
            else if(state == DONE)
                isRunning <= 0;
            else
                isRunning <= isRunning;
        end
    end
   /***** 1 cycle delay Because of BRAM's 1 cycle read delay *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            if_en_delay <= 0;
        end
        else begin
            if_en_delay <= if_en;
        end
    end
    
    /////////////////////////////// Control Line Buffer & Make Window of Pixels ///////////////////////////////
   /***** Line Buffer Variables *****/
//    wire trigger_next_fifo;
    reg [4:0] addr_counter;
//    reg trigger_next_fifo_reg;
    
    wire lb_resetn = (resetn & if_en); 
    
    wire signed [7:0] r0;
    wire signed [7:0] r1;
    wire signed [7:0] r2;
    wire lb_ready_0;
    
    always @(posedge clk or negedge lb_resetn) begin
        if (!lb_resetn) begin
            addr_counter <= 0;
        end
        else if (if_en) begin
            if (addr_counter == (IMG_W - 1)) begin
                addr_counter <= 0;
            end
            else begin
                addr_counter <= addr_counter + 1;
            end
        end
    end

//    always @(posedge clk or negedge lb_resetn) begin
//        if (!lb_resetn) begin
//            addr_counter <= 0;
//            trigger_next_fifo_reg <= 0;
//        end
//        else if (if_en) begin
//            if (addr_counter == (IMG_W - 1)) begin
//                addr_counter <= 0;
//                trigger_next_fifo_reg <= 1;
//            end
//            else begin
//                addr_counter <= addr_counter + 1;
//                trigger_next_fifo_reg <= 0;
//            end
//        end
//    end
    
//    assign trigger_next_fifo = trigger_next_fifo_reg;
    
    /***** Line buffer 0 *****/
    wire [23:0] lb_data_0; ////////////////////////////////////////////////////////////////////////////////////////////////// ??? unsigned ?? ?????¡Æ? ?¡Æ???...??    
    line_buffer #(.DATA_WIDTH(8), .FIFO_DEPTH(IMG_W), .NUM_FIFO(3),. IMG_W(IMG_W)) LBUF0 (
        .clk(clk), .resetn(lb_resetn),
        .ready(lb_ready_0),
        .wren_i(if_en_delay), .rden_i(1'b1), // lb_ready_0
        .data_in(if_dout), .data_out(lb_data_0),
        .addr_counter(addr_counter),
        .wr_sel(wr_sel)
    );

    /***** assign datas to get each pixel data *****/
    // should consider 1cycle read delay from FIFO 8x128 Bram -> every edge when lb_ready is high the first element is 0
    reg lb_ready_0_delay;
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            lb_ready_0_delay <= 0;
        end
        else begin
            lb_ready_0_delay <= lb_ready_0;
        end
    end

    assign r0 = lb_ready_0_delay ? lb_data_0[7:0] : 8'b0;
    assign r1 = lb_ready_0_delay ? lb_data_0[15:8] : 8'b0;
    assign r2 = lb_ready_0_delay ? lb_data_0[23:16] : 8'b0;
                                
    /***** make 3x3 window *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            w00<=0; w01<=0; w02<=0; w10<=0; w11<=0; w12<=0; w20<=0; w21<=0; w22<=0;
        end
        else begin
        // when line buffer pops the data out
            if(lb_ready_0) begin // lb_ready_0 || lb_ready_1 || lb_ready_2
                // window shift
                {w00,w10,w20} <= {w01,w11,w21};                     // col 3, shifted
                {w01,w11,w21} <= {w02,w12,w22};                     // col 2, shifted
                {w02,w12,w22} <= {r2,r1,r0};   // col 1, sign extended, pixel elements has positive value
            end
        end
    end
    
    /////////////////////////////// Write enable for writing Output to ACT_MEMS ///////////////////////////////
    reg [1:0] local_we;
    reg [1:0] local_we_delay;
    reg [1:0] local_we_delay2;
    reg [3:0] state_delay;
    reg [3:0] state_delay2;
    
    /***** refresh address is for controlling multiple iterations to read activations *****/
//    wire refresh_address = (isRunning && (state != n_state)) ? 1: 0;
//    reg refresh_address_delay;
    
    always @(posedge clk or negedge lb_resetn) begin //  or posedge refresh_address_delay
        if(!lb_resetn) begin
            local_we <= 2;
        end
        else if ((addr_counter == IMG_W - 1) && local_we > 0) begin
            local_we <= local_we - 1;
        end 
    end 
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            local_we_delay <= 0;
            local_we_delay2 <= 0;
        end
        else begin 
            local_we_delay <= local_we;
            local_we_delay2 <= local_we_delay;
        end
    end
    
    /***** delay state for write in act_mem2 in correct timing *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            state_delay <= 0;
            state_delay2 <= 0;
        end
        else begin
            state_delay <= state;
            state_delay2 <= state_delay;
        end
    end
    
    /**************************************** Defines what data to write in SRAM2 ****************************************/
    localparam CW = $clog2(IMG_W);                          // Column Width Bit
    reg [CW-1:0] col;                                       // reg to address col
    reg [13:0] row;                                         // reg to address row
    
    reg [ADDR_F1-1:0] out_addr_reg_delay;
//    reg [ADDR_F1-1:0] out_addr_reg_delay2;
    
    reg signed [7:0]  out_din_reg0; //changed
    reg signed [7:0]  out_din_reg1;
    reg signed [7:0]  out_din_reg2;
    reg signed [7:0]  out_din_reg3;
    reg signed [7:0]  out_din_reg4;
    reg signed [7:0]  out_din_reg5;
    reg signed [7:0]  out_din_reg6;
    reg signed [7:0]  out_din_reg7;
    
    wire signed [7:0] out_px_quantized0;    //changed
    wire signed [7:0] out_px_quantized1;
    wire signed [7:0] out_px_quantized2;
    wire signed [7:0] out_px_quantized3;
    wire signed [7:0] out_px_quantized4;
    wire signed [7:0] out_px_quantized5;
    wire signed [7:0] out_px_quantized6;
    wire signed [7:0] out_px_quantized7;
    wire signed [7:0] out_relu0;      //changed
    wire signed [7:0] out_relu1;
    wire signed [7:0] out_relu2;
    wire signed [7:0] out_relu3;
    wire signed [7:0] out_relu4;
    wire signed [7:0] out_relu5;
    wire signed [7:0] out_relu6;
    wire signed [7:0] out_relu7;
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            row <= 0; 
            col <= 0;
        end
        else begin
            // coordinate of SRAM2 to write calculated data
            if(!isRunning) begin
                col <= 0;
                row <= 0;
            end
            else if(row == IMG_H - 2) begin
                row <= 0;
            end
            else if(col == IMG_W - 1) begin
                col <= 0; 
                row <= row + 1'b1;
            end 
            else if (!local_we_delay) begin
                col <= col + 1'b1;
            end
        end
    end
    
   ///////////////////////////////// FOR TIMING CONSTRAINTS ///////////////////////////////////
    reg [ADDR_F1-1:0] col_delay, col_delay2, col_delay3, col_delay4, col_delay5, col_delay6;
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            col_delay <= 0;
            col_delay2 <= 0;
            col_delay3 <= 0;
            col_delay4 <= 0;
            col_delay5 <= 0;
            col_delay6 <= 0;
        end
        else begin
            col_delay <= col;
            col_delay2 <= col_delay;
            col_delay3 <= col_delay2;
            col_delay4 <= col_delay3;
            col_delay5 <= col_delay4;
            col_delay6 <= col_delay5;
        end
    end
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            out_addr_reg <= 0; 
            out_din_reg0 <= 0;
            out_din_reg1 <= 0;
            out_din_reg2 <= 0;
            out_din_reg3 <= 0;
            out_din_reg4 <= 0;
            out_din_reg5 <= 0;
            out_din_reg6 <= 0;
            out_din_reg7 <= 0;
        end
        else if(start) begin
            out_addr_reg <= 0; 
            out_din_reg0 <= 0;
            out_din_reg1 <= 0;
            out_din_reg2 <= 0;
            out_din_reg3 <= 0;
            out_din_reg4 <= 0;
            out_din_reg5 <= 0;
            out_din_reg6 <= 0;
            out_din_reg7 <= 0;
        end
        // if coordinate is in the correct region: Output MEM Write
        // when row is changed, column index should be delayed by 2 (since, window shifting)
        else if(out_addr_reg >= OUT_MAXADDR - 1) begin
            out_addr_reg <= 0;
        end
        else if(col_delay6 >= 2) begin
            out_addr_reg <= out_addr_reg + 1'b1;
            out_din_reg0  <= out_relu0;
            out_din_reg1  <= out_relu1;
            out_din_reg2  <= out_relu2;
            out_din_reg3  <= out_relu3;
            out_din_reg4  <= out_relu4;
            out_din_reg5  <= out_relu5;
            out_din_reg6  <= out_relu6;
            out_din_reg7  <= out_relu7;
        end
        else begin
            out_addr_reg <= out_addr_reg;
            out_din_reg0  <= out_din_reg0;
            out_din_reg1  <= out_din_reg1;
            out_din_reg2  <= out_din_reg2;
            out_din_reg3  <= out_din_reg3;
            out_din_reg4  <= out_din_reg4;
            out_din_reg5  <= out_din_reg5;
            out_din_reg6  <= out_din_reg6;
            out_din_reg7  <= out_din_reg7;
        end
    end
    
    // 1 cycle delay for synchronizing the timing of data & address for writing
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            out_addr_reg_delay <= 0;
        end
        else begin
            out_addr_reg_delay <= out_addr_reg;
        end
    end
    
    ///////////////////////////////// FOR TIMING CONSTRAINTS ///////////////////////////////////
//   reg [ADDR_F1-1:0] out_addr_reg_delay2;
//   reg [ADDR_F1-1:0] out_addr_reg_delay3;
//    always @(posedge clk or negedge resetn) begin
//        if(!resetn) begin
//            out_addr_reg_delay2 <= 0;
//            out_addr_reg_delay3 <= 0;
//        end
//        else begin
//            out_addr_reg_delay2 <= out_addr_reg_delay;
//            out_addr_reg_delay3 <= out_addr_reg_delay2;
//        end
//    end
    
    reg act_mem_2_0_en_reg, act_mem_2_0_we_reg;

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            act_mem_2_0_en_reg <= 1'b0;
            act_mem_2_0_we_reg <= 1'b0;
        end else begin
            if ((state_delay2 == LOAD_LBF || state_delay2 == RUN_CONV) && !local_we_delay2) begin
                act_mem_2_0_en_reg <= 1'b1;
                act_mem_2_0_we_reg <= 1'b1;
            end 
            else begin
                act_mem_2_0_en_reg <= 1'b0;
                act_mem_2_0_we_reg <= 1'b0;
            end
        end
    end

//    reg signed [7:0]  out_din_reg0_delay, out_din_reg1_delay, out_din_reg2_delay, out_din_reg3_delay, out_din_reg4_delay, out_din_reg5_delay, out_din_reg6_delay, out_din_reg7_delay;
//    always @(posedge clk or negedge resetn) begin
//        if(!resetn) begin
//            out_addr_reg_delay2 <= 0;
//            out_din_reg0_delay <= 0; out_din_reg1_delay <= 0; out_din_reg2_delay <= 0; out_din_reg3_delay <= 0; 
//            out_din_reg4_delay <= 0; out_din_reg5_delay <= 0; out_din_reg6_delay <= 0; out_din_reg7_delay <= 0;
//        end
//        else begin
//            out_addr_reg_delay2 <= out_addr_reg_delay;
//            out_din_reg0_delay <= out_din_reg0; out_din_reg1_delay <= out_din_reg1; out_din_reg2_delay <= out_din_reg2; out_din_reg3_delay <= out_din_reg3; 
//            out_din_reg4_delay <= out_din_reg4; out_din_reg5_delay <= out_din_reg5; out_din_reg6_delay <= out_din_reg6; out_din_reg7_delay <= out_din_reg7;
//        end
//    end
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    assign act_mem_2_0_en = act_mem_2_0_en_reg;
    assign act_mem_2_0_we = act_mem_2_0_we_reg;
//    assign act_mem_2_0_en = ((state_delay2 == LOAD_LBF || state_delay2 == RUN_CONV) && !local_we_delay2)? 1'b1 : 1'b0;  //changed
//    assign act_mem_2_0_we = ((state_delay2 == LOAD_LBF || state_delay2 == RUN_CONV) && !local_we_delay2)? 1'b1 : 1'b0;

    assign act_mem_2_addr = out_addr_reg_delay;
    assign act_mem_2_0_din = out_din_reg0;
    assign act_mem_2_1_din = out_din_reg1;
    assign act_mem_2_2_din = out_din_reg2;
    assign act_mem_2_3_din = out_din_reg3;
    assign act_mem_2_4_din = out_din_reg4;
    assign act_mem_2_5_din = out_din_reg5;
    assign act_mem_2_6_din = out_din_reg6;
    assign act_mem_2_7_din = out_din_reg7;
//    assign act_mem_2_addr = out_addr_reg_delay2;
//    assign act_mem_2_0_din = out_din_reg0_delay;
//    assign act_mem_2_1_din = out_din_reg1_delay;
//    assign act_mem_2_2_din = out_din_reg2_delay;
//    assign act_mem_2_3_din = out_din_reg3_delay;
//    assign act_mem_2_4_din = out_din_reg4_delay;
//    assign act_mem_2_5_din = out_din_reg5_delay;
//    assign act_mem_2_6_din = out_din_reg6_delay;
//    assign act_mem_2_7_din = out_din_reg7_delay;
    
   /************************************************************************************************************************/

    assign done = (state == DONE);
    
    // Quantization
    quantize #(
        .INPUT_DATA_WIDTH(21),
        .OUTPUT_DATA_WIDTH(8)
    ) u_quant0 (
        .in_data(out_px0),
        .out_data(out_px_quantized0)
    );
    quantize #(
        .INPUT_DATA_WIDTH(21),
        .OUTPUT_DATA_WIDTH(8)
    ) u_quant1 (
        .in_data(out_px1),
        .out_data(out_px_quantized1)
    );
    quantize #(
        .INPUT_DATA_WIDTH(21),
        .OUTPUT_DATA_WIDTH(8)
    ) u_quant2 (
        .in_data(out_px2),
        .out_data(out_px_quantized2)
    );
    quantize #(
        .INPUT_DATA_WIDTH(21),
        .OUTPUT_DATA_WIDTH(8)
    ) u_quant3 (
        .in_data(out_px3),
        .out_data(out_px_quantized3)
    );
    quantize #(
        .INPUT_DATA_WIDTH(21),
        .OUTPUT_DATA_WIDTH(8)
    ) u_quant4 (
        .in_data(out_px4),
        .out_data(out_px_quantized4)
    );
    quantize #(
        .INPUT_DATA_WIDTH(21),
        .OUTPUT_DATA_WIDTH(8)
    ) u_quant5 (
        .in_data(out_px5),
        .out_data(out_px_quantized5)
    );
    quantize #(
        .INPUT_DATA_WIDTH(21),
        .OUTPUT_DATA_WIDTH(8)
    ) u_quan6 (
        .in_data(out_px6),
        .out_data(out_px_quantized6)
    );
    quantize #(
        .INPUT_DATA_WIDTH(21),
        .OUTPUT_DATA_WIDTH(8)
    ) u_quant7 (
        .in_data(out_px7),
        .out_data(out_px_quantized7)
    );
    
    // ReLU
    relu #(
        .DATA_WIDTH(8)
    ) u_relu0 (
        .in_data(out_px_quantized0),
        .out_data(out_relu0)
    );
    relu #(
        .DATA_WIDTH(8)
    ) u_relu1 (
        .in_data(out_px_quantized1),
        .out_data(out_relu1)
    );
    relu #(
        .DATA_WIDTH(8)
    ) u_relu2 (
        .in_data(out_px_quantized2),
        .out_data(out_relu2)
    );
    relu #(
        .DATA_WIDTH(8)
    ) u_relu3 (
        .in_data(out_px_quantized3),
        .out_data(out_relu3)
    );
    relu #(
        .DATA_WIDTH(8)
    ) u_relu4 (
        .in_data(out_px_quantized4),
        .out_data(out_relu4)
    );
    relu #(
        .DATA_WIDTH(8)
    ) u_relu5 (
        .in_data(out_px_quantized5),
        .out_data(out_relu5)
    );
    relu #(
        .DATA_WIDTH(8)
    ) u_relu6 (
        .in_data(out_px_quantized6),
        .out_data(out_relu6)
    );
    relu #(
        .DATA_WIDTH(8)
    ) u_relu7 (
        .in_data(out_px_quantized7),
        .out_data(out_relu7)
    );
    

endmodule