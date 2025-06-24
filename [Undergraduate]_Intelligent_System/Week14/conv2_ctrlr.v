`timescale 1ns / 1ps

module conv2_ctrlr #(
    // Input Feature Map (IFM) dimensions (output of Conv1)
    parameter IFM_W    = 26,
    parameter IFM_H    = 26,
    // Conv2 dimensions
    parameter CONV2_IC = 8,  // Number of input channels
    parameter CONV2_OC = 16, // Number of output channels
    parameter K_W      = 3,  // Kernel Width
    parameter K_H      = 3,  // Kernel Height
    // SRAM address widths
    parameter ADDR_IFM  = 10,      // Address for one of the CONV2_IC IFM memories (26*26 = 676 < 2^10)
    parameter ADDR_W2   = 11,      // Address for CONV2_OC * CONV2_IC * K_H * K_W weights (16*8*9 = 1152 < 2^11)
    parameter ADDR_OFM  = 10       // Address for one of the CONV2_OC Output FM (OFM) memories (24*24 = 576 < 2^10)
)(
    input                       clk,
    input                       resetn,

    input                       start,          // Start signal for Conv2 operation
    output                      done,           // Conv2 operation finished
//    input                       conv1_done,     // Signal indicating Conv1 has finished writing its output (IFM for Conv2)

    //----------------------------------------------------------
    // Input Feature Map (IFM) Read Interface (from 8 parallel memories, output of Conv1)
    // Assuming common enable and address for all input channels
    output                  ifm_rd_en,      // Read enable for IFM SRAMs
    output  [ADDR_IFM-1:0]  ifm_rd_addr,    // Read address for IFM SRAMs
    
    input   signed [7:0]    ifm_dout_ch0,   // Data from IFM SRAM channel 0
    input   signed [7:0]    ifm_dout_ch1,
    input   signed [7:0]    ifm_dout_ch2,
    input   signed [7:0]    ifm_dout_ch3,
    input   signed [7:0]    ifm_dout_ch4,
    input   signed [7:0]    ifm_dout_ch5,
    input   signed [7:0]    ifm_dout_ch6,
    input   signed [7:0]    ifm_dout_ch7,
    //----------------------------------------------------------
    // Conv2 Weights Read Interface
    output                  w2_rd_en,       // Read enable for Conv2 weight memory
    output  [ADDR_W2-1:0]   w2_rd_addr,     // Read address for Conv2 weight memory
    // conv2_weight_dout_from_top is implicitly used by the PEs, driven by top module
    //----------------------------------------------------------
//    // Output Feature Map (OFM) Write Interface (to 16 parallel memories)
//    output                  ofm_wr_en,      // Write enable for OFM SRAMs
//    output  [ADDR_OFM-1:0]  ofm_wr_addr,    // Write address for OFM SRAMs
//    output  signed [7:0]    ofm_dout,    // Data to OFM SRAM channel 0
    //----------------------------------------------------------
    // Interface to 8 PEs (Processing Elements)
    // Output: 3x3 windows for each of the 8 input channels
    output reg signed [7:0] w00_ic0, w01_ic0, w02_ic0, w10_ic0, w11_ic0, w12_ic0, w20_ic0, w21_ic0, w22_ic0,
    output reg signed [7:0] w00_ic1, w01_ic1, w02_ic1, w10_ic1, w11_ic1, w12_ic1, w20_ic1, w21_ic1, w22_ic1,
    output reg signed [7:0] w00_ic2, w01_ic2, w02_ic2, w10_ic2, w11_ic2, w12_ic2, w20_ic2, w21_ic2, w22_ic2,
    output reg signed [7:0] w00_ic3, w01_ic3, w02_ic3, w10_ic3, w11_ic3, w12_ic3, w20_ic3, w21_ic3, w22_ic3,
    output reg signed [7:0] w00_ic4, w01_ic4, w02_ic4, w10_ic4, w11_ic4, w12_ic4, w20_ic4, w21_ic4, w22_ic4,
    output reg signed [7:0] w00_ic5, w01_ic5, w02_ic5, w10_ic5, w11_ic5, w12_ic5, w20_ic5, w21_ic5, w22_ic5,
    output reg signed [7:0] w00_ic6, w01_ic6, w02_ic6, w10_ic6, w11_ic6, w12_ic6, w20_ic6, w21_ic6, w22_ic6,
    output reg signed [7:0] w00_ic7, w01_ic7, w02_ic7, w10_ic7, w11_ic7, w12_ic7, w20_ic7, w21_ic7, w22_ic7,

    // Output: Weight enable signals for the 8 PEs
    output wire             w_en_pe0, w_en_pe1, w_en_pe2,  w_en_pe3,  w_en_pe4,  w_en_pe5,  w_en_pe6,  w_en_pe7,

    // Input: Raw accumulated output from the 8 PEs
    input wire signed [20:0] out_px0, out_px1, out_px2,  out_px3,  out_px4,  out_px5,  out_px6,  out_px7,
    
    output wire [4:0] col_out,
    output wire [9:0] out_addr // OFM_MAX_ADDR_LOG = 10
);
    localparam CONV2_OFM_W = IFM_W - K_W + 1; // Output Feature Map Width = 26 - 3 + 1 = 24
    localparam CONV2_OFM_H = IFM_H - K_H + 1; // Output Feature Map Height = 26 - 3 + 1 = 24
    
    localparam IFM_MAX_ADDR  = IFM_W * IFM_H;                                       // 26 * 26 = 676
    localparam OFM_MAX_ADDR  = CONV2_OFM_W * CONV2_OFM_H;                           // 24 * 24 = 576
    localparam TOTAL_WEIGHTS_PER_PE = K_H * K_W;                                    // 3 * 3 = 9 
    localparam TOTAL_WEIGHTS_CONV2  = CONV2_OC * CONV2_IC * TOTAL_WEIGHTS_PER_PE;   // 16 * 8 * 9 = 1152

    // Weight loading delay before computation starts full swing
    // Adjusted to start processing once first PE's weights are loaded and enough data is in line buffers
    // Cycles for every PE to load its weights
    localparam NEEDED_WEIGHT_CYCLES = CONV2_IC * TOTAL_WEIGHTS_PER_PE - IFM_W*2; // 8*9 - 26*2 = 20
    localparam MAX_WEIGHT_CYCLES     = CONV2_IC * TOTAL_WEIGHTS_PER_PE;           // 8*9 = 72
    localparam NUM_ITER = 16;
    
    /////////////////////////////// STATE Transition Logic /////////////////////////////// 
    localparam IDLE         = 3'd0;
    localparam LOAD_WEIGHTS = 3'd1;
    localparam LOAD_LBF     = 3'd2;
    localparam RUN_CONV     = 3'd3;
    localparam IS_DONE      = 3'd4;
    localparam DONE         = 3'd5;
    
    reg [2:0] state, n_state;
    reg [6:0] wait_cnt;             // TOTAL_WEIGHTS_TO_LOAD per itearation = 72 > 2^7 = 128
    reg [3:0] num_iteration;        // Need to iterate IFM 16 times
    reg [ADDR_IFM-1:0] cnt_run;
    
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
    
    /***** counter for cnt_run; IFM addr *****/
//    always @(posedge clk or negedge resetn) begin
//        if(~resetn) 
//            cnt_run <= 0;
//        else begin
//           case (state)     
//                LOAD_LBF, RUN_CONV : begin
//                    cnt_run <= cnt_run + 1;
//                end
//                default : cnt_run <= 0;
//            endcase
//        end
//    end
    
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
            if(wait_cnt == NEEDED_WEIGHT_CYCLES-1)  // NEEDED_WEIGHT_CYCLES - 2 = 18 / NEEDED_WEIGHT_CYCLES - 1 = 19
                n_state = LOAD_LBF;
            else n_state = LOAD_WEIGHTS;
        end
        LOAD_LBF : begin
            if(wait_cnt == MAX_WEIGHT_CYCLES-1)     // MAX_WEIGHT_CYCLES - 2 = 70 / MAX_WEIGHT_CYCLES - 1 = 71
                n_state = RUN_CONV;
            else n_state = LOAD_LBF;
        end
        RUN_CONV : begin
            if(cnt_run == IFM_MAX_ADDR - 1) begin
                n_state = IS_DONE;
            end
            else n_state = RUN_CONV;
        end
        IS_DONE : begin
            if(num_iteration == (NUM_ITER - 1))                    
                n_state = DONE;
            else
                n_state = LOAD_WEIGHTS;                
        end
        DONE : begin
            n_state = IDLE;
        //                if(~resetn) n_state = IDLE;
        //                else n_state = DONE;
        end
        default :  n_state = IDLE;
        endcase
    end
//   always @(*) begin
//        case (state)
//            IDLE : begin 
//                if(start) begin
//                    n_state <= LOAD_WEIGHTS;
//                end               
//                else n_state <= IDLE;
//            end            
//            LOAD_WEIGHTS : begin
//                if(wait_cnt == NEEDED_WEIGHT_CYCLES-1)  // NEEDED_WEIGHT_CYCLES - 2 = 18 / NEEDED_WEIGHT_CYCLES - 1 = 19
//                    n_state <= LOAD_LBF;
//                else n_state <= LOAD_WEIGHTS;
//            end
//            LOAD_LBF : begin
//                if(wait_cnt == MAX_WEIGHT_CYCLES-1)     // MAX_WEIGHT_CYCLES - 2 = 70 / MAX_WEIGHT_CYCLES - 1 = 71
//                    n_state <= RUN_CONV;
//                else n_state <= LOAD_LBF;
//            end
//            RUN_CONV : begin
//                if(cnt_run == IFM_MAX_ADDR - 1) begin
//                    n_state <= IS_DONE;
//                end
//                else n_state <= RUN_CONV;
//            end
//            IS_DONE : begin
//                if(num_iteration == (NUM_ITER - 1))                    
//                    n_state <= DONE;
//                else
//                    n_state <= LOAD_WEIGHTS;                
//            end
//            DONE : begin
//                n_state <= IDLE;
////                if(~resetn) n_state = IDLE;
////                else n_state = DONE;
//            end
//            default :  n_state <= IDLE;
//        endcase
//    end
    
   /***** count iterations for Multi Channel Multi Kernels *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            num_iteration <= 0;
        end
        else begin
            case (state)         
                IS_DONE : begin
                    num_iteration <= num_iteration + 1;
                end
                default :  begin
                    num_iteration <= num_iteration;
                end
            endcase
        end
    end
    
    /////////////////////////////// Load Weights to each PE ///////////////////////////////
    reg [6:0]           weight_en_cnt = 0;
    reg [ADDR_W2-1:0]   weight_addr_reg, weight_addr_reg_delay;
    reg [7:0]           which_weight_en;
    reg [3:0]           weight_en_trigger_cnt;
//    reg                 weight_en_trigger;
    reg                 w2_rd_en_reg;
    
    
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
//    always @(posedge clk or negedge resetn) begin
//        if(!resetn) begin
////            weight_en_trigger <= 0;
//            weight_en_trigger_cnt <= 0;
//        end
//        else begin
//            case (state)         
//                LOAD_WEIGHTS, LOAD_LBF: begin
//                    if(weight_en_trigger_cnt == TOTAL_WEIGHTS_PER_PE - 1) begin // TOTAL_WEIGHTS_PER_PE - 1 = 8
//                        weight_en_trigger_cnt <= 0;
//                    end
//                    else begin
////                        weight_en_trigger <= 0;
//                        weight_en_trigger_cnt <= weight_en_trigger_cnt + 1;
//                    end
                    
////                    if(weight_en_trigger_cnt == 0) begin
////                        weight_en_trigger <= 1;
////                    end
//                end
//                default :  begin
////                    weight_en_trigger <= 0;
//                    weight_en_trigger_cnt <= 0; 
//                end
//            endcase
//        end
//    end
    
//    /**** shift which_weight_en to figure out which weight load needed *****/
//    always @(posedge clk or negedge resetn) begin
//        if (!resetn) begin
//            which_weight_en <= 8'd0;
//        end
//        else begin
//            if((weight_en_trigger_cnt == 0) && (which_weight_en == 8'd0 || which_weight_en == 8'd1))
//                which_weight_en <= 8'b10000000;
//            else
//                which_weight_en <= (which_weight_en >> 1);
//        end
//    end
    
   /***** Increment Address to read Weight MEM & make weight read enable High *****/
   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            weight_addr_reg <= 0;
            w2_rd_en_reg <= 0;
        end
        else begin
            case (state)         
                    LOAD_WEIGHTS, LOAD_LBF: begin
                        weight_addr_reg <= weight_addr_reg + 1;
                        w2_rd_en_reg <= 1;
                    end
                    DONE : begin
                        weight_addr_reg <= 0;
                        w2_rd_en_reg <= 0;
                    end
                    default :  begin
                        weight_addr_reg <= weight_addr_reg;
                        w2_rd_en_reg <= 0;
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
    
    assign {w_en_pe0, w_en_pe1, w_en_pe2,  w_en_pe3,  w_en_pe4,  w_en_pe5,  w_en_pe6,  w_en_pe7} = (w2_rd_en_reg) ? which_weight_en : 0;
    
    assign w2_rd_en = w2_rd_en_reg; //(state == LOAD_WEIGHTS | state == LOAD_LBF);
    assign w2_rd_addr = weight_addr_reg_delay;

    /////////////////////////////// Read Activations from ACT_MEM_0 ~ 7 ///////////////////////////////
    reg isRunning;
    reg ifm_rd_en_delay;
    
    assign ifm_rd_en = isRunning;
    assign ifm_rd_we = 1'b0;
    assign ifm_rd_addr = (isRunning) ? cnt_run: {ADDR_IFM{1'b0}};
    
    /***** Defines what address to read *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) cnt_run <= 0;
        else begin
            if(isRunning)
                if(cnt_run == IFM_MAX_ADDR - 1)
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
    /***** Line Buffer Variables *****/
//    wire trigger_next_fifo;
    reg [4:0] addr_counter;
//    reg trigger_next_fifo_reg;
    
    wire lb_resetn = (resetn & ifm_rd_en);

    always @(posedge clk or negedge lb_resetn) begin
        if (!lb_resetn) begin
            addr_counter <= 0;
        end
        else if (ifm_rd_en) begin
            if (addr_counter == (IFM_W - 1)) begin
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
//        else if (ifm_rd_en) begin
//            if (addr_counter == (IFM_W - 1)) begin
//                addr_counter <= 0;
//                trigger_next_fifo_reg <= 1;
//            end
//            else begin
//                addr_counter <= addr_counter + 1;
//                trigger_next_fifo_reg <= 0;
//            end
//        end
//        else begin
//            addr_counter <= 0;
//            trigger_next_fifo_reg <= 0;
//        end
//    end
    
//    assign trigger_next_fifo = trigger_next_fifo_reg;
    
    /***** Line buffer 0 *****/
    wire [23:0] lb_data_0;
    wire lb_ready_0;
    line_buffer #(.DATA_WIDTH(8), .FIFO_DEPTH(IFM_W), .NUM_FIFO(3), .IMG_W(IFM_W)) LBUF1 (
        .clk(clk), .resetn(lb_resetn),
        .ready(lb_ready_0),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_dout_ch0), .data_out(lb_data_0),
        .addr_counter(addr_counter),
//        .refresh_n(ifm_rd_en_delay),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 1 *****/
    wire [23:0] lb_data_1;
    wire lb_ready_1;
    line_buffer #(.DATA_WIDTH(8), .FIFO_DEPTH(IFM_W), .NUM_FIFO(3), .IMG_W(IFM_W)) LBUF2 (
        .clk(clk), .resetn(lb_resetn),
        .ready(lb_ready_1),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_dout_ch1), .data_out(lb_data_1),
        .addr_counter(addr_counter),
//        .refresh_n(ifm_rd_en_delay),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 2 *****/
    wire [23:0] lb_data_2;
    wire lb_ready_2;
    line_buffer #(.DATA_WIDTH(8), .FIFO_DEPTH(IFM_W), .NUM_FIFO(3), .IMG_W(IFM_W)) LBUF3 (
        .clk(clk), .resetn(lb_resetn),
        .ready(lb_ready_2),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_dout_ch2), .data_out(lb_data_2),
        .addr_counter(addr_counter),
//        .refresh_n(ifm_rd_en_delay),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 3 *****/
    wire [23:0] lb_data_3;
    wire lb_ready_3;  
    line_buffer #(.DATA_WIDTH(8), .FIFO_DEPTH(IFM_W), .NUM_FIFO(3), .IMG_W(IFM_W)) LBUF4 (
        .clk(clk), .resetn(lb_resetn),
        .ready(lb_ready_3),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_dout_ch3), .data_out(lb_data_3),
        .addr_counter(addr_counter),
//        .refresh_n(ifm_rd_en_delay),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 4 *****/
    wire [23:0] lb_data_4;
    wire lb_ready_4;
    line_buffer #(.DATA_WIDTH(8), .FIFO_DEPTH(IFM_W), .NUM_FIFO(3), .IMG_W(IFM_W)) LBUF5 (
        .clk(clk), .resetn(lb_resetn),
        .ready(lb_ready_4),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_dout_ch4), .data_out(lb_data_4),
        .addr_counter(addr_counter),
//        .refresh_n(ifm_rd_en_delay),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 5 *****/
    wire [23:0] lb_data_5;
    wire lb_ready_5;
    line_buffer #(.DATA_WIDTH(8), .FIFO_DEPTH(IFM_W), .NUM_FIFO(3), .IMG_W(IFM_W)) LBUF6 (
        .clk(clk), .resetn(lb_resetn),
        .ready(lb_ready_5),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_dout_ch5), .data_out(lb_data_5),
        .addr_counter(addr_counter),
//        .refresh_n(ifm_rd_en_delay),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 6 *****/
    wire [23:0] lb_data_6;
    wire lb_ready_6;
    line_buffer #(.DATA_WIDTH(8), .FIFO_DEPTH(IFM_W), .NUM_FIFO(3), .IMG_W(IFM_W)) LBUF7 (
        .clk(clk), .resetn(lb_resetn),
        .ready(lb_ready_6),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_dout_ch6), .data_out(lb_data_6),
        .addr_counter(addr_counter),
//        .refresh_n(ifm_rd_en_delay),
        .wr_sel(wr_sel)
    );
    /***** Line buffer 7 *****/
    wire [23:0] lb_data_7;
    wire lb_ready_7;
    line_buffer #(.DATA_WIDTH(8), .FIFO_DEPTH(IFM_W), .NUM_FIFO(3), .IMG_W(IFM_W)) LBUF8 (
        .clk(clk), .resetn(lb_resetn),
        .ready(lb_ready_7),
        .wren_i(ifm_rd_en_delay), .rden_i(1'b1),
        .data_in(ifm_dout_ch7), .data_out(lb_data_7),
        .addr_counter(addr_counter),
//        .refresh_n(ifm_rd_en_delay),
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

    wire signed [23:0] row_ch0, row_ch1, row_ch2, row_ch3, row_ch4, row_ch5, row_ch6, row_ch7;

    assign row_ch0 = lb_ready_0_delay ? lb_data_0[23:0] : 8'b0;
    assign row_ch1 = lb_ready_0_delay ? lb_data_1[23:0] : 8'b0;
    assign row_ch2 = lb_ready_0_delay ? lb_data_2[23:0] : 8'b0;
    assign row_ch3 = lb_ready_0_delay ? lb_data_3[23:0] : 8'b0;
    assign row_ch4 = lb_ready_0_delay ? lb_data_4[23:0] : 8'b0;
    assign row_ch5 = lb_ready_0_delay ? lb_data_5[23:0] : 8'b0;
    assign row_ch6 = lb_ready_0_delay ? lb_data_6[23:0] : 8'b0;
    assign row_ch7 = lb_ready_0_delay ? lb_data_7[23:0] : 8'b0;
                                
    /***** make 3x3 window *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            w00_ic0<=0; w01_ic0<=0; w02_ic0<=0; w10_ic0<=0; w11_ic0<=0; w12_ic0<=0; w20_ic0<=0; w21_ic0<=0; w22_ic0<=0;
            w00_ic1<=0; w01_ic1<=0; w02_ic1<=0; w10_ic1<=0; w11_ic1<=0; w12_ic1<=0; w20_ic1<=0; w21_ic0<=0; w22_ic1<=0;
            w00_ic2<=0; w01_ic2<=0; w02_ic2<=0; w10_ic2<=0; w11_ic2<=0; w12_ic2<=0; w20_ic2<=0; w21_ic0<=0; w22_ic2<=0;
            w00_ic3<=0; w01_ic3<=0; w02_ic3<=0; w10_ic3<=0; w11_ic3<=0; w12_ic3<=0; w20_ic3<=0; w21_ic0<=0; w22_ic3<=0;
            w00_ic4<=0; w01_ic4<=0; w02_ic4<=0; w10_ic4<=0; w11_ic4<=0; w12_ic4<=0; w20_ic4<=0; w21_ic0<=0; w22_ic4<=0;
            w00_ic5<=0; w01_ic5<=0; w02_ic5<=0; w10_ic5<=0; w11_ic5<=0; w12_ic5<=0; w20_ic5<=0; w21_ic0<=0; w22_ic5<=0;
            w00_ic6<=0; w01_ic6<=0; w02_ic6<=0; w10_ic6<=0; w11_ic6<=0; w12_ic6<=0; w20_ic6<=0; w21_ic0<=0; w22_ic6<=0;
            w00_ic7<=0; w01_ic7<=0; w02_ic7<=0; w10_ic7<=0; w11_ic7<=0; w12_ic7<=0; w20_ic7<=0; w21_ic0<=0; w22_ic7<=0;
        end
        else begin
        // when line buffer pops the data out
            if(lb_ready_0) begin
                // window shift
                {w00_ic0,w10_ic0,w20_ic0} <= {w01_ic0,w11_ic0,w21_ic0};     // col 3, shifted
                {w01_ic0,w11_ic0,w21_ic0} <= {w02_ic0,w12_ic0,w22_ic0};     // col 2, shifted
                {w02_ic0,w12_ic0,w22_ic0} <= {row_ch0};                     // col 1, sign extended, pixel elements has positive value
                
                {w00_ic1,w10_ic1,w20_ic1} <= {w01_ic1,w11_ic1,w21_ic1};     // col 3, shifted
                {w01_ic1,w11_ic1,w21_ic1} <= {w02_ic1,w12_ic1,w22_ic1};     // col 2, shifted
                {w02_ic1,w12_ic1,w22_ic1} <= {row_ch1};                     // col 1, sign extended, pixel elements has positive value
                
                {w00_ic2,w10_ic2,w20_ic2} <= {w01_ic2,w11_ic2,w21_ic2};     // col 3, shifted
                {w01_ic2,w11_ic2,w21_ic2} <= {w02_ic2,w12_ic2,w22_ic2};     // col 2, shifted
                {w02_ic2,w12_ic2,w22_ic2} <= {row_ch2};                     // col 1, sign extended, pixel elements has positive value
                
                {w00_ic3,w10_ic3,w20_ic3} <= {w01_ic3,w11_ic3,w21_ic3};     // col 3, shifted
                {w01_ic3,w11_ic3,w21_ic3} <= {w02_ic3,w12_ic3,w22_ic3};     // col 2, shifted
                {w02_ic3,w12_ic3,w22_ic3} <= {row_ch3};                     // col 1, sign extended, pixel elements has positive value
                
                {w00_ic4,w10_ic4,w20_ic4} <= {w01_ic4,w11_ic4,w21_ic4};     // col 3, shifted
                {w01_ic4,w11_ic4,w21_ic4} <= {w02_ic4,w12_ic4,w22_ic4};     // col 2, shifted
                {w02_ic4,w12_ic4,w22_ic4} <= {row_ch4};                     // col 1, sign extended, pixel elements has positive value
                
                {w00_ic5,w10_ic5,w20_ic5} <= {w01_ic5,w11_ic5,w21_ic5};     // col 3, shifted
                {w01_ic5,w11_ic5,w21_ic5} <= {w02_ic5,w12_ic5,w22_ic5};     // col 2, shifted
                {w02_ic5,w12_ic5,w22_ic5} <= {row_ch5};                     // col 1, sign extended, pixel elements has positive value
                
                {w00_ic6,w10_ic6,w20_ic6} <= {w01_ic6,w11_ic6,w21_ic6};     // col 3, shifted
                {w01_ic6,w11_ic6,w21_ic6} <= {w02_ic6,w12_ic6,w22_ic6};     // col 2, shifted
                {w02_ic6,w12_ic6,w22_ic6} <= {row_ch6};                     // col 1, sign extended, pixel elements has positive value
               
                {w00_ic7,w10_ic7,w20_ic7} <= {w01_ic7,w11_ic7,w21_ic7};     // col 3, shifted
                {w01_ic7,w11_ic7,w21_ic7} <= {w02_ic7,w12_ic7,w22_ic7};     // col 2, shifted
                {w02_ic7,w12_ic7,w22_ic7} <= {row_ch7};                     // col 1, sign extended, pixel elements has positive value
            end
        end
    end
    
    /////////////////////////////// Control Address of output piexels ///////////////////////////////
    localparam OFM_MAX_ADDR_LOG = 10;   // 576 < 2^10 = 1024
    localparam IFM_W_LOG        = 5;    // 26 < 2^5 = 32
    localparam IFM_H_LOG        = 5;    // 26 < 2^5 = 32
    
    reg [OFM_MAX_ADDR_LOG-1 : 0]    out_addr_reg;
    reg [OFM_MAX_ADDR_LOG-1 : 0]    out_addr_reg_delay;
    reg [OFM_MAX_ADDR_LOG-1 : 0]    out_addr_reg_delay2;
    
    reg [IFM_W_LOG-1 : 0]           row;
    reg [IFM_H_LOG-1 : 0]           col;
    reg [1:0]                       local_we;
    reg [1:0]                       local_we_delay;
    
//    assign out_addr = out_addr_reg_delay;
            
    always @(posedge clk or negedge lb_resetn) begin //  or posedge refresh_address_delay
        if(!lb_resetn) begin
            local_we <= 2;
        end
        else if ((addr_counter == IFM_W - 1) && local_we > 0) begin
            local_we <= local_we - 1;
        end 
    end 
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            local_we_delay <= 0;
        end
        else begin 
            local_we_delay <= local_we;
        end
    end
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            row <= 0; 
            col <= 0;
        end
        else begin
            // coordinate of SRAM2 to write calculated data
            if(!isRunning) begin
                col <= 0;
                row <= row;
            end
            else if(row == IFM_H - 2) begin
                row <= 0;
            end
            else if(col == IFM_W - 1) begin             // OFM_W = 24 ->  OFM_W - 1 = 23 = IFM_W - 3
                col <= 0; 
                row <= row + 1'b1;
            end 
            else if (!local_we_delay) begin
                col <= col + 1'b1;
            end
        end
    end
    
   ///////////////////////////////// FOR TIMING CONSTRAINTS ///////////////////////////////////
    reg [IFM_H_LOG-1:0] col_delay, col_delay2, col_delay3, col_delay4, col_delay5, col_delay6, col_delay7;
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            col_delay <= 0;
            col_delay2 <= 0;
            col_delay3 <= 0;
            col_delay4 <= 0;
            col_delay5 <= 0;
            col_delay6 <= 0;
            col_delay7 <= 0;
        end
        else begin
            col_delay <= col;
            col_delay2 <= col_delay;
            col_delay3 <= col_delay2;
            col_delay4 <= col_delay3;
            col_delay5 <= col_delay4;
            col_delay6 <= col_delay5;
            col_delay7 <= col_delay6;
        end
    end
    //////////////////////////////////////////////////////////////////////////////////////////////
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            out_addr_reg <= 0;
        end
        else begin
            if(out_addr_reg >= OFM_MAX_ADDR - 1) begin
                out_addr_reg <= 0;
            end
            else if(col_delay7 >= 2) begin
                out_addr_reg <= out_addr_reg + 1'b1;
            end
        end
    end
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            out_addr_reg_delay <= 0;
            out_addr_reg_delay2 <= 0;
        end
        else begin
            out_addr_reg_delay <= out_addr_reg;
            out_addr_reg_delay2 <= out_addr_reg_delay;
        end
    end
    
    assign out_addr = out_addr_reg_delay2;
    assign col_out = col_delay7;
    
    assign done = (state == DONE);

//    // 1 Channel
//    quantize #( .INPUT_DATA_WIDTH(21), .OUTPUT_DATA_WIDTH(8) ) i_quantize_0 ( .in_data(out_px0), .out_data(out_px_quantized0) );
//    relu #( .DATA_WIDTH(8) ) i_relu_0 ( .in_data(out_px_quantized0), .out_data(out_relu0) );

endmodule