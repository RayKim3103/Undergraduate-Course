`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/27 15:02:20
// Design Name: 
// Module Name: memory_ctrlr
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


module memory_ctrlr #(
        parameter IMG_W  = 102,
        parameter IMG_H  = 102,
        parameter S1_ADDR_W = 14,     // 102��102 < 2^14 = 16384
        parameter S2_ADDR_W = 14
    )(
        input                       clk,
        input                       resetn,
        // CSR
        input                       start,
        output                      done,
        output                      done_led,
        //----------------------------------------------------------
        // SRAM1  (Port-B : PL side, Read Only)
        output                  s1_en,
        output                  s1_we,
        output  [S1_ADDR_W-1:0] s1_addr,
        input   [7:0]           s1_dout,
        //----------------------------------------------------------
        // SRAM2  (Port-B : PL side, Write Only)
        output                  s2_en,
        output                  s2_we,            // 1 = write (byte0 ���)
        output  [S2_ADDR_W-1:0] s2_addr,
        output  [7:0]           s2_din
    );
    
    localparam IDLE = 2'b00;
    localparam RUN  = 2'b01;
    localparam DONE = 2'b10;

    localparam s2_MAXADDR = 14'd10000;

    reg [1:0] state, n_state;

    reg [13:0] cnt_run;
//    wire done_run;
    
    reg [7:0] s2_din_reg;
    reg [13:0] s2_addr_reg;
    reg [13:0] s2_addr_reg_delay;
    reg lb_ready_0_delay2; // , lb_ready_1_delay2, lb_ready_2_delay2;

//    assign done_run = (cnt_run == 14'd10404);

    assign s1_en = (state == RUN);
    assign s1_we = 1'b0;
    assign s1_addr = (state == RUN)? cnt_run: {S2_ADDR_W{1'b0}};

    assign s2_addr = s2_addr_reg_delay; // s2_addr_reg;
    assign s2_en = (lb_ready_0_delay2)? 1'b1 : 1'b0; //  || lb_ready_1_delay2 || lb_ready_2_delay2
    assign s2_we = (lb_ready_0_delay2)? 1'b1 : 1'b0; //  || lb_ready_1_delay2 || lb_ready_2_delay2

//    reg [7:0] s2_din_reg_delay;
    assign s2_din = s2_din_reg;

    assign done = (state == DONE);
    assign done_led = done;
    
    /**************************************** Defines what data to write in SRAM2 ****************************************/
    wire [7:0] r0;
    wire [7:0] r1;
    wire [7:0] r2;
    wire lb_ready_0;
    reg [8:0] w00,w01,w02, w10,w11,w12, w20,w21,w22;
     wire [7:0] edge_px;
   /************************************************************************************************************************/
   
   /***** Defines when does the state change *****/ 
   always @(*) begin
        case (state)
            IDLE : begin 
                if(start) n_state = RUN;
                else n_state = IDLE;
            end            
            RUN : begin
//                if(done_run) n_state = DONE;
                if(s2_addr >= s2_MAXADDR) n_state = DONE;
                else n_state = RUN;
            end
            DONE : begin
                if(~resetn) n_state = IDLE;
                else n_state = DONE;
            end
            default :  n_state = IDLE;
        endcase
    end
    
    /***** Defines what address to read *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) cnt_run <= 7'd0;
        else begin
            if(state == RUN) 
                cnt_run <= cnt_run + 7'd1;
            else
                cnt_run <= 7'd0;
        end
    end
    
    /***** state transition *****/
    always @(posedge clk or negedge resetn) begin
        if(~resetn) state <= IDLE;
        else state <= n_state;
    end

    /***** Line Buffer Variables *****/
    wire trigger_next_fifo;
    reg s1_en_delay;
    wire [1:0] wr_sel;
    reg wren_i_LBUF1, wren_i_LBUF2;
    
    reg [6:0] addr_counter;  // 102 < 2^7
    reg trigger_next_fifo_reg;
    always @(posedge clk) begin
        if (!resetn) begin
            addr_counter <= 0;
            trigger_next_fifo_reg <= 0;
        end
        else if (s1_en) begin
            if (addr_counter == 101) begin
                addr_counter <= 0;
                trigger_next_fifo_reg <= 1;
            end
            else begin
                addr_counter <= addr_counter + 1;
                trigger_next_fifo_reg <= 0;
            end
        end
    end
    
    assign trigger_next_fifo = trigger_next_fifo_reg;  // ���� �ֱ⿡ ���ο� row ����    
//    assign trigger_next_fifo = (s1_addr % IMG_H == 0) ? 1'b1 : 1'b0;
    
    // 1 cycle delay Because of BRAM's 1 cycle read delay
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            s1_en_delay <= 0;
        end
        else begin
            s1_en_delay <= s1_en;
        end
    end
    
//    // turn on Line Buffer 1, 2 in correct timing
//    always @(posedge clk or negedge resetn) begin
//        if(!resetn) begin
//            wren_i_LBUF1 <= 0; 
//            wren_i_LBUF2 <= 0;
//        end
//        else if(wr_sel == 1) wren_i_LBUF1 <= 1;
//        else if(wr_sel == 2) wren_i_LBUF2 <= 1;
//        else if (s1_en_delay)begin
//            wren_i_LBUF1 <= wren_i_LBUF1; 
//            wren_i_LBUF2 <= wren_i_LBUF2;
//        end
//        else begin
//            wren_i_LBUF1 <= 0;
//            wren_i_LBUF2 <= 0;
//        end
//    end
     
    /***** Line buffer 0 *****/
    wire lb_ready_0;
    wire [23:0] lb_data_0;    
    line_buffer #(.DATA_WIDTH(8), .FIFO_DEPTH(IMG_W), .NUM_FIFO(3)) LBUF0 (
        .clk(clk), .resetn(resetn),
        .ready(lb_ready_0),
        .wren_i(s1_en_delay), .rden_i(1'b1), // lb_ready_0
        .data_in(s1_dout), .data_out(lb_data_0),
        .trigger_next_fifo(trigger_next_fifo),
        .wr_sel(wr_sel)
    );

    /***** assign datas to get each pixel data *****/
    // should consider 1cycle read delay from FIFO 8x128 Bram -> every edge when lb_ready is high the first element is 0
    reg lb_ready_0_delay, lb_ready_1_delay, lb_ready_2_delay;
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
    
    /***** variables for making 3x3 window *****/ 
    // 3x3 window elements, consider sign-extension
//    reg [8:0] w00,w01,w02, w10,w11,w12, w20,w21,w22;

    /***** sobel_pixel computer instance *****/
    // compute with sobel_pixel.v
//    wire [7:0] edge_px;
    sobel_pixel PIX (.p00(w00),.p01(w01),.p02(w02),
                     .p10(w10),.p11(w11),.p12(w12),
                     .p20(w20),.p21(w21),.p22(w22),
                     .edge_out(edge_px));
    
    /***** make 3x3 window *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            w00<=0; w01<=0; w02<=0; w10<=0; w11<=0; w12<=0; w20<=0; w21<=0; w22<=0;
        end
        else begin
        // when line buffer pops the data out
            if(lb_ready_0) begin // lb_ready_0 || lb_ready_1 || lb_ready_2
                // window shift
                {w02,w12,w22} <= {w01,w11,w21};                     // col 3, shifted
                {w01,w11,w21} <= {w00,w10,w20};                     // col 2, shifted
                {w00,w10,w20} <= {{1'b0,r0},{1'b0,r1},{1'b0,r2}};   // col 1, sign extended, pixel elements has positive value
            end
        end
    end 
    
    /***** write computed data to SRAM2 *****/
    localparam CW = $clog2(IMG_W);                          // Column Width Bit
//    localparam RW = $clog2(IMG_H);                          // Row Width Bit
    reg [CW-1:0] col;                                       // reg to address col
    reg [13:0] row;                                         // reg to address row
    
    // should consider computation window shifting delay 1 cycle
//    reg lb_ready_0_delay2, lb_ready_1_delay2, lb_ready_2_delay2;
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            lb_ready_0_delay2 <= 0;
        end
        else begin
            lb_ready_0_delay2 <= lb_ready_0_delay;
        end
    end
    
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            row <= 0; 
            col <= 0;
            s2_addr_reg <= 0; 
            s2_din_reg <= 0;
        end
        else begin
            // coordinate of SRAM2 to write calculated data
            if(row == (IMG_H - 2)) begin
                row <= row;
            end
            else if(col == IMG_W-1) begin
                col <= 0; 
                row <= row + 1'b1;
            end 
            else if (lb_ready_0_delay2) begin //  || lb_ready_1_delay2 || lb_ready_2_delay2
                col <= col + 1'b1;
            end
            
            // if coordinate is in the correct region: SRAM2 Write
            // when row is changed, column index should be delayed by 2 (since, window shifting)
            if((col>=2)) begin //  (row>=0)&&     &&(col<=IMG_W)
                s2_addr_reg <= s2_addr_reg + 1'b1; // (row << 6) + (row << 5) + (row << 2) + (col - 2); // (row)*(IMG_W-2) + (col-2);
                s2_din_reg  <= edge_px;
            end
        end
    end
    
    // 1 cycle delay for synchronizing the timing of data & address for writing
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            s2_addr_reg_delay <= 0;
        end
        else begin
            s2_addr_reg_delay <= s2_addr_reg;
        end
    end
  
endmodule
