`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/04/04 17:35:53
// Design Name: 
// Module Name: line_buffer
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


module line_buffer#(
    parameter DATA_WIDTH = 32,   // FIFO bit width
    parameter FIFO_DEPTH = 32,   //  2, 4, 8 ... 
    parameter NUM_FIFO   = 3,    // number of FIFO
    parameter IMG_W = 28
    )(
    input   wire    clk,
    input   wire    resetn,
    input   wire    [4:0]               CURRENT_IMG_W,
    output  wire    ready,      // FIFO pops out data in parallel when ready is high
    input   wire    wren_i,     // write enable 
    input   wire    rden_i,     // read enable
    input   wire    [DATA_WIDTH-1:0]    data_in,    // input data (8bits) in to FIFO 
    input   wire    [4:0]               addr_counter,
//    input   wire    refresh_n,
    
    output  wire    [3*DATA_WIDTH-1:0]    data_out,  // output data in parallel (24bits)
    output  wire    [1:0]                 wr_sel
    );    
    wire full_0, full_1, full_2;    // High when each FIFO is full (can't put data in)
    wire empty_0, empty_1, empty_2; // High when each FIFO is empty (No data to pop out)
    
    // To do
    /////////////////////////////////////

    // data which has been poped out from FIFO 0, 1, 2
    wire [DATA_WIDTH-1:0] data_out_0, data_out_1, data_out_2;
    
    // ready when All the 3 FIFOs have FIFO_DEPTH amount of elements
    reg ready_count;
    
    /***** write: count for when All the 2 FIFOs have FIFO_DEPTH amount of elements *****/
    always @(posedge clk or negedge resetn) begin
        if(!resetn) //  || !refresh_n
            ready_count <= 1; // 0
        else if((addr_counter == (CURRENT_IMG_W-1)) && wren_i && ready_count > 0) //  && ready_count < NUM_FIFO
            ready_count <= ready_count - 1'b1;
        else
            ready_count <= ready_count; // 0
    end
    
    /***** read: count for when All the 2 FIFOs have FIFO_DEPTH amount of elements *****/
    reg [1:0] count_read;
    always @(posedge clk or negedge resetn) begin
        if(!resetn)
            count_read <= 2'd2; 
        else if(addr_counter == (CURRENT_IMG_W-1))
            count_read <= ready_count;
    end
    
    /***** Delay Beacause of BRAM 1 cycle read delay *****/
    reg [1:0] wr_sel_reg_delay;
    reg ready_count_delay;
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin                           //  || !refresh_n
            ready_count_delay <= 1;
        end
        else begin
            ready_count_delay <= ready_count;
        end
    end
    
//    /***** write: count for when All the 2 FIFOs have FIFO_DEPTH amount of elements *****/
//    always @(posedge trigger_next_fifo or negedge resetn) begin
//        if(!resetn) //  || !refresh_n
//            ready_count <= 1; // 0
//        else if(wren_i && ready_count > 0) //  && ready_count < NUM_FIFO
//            ready_count <= ready_count - 1'b1;
//        else
//            ready_count <= ready_count; // 0
//    end
    
//    /***** read: count for when All the 2 FIFOs have FIFO_DEPTH amount of elements *****/
//    reg [1:0] count_read;
//    always @(posedge trigger_next_fifo or negedge resetn) begin
//        if(!resetn)
//            count_read <= 2'd2; 
//        else
//            count_read <= ready_count;
//    end
    
//    /***** Delay Beacause of BRAM 1 cycle read delay *****/
//    reg [1:0] wr_sel_reg_delay;
//    reg ready_count_delay;
//    always @(posedge clk or negedge resetn) begin
//        if(!resetn) begin //  || !refresh_n
////            wr_sel_reg_delay <= 0;
//            ready_count_delay <= 1;
//        end
//        else begin
////            wr_sel_reg_delay <= wr_sel_reg;
//            ready_count_delay <= ready_count;
//        end
//    end
    
    // when select signal is high & wren_i is high, 
    // write data into corresponding FIFO
    wire wren_0 = (ready_count_delay == 2'd1 || ready_count_delay == 2'd0) && wren_i;
    wire wren_1 = (ready_count_delay == 2'd0) && wren_i;
    
    wire rden_0 = (count_read == 1 || count_read == 0) && rden_i; // ready && rden_i;
    wire rden_1 = (count_read == 0) && rden_i; // ready && rden_i;
    
    
    // need to concatenate the data_out from each FIFO
    // which will pop out data in parallel (24bits)
    assign data_out = {data_out_1, data_out_0, data_out_2};
    
    // when 1st & 2nd FIFO has 102 elements
    assign ready = (rden_1);
    
    // Last Row of FIFO, just put data
    assign data_out_2 = data_in;
    ////////////////////////////////////



    fifo #(
    .DATA_WIDTH(DATA_WIDTH),
    .FIFO_DEPTH(FIFO_DEPTH) 
    ) Ufifo_0(
        .clk(clk),
        .rst_n(resetn),
        .wren_i(wren_0),
        .rden_i(rden_0), // rden_0
        .wdata_i(data_in),
        .rdata_o(data_out_0), // data_out[DATA_WIDTH-1:0]
        .full_o(full_0),
        .empty_o(empty_0)
    );

    fifo #(
    .DATA_WIDTH(DATA_WIDTH),
    .FIFO_DEPTH(FIFO_DEPTH)
    ) Ufifo_1(
        .clk(clk),
        .rst_n(resetn),
        .wren_i(wren_1), // wren_1
        .rden_i(rden_1),
        .wdata_i(data_out_0), // data_in
        .rdata_o(data_out_1), // data_out[2*DATA_WIDTH-1:DATA_WIDTH]
        .full_o(full_1),
        .empty_o(empty_1)
    );

//    fifo #(
//    .DATA_WIDTH(8),
//    .FIFO_DEPTH(102)
//    ) Ufifo_2(
//        .clk(clk),
//        .rst_n(resetn),
//        .wren_i(wren_2), // wren_2
//        .rden_i(rden_2),
//        .wdata_i(data_out_1), // data_in
//        .rdata_o(data_out_2), // data_out[3*DATA_WIDTH-1:2*DATA_WIDTH]
//        .full_o(full_2),
//        .empty_o(empty_2)
//    );
    

endmodule
