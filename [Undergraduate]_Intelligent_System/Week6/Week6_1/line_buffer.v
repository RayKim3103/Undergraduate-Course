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
    parameter DATA_WIDTH = 8,   // FIFO bit width
    parameter FIFO_DEPTH = 8,   //  2, 4, 8 ... 
    parameter NUM_FIFO   = 3    // number of FIFO
    )(
    input   wire    clk,
    input   wire    resetn,
    output  wire    ready,      // FIFO pops out data in parallel when ready is high
    input   wire    wren_i,     // write enable 
    input   wire    rden_i,     // read enable
    input   wire    [DATA_WIDTH-1:0]    data_in,    // input data (8bits) in to FIFO 
    output  wire    [3*DATA_WIDTH-1:0]    data_out  // output data in parallel (24bits)
    );    
    wire full_0, full_1, full_2;    // High when each FIFO is full (can't put data in)
    wire empty_0, empty_1, empty_2; // High when each FIFO is empty (No data to pop out)




    // To do
    /////////////////////////////////////

    // data which has been poped out from FIFO 0, 1, 2
    wire [DATA_WIDTH-1:0] data_out_0, data_out_1, data_out_2;
    // select signal to select which FIFO to write into
    reg [1:0] wr_sel;
    
    // when select signal is high & wren_i is high, 
    // write data into corresponding FIFO
    wire wren_0 = (wr_sel == 2'd0) && wren_i;
    wire wren_1 = (wr_sel == 2'd1) && wren_i;
    wire wren_2 = (wr_sel == 2'd2) && wren_i;
    
    wire rden_0 = ready && rden_i;
    wire rden_1 = ready && rden_i;
    wire rden_2 = ready && rden_i;
    
    // since wren_i is wire, 
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin  // initialize
            wr_sel <= 0;
        end
        // FIFO 0 will be written first
        // if FIFO 0 is full, then FIFO 1 will be written
        // if FIFO 1 is full, then FIFO 2 will be written
        case ({full_2, full_1, full_0})
            3'b000: wr_sel <= 2'd0;
            3'b001: wr_sel <= 2'd1;
            3'b011: wr_sel <= 2'd2;

            // if each FIFO is not full, keep the same
            default: wr_sel <= wr_sel;  
        endcase
            
    end
    
    // need to concatenate the data_out from each FIFO
    // which will pop out data in parallel (24bits)
    assign data_out = {data_out_2, data_out_1, data_out_0};
    
    // when all FIFO are full, ready is high
    // when all FIFO are empty, ready is low
    // else, keep the same
    assign ready = (full_0 && full_1 && full_2) ? 1'b1 :
                   (empty_0 && empty_1 && empty_2) ? 1'b0 : ready;
    ////////////////////////////////////



    fifo #(
    .DATA_WIDTH(8),
    .FIFO_DEPTH(8) 
    ) Ufifo_0(
        .clk(clk),
        .rst_n(resetn),
        .wren_i(wren_0),
        .rden_i(rden_0),
        .wdata_i(data_in),
        .rdata_o(data_out[DATA_WIDTH-1:0]),
        .full_o(full_0),
        .empty_o(empty_0)
    );

    fifo #(
    .DATA_WIDTH(8),
    .FIFO_DEPTH(8)
    ) Ufifo_1(
        .clk(clk),
        .rst_n(resetn),
        .wren_i(wren_1),
        .rden_i(rden_1),
        .wdata_i(data_in),
        .rdata_o(data_out[2*DATA_WIDTH-1:DATA_WIDTH]),
        .full_o(full_1),
        .empty_o(empty_1)
    );

    fifo #(
    .DATA_WIDTH(8),
    .FIFO_DEPTH(8)
    ) Ufifo_2(
        .clk(clk),
        .rst_n(resetn),
        .wren_i(wren_2),
        .rden_i(rden_2),
        .wdata_i(data_in),
        .rdata_o(data_out[3*DATA_WIDTH-1:2*DATA_WIDTH]),
        .full_o(full_2),
        .empty_o(empty_2)
    );
    

endmodule
