`timescale 1ns / 1ps
`define CLOCK_PERIOD 10
`define DELTA 1
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/04/04 19:29:02
// Design Name: 
// Module Name: tb_line_buffer
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


module tb_line_buffer();

    localparam DATA_WIDTH = 8; // bit width
    localparam FIFO_DEPTH = 8; // FIFO depth (2^n)
    
    reg clk; // generate clock signal
    initial begin
        clk = 1'b0;
        forever begin
            #(`CLOCK_PERIOD/2) clk = ~clk;
        end
    end

    reg resetn;
    reg wren_i, rden_i;
    reg [DATA_WIDTH-1:0]    data_in;

    wire ready;
    wire [3*DATA_WIDTH-1:0]    data_out;

    reg [DATA_WIDTH-1:0] mem_to_DUT [3*FIFO_DEPTH-1:0];
    reg [3*DATA_WIDTH-1:0] mem_from_DUT [FIFO_DEPTH-1:0];

    // initialize memory with random values
    integer i;
    initial begin
        for (i = 0; i< 3*FIFO_DEPTH; i = i+1) begin
            mem_to_DUT[i] <= $random;
        end
    end

    initial begin
        // 0. initialize
        resetn = 1'b1;
        wren_i = 1'b0;
        rden_i = 1'b0;
        data_in = {(DATA_WIDTH){1'b0}};
        
        // 1. reset
        @(posedge clk);
        #(`DELTA)
        resetn = 1'b0;   // reset on
        
        @(posedge clk);
        #(`DELTA)
        resetn = 1'b1;   // reset off
    end

    // in this loop by using generated write enable signal 
    // it will write data into FIFO
    // the data was generated in the previous loop (random values)
    integer k;
    initial begin
        #300
        for (k=0; k < 3*FIFO_DEPTH; k=k+1) begin
            #(`CLOCK_PERIOD * ($urandom % 20 + 1)); //  1~21 clock cycle dealy
            @(posedge clk); // wait for clock
            #(`DELTA)       // 1ns delay
            wren_i = 1'b1;  // set write enable High
            data_in = mem_to_DUT[k];
            @(posedge clk); // wait for next clock
            #(`DELTA)       // 1ns delay
            wren_i = 1'b0;  // set write enable Low
        end       
    end

    // when ready is High, it means that all FIFO are full
    // and it is ready to read data out
    initial begin
        @(posedge ready) begin
            $display ("DUT ready is HIGH!");
            #(`DELTA)
            compare_memory();
            #(`DELTA)
            $finish();
        end

    end


    integer j;
    task compare_memory;
        begin
            rden_i <= 1'b1;
            for(j=0; j<FIFO_DEPTH; j=j+1) begin
                @(posedge clk);
                #(`DELTA)
                mem_from_DUT[j] <= data_out;
                $display ("[%d] IDEAL : %h DUT : %h", j, {mem_to_DUT[2*FIFO_DEPTH+j], mem_to_DUT[FIFO_DEPTH+j], mem_to_DUT[j]},data_out );
                if({mem_to_DUT[2*FIFO_DEPTH+j], mem_to_DUT[FIFO_DEPTH+j], mem_to_DUT[j]} != data_out) begin
                    $display("Error: memory comparison failed @ %8dns", $time);
//                    $writememh("YOUR PROJECT PATH/mem_to_DUT.hex", mem_to_DUT);
//                    $writememh("YOUR PROJECT PATH/mem_from_DUT.hex", mem_from_DUT);
                    $writememh("C:/2025exp_accelerator/week6_1/mem_to_DUT.hex", mem_to_DUT);
                    $writememh("C:/2025exp_accelerator/week6_1/mem_from_DUT.hex", mem_from_DUT);
                    $finish();
                end
            end
            $display("PASS: memory comparison succeed @ %8dns", $time);
        end
    endtask


    // line_buffer DUT instantiation
    line_buffer#(
        .DATA_WIDTH(8),
        .FIFO_DEPTH(8),    //  2, 4, 8 ... 
        .NUM_FIFO  (3)
        ) Uline_buffer(
        .clk(clk),
        .resetn(resetn),
        .ready(ready),
        .wren_i(wren_i),    
        .rden_i(rden_i),    
        .data_in(data_in),
        .data_out(data_out)
        );    

endmodule
