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


module top_memory_ctrlr(
    input wire clk,     // input clk port
    input wire resetn,  // input reset port
    
    input  wire start,  // input start signal port
    output wire done    // output done signal port
    );
    
    // BW: Bit Width
    // AMAX: Max number of Address line (memory depth¸¦ ÀÇ¹Ì)
    // ADR: Number of bits to represent address
    parameter SRAM1_BW = 16;
    parameter SRAM1_AMAX = 256;
    parameter SRAM1_ADR = $clog2(SRAM1_AMAX);
    parameter SRAM2_BW = 32;
    parameter SRAM2_AMAX = 192;
    parameter SRAM2_ADR = $clog2(SRAM2_AMAX);


    wire s1_en, s2_en;  // SRAM 1¡¢ £² enable signal
    wire s1_we, s2_we;  // SRAM 1¡¢ £² write enable signal
    wire [SRAM1_ADR-1:0] s1_addr;   // SRAM 1 read address
    wire [SRAM2_ADR-1:0] s2_addr;   // SRAM 2 write address
    wire [SRAM1_BW-1:0] s1_din, s1_dout;    // SRAM 1 read data
    wire [SRAM2_BW-1:0] s2_din, s2_dout;    // SRAM 2 write data

    // module which controls SRAM interface. It controls operation of SRAM1 & SRAM 2
    memory_ctrlr #(.SRAM1_BW(SRAM1_BW), .SRAM1_AMAX(SRAM1_AMAX), .SRAM2_BW(SRAM2_BW), .SRAM2_AMAX(SRAM2_AMAX)) 
    Umemory_ctrlr (.clk(clk), .resetn(resetn), .start(start), .done(done),
                   .s1_en(s1_en), .s1_we(s1_we), .s1_addr(s1_addr), .s1_dout(s1_dout),
                   .s2_en(s2_en), .s2_we(s2_we), .s2_addr(s2_addr), .s2_din(s2_din));
    
    // SRAM 1 & 2 modeling, it can operate read operation and write operation with #1 delay
    sram_test #(.BW(SRAM1_BW), .AMAX(SRAM1_AMAX)) SRAM1(.clk(clk), .en(s1_en), .we(s1_we), .addr(s1_addr), .din(s1_din), .dout(s1_dout));
    sram_test #(.BW(SRAM2_BW), .AMAX(SRAM2_AMAX)) SRAM2(.clk(clk), .en(s2_en), .we(s2_we), .addr(s2_addr), .din(s2_din), .dout(s2_dout));
 
endmodule
