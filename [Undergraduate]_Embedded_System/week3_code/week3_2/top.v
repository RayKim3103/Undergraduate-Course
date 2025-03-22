
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: HUINS
// Engineer: 
// 
// Create Date: 2012/11/28 13:46:07
// Design Name: segment
// Module Name: easy_clock
// Project Name: segment
// Target Devices: xc7z020clg484-1
// Tool Versions: Xilinx PlanAhead 14.3
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module top(
input clk_in,
input resetn,
output [7:0] led_out,
output [7:0] segout,
output [7:0] segcom);

  wire [31:0] segdata;

  clock clock_inst (
    .clk_in(clk_in),
    .resetn(resetn),
    .segdata(segdata));

  seven_seg seven_seg_inst (
    .clk(clk_in),
    .resetn(resetn),
    .data(segdata),
    .segout(segout),
    .segcom(segcom),
    .led_out(led_out));

endmodule
