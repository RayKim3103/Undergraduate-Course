
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
input clk_in,           // input clock
input resetn,           // reset (Active Low)
output [7:0] led_out,   // LED 출력
output [7:0] segout,    // 현재 표시할 7-segment 출력 데이터
output [7:0] segcom);   // 7-segment select 신호 (8개의 디스플레이 패널 중 1개 선택)

  wire [31:0] segdata;  // 32bit 데이터: 4bit씩 디스플레이 패널(7-segment & DP) 1개씩 담당

  // clk에 따라 시간을 계산하여 그 결과를 32bit segdata로 출력
  clock clock_inst (
    .clk_in(clk_in),        // input clock
    .resetn(resetn),        // reset signal
    .segdata(segdata));     // output segdata

  // 32bit segdata를 받아 clk에 따라 4bit 이진수를 7-segment 디스플레이와 LED로 바꿈
  seven_seg seven_seg_inst (
    .clk(clk_in),           // input clock
    .resetn(resetn),        // reset signal
    .data(segdata),         // input segdata
    .segout(segout),        // 8bit 7-segment 디스플레이 신호
    .segcom(segcom),        // 8bit 7-segment 패널 select 신호
    .led_out(led_out));     // 8개의 LED 출력 결정

endmodule
