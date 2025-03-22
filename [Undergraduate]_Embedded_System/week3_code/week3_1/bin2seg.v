`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: HUINS
// Engineer: 
// 
// Create Date: 2012/11/28 13:46:07
// Design Name: segment
// Module Name: bin2seg
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


module bin2seg(
input [3:0] bin_data,           // 4비트 이진수 입력
output reg [7:0] seg_data);     // 8비트 7-segment 디스플레이 출력

  // bin_data가 변경될 때마다 동작
  always @(bin_data)
  begin
    case (bin_data)
      4'b0000 : seg_data <= 8'b11111100 ;   // 이진수 0000은 7-segment에서 '0'을 표시
      4'b0001 : seg_data <= 8'b01100000 ;   // 이진수 0001은 7-segment에서 '1'을 표시
      4'b0010 : seg_data <= 8'b11011010 ;   // 이진수 0010은 7-segment에서 '2'를 표시
      4'b0011 : seg_data <= 8'b11110010 ;   // 이진수 0011은 7-segment에서 '3'을 표시
      4'b0100 : seg_data <= 8'b01100110 ;   // 이진수 0100은 7-segment에서 '4'를 표시
      4'b0101 : seg_data <= 8'b10110110 ;   // 이진수 0101은 7-segment에서 '5'를 표시
      4'b0110 : seg_data <= 8'b10111110 ;   // 이진수 0110은 7-segment에서 '6'을 표시
      4'b0111 : seg_data <= 8'b11100100 ;   // 이진수 0111은 7-segment에서 '7'을 표시
      4'b1000 : seg_data <= 8'b11111110 ;   // 이진수 1000은 7-segment에서 '8'을 표시
      4'b1001 : seg_data <= 8'b11110110 ;   // 이진수 1001은 7-segment에서 '9'를 표시
      4'b1010 : seg_data <= 8'b00000010 ;   // 이진수 1010은 7-segment에서 '-'를 표시
      default : seg_data <= 8'b11111100 ;   // 그 외 기본 값은 '0'을 표시
    endcase 
  end 

endmodule