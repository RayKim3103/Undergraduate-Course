`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/12 19:06:27
// Design Name: 
// Module Name: Shift_74LS138
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


module Shift_74LS138(
    input   wire    G1, G2A, G2B,
    input   wire    C, B, A,  
    output  wire    [7:0] Y
);
    assign Y = (G1 & ~G2A & ~G2B) ? ~(8'b00000001 << {C, B, A}) : 8'b11111111;
    
endmodule
