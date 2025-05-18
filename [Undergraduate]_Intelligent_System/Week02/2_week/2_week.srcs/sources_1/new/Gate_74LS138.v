`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/10 10:19:43
// Design Name: 
// Module Name: Gate_74LS138
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


module Gate_74LS138(

    input   wire    G1, G2A, G2B,
    input   wire    A, B, C,
    output  wire    [7:0] Y
);

    wire enable;

    // Define Enabel Signal
    assign enable = G1 & ~G2A & ~G2B;

    // Define Output
    assign Y[0] = ~(enable & ~A & ~B & ~C); // Y[0] = 0 when, enable = 1, C = 0, B = 0, A = 0
    assign Y[1] = ~(enable & A & ~B & ~C);  // Y[1] = 0 when, enable = 1, C = 0, B = 0, A = 1
    assign Y[2] = ~(enable & ~A & B & ~C);  // Y[2] = 0 when, enable = 1, C = 0, B = 1, A = 0
    assign Y[3] = ~(enable & A & B & ~C);   // Y[3] = 0 when, enable = 1, C = 0, B = 1, A = 1
    assign Y[4] = ~(enable & ~A & ~B & C);  // Y[4] = 0 when, enable = 1, C = 1, B = 0, A = 0 
    assign Y[5] = ~(enable & A & ~B & C);   // Y[5] = 0 when, enable = 1, C = 1, B = 0, A = 1 
    assign Y[6] = ~(enable & ~A & B & C);   // Y[6] = 0 when, enable = 1, C = 1, B = 1, A = 0 
    assign Y[7] = ~(enable & A & B & C);    // Y[7] = 0 when, enable = 1, C = 1, B = 1, A = 1

endmodule

