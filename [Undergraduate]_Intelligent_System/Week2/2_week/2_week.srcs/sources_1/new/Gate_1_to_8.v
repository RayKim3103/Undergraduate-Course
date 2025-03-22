`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/12 19:32:52
// Design Name: 
// Module Name: Gate_1_to_8
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


module Gate_1_to_8(
    input   wire    G1,
    input   wire    A, B, C,
    output  wire    [7:0] Y
    );
    
    // Define Output
    assign Y[0] = ~(G1 & ~A & ~B & ~C); // Y[0] = 0 when, G1 = 1, C = 0, B = 0, A = 0
    assign Y[1] = ~(G1 & A & ~B & ~C);  // Y[1] = 0 when, G1 = 1, C = 0, B = 0, A = 1
    assign Y[2] = ~(G1 & ~A & B & ~C);  // Y[2] = 0 when, G1 = 1, C = 0, B = 1, A = 0
    assign Y[3] = ~(G1 & A & B & ~C);   // Y[3] = 0 when, G1 = 1, C = 0, B = 1, A = 1
    assign Y[4] = ~(G1 & ~A & ~B & C);  // Y[4] = 0 when, G1 = 1, C = 1, B = 0, A = 0 
    assign Y[5] = ~(G1 & A & ~B & C);   // Y[5] = 0 when, G1 = 1, C = 1, B = 0, A = 1 
    assign Y[6] = ~(G1 & ~A & B & C);   // Y[6] = 0 when, G1 = 1, C = 1, B = 1, A = 0 
    assign Y[7] = ~(G1 & A & B & C);    // Y[7] = 0 when, G1 = 1, C = 1, B = 1, A = 1
endmodule
