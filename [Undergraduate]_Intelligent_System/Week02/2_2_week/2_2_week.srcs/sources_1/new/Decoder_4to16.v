`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/12 20:01:59
// Design Name: 
// Module Name: Decoder_4to16
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


module Decoder_4to16 (
    input wire D, C, B, A,          // Input 4 bit
    input wire G1A, G1B, G2A, G2B,  // Enable 4 bit
    output wire [15:0] Y            // Output 16bit
);
    wire [7:0] Y_lower, Y_upper;  // 2 wires, one comes from first 74LS138 and the other comes from second 74LS138

    // first 74LS138 (Activate when D=0)
    Gate_74LS138 M1 (
        .C(C), .B(B), .A(A),
        .G1(~D), .G2A(G1A), .G2B(G1B),
        .Y(Y_lower)
    );

    // second 74LS138 (Activate when D=1)
    Gate_74LS138 M2 (
        .C(C), .B(B), .A(A),
        .G1(D), .G2A(G2A), .G2B(G2B),
        .Y(Y_upper)
    );

    // Output (each 8bits together)
    assign Y = {Y_upper, Y_lower};

endmodule
