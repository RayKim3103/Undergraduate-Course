`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/12 19:47:03
// Design Name: 
// Module Name: Verilog_1_to_8
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


module Verilog_1_to_8(
    input   wire    G1,
    input   wire    A, B, C,
    output  reg     [7:0] Y // define as "reg"
    );
    
    always @(*) begin
        if (G1) begin
            case ({C, B, A})
                3'b000: Y = 8'b11111110;    // Y[0] = 0 when, (G1) = 1, C = 0, B = 0, A = 0
                3'b001: Y = 8'b11111101;    // Y[1] = 0 when, (G1) = 1, C = 0, B = 0, A = 1
                3'b010: Y = 8'b11111011;    // Y[2] = 0 when, (G1) = 1, C = 0, B = 1, A = 0
                3'b011: Y = 8'b11110111;    // Y[3] = 0 when, (G1) = 1, C = 0, B = 1, A = 1
                3'b100: Y = 8'b11101111;    // Y[4] = 0 when, (G1) = 1, C = 1, B = 0, A = 0 
                3'b101: Y = 8'b11011111;    // Y[5] = 0 when, (G1) = 1, C = 1, B = 0, A = 1 
                3'b110: Y = 8'b10111111;    // Y[6] = 0 when, (G1) = 1, C = 1, B = 1, A = 0 
                3'b111: Y = 8'b01111111;    // Y[7] = 0 when, (G1) = 1, C = 1, B = 1, A = 1
                default: Y = 8'b11111111;   // Default, Every bit is high
            endcase
        end 
        else begin
            Y = 8'b11111111;                // (G1 & ~G2A & ~G2B) = 0 then, Every bit is high 
        end
    end  
    
endmodule
