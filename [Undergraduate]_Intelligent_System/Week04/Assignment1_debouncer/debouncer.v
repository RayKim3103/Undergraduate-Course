`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/23 12:03:22
// Design Name: 
// Module Name: debouncer
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


module debouncer(
    input wire clk,
    input wire in,
    output wire out_pos,
    output wire out_neg
    );

    wire Q, Q2;
    dff Udff0(clk, in, Q);
    dff Udff1(clk, Q ,Q2 );

    assign out_pos = ~Q2 & Q;
    assign out_neg = Q2 & ~Q;
endmodule


module dff(
    input wire clk, 
    input wire D, 
    output reg Q
    );

    always @ (posedge clk) begin
        Q <= D;
    end
endmodule