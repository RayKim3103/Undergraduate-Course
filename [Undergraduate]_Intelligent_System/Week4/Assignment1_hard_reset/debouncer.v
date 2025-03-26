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
    input wire btn_in,    // 버튼 입력 신호 (Jitter 존재)
    output wire btn_out   // Debounce된 버튼 출력
    );
    
    wire q1, q2; 
    
    // 두 개의 D Flip Flop
    dff Udff0(clk, btn_in, q1);
    dff Udff1(clk, q1 ,q2 );
    
    assign btn_out = q1 & ~q2; // q1과 ~q2가 AND게이트 지남
    
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