`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/18 21:50:50
// Design Name: 
// Module Name: debouncer_2
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


module debouncer_2(
    input wire clk,
    input wire btn_in,      // 버튼 입력 신호 (Jitter 존재)
    output wire btn_out     // Debounce된 버튼 출력
    );
    
    wire q1, q2; // 두 개의 D flip-flop 출력 신호

    // 첫 번째 D flip-flop (DFF1)
    d_ff dff1 (
        .clk(clk),          // clcok
        .resetn(1'b1),      // No Reset
        .D(btn_in),         // 버튼 입력
        .Q(q1)              // 첫 번째 플립플롭 출력
    );

    // 두 번째 D flip-flop (DFF2)
    d_ff dff2 (
        .clk(clk),          // clock
        .resetn(1'b1),      // No Reset 
        .D(q1),             // 첫 번째 Flip Flop 출력
        .Q(q2)              // 두 번째 Flip Flop 출력
    );
    
    assign btn_out = q1 & ~q2; // q1과 ~q2가 AND 게이트 지남
    
endmodule
