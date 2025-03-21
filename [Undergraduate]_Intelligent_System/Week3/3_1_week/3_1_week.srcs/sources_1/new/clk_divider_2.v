`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/18 21:58:23
// Design Name: 
// Module Name: clk_divider_2
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


module clk_divider_2(
    input wire clk,         // input clock
    input wire resetn,      // reset 신호 (active low)
    output wire clk_50hz    // 분기된 clock
    );

    wire q1, q2;            // 두 개의 D flip-flop 출력 신호
    wire q1_bar, q2_bar;    // 두 개의 negative D flip-flop 출력 신호

    // 첫 번째 D flip-flop (DFF1)
    d_ff dff1 (
        .clk(clk),          // clock
        .resetn(resetn),    // reset
        .D(q1_bar),         // 버튼 입력
        .Q(q1),             // 첫 번째 Flip Flop 출력
        .Q_bar(q1_bar)      // 두 번째 Flip Flop 반전된 출력
    );

    // 두 번째 D flip-flop (DFF2)
    d_ff dff2 (
        .clk(q1),               // clock
        .resetn(resetn),        // reset
        .D(q2_bar),             // 두 번째 Flip Flop의 출력을 반전시켜 받음
        .Q(q2),                 // 두 번째 Flip Flop 출력
        .Q_bar(q2_bar)          // 두 번째 Flip Flop 반전된 출력
    );
    
    assign clk_50hz = q2;
    
endmodule
