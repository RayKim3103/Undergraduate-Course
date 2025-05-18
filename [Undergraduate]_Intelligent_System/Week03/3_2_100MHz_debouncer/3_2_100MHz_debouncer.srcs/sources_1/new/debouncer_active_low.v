`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/19 13:31:39
// Design Name: 
// Module Name: debouncer_active_low
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


module debouncer_active_low(
    input wire clk,
    input wire btn_in,    // 버튼 입력 신호 (노이즈 존재)
    output wire btn_out    // Debounce된 버튼 출력
    );
        
    reg q1, q2; // 두 개의 D 플립플롭

    always @(posedge clk) begin
        q1 <= btn_in;  // 첫 번째 DFF
        q2 <= q1;      // 두 번째 DFF
    end
    
    assign btn_out = q1 | ~q2; // q1과 ~q2가 OR게이트 지남 
endmodule
