`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/18 21:48:59
// Design Name: 
// Module Name: d_ff
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


module d_ff(
    input wire clk,         // 클록 신호
    input wire resetn,      // 비동기 리셋 (active low)
    input wire D,           // 데이터 입력
    output reg Q,           // 출력
    output wire Q_bar
    );
    
    // resetn이 0일 때 Q는 0
    always @(posedge clk or negedge resetn) begin
        if (~resetn)        // resetn이 0일 때 리셋
            Q <= 0;
        else
            Q <= D;         // rising edge에서 D 값을 Q로 전달
    end
    
    assign Q_bar = ~Q;
endmodule
