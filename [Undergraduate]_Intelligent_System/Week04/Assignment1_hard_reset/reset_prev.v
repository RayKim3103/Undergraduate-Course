`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/23 21:20:09
// Design Name: 
// Module Name: reset_prev
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


module reset_prev(
    input wire clk,
    input wire reset,
    
    output wire reset_prev 
    );
    
    reg q1, q2; // 신호 delay를 위한 reg들
    
    // Flip Flop을 차례로 써서 reset신호를 delay시키면 유지한다.
    // 이 delay된 신호를 debouncer에서 썼던 logic과 같이 둘다 0인 부분만 살려 INV를 지나도록 한다.
    // 즉, reset끝나고 reset_prev는 1 clk cycle동안 1이 됨.
    // 결국, reset_prev의 rising edge를 감지하면 reset이 끝나고 바로 동작 시작 시 초기화 가능 
    always @(posedge clk or negedge reset) begin 
        if(!reset) begin
            q1 <= reset;
            q2 <= q1;
        end
        else begin
            q1 <= reset;
            q2 <= q1; // reset 이전 값 저장
        end
    end
    
    assign reset_prev = ~(q1 | ~q2); 
endmodule