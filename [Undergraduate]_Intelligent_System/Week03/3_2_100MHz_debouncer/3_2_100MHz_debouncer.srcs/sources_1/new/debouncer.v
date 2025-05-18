`timescale 1ns / 1ps

module debouncer(
    input wire clk,
    input wire btn_in,    // 버튼 입력 신호 (Jitter 존재)
    output wire btn_out    // Debounce된 버튼 출력
    );
    
    reg q1, q2; // 두 개의 D 플립플롭

    always @(posedge clk) begin
        q1 <= btn_in;  // 첫 번째 DFF
        q2 <= q1;      // 두 번째 DFF
    end
    assign btn_out = q2;    
//    assign btn_out = q1 & ~q2; // q1과 ~q2가 AND게이트 지남
endmodule