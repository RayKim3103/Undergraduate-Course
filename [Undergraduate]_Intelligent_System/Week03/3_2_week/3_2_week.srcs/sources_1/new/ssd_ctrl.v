`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/17 20:14:59
// Design Name: 
// Module Name: ssd_crtl
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


module ssd_ctrl(
    input wire clk_50hz,                    // 50Hz 클럭
    input wire [3:0] count,                 // 0~15 (4-bit) count 
    input wire resetn,
    
    output wire aa, ab, ac, ad, ae, af, ag, // 7-segment output
    output wire cat                         // 7-segment 2개의 출력 중 1개 선택
    );

    // cat 신호는 50Hz 주기로 설정
    assign cat = clk_50hz;
    
    // 7-Segment Display Mapping
    // cat이 0일 때는 10의 자리 수 표기
    // cat이 1일 때는 1의 자리 수 표기
    assign {aa, ab, ac, ad, ae, af, ag} = 
        (cat == 0) ? (
            (count == 4'd0)  ? 7'b1111110 : // 0
            (count == 4'd1)  ? 7'b0110000 : // 1
            (count == 4'd2)  ? 7'b1101101 : // 2
            (count == 4'd3)  ? 7'b1111001 : // 3
            (count == 4'd4)  ? 7'b0110011 : // 4
            (count == 4'd5)  ? 7'b1011011 : // 5
            (count == 4'd6)  ? 7'b1011111 : // 6
            (count == 4'd7)  ? 7'b1110000 : // 7
            (count == 4'd8)  ? 7'b1111111 : // 8
            (count == 4'd9)  ? 7'b1110011 : // 9
            (count == 4'd10) ? 7'b1111110 : // 0
            (count == 4'd11) ? 7'b0110000 : // 1
            (count == 4'd12) ? 7'b1101101 : // 2
            (count == 4'd13) ? 7'b1111001 : // 3
            (count == 4'd14) ? 7'b0110011 : // 4
            (count == 4'd15) ? 7'b1011011 : // 5
            7'b1111110                      // OFF 상태 (기본값)
        ) : (
            (count == 4'd0)  ? 7'b1111110 : // 0
            (count == 4'd1)  ? 7'b1111110 : // 0
            (count == 4'd2)  ? 7'b1111110 : // 0
            (count == 4'd3)  ? 7'b1111110 : // 0
            (count == 4'd4)  ? 7'b1111110 : // 0
            (count == 4'd5)  ? 7'b1111110 : // 0
            (count == 4'd6)  ? 7'b1111110 : // 0
            (count == 4'd7)  ? 7'b1111110 : // 0
            (count == 4'd8)  ? 7'b1111110 : // 0
            (count == 4'd9)  ? 7'b1111110 : // 0
            (count == 4'd10) ? 7'b0110000 : // 1
            (count == 4'd11) ? 7'b0110000 : // 1
            (count == 4'd12) ? 7'b0110000 : // 1
            (count == 4'd13) ? 7'b0110000 : // 1
            (count == 4'd14) ? 7'b0110000 : // 1
            (count == 4'd15) ? 7'b0110000 : // 1
            7'b1111110                      // OFF 상태 (기본값)
        );
    
endmodule
