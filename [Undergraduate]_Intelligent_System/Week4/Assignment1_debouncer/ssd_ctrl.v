`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/23 12:03:10
// Design Name: 
// Module Name: ssd_ctrl
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
    input wire clk,
//    input wire reset,
    input wire reset_prev,
    input wire fill_en,
    input wire [1:0] item_select,
    input wire [2:0] stock1, stock2, stock3,
    input wire [5:0] balance,
    input wire cat,
    output wire aa, ab, ac, ad, ae, af, ag
    );
    
    // 0~50범위의 숫자를 저장할 register (= count)
    reg [5:0] count;    
    // 미리 1의 자리 수를 계산한 것을 전달할 wire
    wire [3:0] display_one;   
    // 미리 10의 자리 수를 계산한 것을 전달할 wire
    wire [3:0] display_ten;   

    // 표시할 값 결정
    always @(posedge clk or posedge reset_prev) begin
        if (reset_prev) begin  // reset 스위치를 내렸을 때를 감지
            count <= 4'd0;
        end
       else if (fill_en == 1) begin  // ITEM FILLING 모드
            // SSD는 최근 Item의 stock을 보여줌 
            // (item_select: 최근에 선택한 Item을 저장하는 reg)
            case (item_select)              
                2'b01: count <= stock1;
                2'b10: count <= stock2;
                2'b11: count <= stock3;
                default: count <= 4'd0;
            endcase
        end 
        else begin  // SELLING & COIN INSERTING 모드
            count <= balance;  // SSD는 balance의 값을 보여줌
        end
    end
    
    // 미리 1의 자리 수를 계산
    assign display_one = (count % 10);  
    // 미리 10의 자리 수를 계산
    assign display_ten = (count / 10);      
    
    // SSD의 cat에 따른 aa~ag신호 할당
    assign {aa, ab, ac, ad, ae, af, ag} = 
        (cat == 0) ? (
            (display_one == 4'd0)  ? 7'b1111110 : // 0
            (display_one == 4'd1)  ? 7'b0110000 : // 1
            (display_one == 4'd2)  ? 7'b1101101 : // 2
            (display_one == 4'd3)  ? 7'b1111001 : // 3
            (display_one == 4'd4)  ? 7'b0110011 : // 4
            (display_one == 4'd5)  ? 7'b1011011 : // 5
            (display_one == 4'd6)  ? 7'b1011111 : // 6
            (display_one == 4'd7)  ? 7'b1110000 : // 7
            (display_one == 4'd8)  ? 7'b1111111 : // 8
            (display_one == 4'd9)  ? 7'b1110011 : // 9
            7'b1111110                      // OFF 상태 (기본값)
        ) : (
            (display_ten == 4'd0) ? 7'b1111110 :  // 0
            (display_ten == 4'd1) ? 7'b0110000 :  // 1
            (display_ten == 4'd2) ? 7'b1101101 :  // 2
            (display_ten == 4'd3) ? 7'b1111001 :  // 3
            (display_ten == 4'd4) ? 7'b0110011 :  // 4
            (display_ten == 4'd5) ? 7'b1011011 :  // 5
            7'b1111110                      // OFF 상태 (기본값)
        );
    
endmodule
