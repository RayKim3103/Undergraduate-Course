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
    input wire reset,
//    input wire [1:0] mode,
    input wire fill_en,
    input wire [1:0] item_select,
    input wire [2:0] stock1, stock2, stock3,
    input wire [6:0] balance,
    input wire cat,
    output wire aa, ab, ac, ad, ae, af, ag
//    output wire cat
    );

    reg [5:0] count;

    // 표시할 값 결정
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            count <= 4'd0;
        end
//        else if (mode == 2'b10) begin // ITEM FILLING 모드
       else if (fill_en == 1) begin
            case (item_select)  // count Max Value: 5
                2'b01: count <= stock1;
                2'b10: count <= stock2;
                2'b11: count <= stock3;
                default: count <= 4'd0;
            endcase
        end 
        else begin              // SELLING & COIN INSERTING 모드
            count <= balance;    // Max Value: 50
        end
    end

    // 
    assign {aa, ab, ac, ad, ae, af, ag} = 
        (cat == 0) ? (
            ((count%10) == 4'd0)  ? 7'b1111110 : // 0
            ((count%10) == 4'd1)  ? 7'b0110000 : // 1
            ((count%10) == 4'd2)  ? 7'b1101101 : // 2
            ((count%10) == 4'd3)  ? 7'b1111001 : // 3
            ((count%10) == 4'd4)  ? 7'b0110011 : // 4
            ((count%10) == 4'd5)  ? 7'b1011011 : // 5
            ((count%10) == 4'd6)  ? 7'b1011111 : // 6
            ((count%10) == 4'd7)  ? 7'b1110000 : // 7
            ((count%10) == 4'd8)  ? 7'b1111111 : // 8
            ((count%10) == 4'd9)  ? 7'b1110011 : // 9
            7'b1111110                      // OFF 상태 (기본값)
        ) : (
            ((count/10) == 4'd0) ? 7'b1111110 :  // 0
            ((count/10) == 4'd1) ? 7'b0110000 :  // 1
            ((count/10) == 4'd2) ? 7'b1101101 :  // 2
            ((count/10) == 4'd3) ? 7'b1111001 :  // 3
            ((count/10) == 4'd4) ? 7'b0110011 :  // 4
            ((count/10) == 4'd5) ? 7'b1011011 :  // 5
            7'b1111110                      // OFF 상태 (기본값)
        );
    
endmodule
