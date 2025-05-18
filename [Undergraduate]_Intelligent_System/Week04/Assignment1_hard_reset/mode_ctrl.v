`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/23 12:01:35
// Design Name: 
// Module Name: mode_ctrl
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


module mode_ctrl(
    input wire clk,
//    input wire reset,
    input wire reset_prev,      // reset의 negedge를 감지하여 High로 가는 신호
    input wire sw2, sw1,        // switch입력
    
    output reg [1:0] mode,      // switch에 따른 모드 4가지(IDLE 포함) 따라서, 2bit
    output wire enable          // sw3를 enable로 바꿔주기 위한 wire
    );
    
    always @(posedge clk or posedge reset_prev) begin
        if(reset_prev)
            mode <= 2'b11;              // reset시 모드 초기화 (IDLE모드라고 생각하면 됨)
        else begin
            case ({sw2, sw1})
                2'b00: mode <= 2'b00;   // SELLING
                2'b01: mode <= 2'b01;   // COIN INSERT
                2'b11: mode <= 2'b01;   // COIN INSERT
                2'b10: mode <= 2'b10;   // ITEM FILLING
                default: mode <= 2'b11; // IDLE모드로 잡음
            endcase
        end
    end
endmodule
