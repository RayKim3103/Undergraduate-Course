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
    input wire reset,
    input wire sw3, sw2, sw1,
    
    output reg [1:0] mode,
    output wire enable
    );
    
    assign enable = sw3;
    
    always @(posedge clk or posedge reset) begin
        if(reset)
            mode <= 2'b11;              // 동작 X모드 --------------> 디버깅용
//        else if (!enable)             // !enable이면 clock = 0이 되어 이 구문 없어도 될 듯    
//            mode <= 2'b11;            // 동작 X모드 --------------> 디버깅용
        else begin
            case ({sw2, sw1})
                2'b00: mode <= 2'b00;   // SELLING
                2'b01: mode <= 2'b01;   // COIN INSERT
                2'b11: mode <= 2'b01;   // COIN INSERT
                2'b10: mode <= 2'b10;   // ITEM FILLING
                default: mode <= 2'b11; // 동작 X모드 --------------> 디버깅용
            endcase
        end
    end
endmodule
