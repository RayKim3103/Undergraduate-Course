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
//    input wire sw2, sw1,        // switch입력
    input   wire sw1_pos,
    input   wire sw1_neg,
    input   wire sw2_pos,
    input   wire sw2_neg,
    
    output reg [1:0] mode,      // switch에 따른 모드 4가지(IDLE 포함) 따라서, 2bit
    output wire enable          // sw3를 enable로 바꿔주기 위한 wire
    );
    
    localparam  COIN_2  = 2'b11; 
    localparam  SELLING = 2'b00;
    localparam  COIN    = 2'b01;
    localparam  FILLING  = 2'b10;
    
    reg [1:0] n_mode;
    
    always @(posedge clk or posedge reset_prev) begin 
        if (reset_prev) mode <= SELLING;
        else mode <= n_mode;
    end
    
    always @(*) begin
        case(mode)
            COIN_2    : begin
                if(sw1_neg && sw2_neg) n_mode = SELLING;
                else if(sw1_neg) n_mode = FILLING;
                else if(sw2_neg) n_mode = COIN;
                else n_mode = COIN_2;
            end
            SELLING    : begin
                if(sw1_pos && sw2_pos) n_mode = COIN_2;
                else if(sw1_pos) n_mode = COIN;
                else if(sw2_pos) n_mode = FILLING;
                else n_mode = SELLING;
            end
            COIN: begin 
                if(sw1_neg && sw2_pos) n_mode = FILLING;
                else if (sw1_neg) n_mode = SELLING;
                else if (sw2_pos) n_mode = COIN_2;
                else n_mode = COIN;
            end
            FILLING: begin
                if(sw1_pos && sw2_neg) n_mode = COIN; 
                else if(sw1_pos) n_mode = COIN_2;
                else if(sw2_neg) n_mode = SELLING;
                else n_mode = FILLING;
            end
            default  : n_mode = SELLING;
        endcase
    end

endmodule
