`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/23 12:02:13
// Design Name: 
// Module Name: fsm_ctrl
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


    module mode_ctrl_signals(
    input wire clk,
    input wire reset,
    input wire [1:0] mode,
    input wire btn1, btn2, btn3,
    
//    input wire [3:0] price,
    
    output reg [1:0] item_select,
    output reg fill_en,
    
    output reg [2:0] stock1, stock2, stock3,
    output reg [6:0] balance
    );
    parameter coin_1 = 1, coin_2 = 5, coin_3 = 10;
    parameter price_1 = 3, price_2 = 5, price_3 = 7;
    reg coin_en, sell_en;    //, fill_en;
    
    // set enable signals according to corresponding mode
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            fill_en <= 0; 
            coin_en <= 0; 
            sell_en <= 0;
//            item_select <= 0;
        end 
        else begin
            fill_en <= (mode == 2'b10);
            coin_en <= (mode == 2'b01);
            sell_en <= (mode == 2'b00);
        end
    end
    
    // store what is the most recently selected Item
    always @(posedge clk or posedge reset) begin
        if(reset) begin
            item_select <= 2'b00;
        end
        else if (btn1) begin
            if (fill_en) item_select <= 2'b01;
            else if (sell_en) item_select <= 2'b01;
        end 
        else if (btn2) begin
            if (fill_en) item_select <= 2'b10;
            else if (sell_en) item_select <= 2'b10;
        end 
        else if (btn3) begin
            if (fill_en) item_select <= 2'b11;
            else if (sell_en) item_select <= 2'b11;
        end 
        else begin
            item_select <= item_select;
        end
    end
    
    // Item stock managing
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            stock1 <= 0; stock2 <= 0; stock3 <= 0;
        end
        else begin
                if (btn1) begin
                    if (fill_en && stock1 < 5) stock1 <= stock1 + 1;
                    else if (sell_en && stock1 > 0) stock1 <= stock1 - 1;
                end
                if (btn2) begin
                    if (fill_en && stock2 < 5) stock2 <= stock2 + 1;
                    else if (sell_en && stock2 > 0) stock2 <= stock2 - 1;
                end
                if (btn3) begin
                    if (fill_en && stock3 < 5) stock3 <= stock3 + 1;
                    else if (sell_en && stock3 > 0) stock3 <= stock3 - 1;
                end
        end
    end
    
    // coin insertion managing
    always @(posedge clk or posedge reset) begin
        if (reset)
            balance <= 0;
        else begin
            if (btn1) begin
                if (coin_en && balance < 50)       // 초과 시에는 50으로 맞추나 ????? (ex. 47 + 10 -> 47 or 50??????)
                    balance <= balance + coin_1;
                else if (sell_en && balance >= price_1)
                    balance <= balance - price_1;
            end
            else if (btn2) begin
                if (coin_en && balance < 46)       // 초과 시에는 50으로 맞추나 ????? (ex. 47 + 10 -> 47 or 50??????)
                    balance <= balance + coin_2;
                else if (sell_en && balance >= price_2)
                    balance <= balance - price_2;
            end
            else if (btn3) begin
                if (coin_en && balance < 41)       // 초과 시에는 50으로 맞추나 ????? (ex. 47 + 10 -> 47 or 50??????)
                    balance <= balance + coin_3;
                else if (sell_en && balance >= price_3)
                    balance <= balance - price_3;
            end
        end
    end
endmodule
