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
    input wire reset_prev,
    input wire [1:0] mode,
    input wire btn1, btn2, btn3,

    output reg [1:0] item_select,               // 최근에 선택한 item 저장
    output wire fill_en,                         // FILLING모드인지 저장
    output reg [2:0] stock1, stock2, stock3,    // 각각의 Item의 재고
    output reg [6:0] balance                    // vending machine에 충전된 돈의 양
    );
    
    parameter coin_1 = 1, coin_2 = 5, coin_3 = 10;      // 가독성을 위해, 코인의 가격을 미리 정의
    parameter price_1 = 3, price_2 = 5, price_3 = 7;    // 가독성을 위해, 아이템의 가격을 미리 정의
    wire coin_en, sell_en;                               // COIN INSERTING 모드, SELLING 모드 각각의 enable 신호 저장하는 register
    
    assign sell_en = (mode == 2'b00);
    assign coin_en = (mode == 2'b01);
    assign fill_en = (mode == 2'b10);
    
    // store what is the most recently selected Item
    always @(posedge clk or posedge reset_prev) begin                
        if(reset_prev) begin                  // reset 스위치를 내렸을 때를 감지
            item_select <= 2'b00;
        end
        else if (btn1) begin                  // 버튼 입력에 따른 최근 item선택을 저장하는 register값 변경
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
            item_select <= item_select;        // 버튼 입력이 없으면 최근 item선택 유지
        end
    end
    
    wire stock1_fill_en;    
    wire stock2_fill_en;
    wire stock3_fill_en;
    
    wire stock1_sell_en;
    wire stock2_sell_en;
    wire stock3_sell_en;
    
    assign stock1_fill_en = (fill_en && stock1 < 5);                        // FILLING모드이고, 재고 넘치지 않을 때, stock1 충전 가능
    assign stock2_fill_en = (fill_en && stock2 < 5);                        // FILLING모드이고, 재고 넘치지 않을 때, stock2 충전 가능
    assign stock3_fill_en = (fill_en && stock3 < 5);                        // FILLING모드이고, 재고 넘치지 않을 때, stock3 충전 가능
    
    assign stock1_sell_en = (sell_en && stock1 > 0 && balance >= price_1);  // SELLING모드이고, 재고 있고, 돈 있을 때 stock1 구매 가능
    assign stock2_sell_en = (sell_en && stock2 > 0 && balance >= price_2);  // SELLING모드이고, 재고 있고, 돈 있을 때 stock2 구매 가능
    assign stock3_sell_en = (sell_en && stock3 > 0 && balance >= price_3);  // SELLING모드이고, 재고 있고, 돈 있을 때 stock3 구매 가능
    
    // Item stock managing
    always @(posedge clk or posedge reset_prev) begin
        if (reset_prev) begin
            stock1 <= 0; stock2 <= 0; stock3 <= 0;
        end
        else begin  // stock* Max Value: 5 (stock은 5이하이다.), min Value: 0
                if (btn1) begin
                    if (stock1_fill_en) 
                        stock1 <= stock1 + 1;      // FILL 모드이면 1개 채워 넣음
                    else if (stock1_sell_en)       // SELL 모드이고 돈이 있으면, 1개 팔음
                        stock1 <= stock1 - 1;
                end
                if (btn2) begin
                    if (stock2_fill_en) 
                        stock2 <= stock2 + 1;      // FILL 모드이면 1개 채워 넣음
                    else if (stock2_sell_en)       // SELL 모드이고 돈이 있으면, 1개 팔음
                        stock2 <= stock2 - 1;
                end
                if (btn3) begin
                    if (stock3_fill_en) 
                        stock3 <= stock3 + 1;      // FILL 모드이면 1개 채워 넣음
                    else if (stock3_sell_en)       // SELL 모드이고 돈이 있으면, 1개 팔음
                        stock3 <= stock3 - 1;
                end
        end
    end
    
    // coin insertion managing
    always @(posedge clk or posedge reset_prev) begin
        if (reset_prev)
            balance <= 0;
        else begin
            if (btn1) begin
                if (coin_en && ((balance + coin_1) <= 50))       // balance + coin_1(= 돈: \1)이 50 이하이면 \1 채움
                    balance <= balance + coin_1;
                else if (stock1_sell_en)                         // SELL 모드이고 돈이 있으면, 물건 구매: balacne - \1
                    balance <= balance - price_1;
            end
            else if (btn2) begin
                if (coin_en && ((balance + coin_2) <= 50))       // balance + coin_2(= 돈: \5)이 50 이하이면 \5 채움
                    balance <= balance + coin_2;
                else if (stock2_sell_en)                         // SELL 모드이고 돈이 있으면, 물건 구매: balacne - \5
                    balance <= balance - price_2;
            end
            else if (btn3) begin
                if (coin_en && ((balance + coin_3) <= 50))       // balance + coin_3(= 돈: \10)이 50 이하이면 \10 채움
                    balance <= balance + coin_3;
                else if (stock3_sell_en)                         // SELL 모드이고 돈이 있으면, 물건 구매: balacne - \10
                    balance <= balance - price_3;
            end
        end
    end
endmodule
