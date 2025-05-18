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
//    input wire reset,
    input wire reset_prev,
    input wire [1:0] mode,
    input wire btn1, btn2, btn3,
    
    // 최근에 선택한 item 저장
    output reg [1:0] item_select,               
    // FILLING모드인지 저장
    output wire fill_en,                        
    // 각각의 Item의 재고 
    output reg [2:0] stock1, stock2, stock3,    
    // vending machine에 충전된 돈의 양
    output reg [5:0] balance                    
    );
    
    // 가독성을 위해, 코인의 가격을 미리 정의
    parameter coin_1 = 1, coin_2 = 5, coin_3 = 10;      
    // 가독성을 위해, 아이템의 가격을 미리 정의
    parameter price_1 = 3, price_2 = 5, price_3 = 7;    
    // COIN INSERTING 모드, SELLING 모드 각각의 enable 신호 저장하는 register
    wire coin_en, sell_en;                               
    
    // 각각의 Mode를 가독성이 좋게, ~~~_en과 같이 이름을 다르게 하여 다시 할당
    assign sell_en = (mode == 2'b00);
    assign coin_en = (mode == 2'b01 || mode == 2'b11);
    assign fill_en = (mode == 2'b10);
    
    // store what is the most recently selected Item in FILLING mode
    always @(posedge clk or posedge reset_prev) begin                
        if(reset_prev) begin    // reset 스위치를 내렸을 때를 감지
            item_select <= 2'b00;
        end
        // 버튼 입력에 따른 최근 item선택을 저장하는 register값 변경
        else if (fill_en) begin    
            if (btn1) item_select <= 2'b01;
            else if (btn2) item_select <= 2'b10;
            else if (btn3) item_select <= 2'b11;
            else item_select <= item_select;
        end 
        else begin  // 버튼 입력이 없으면 최근 item선택 유지
            item_select <= item_select;        
        end
    end
    
    
    // Mode와 stock의 양, balance의 양에 따라 채우거나 구매가 가능한지
    wire stock1_fill_en;    
    wire stock2_fill_en;
    wire stock3_fill_en;
    
    wire stock1_sell_en;
    wire stock2_sell_en;
    wire stock3_sell_en;
    
    // FILLING모드이고, 재고 넘치지 않을 때, stock1 충전 가능
    assign stock1_fill_en = (fill_en && stock1 < 5);                        
    // FILLING모드이고, 재고 넘치지 않을 때, stock2 충전 가능
    assign stock2_fill_en = (fill_en && stock2 < 5);                        
    // FILLING모드이고, 재고 넘치지 않을 때, stock3 충전 가능
    assign stock3_fill_en = (fill_en && stock3 < 5);                        

    // SELLING모드이고, 재고 있고, 돈 있을 때 stock1 구매 가능
    assign stock1_sell_en = (sell_en && stock1 > 0 && balance >= price_1);  
    // SELLING모드이고, 재고 있고, 돈 있을 때 stock2 구매 가능
    assign stock2_sell_en = (sell_en && stock2 > 0 && balance >= price_2);  
    // SELLING모드이고, 재고 있고, 돈 있을 때 stock3 구매 가능
    assign stock3_sell_en = (sell_en && stock3 > 0 && balance >= price_3);  
    
    // Item stock managing
    always @(posedge clk or posedge reset_prev) begin
        if (reset_prev) begin
            stock1 <= 0; stock2 <= 0; stock3 <= 0;
        end
        else begin  // stock*: Max Value: 5 / min Value: 0
            case({btn1, btn2, btn3})
                3'b100:begin  // FILL 모드이면 1개 채워 넣음
                    if (stock1_fill_en) 
                        stock1 <= stock1 + 1;      
                    else if (stock1_sell_en)  // SELL모드 돈이 있으면, 1개 팔음
                        stock1 <= stock1 - 1;
                    else stock1 <= stock1;
                end
                3'b010:begin
                    if (stock2_fill_en)  // FILL 모드이면 1개 채워 넣음
                        stock2 <= stock2 + 1;      
                    else if (stock2_sell_en)  // SELL모드 돈이 있으면, 1개 팔음
                        stock2 <= stock2 - 1;
                    else stock2 <= stock2;
                end
                3'b001:begin
                    if (stock3_fill_en)  // FILL 모드이면 1개 채워 넣음
                        stock3 <= stock3 + 1;      
                    else if (stock3_sell_en)  // SELL모드 돈이 있으면, 1개 팔음
                        stock3 <= stock3 - 1;
                    else stock3 <= stock3;
                end
                default: begin 
                    stock1 <= stock1;
                    stock2 <= stock2;
                    stock3 <= stock3;
                end 
            endcase
        end
    end
    
    // coin insertion managing
    always @(posedge clk or posedge reset_prev) begin
        if (reset_prev)
            balance <= 0;
        else begin
            case({btn1, btn2, btn3})
                3'b100:begin
                    // balance + coin_1(= 돈: 1)이 50 이하이면 1 채움
                    if(coin_en && ((balance + coin_1) <= 50)) 
                        balance <= balance + coin_1;
                    // SELL 모드이고 돈이 있으면, 물건 구매: balacne - 1
                    else if(stock1_sell_en)         
                        balance <= balance - price_1;
                    else balance <= balance;
                end
                3'b010:begin
                    // balance + coin_2(= 돈: 5)이 50 이하이면 5 채움
                    if(coin_en && ((balance + coin_2) <= 50))  
                        balance <= balance + coin_2;
                    // SELL 모드이고 돈이 있으면, 물건 구매: balacne - 5
                    else if (stock2_sell_en)                         
                        balance <= balance - price_2;
                    else balance <= balance;
                end
                3'b001:begin
                    // balance + coin_3(= 돈: 10)이 50 이하이면 10 채움
                    if(coin_en && ((balance + coin_3) <= 50))   
                        balance <= balance + coin_3;
                    // SELL 모드이고 돈이 있으면, 물건 구매: balacne - 10
                    else if (stock3_sell_en)                         
                        balance <= balance - price_3;
                    else balance <= balance;
                end
                default: balance <= balance;
            endcase         
        end
    end
endmodule
