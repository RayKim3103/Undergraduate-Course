`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/10/05 13:15:21
// Design Name: 
// Module Name: vending_machine
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


module vending_machine(
    input wire clk,
    input wire reset,
    input wire btn1, btn2, btn3,
    input wire sw3, sw2, sw1,
    output wire LED5, 
    output reg LED4, LED3, LED2,
    output wire aa, ab, ac, ad, ae, af, ag, cat
    );
    
    wire clk_in;
    wire clk_50Mhz, clk_50hz;
    wire [1:0] mode;
    wire enable;
    wire [1:0] item_select;
    wire fill_en;
    wire [6:0] balance;
    wire [2:0] stock1, stock2, stock3;
//    wire [3:0] price;
    
    reg reset_prev;
    
    assign clk_in = (enable && ~reset) ? clk : 1'b0;
    assign LED5 = enable;
    assign cat = clk_50hz;      // -> 맞겠지???????????
    
//    assign price = (item_select == 2'b01) ? 3 :
//                (item_select == 2'b10) ? 5 :
//                (item_select == 2'b11) ? 7 : 0;
                
    always @(posedge clk)
        reset_prev <= reset; // reset 이전 값 저장
    
    clk_divider u_clk_divider(
        .clk(clk_in),
        .reset(reset),
        .clk_50hz(clk_50hz),
        .clk_50Mhz(clk_50Mhz)
    );
    
    debouncer debouncer1(
        .clk(clk_50Mhz),
        .btn_in(btn1),
        .btn_out(btn1_clean)
    );
    debouncer debouncer2(
        .clk(clk_50Mhz),
        .btn_in(btn2),
        .btn_out(btn2_clean)
    );
    debouncer debouncer3(
        .clk(clk_50Mhz),
        .btn_in(btn3),
        .btn_out(btn3_clean)
    );

    mode_ctrl u_mode_ctrl(
        .clk(clk_50Mhz),
        .reset(reset),
        .sw3(sw3), .sw2(sw2), .sw1(sw1),
         
        .mode(mode), .enable(enable)
    );
    
    mode_ctrl_signals u_mode_ctrl_signals(
        .clk(clk_50Mhz), .reset(reset), 
        .mode(mode),
        .btn1(btn1_clean), .btn2(btn2_clean), .btn3(btn3_clean),
        
//        .price(price),
        
        .item_select(item_select),
        .fill_en(fill_en),
        
        .stock1(stock1), .stock2(stock2), .stock3(stock3),
        .balance(balance)
    );

    ssd_ctrl u_ssd_ctrl(
        .clk(clk_50Mhz),
        .reset(reset),
//        .mode(mode),
        .fill_en(fill_en),
        .item_select(item_select),
        .stock1(stock1),
        .stock2(stock2),
        .stock3(stock3),
        .balance(balance),
        .cat(cat),
        
        .aa(aa), .ab(ab), .ac(ac), .ad(ad), .ae(ae), .af(af), .ag(ag)
    );
    
    always @(*) begin
        if (mode == 2'b00) begin            // 판매 모드
            LED2 = (stock1 == 0);
            LED3 = (stock2 == 0);
            LED4 = (stock3 == 0);
        end 
        else if (mode == 2'b10) begin   // FILL 모드
             case (item_select)
                2'b01: {LED2, LED3, LED4} <= 3'b100;
                2'b10: {LED2, LED3, LED4} <= 3'b010;
                2'b11: {LED2, LED3, LED4} <= 3'b001;
                default: {LED2, LED3, LED4} <= 3'b000;
             endcase
        end
        else begin
            LED2 = 0; LED3 = 0; LED4 = 0;
        end
    end

endmodule
