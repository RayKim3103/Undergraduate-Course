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
    
/*** 입출력 포트 설명 ***/
/* reset: */ 
/*    reset이 1일 때는 모듈 동작 X, */ 
/*    reset이 1->0으로 갈 때 모든 register들 초기화 */
/* btn1, btn2, btn3: */ 
/*    ITEM FILLING, SELLING, COIN INSERTING 때 어떤 ITEM or COIN을 선택하는 지 고르는 신호 */
/* LED들: */
/*    vending machine이 켜졌는가, */ 
/*    그리고, 현재 vending machine이 어떤 모드인가를 나타냄 */
/* aa~ag, cat: */ 
/*    SSD 디스플레이를 위한 신호 */
/**********************/

    // sw3와 같음, 하지만, enable신호로 두는 것이 가독성이 좋아서 enable wire를 만듬
    wire enable;
    // sw3에 따라 clk을 그대로 받거나 clk이 끊기도록 하는 역할 (Hard Reset)                        
    wire clk_in; 
    // 모듈에 사용하는 clk_50Mhz 및 cat신호에 사용하는 clk_50hz
    wire clk_50Mhz, clk_50hz;           
    // switch에 따른 state 구분을 위해 2bit mode wire사용
    wire [1:0] mode;                    
    // 가장 최근에 선택한 item을 구분해주는 2bit 신호
    wire [1:0] item_select;             
    // FILLING모드 enable신호
    wire fill_en;                       
    // vending machine에 넣은 돈
    wire [5:0] balance;                 
    // vending machine의 아이템 재고
    wire [2:0] stock1, stock2, stock3; 
    // Hard reset 감지를 위해 1cycle이전의 reset을 저장 
    wire reset_prev;         
    
    wire sw1_pos, sw1_neg, sw2_pos, sw2_neg;          
    
    // 가독성을 위해 sw3를 enable로 표기. assign 사용
    assign enable = sw3;                              
    // Vending Machine OFF시 clk을 차단
    assign clk_in = (enable && ~reset) ? clk : 1'b0;
    // LED5는 Vending Machine이 켜져 있는지를 판단, (sw3 = enable)  
    assign LED5 = enable; 
    // SSD를 위한 cat신호             
    assign cat = clk_50hz;
    
    clk_divider u_clk_divider(
        .clk(clk_in),
//        .reset(reset),
        .reset_prev(reset_prev),
        .clk_50hz(clk_50hz),
        .clk_50Mhz(clk_50Mhz)
    );
    
    reset_prev u_reset_prev(
        .clk(clk),
        .reset(reset),
        .reset_prev(reset_prev)
    );
    
    debouncer debouncer1(
        .clk(clk_50Mhz),
        .in(btn1),
        .out_pos(btn1_clean)
//        .out_neg()
    );
    debouncer debouncer2(
        .clk(clk_50Mhz),
        .in(btn2),
        .out_pos(btn2_clean)
    );
    debouncer debouncer3(
        .clk(clk_50Mhz),
        .in(btn3),
        .out_pos(btn3_clean)
    );
    
    debouncer debouncer4(
        .clk(clk_50Mhz),
        .in(sw1),
        .out_pos(sw1_pos),
        .out_neg(sw1_neg)
    );
    debouncer debouncer5(
        .clk(clk_50Mhz),
        .in(sw2),
        .out_pos(sw2_pos),
        .out_neg(sw2_neg)
    );

    mode_ctrl u_mode_ctrl(
        .clk(clk_50Mhz),
//        .reset(reset),
        .reset_prev(reset_prev ),
        .sw2_pos(sw2_pos), 
        .sw2_neg(sw2_neg),
        .sw1_pos(sw1_pos),
        .sw1_neg(sw1_neg),
         
        .mode(mode), .enable(enable)
    );
    
    mode_ctrl_signals u_mode_ctrl_signals(
        .clk(clk_50Mhz), 
//        .reset(reset), 
        .reset_prev(reset_prev),
        .mode(mode),
        .btn1(btn1_clean), .btn2(btn2_clean), .btn3(btn3_clean),
        
        .item_select(item_select),
        .fill_en(fill_en),
        .stock1(stock1), .stock2(stock2), .stock3(stock3),
        .balance(balance)
    );

    ssd_ctrl u_ssd_ctrl(
        .clk(clk_50Mhz),
//        .reset(reset),
        .reset_prev(reset_prev),
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
        if (mode == 2'b00) begin        // SELLING 모드
            LED2 <= (stock1 == 0);
            LED3 <= (stock2 == 0);
            LED4 <= (stock3 == 0);
        end 
        else if (mode == 2'b10) begin   // FILLING 모드
            case (item_select)              
                2'b01:  begin 
                    LED2 <= 1; LED3 <= 0; LED4 <= 0; 
                end
                2'b10:  begin 
                    LED2 <= 0; LED3 <= 1; LED4 <= 0; 
                end
                2'b11:  begin 
                    LED2 <= 0; LED3 <= 0; LED4 <= 1; 
                end
                default: begin 
                    LED2 <= 0; LED3 <= 0; LED4 <= 0; 
                end
            endcase
        end 
        else begin  // SELLING, FILLING 모드 아니면 다 꺼지도록
            LED2 <= 0; LED3 <= 0; LED4 <= 0;
        end
    end

endmodule
