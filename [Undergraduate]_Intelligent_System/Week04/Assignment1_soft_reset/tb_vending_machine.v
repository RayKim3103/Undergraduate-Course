`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/23 13:48:46
// Design Name: 
// Module Name: tb_vending_machine
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


module tb_vending_machine;

    // Inputs
    reg clk;
    reg reset;
    reg btn1, btn2, btn3;
    reg sw3, sw2, sw1;

    // Outputs
    wire LED5, LED4, LED3, LED2;
    wire aa, ab, ac, ad, ae, af, ag, cat;

    // Instantiate the Unit Under Test (UUT)
    vending_machine uut (
        .clk(clk),
        .reset(reset),
        .btn1(btn1), .btn2(btn2), .btn3(btn3),
        .sw3(sw3), .sw2(sw2), .sw1(sw1),
        .LED5(LED5), .LED4(LED4), .LED3(LED3), .LED2(LED2),
        .aa(aa), .ab(ab), .ac(ac), .ad(ad), .ae(ae), .af(af), .ag(ag), .cat(cat)
    );

    // Clock generation (100MHz)
    initial clk = 0;
    always #5 clk = ~clk; // 10ns period → 100MHz

    initial begin
        // 초기화
        reset = 1; btn1 = 0; btn2 = 0; btn3 = 0;
        sw3 = 0; sw2 = 0; sw1 = 0;
        #50;

        reset = 0; // 시스템 시작
        #50
        sw3 = 1;   // 자판기 ON
        #50;

        // === 아이템 채우기 모드 ===
        sw2 = 1; sw1 = 0; // FILLING MODE
        #20;

        btn1 = 1; // Item 1 충전
        #20;
        btn1 = 0;
        #50;
        btn1 = 1; // Item 1 충전
        #20;
        btn1 = 0;
        #50;
        btn2 = 1; // Item 2 충전
        #20;
        btn2 = 0;
        #50; 
        btn3 = 1; // Item 3 충전
        #20;
        btn3 = 0;
        #50; 

        // === 동전 삽입 모드 ===
        sw2 = 1; sw1 = 1; // COIN MODE
        #20;

        btn1 = 1; // \1
        #20;
        btn1 = 0;
        #50;
        btn2 = 1; // \5
        #20;
        btn2 = 0;
        #50; 
        btn3 = 1; // \10
        #20;
        btn3 = 0;
        #50; 

        // === 판매 모드 ===
        sw2 = 0; sw1 = 0; // SELLING MODE
        #20;

        btn2 = 1; // Item 2 선택 (가격: 5)
        #20;
        btn2 = 0;
        #50; 
        btn3 = 1; // Item 3 선택 (가격: 7)
        #20;
        btn3 = 0;
        #50; 
        
        // === Vending Machine OFF & ON ===
        sw3 = 0;   // 자판기 OFF
        #300
        // === 동전 삽입 모드 ===
        sw2 = 1; sw1 = 1; // COIN MODE
        #20;

        btn1 = 1; // \1
        #20;
        btn1 = 0;
        #50;
        btn2 = 1; // \5
        #20;
        btn2 = 0;
        #50; 
        btn3 = 1; // \10
        #20;
        btn3 = 0;
        #50; 
        // =======================
        sw3 = 1;   // 자판기 ON
        #300;
        // ========================
        
        // === 아이템 채우기 모드 ===
        sw2 = 1; sw1 = 0; // FILLING MODE
        #200;

        btn1 = 1; // Item 1 충전
        #20;
        btn1 = 0;
        #50;
        btn2 = 1; // Item 2 충전
        #20;
        btn2 = 0;
        #50; 
        btn3 = 1; // Item 3 충전
        #20;
        btn3 = 0;
        #50; 
        
        // === reset ===
        reset = 1;
        #50;
        reset = 0;
        #50;

        // === 상태 확인을 위한 지연 ===
        #200;

        $display("Test Finished");
        $stop;
    end

endmodule
