`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/17 10:27:02
// Design Name: 
// Module Name: tb_top_counter
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


module tb_top_counter;

    // 테스트를 위한 신호 선언
    reg clk;
    reg resetn;
    reg in_up;
    reg in_down;

    wire aa, ab, ac, ad, ae, af, ag, cat;
    wire in_up_clean;
    wire in_down_clean;
    wire clk_50hz;
    wire [3:0] count;
//    wire resetn_clean;

    // DUT (Device Under Test) 인스턴스화
    top_counter uut (
        .clk(clk),
        .resetn(resetn),
        .in_up(in_up),
        .in_down(in_down),
        .aa(aa),
        .ab(ab),
        .ac(ac),
        .ad(ad),
        .ae(ae),
        .af(af),
        .ag(ag),
        .cat(cat),

        .in_up_clean(in_up_clean),
        .in_down_clean(in_down_clean),
        .clk_50hz(clk_50hz),
        .count(count)
//        .resetn_clean(resetn_clean),
    );

    // 100MHz 클럭 생성 (10ns 주기)
    always #5 clk = ~clk;

    // 테스트 시나리오
    initial begin
        // 초기화
        clk = 0;
        resetn = 1;
        in_up = 0;
        in_down = 0;

        // 리셋 신호 활성화
        #200 resetn = 0;
        #120 resetn = 1;

        // Jitter 발생 테스트
        // 정상적인 버튼 입력 전 노이즈를 발생시킴
        #100 in_up = 1; 
        #5 in_up = 0; #5 in_up = 1; #5 in_up = 0; #5 in_up = 1; // Jitter (빠른 변동)
        #100 in_up = 0; 
        #5 in_up = 1; #5 in_up = 0; #5 in_up = 1; #5 in_up = 0; // Jitter (빠른 변동)
        
        #100 in_up = 1; // 정상 입력
        #100 in_up = 0;

        
        #100 in_down = 1; 
        #5 in_down = 0; #5 in_down = 1; #5 in_down = 0; #5 in_down = 1; // Jitter
        #100 in_down = 0;
        #5 in_down = 1; #5 in_down = 0; #5 in_down = 1; #5 in_down = 0; // Jitter
        
        #100 in_down = 1; // 정상 입력
        #100 in_down = 0; 

        // 증가 버튼을 여러 번 눌러 최대값(9) 테스트
        repeat(8) begin
            #100 in_up = 1; #100 in_up = 0;
        end

        // 감소 버튼을 여러 번 눌러 최소값(0) 테스트
        repeat(10) begin
            #100 in_down = 1; #100 in_down = 0;
        end

        // 시뮬레이션 종료
        #1000;
        $finish;
    end

    // 결과를 모니터링
    initial begin
        $monitor("Time = %t | Reset = %b | Count = %b | SSD = %b%b%b%b%b%b%b | cat = %b",
                 $time, resetn, uut.count, aa, ab, ac, ad, ae, af, ag, cat);
    end

endmodule
