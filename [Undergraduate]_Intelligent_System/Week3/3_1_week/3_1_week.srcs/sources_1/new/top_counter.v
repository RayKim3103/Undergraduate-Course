`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/17 09:53:00
// Design Name: 
// Module Name: top_counter
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


module top_counter(
    input wire in_up,       // 증가 버튼
    input wire in_down,     // 감소 버튼
    input wire clk,         // 100 MHz FPGA 클럭 입력
    input wire resetn,      // Active-low 리셋 신호
    
    // 7-segment 디스플레이 출력
    output wire aa,
    output wire ab,
    output wire ac,
    output wire ad,
    output wire ae,
    output wire af,
    output wire ag,
    output wire cat
    
    // reg로 설계 시
//    output reg aa,
//    output reg ab,
//    output reg ac,
//    output reg ad,
//    output reg ae,
//    output reg af,
//    output reg ag,
//    output wire cat,
    
    /////////////// 시뮬레이션 용 ////////////
//    // 내부 신호를 관찰할 수 있도록 출력 추가
//    output wire in_up_clean,
//    output wire in_down_clean,
//    output wire clk_50hz,
//    output reg [3:0] count
    /////////////////////////////////////////
//    output wire resetn_clean
    );

/////////////// 보드 Bitstream 용 ////////////
    // debounce된 버튼 신호 & 분기된 clk
    wire in_up_clean;
    wire in_down_clean;
    reg [3:0] count;         
    wire clk_50hz;
//////////////////////////////////////////////
//    wire resetn_clean;        
    
    // Clock Divider 인스턴스화
    clk_divider clk_gen (
        .clk(clk),
        .resetn(resetn),
        .clk_50hz(clk_50hz)
    );
    
    // Debouncer 인스턴스화 -> clk or clk_50hz??
    debouncer_2 db_up (
        .clk(clk),
        .btn_in(in_up),
        .btn_out(in_up_clean)
    );

    debouncer_2 db_down (
        .clk(clk),
        .btn_in(in_down),
        .btn_out(in_down_clean)
    );

/***** OutPut을 wire형으로 선언시 *****/
    // 100MHz 클럭을 기반으로 카운터 증가/감소
    always @(posedge clk or negedge resetn) begin
        if (!resetn)                            // resetn 시 초기화
            count <= 4'd0;
        else if (in_up_clean && count < 4'd9)   // up버튼 입력 시(debounce됨) & counter가 9미만이면 최대 9까지는 +1
            count <= count + 1;
        else if (in_down_clean && count > 4'd0) // down버튼 입력 시(debounce됨) & counter가 0초과이면 최대 0까지는 -1
            count <= count - 1;
    end

    // cat 신호는 50Hz 주기로 토글되도록 설정
    assign cat = clk_50hz;

    // 7-segment Mapping (각 숫자에 대한 비트 설정)
    assign {aa, ab, ac, ad, ae, af, ag} = 
        (cat == 0) ? (
            (count == 4'd0) ? 7'b1111110 :  // 0
            (count == 4'd1) ? 7'b0110000 :  // 1
            (count == 4'd2) ? 7'b1101101 :  // 2
            (count == 4'd3) ? 7'b1111001 :  // 3
            (count == 4'd4) ? 7'b0110011 :  // 4
            (count == 4'd5) ? 7'b1011011 :  // 5
            (count == 4'd6) ? 7'b1011111 :  // 6
            (count == 4'd7) ? 7'b1110000 :  // 7
            (count == 4'd8) ? 7'b1111111 :  // 8
            (count == 4'd9) ? 7'b1110011 :  // 9
            7'b1111110                      // OFF 상태 (기본값)
        ) : 7'b1111110;                     // cat이 1일 때는 숫자 0을 표시

/***** OutPut을 reg형으로 선언시 *****/
    
//    // cat 신호는 50Hz 주기로 토글되도록 설정
//    assign cat = clk_50hz;
    
//    // PushButton 시 7-segement를 1씩 증가 or 감소
//    always @(posedge clk or negedge resetn) begin
//        if (!resetn)
//            count <= 4'd0;
//        else if (in_up_clean && count < 4'd9)
//            count <= count + 1;
//        else if (in_down_clean && count > 4'd0)
//            count <= count - 1;
//    end
    
//    // 7-segement Mapping, 강의안에 나와있는 Mapping을 기준으로 함
//    always @(posedge clk_50hz) begin
//        if(cat == 1) begin
//            case (count)
//                4'd0: {aa, ab, ac, ad, ae, af, ag} = 7'b1111110;
//                4'd1: {aa, ab, ac, ad, ae, af, ag} = 7'b0110000;
//                4'd2: {aa, ab, ac, ad, ae, af, ag} = 7'b1101101;
//                4'd3: {aa, ab, ac, ad, ae, af, ag} = 7'b1111001;
//                4'd4: {aa, ab, ac, ad, ae, af, ag} = 7'b0110011;
//                4'd5: {aa, ab, ac, ad, ae, af, ag} = 7'b1011011;
//                4'd6: {aa, ab, ac, ad, ae, af, ag} = 7'b1011111;
//                4'd7: {aa, ab, ac, ad, ae, af, ag} = 7'b1110000;
//                4'd8: {aa, ab, ac, ad, ae, af, ag} = 7'b1111111;
//                4'd9: {aa, ab, ac, ad, ae, af, ag} = 7'b1110011;
//                default: {aa, ab, ac, ad, ae, af, ag} = 7'b0000000;
//            endcase
//        end
//        else begin
//            {aa, ab, ac, ad, ae, af, ag} = 7'b1111110;
//        end
//    end

/***** resetn 신호가 button으로 입력되면 debouncing을 해야하나? *****/
//    debouncer_active_low db_resetn (
//        .clk(clk_50hz),
//        .btn_in(resetn),
//        .btn_out(resetn_clean)
//    );

        
endmodule
