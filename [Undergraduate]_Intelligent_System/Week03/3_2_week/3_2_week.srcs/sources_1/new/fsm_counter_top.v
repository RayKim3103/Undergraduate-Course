`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/17 20:38:18
// Design Name: 
// Module Name: fsm_counter_top
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

module fsm_counter_top(

    input wire clk,     // 100MHz FPGA 클럭
    input wire resetn,  // Active Low 리셋
    input wire sw0,     // UP을 담당하는 switch 0
    input wire sw1,     // Down을 담당하는 switch 1

    // 7-segment 디스플레이 출력 및 led 출력
    output wire aa,
    output wire ab,
    output wire ac,
    output wire ad,
    output wire ae,
    output wire af,
    output wire ag,
    output wire cat,
    output wire led0,
    output wire led1
        
    /////////////// 시뮬레이션 용 ////////////
//    output wire clk_1hz, clk_50hz,
//    output wire [3:0] count,
//    output wire [1:0] state

);

/////////////// 보드 Bitstream 용 ////////////
    wire clk_1hz, clk_50hz;     // 분기된 1hz 클럭, 50hz 클럭
    wire [3:0] count;           // 숫자 0~15까지의 정보를 전달하는 count 
//    reg [1:0] state;            // fsm_ctrl에서 결정된 state를 저장 -> 시뮬레이션용
//////////////////////////////////////////////

    // Clock Divider 인스턴스: 100MHz -> 50Hz, 1Hz
    clk_divider clk_div_inst (
        .clk(clk),
        .resetn(resetn),
        
        .clk_1hz(clk_1hz),
        .clk_50hz(clk_50hz)
    );
    
    // FSM Controller 인스턴스
    fsm_ctrl fsm_ctrl_inst (
        .clk_1hz(clk_1hz),
        .clk(clk),
        .resetn(resetn),
        .sw0(sw0),
        .sw1(sw1),
        
        .count(count),
//        .state(state),    // -> 시뮬레이션 용
        .led0(led0),
        .led1(led1)
    );
    
    // SSD Controller 인스턴스
    ssd_ctrl ssd_ctrl_inst (
        .clk_50hz(clk_50hz),
        .count(count),
        .resetn(resetn),
        
        .aa(aa), .ab(ab), .ac(ac), .ad(ad), .ae(ae), .af(af), .ag(ag),
        .cat(cat)
    );

endmodule

