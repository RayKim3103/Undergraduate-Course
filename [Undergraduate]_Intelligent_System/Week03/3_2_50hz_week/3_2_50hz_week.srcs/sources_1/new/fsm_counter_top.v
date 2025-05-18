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

    input wire clk,
    input wire resetn,
    input wire sw0,
    input wire sw1,

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
    wire clk_1hz, clk_50hz;
    wire [3:0] count;
    reg [1:0] state;
//////////////////////////////////////////////

    // Clock Divider: 100MHz -> 50Hz, 1Hz
    clk_divider clk_div_inst (
        .clk(clk),
        .resetn(resetn),
        
        .clk_1hz(clk_1hz),
        .clk_50hz(clk_50hz)
    );
    
    // FSM Controller
    fsm_ctrl fsm_ctrl_inst (
        .clk_1hz(clk_1hz),
        .clk_50hz(clk_50hz),
        .resetn(resetn),
        .sw0(sw0),
        .sw1(sw1),
        
        .cat(cat),
        .count(count),
//        .state(state),    // -> 시뮬레이션 용
        .led0(led0),
        .led1(led1)
    );
    
    // SSD Controller
    ssd_ctrl ssd_ctrl_inst (
        .clk_50hz(clk_50hz),
        .count(count),
        .resetn(resetn),
        .cat(cat),
        
        .aa(aa), .ab(ab), .ac(ac), .ad(ad), .ae(ae), .af(af), .ag(ag)
//        .cat(cat)
    );

endmodule

