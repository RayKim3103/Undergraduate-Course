`timescale 1ns / 1ps

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
//    output wire [1:0] state,
//    output wire resetn_clean, sw0_clean, sw1_clean

);

/////////////// 보드 Bitstream 용 ////////////
    wire clk_1hz, clk_50hz;
    wire [3:0] count;
    reg [1:0] state;
    wire resetn_clean, sw0_clean, sw1_clean;
//////////////////////////////////////////////

    //    assign resetn_clean = resetn;
    //    assign sw0_clean = sw0;
    //    assign sw1_clean = sw1;

    debouncer debouncer_1(
        .clk(clk),
        .btn_in(resetn),
        .btn_out(resetn_clean)
    );
    
    debouncer debouncer_2(
        .clk(clk),
        .btn_in(sw0),
        .btn_out(sw0_clean)
    );
    
    debouncer debouncer_3(
        .clk(clk),
        .btn_in(sw1),
        .btn_out(sw1_clean)
    );

    // Clock Divider: 100MHz -> 50Hz, 1Hz
    clk_divider clk_div_inst (
        .clk(clk),
        .resetn(resetn_clean),
        
        .clk_1hz(clk_1hz),
        .clk_50hz(clk_50hz)
    );
    
    // FSM Controller
    fsm_ctrl fsm_ctrl_inst (
        .clk_1hz(clk_1hz),
        .clk(clk),
        .resetn(resetn_clean),
        .sw0(sw0_clean),
        .sw1(sw1_clean),
        
        .count(count),
//        .state(state),    // -> 시뮬레이션 용
        .led0(led0),
        .led1(led1)
    );
    
    // SSD Controller
    ssd_ctrl ssd_ctrl_inst (
        .clk_50hz(clk_50hz),
        .count(count),
        .resetn(resetn_clean),
        
        .aa(aa), .ab(ab), .ac(ac), .ad(ad), .ae(ae), .af(af), .ag(ag),
        .cat(cat)
    );

endmodule
