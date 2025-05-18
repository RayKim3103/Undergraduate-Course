`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/19 14:08:09
// Design Name: 
// Module Name: tb_fsm_counter_top
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


module tb_fsm_counter_top;

    reg clk;
    reg resetn;
    reg sw0, sw1;
    
    wire aa, ab, ac, ad, ae, af, ag;
    wire cat;
    wire led0, led1;
    wire clk_1hz, clk_50hz;
    wire [3:0] count;
    wire [1:0] state;
    wire resetn_clean, sw0_clean, sw1_clean;
    
    // Instantiate the top module
    fsm_counter_top uut (
        .clk(clk),
        .resetn(resetn),
        .sw0(sw0),
        .sw1(sw1),
        .aa(aa), .ab(ab), .ac(ac), .ad(ad), .ae(ae), .af(af), .ag(ag),
        .cat(cat),
        .led0(led0),
        .led1(led1),
        .clk_1hz(clk_1hz),
        .clk_50hz(clk_50hz),
        .count(count),
        .state(state),
        .resetn_clean(resetn_clean),
        .sw0_clean(sw0_clean),
        .sw1_clean(sw1_clean)
    );

    // Clock generation
    always #5 clk = ~clk;  // 100MHz -> 10ns period

    initial begin
        // Initialize inputs
        clk = 0;
        resetn = 0;
        sw0 = 0;
        sw1 = 0;

        // Reset sequence
        #20 resetn = 1; 
        #20 resetn = 0;
        #20 resetn = 1;
        
        // Test IDLE to READY transition
        #50 sw0 = 1;
        #5 sw0 = 0;
        #5 sw0 = 1;
        #5 sw0 = 0;
        #5 sw0 = 1;
        
        #500 sw0 = 0;
        #5 sw0 = 1;
        #5 sw0 = 0;
        #5 sw0 = 1;
        #5 sw0 = 0;
        
        // Test READY to UP
        #500 sw0 = 1;
        #500 sw0 = 0;
        
        // Wait to observe counting up
        #500;

        // Test READY to DOWN
        #500 sw1 = 1;
        #500 sw1 = 0;
        
        // Wait to observe counting down
        #500;

        // Reset again
        #50 resetn = 0;
        #500 resetn = 1;

        // End simulation
        #1000 $finish;
    end
endmodule
