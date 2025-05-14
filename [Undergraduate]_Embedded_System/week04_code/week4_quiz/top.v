`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:
// Engineer:
//
// Create Date: 2014/07/28 15:03:27
// Design Name:
// Module Name: top
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


module top(
   // Processor System
   // inout [14:0]DDR_addr,
   // inout [2:0]DDR_ba,
   // inout DDR_cas_n,
   // inout DDR_ck_n,
   // inout DDR_ck_p,
   // inout DDR_cke,
   // inout DDR_cs_n,
   // inout [3:0]DDR_dm,
   // inout [31:0]DDR_dq,
   // inout [3:0]DDR_dqs_n,
   // inout [3:0]DDR_dqs_p,
   // inout DDR_odt,
   // inout DDR_ras_n,
   // inout DDR_reset_n,
   // inout DDR_we_n,
   // inout FIXED_IO_ddr_vrn,
   // inout FIXED_IO_ddr_vrp,
   // inout [53:0]FIXED_IO_mio,
   // inout FIXED_IO_ps_clk,
   // inout FIXED_IO_ps_porb,
   // inout FIXED_IO_ps_srstb,
   // Programmable Logic
   input  resetn,
   input  lcdclk,
   input [2:0] PushButton,
   output lcd_rs,
   output lcd_rw,
   output lcd_en,
   output [7:0] lcd_data
 );
 
// wire [2:0] PushButton_clean;
 
//  debouncer udebouncer1 (
//    .in(PushButton[0]),
//    .clk(lcdclk),
//    .resetn(resetn),
//    .out_pos(PushButton_clean[0])
//  );
 
//  debouncer udebouncer2 (
//      .in(PushButton[1]),
//      .clk(lcdclk),
//      .resetn(resetn),
//      .out_pos(PushButton_clean[1])
//    );
   
//    debouncer udebouncer3 (
//        .in(PushButton[2]),
//        .clk(lcdclk),
//        .resetn(resetn),
//        .out_pos(PushButton_clean[2])
//      );
     
  textlcd utextlcd (
    .PushButton(PushButton),
    .resetn(resetn),
    .lcdclk(lcdclk),
    .lcd_rs(lcd_rs),
    .lcd_rw(lcd_rw),
    .lcd_en(lcd_en),
    .lcd_data(lcd_data)
  );
 
endmodule