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
   input  resetn,         // reset 신호 port
   input  lcdclk,         // input clock 신호 port
   output lcd_rs,         // text LCD의 register select 신호 port
   output lcd_rw,         // text LCD의 read/write 신호 port
   output lcd_en,         // text LCD enable 신호 port
   output [7:0] lcd_data  // 각각의 text LCD segment가 나타낼 문자 (8bit)
 );
 
  
  // textlcd 모듈을 인스턴스화 하여 사용
  textlcd utextlcd (
    .resetn(resetn),
    .lcdclk(lcdclk),
    .lcd_rs(lcd_rs),
    .lcd_rw(lcd_rw),
    .lcd_en(lcd_en),
    .lcd_data(lcd_data)
  );
  
  endmodule