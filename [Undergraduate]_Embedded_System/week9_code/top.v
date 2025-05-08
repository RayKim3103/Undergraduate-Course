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
  inout [14:0]DDR_addr,
  inout [2:0]DDR_ba,
  inout DDR_cas_n,
  inout DDR_ck_n,
  inout DDR_ck_p,
  inout DDR_cke,
  inout DDR_cs_n,
  inout [3:0]DDR_dm,
  inout [31:0]DDR_dq,
  inout [3:0]DDR_dqs_n,
  inout [3:0]DDR_dqs_p,
  inout DDR_odt,
  inout DDR_ras_n, 
  inout DDR_reset_n,
  inout DDR_we_n,
  inout FIXED_IO_ddr_vrn,
  inout FIXED_IO_ddr_vrp,
  inout [53:0]FIXED_IO_mio,
  inout FIXED_IO_ps_clk,
  inout FIXED_IO_ps_porb,
  inout FIXED_IO_ps_srstb,
  // Programmable Logic
  input  resetn,
  input  lcdclk,
  output lcd_rs,
  output lcd_rw,
  output lcd_en,
  output [7:0] lcd_data
);

wire [31:0] reg_a;
wire [31:0] reg_b;
wire [31:0] reg_c;
wire [31:0] reg_d;
wire [31:0] reg_e;
wire [31:0] reg_f;
wire [31:0] reg_g;
wire [31:0] reg_h;

system_wrapper usystem_wrapper (
  .DDR_addr(DDR_addr),
  .DDR_ba(DDR_ba),
  .DDR_cas_n(DDR_cas_n),
  .DDR_ck_n(DDR_ck_n),
  .DDR_ck_p(DDR_ck_p),
  .DDR_cke(DDR_cke),
  .DDR_cs_n(DDR_cs_n),
  .DDR_dm(DDR_dm),
  .DDR_dq(DDR_dq),
  .DDR_dqs_n(DDR_dqs_n),
  .DDR_dqs_p(DDR_dqs_p),
  .DDR_odt(DDR_odt),
  .DDR_ras_n(DDR_ras_n),
  .DDR_reset_n(DDR_reset_n),
  .DDR_we_n(DDR_we_n),
  .FIXED_IO_ddr_vrn(FIXED_IO_ddr_vrn),
  .FIXED_IO_ddr_vrp(FIXED_IO_ddr_vrp),
  .FIXED_IO_mio(FIXED_IO_mio),
  .FIXED_IO_ps_clk(FIXED_IO_ps_clk),
  .FIXED_IO_ps_porb(FIXED_IO_ps_porb),
  .FIXED_IO_ps_srstb(FIXED_IO_ps_srstb),
  .REG_A(reg_a),
  .REG_B(reg_b),
  .REG_C(reg_c),
  .REG_D(reg_d),
  .REG_E(reg_e),
  .REG_F(reg_f),
  .REG_G(reg_g),
  .REG_H(reg_h)
);

textlcd utextlcd (
  .resetn(resetn),
  .lcdclk(lcdclk),
  .lcd_rs(lcd_rs),
  .lcd_rw(lcd_rw),
  .lcd_en(lcd_en),
  .lcd_data(lcd_data),
  .reg_a(reg_a),
  .reg_b(reg_b),
  .reg_c(reg_c),
  .reg_d(reg_d),
  .reg_e(reg_e),
  .reg_f(reg_f),
  .reg_g(reg_g),
  .reg_h(reg_h)
);

endmodule
