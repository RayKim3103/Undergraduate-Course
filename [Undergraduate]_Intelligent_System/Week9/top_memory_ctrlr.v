`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/27 15:03:20
// Design Name: 
// Module Name: top_memory_ctrlr
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


module top_memory_ctrlr(
    input wire clk,
    input wire resetn,
    
    input  wire start,
    output wire done,
    
    input wire s1_clka,
    input wire s1_ena,
    input wire [1:0] s1_wea,
    input wire [7:0] s1_addra,
    input wire [15:0] s1_dina,
    output wire [15:0] s1_douta,
    
    input wire s2_clka,
    input wire s2_ena,
    input wire [3:0] s2_wea,
    input wire [7:0] s2_addra,
    input wire [31:0] s2_dina,
    output wire [31:0] s2_douta
    );

    wire s1_en, s2_en;
    wire s1_we, s2_we;
    wire [1:0] s1_web;
    wire [3:0] s2_web;
    wire [7:0] s1_addr;
    wire [7:0] s2_addr;
    wire [15:0] s1_din, s1_dout;
    wire [31:0] s2_din, s2_dout;
    
    assign s1_web = {2{s1_we}};
    assign s2_web = {4{s2_we}};


    memory_ctrlr #(.SRAM1_BW(16), .SRAM1_AMAX(256), .SRAM2_BW(32), .SRAM2_AMAX(192)) 
    Umemory_ctrlr (.clk(clk), .resetn(resetn), .start(start), .done(done),
                   .s1_en(s1_en), .s1_we(s1_we), .s1_addr(s1_addr), .s1_dout(s1_dout),
                   .s2_en(s2_en), .s2_we(s2_we), .s2_addr(s2_addr), .s2_din(s2_din));
    
    SRAM1 SRAM1 (
      .clka(s1_clka),    // input wire clka
      .ena(s1_ena),      // input wire ena
      .wea(s1_wea),      // input wire [1 : 0] wea
      .addra(s1_addra),  // input wire [7 : 0] addra
      .dina(s1_dina),    // input wire [15 : 0] dina
      .douta(s1_douta),  // output wire [15 : 0] douta
      .clkb(clk),    // input wire clkb
      .enb(s1_en),      // input wire enb
      .web(s1_web),      // input wire [1 : 0] web
      .addrb(s1_addr),  // input wire [7 : 0] addrb
      .dinb(s1_din),    // input wire [15 : 0] dinb
      .doutb(s1_dout)  // output wire [15 : 0] doutb
    );
    SRAM2 SRAM2 (
      .clka(s2_clka),    // input wire clka
      .ena(s2_ena),      // input wire ena
      .wea(s2_wea),      // input wire [3 : 0] wea
      .addra(s2_addra),  // input wire [7 : 0] addra
      .dina(s2_dina),    // input wire [31 : 0] dina
      .douta(s2_douta),  // output wire [31 : 0] douta
      .clkb(clk),    // input wire clkb
      .enb(s2_en),      // input wire enb
      .web(s2_web),      // input wire [3 : 0] web
      .addrb(s2_addr),  // input wire [7 : 0] addrb
      .dinb(s2_din),    // input wire [31 : 0] dinb
      .doutb(s2_dout)  // output wire [31 : 0] doutb
    );
    
endmodule
