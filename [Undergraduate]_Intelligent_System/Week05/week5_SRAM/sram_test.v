`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/27 14:59:57
// Design Name: 
// Module Name: sram_test
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


module sram_test #(
  parameter BW = 32,            // BW: Bit Width
  parameter AMAX = 512,         // AMAX: Max number of Address line (memory depth¸¦ ÀÇ¹Ì)
  parameter ADR = $clog2(AMAX)  // ADR: Number of bits to represent address
)(
  input wire clk,               // input clock
  input wire en,                // sram enable signal
  input wire we,                // sram write enable signal (1: write / 0: read)
  input wire [ADR-1:0] addr,    // input address
  input wire [BW-1:0] din,      // input data
  output wire [BW-1:0] dout     // ouput data
);
  reg [BW-1:0] mem [0:AMAX-1];   // 2D array representation of memory(SRAM) / Size: BW * AMAX

  // write operation
  always @(posedge clk) begin   
    if(en) begin                // if sram is enable
      if(we) begin              // if sram does write operation
        // write the data into SRAM, the value is stored in specific address
        mem[addr] <= #1 din;       
      end
    end
  end

  // read operation
  // if sram is enable and does read operation, read data for specific memory address 
//  assign #1 dout =  (en)? mem[addr] : 32'hx;
    assign #1 dout = (en) ? (!we ? mem[addr] : 32'hx) : 32'hx;
endmodule

