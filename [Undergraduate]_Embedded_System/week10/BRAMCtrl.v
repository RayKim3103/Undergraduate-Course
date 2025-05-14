`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2014/07/31 12:56:45
// Design Name: 
// Module Name: BRAMCtrl
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


`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2013/02/19 14:59:36
// Design Name: 
// Module Name: BRAMCtrl
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


module BRAMCtrl(
  input CLK,
  input RESET,
  input Vsync,
  input Hsync,
  input DE,
  output BRAMCLK,
  output [17:0] BRAMADDR,
  input [15:0] BRAMDATA,
  output [7:3] R,
  output [7:2] G,
  output [7:3] B,
  input Reverse_SW);

  parameter HSIZE = 480;
  parameter VSIZE = 272;

  reg [13:0] hcnt;
  reg [23:0] vcnt;
  reg DE1d;

always @ (posedge CLK or posedge RESET)
begin
  if (RESET)
  begin
    hcnt <= 14'd0;
    vcnt <= 24'd0;
    DE1d <= 1'b0;
  end
  else
  begin
    DE1d <= DE;
    if (Reverse_SW)
    begin
      if (!Vsync)
        vcnt <= (VSIZE-1)*HSIZE;
      else if ((!DE) && (DE1d))
        vcnt <= vcnt - HSIZE;
    end
    else
    begin
      if (!Vsync)
        vcnt <= 18'd0;
      else if ((!DE) && (DE1d))
        vcnt <= vcnt + HSIZE;
    end


    if (!DE)
      hcnt <= 14'd0;
    else if (hcnt < HSIZE)
      hcnt <= hcnt + 14'd1;

  end
end

  assign BRAMCLK = CLK;
  assign BRAMADDR = vcnt + hcnt;
  assign R[7:3] = BRAMDATA[15:11];
  assign G[7:2] = BRAMDATA[10:5];
  assign B[7:3] = BRAMDATA[4:0];

endmodule
