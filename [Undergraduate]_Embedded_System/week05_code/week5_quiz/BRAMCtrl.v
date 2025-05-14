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

// YOUR CODE GOES IN HERE
// NOTICE
// Reverse_SW IS ALREADY IMPLEMENTED (MAKE USE OF IT)

module BRAMCtrl(
  input CLK,
  input RESET,
  input Vsync,            // vertical sync
  input Hsync,            // horizontal sync
  input DE,               // display enable (valid pixel area)
  output BRAMCLK,         // BRAM clock
  output [17:0] BRAMADDR, // BRAM address
  input [15:0] BRAMDATA,  // BRAM data 16 bits(R/G/B=5/6/5)
  output [7:3] R,         // red ouput 5 bits
  output [7:2] G,         // green output 6 bits
  output [7:3] B,         // blue output 5 bits
  input Reverse_SW);      // image reverse switch

  parameter HSIZE = 480;  // horizontal size (piexel size)
  parameter VSIZE = 272;  // vertical size  (line size)

  reg [13:0] hcnt;        // horizontal counter (0~479)
  reg [23:0] vcnt;        // vertical counter (0~271)
  reg DE1d;               // DE delay 1 clock (detect the end of line)
  
  always @ (posedge CLK or posedge RESET) begin
    if (RESET) begin      // initialization
      hcnt <= 14'd0;
      vcnt <= 24'd0;
      DE1d <= 1'b0;
    end

    else begin
      DE1d <= DE;       // DE delay 1 clock

    /*** original code ***/
      // if (Reverse_SW) begin
    /*********************/
      // Reverse Image: count address from bottom to top
      if (!Reverse_SW) begin  
          // Vsync is low when the display is in the blanking period
          if (!Vsync)         
            // start at last line 
            // (set vcnt to the last address of the display area)
            vcnt <= (VSIZE-1)*HSIZE;
          else if ((!DE) && (DE1d)) // end of line → move 1 line up
            vcnt <= vcnt - HSIZE;
        end
      // Normal Image: count address from top to bottom
      else begin        
        if (!Vsync)
          // start at first line 
          // (set vcnt to the first address of the display area)
          vcnt <= 18'd0;  
        else if ((!DE) && (DE1d)) // end of line → move 1 line down
          vcnt <= vcnt + HSIZE;
      end
      // end of line → reset horizontal counter
      if (!DE)  
        hcnt <= 14'd0;
      else if (hcnt < HSIZE)  // horizontal counter (0~479)
        hcnt <= hcnt + 14'd1;

    end
  end
  
    assign BRAMCLK = CLK;             // BRAM clock = input clock

    // BRAM address = vertical address + horizontal address
    assign BRAMADDR = vcnt + hcnt;    
    assign R[7:3] = BRAMDATA[15:11];  // red from BRAM data
    assign G[7:2] = BRAMDATA[10:5];   // green from BRAM data
    assign B[7:3] = BRAMDATA[4:0];    // blue from BRAM data
  endmodule