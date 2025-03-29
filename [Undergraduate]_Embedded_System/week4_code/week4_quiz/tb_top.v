`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/28 11:59:09
// Design Name: 
// Module Name: tb_top
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


module tb_top;

  // Inputs
  reg resetn;
  reg lcdclk;
  reg [2:0] PushButton;

  // Outputs
  wire lcd_rs;
  wire lcd_rw;
  wire lcd_en;
  wire [7:0] lcd_data;

  // Instantiate the Unit Under Test (UUT)
  top uut (
    .resetn(resetn),
    .lcdclk(lcdclk),
    .PushButton(PushButton),
    .lcd_rs(lcd_rs),
    .lcd_rw(lcd_rw),
    .lcd_en(lcd_en),
    .lcd_data(lcd_data)
  );

  // Clock generation: 10ns аж╠Б (100MHz)
  initial begin
    lcdclk = 0;
    forever #1 lcdclk = ~lcdclk;
  end

  // Test procedure
  initial begin
    // Initialize Inputs
    resetn = 0;
    PushButton = 3'b000;

    // Wait 100 ns for global reset to finish
    #1000;
    resetn = 1;
    #200
    resetn = 0;
    #200
    resetn = 1;
    
    #300000;
    // Simulate button press sequences
    #5000 PushButton[0] = 1;
    #2000 PushButton[0] = 0;
    
    #300000;
    
    #10000 PushButton[1] = 1;
    #2000 PushButton[1] = 0;
    
    #300000;

    #10000 PushButton[2] = 1;
    #2000 PushButton[2] = 0;
    
    #300000;

    // Add more interactions if needed
    #5000 $finish;
  end

endmodule
