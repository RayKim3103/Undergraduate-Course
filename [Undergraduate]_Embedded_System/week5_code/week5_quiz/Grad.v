`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/04/04 09:29:23
// Design Name: 
// Module Name: Grad
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


module Grad(R,G,B,H_COUNT,V_COUNT,DE,RESET);
    
    
    input [9:0] H_COUNT;
    input [9:0] V_COUNT;
    input DE;
    input RESET;
    
    output reg [7:3] R;
    output reg [7:2] G;
    output reg [7:3] B;
    

    always@(DE,H_COUNT,V_COUNT,RESET) begin
        if(RESET == 1) begin
            R = 0;
            G = 0;
            B = 0;
        end
  
        else if(DE == 1) begin
            R = ((V_COUNT - 10'd12) / 17 * 2);
            G = ((H_COUNT - 10'd43) / 15) * 2;
            B = 0;

        end
    end
endmodule
