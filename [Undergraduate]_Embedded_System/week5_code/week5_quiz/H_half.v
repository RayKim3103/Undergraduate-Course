`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/04/04 09:29:23
// Design Name: 
// Module Name: H_half
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


module H_half(R,G,B,H_COUNT,V_COUNT,DE,RESET);
    
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
            if(H_COUNT >= 10'd1) begin
                if ((V_COUNT > 10'd11) && (V_COUNT <= 10'd28)) begin    /* white */ 
                    R = 5'b11111;
                    G = 6'b111111;
                    B = 5'b11111;
                end

                else if ((V_COUNT > 10'd45) && (V_COUNT <= 10'd62)) begin     /* yellow */                
                    R = 5'b11111;
                    G = 6'b111111;
                    B = 0;
                end

                else if ((V_COUNT > 10'd79) && (V_COUNT <= 10'd96)) begin     /* cyan */
                    R = 0;
                    G = 6'b111111;
                    B = 5'b11111;
                end

                else if ((V_COUNT > 10'd113) && (V_COUNT <= 10'd130)) begin    /* green */
                    R = 0;
                    G = 6'b111111;
                    B = 0;
                end
                
                else if ((V_COUNT > 10'd147) && (V_COUNT <= 10'd164)) begin    /* purple */
                    R = 5'b11111;
                    G = 0;
                    B = 5'b11111;
                end

                else if ((V_COUNT > 10'd181) && (V_COUNT <= 10'd198)) begin   /* red */
                    R = 5'b11111;
                    G = 0;
                    B = 0;
                end  

                else if ((V_COUNT > 10'd215) && (V_COUNT <= 10'd232)) begin     /* blue */
                    R = 0;
                    G = 0;
                    B = 5'b11111;
                end

                else if ((V_COUNT > 10'd249) && (V_COUNT <= 10'd266)) begin    /* white */
                    R = 5'b11111;
                    G = 6'b111111;
                    B = 5'b11111;
                end

                else begin
                    R = 0;
                    G = 0;
                    B = 0;
                end
            end
        end
        
        else begin
            R = 0;
            G = 0;
            B = 0;
        end
                       
    end
endmodule
