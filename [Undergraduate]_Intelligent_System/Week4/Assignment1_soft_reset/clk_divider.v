`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/23 12:03:35
// Design Name: 
// Module Name: clk_divider
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


module clk_divider(
    input wire clk,         // 100MHz FPGA clock
    input wire reset,       // Active-high 리셋 신호
    output reg clk_50hz,    // 50Hz로 변환된 클럭 출력
    output reg clk_50Mhz
    );

    
    reg [19:0] counter_50hz;                     // 50Hz 클럭 생성을 위한 카운터, 2^20 =  1,048,576
    
    // 100MHz(= 100,000,000Hz) 클럭을 50Hz로 변환하려면, 클럭이 2,000,000마다 들어와야 한다.
    // 즉, 1,000,000번 클럭마다 출력을 토글하면 50Hz 신호가 생성
    always @(posedge clk or posedge reset) begin
        if (reset) begin                       // resetn 시 초기화
            counter_50hz <= 0;
            clk_50hz <= 0;
        end 
        else if (counter_50hz >= 999_999) begin        // counter_50hz가 999_999를 찍을 시 1,000,000번째 clk이므로 토글 & counter 초기화 / 시뮬레이션 : 3
            counter_50hz <= 0;
            clk_50hz <= ~clk_50hz;      
        end 
        else begin
            counter_50hz <= counter_50hz + 1;    // counter_50hz가 넉넉하면, counter += 1
        end
    end
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin                       // resetn 시 초기화
            clk_50Mhz <= 0;
        end
        else
            clk_50Mhz <= ~clk_50Mhz;
    end
endmodule
