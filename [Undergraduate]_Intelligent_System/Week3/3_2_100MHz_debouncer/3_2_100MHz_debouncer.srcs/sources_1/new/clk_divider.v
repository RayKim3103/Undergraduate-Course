`timescale 1ns / 1ps

module clk_divider(
    input wire clk,         // 100MHz FPGA 클럭
    input wire resetn,      // Active-low 리셋 신호
    output reg clk_50hz,    // 50Hz로 변환된 클럭 출력
    output reg clk_1hz      // 1Hz로 변환된 클럭 출력
    );
    
    reg [25:0] counter_1hz;     // 1Hz 클럭 생성을 위한 카운터, 2^26 = 67,108,864
    reg [19:0] counter_50hz;    // 50Hz 클럭 생성을 위한 카운터, 2^20 =  1,048,576
    
    // 100MHz(= 100,000,000Hz) 클럭을 50Hz로 변환하려면, 클럭이 2,000,000마다 들어와야 한다.
    // 즉, 1,000,000번 클럭마다 출력을 토글하면 50Hz 신호가 생성됨
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin                                  // resetn 시 초기화
            counter_50hz <= 0;
            clk_50hz <= 0;
        end else if (counter_50hz >= 999_999) begin         // counter_50hz가 999_999를 찍을 시 1,000,000번째 clk이므로 토글 & counter 초기화 / 시뮬레이션 : 3
            counter_50hz <= 0;
            clk_50hz <= ~clk_50hz;
        end else begin                                      // counter_50hz가 넉넉하면, counter += 1
            counter_50hz <= counter_50hz + 1;
        end
    end
    
    // 100MHz(= 100,000,000Hz) 클럭을 1Hz로 변환하려면, 클럭이 100,000,000마다 들어와야 한다.
    // 즉, 50,000,000번 클럭마다 출력을 토글하면 1Hz 신호가 생성됨
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin                                  // resetn 시 초기화
            counter_1hz <= 0;
            clk_1hz <= 0;
        end else if (counter_1hz >= 49_999_999) begin       // counter_1hz가 49_999_999를 찍을 시 50,000,000번째 clk이므로 토글 & counter 초기화 / 시뮬레이션 : 10
            counter_1hz <= 0;
            clk_1hz <= ~clk_1hz;
        end else begin
            counter_1hz <= counter_1hz + 1;                 // counter_1hz가 넉넉하면, counter += 1
        end
    end
endmodule