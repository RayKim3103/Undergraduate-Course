`timescale 1ns / 1ps

module fsm_ctrl(
    input wire clk_1hz,     // 1Hz clock input
    input wire clk,         // clock input
    input wire resetn,      // Active-low reset
    input wire sw0,         // State control switch
    input wire sw1,         // State control switch

    output reg [3:0] count,   // Counter value
//    output reg [1:0] state,   // Current state -> tb 시뮬레이션 확인 용 신호
    output reg led0, led1     // LED indicators
    );

/////////////// 보드 Bitstream 용 ////////////
    reg [1:0] state;
//////////////////////////////////////////////

    // state 구분을 위한 파라미터 선언
    parameter IDLE = 2'b00, UP = 2'b01, DOWN = 2'b10, READY = 2'b11;

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin                      // resetn 시 초기화 (IDLE 및 0)
            state <= IDLE;
//            count <= 4'b0000;
            led0 <= 0;
            led1 <= 0;
        end 
        else begin
            case (state)
                IDLE: begin                     // IDLE: 바로 READY state로 transition / led0 = 0, led1 = 0 
                    state <= READY;
                    led0 <= 0;
                    led1 <= 0;
                end
                READY: begin                    // READY: sw1 && !sw0 이면 DOWN / sw0 이면 UP / led0 = 0, led1 = 0 
                    led0 <= 0;
                    led1 <= 0;
                    if (sw1 && !sw0)
                        state <= DOWN;
                    else if (sw0)
                        state <= UP;
                    else
                        state <= state;
                end
                UP: begin                       // UP: !sw0 이면 READY / led0 = 1, led1 = 0 
                    led0 <= 1;
                    led1 <= 0;
                    if (!sw0)
                         state <= READY;
                    else 
                        state <= state;
                end
                DOWN: begin                     // DOWN: !sw1 & !sw0 이면 READY / sw0 이면 UP / led0 = 0, led1 = 1 
                    led0 <= 0;
                    led1 <= 1;
                    if (!sw1 & !sw0) 
                        state <= READY;
                    else if (sw0)
                        state <= UP;
                    else 
                        state <= state;
                end
            endcase
        end
    end
    
    // 1Hz 클럭을 기반으로 카운터 증가/감소
    always @(posedge clk_1hz or negedge resetn) begin
        if(!resetn)
            count <= 4'b0000;
        else if (state == UP && count < 4'd15) begin           // state == UP && count < 4'd15
            count <= count + 1;
        end
        else if (state == DOWN && count > 4'd0) begin       // state == DOWN && count > 4'd0
            count <= count - 1;
        end
        else
            count <= count;
    end
endmodule