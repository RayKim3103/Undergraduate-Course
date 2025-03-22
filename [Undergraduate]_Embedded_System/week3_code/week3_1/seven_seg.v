`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: HUINS
// Engineer: 
// 
// Create Date: 2012/11/28 13:46:07
// Design Name: segment
// Module Name: seven_seg
// Project Name: segment
// Target Devices: xc7z020clg484-1
// Tool Versions: Xilinx PlanAhead 14.3
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module seven_seg(
input clk,                  // input clock
input resetn,               // reset 신호 (active low)
input [31:0] data,          // 32비트 데이터 입력
output reg [7:0] segout,    // 현재 표시할 7-segment 데이터
output reg [7:0] segcom,    // 8개의 7-segment 패널 중 1개 select 신호
output [7:0] led_out);      // 8bit 데이터는 8개의 LED 출력 결정

  // 각 7세그먼트 출력에 대한 wire 선언
  wire [7:0] seg1;
  wire [7:0] seg2;
  wire [7:0] seg3;
  wire [7:0] seg4;
  wire [7:0] seg5;
  wire [7:0] seg6;
  wire [7:0] seg7;
  wire [7:0] seg8;

  // counter 선언
  reg[14:0] clk_cnt;    // 클럭 카운터
  reg[2:0] com_cnt;     // 7-segment 선택용 카운터

  // bin2seg 모듈을 이용해 데이터를 4비트씩 각각 7-segment로 변환 (디스플레이 패널이 총 8개)
  bin2seg bin2seg_1 (.bin_data(data[31:28]), .seg_data(seg1));
  bin2seg bin2seg_2 (.bin_data(data[27:24]), .seg_data(seg2));
  bin2seg bin2seg_3 (.bin_data(data[23:20]), .seg_data(seg3));
  bin2seg bin2seg_4 (.bin_data(data[19:16]), .seg_data(seg4));
  bin2seg bin2seg_5 (.bin_data(data[15:12]), .seg_data(seg5));
  bin2seg bin2seg_6 (.bin_data(data[11:8]), .seg_data(seg6));
  bin2seg bin2seg_7 (.bin_data(data[7:4]), .seg_data(seg7));
  bin2seg bin2seg_8 (.bin_data(data[3:0]), .seg_data(seg8));

  assign led_out = data[7:0];   // 하위 8비트 데이터를 LED로 출력

  // clk와 resetn에 따른 동작: clk_cnt와 com_cnt의 값을 설정
  // 16384번 clock cycle마다 com_cnt를 1 증가시키고 com_cnt가 7이면 다시 0부터 시작
  // 7-segment 디스플레이 패널 각각은 해당하는 com_cnt가 1이면 켜진다.(총 패널 8개 -> com_cnt범위: 0~7)
  // 따라서, com_cnt를 clk cycle에 맞춰 빠르게 변화시키며 사람의 눈에는 연속적으로 보이게 한다. 
  always @ (negedge resetn or posedge clk)
  begin
    if (!resetn)                                // reset 시 counter 초기화
    begin
      clk_cnt <= 15'd0;
      com_cnt <= 3'd0;
    end
    else
    begin
      if (clk_cnt == 15'd16384 )                // 16384번 clock cycle마다 true 
      begin                                     
        clk_cnt <= 15'd0;                       // clk_cnt 초기화
        if (com_cnt == 3'd7) com_cnt <= 3'd0;   // com_cnt가 7일 때 초기화
        else com_cnt <= com_cnt + 3'd1;         // 그 외에는 com_cnt += 1
      end
      else                                      // clock cycle이 16384가 아니면 동작
      begin
        clk_cnt <= clk_cnt + 15'd1;             // clk_cnt += 1
      end
    end
  end

  // com_cnt가 변하거나 seg1 ~ seg8이 변할 때마다 동작 
  // -> com_cnt를 이용해 디스플레이 패널 8개를 빠르게 순환시키고, 새로운 bin_data가 들어오면 바로 seg1~seg8를 업데이트해야 하기 때문
  // com_cnt 값에 따라 표시할 7-segment(seg1 ~ seg8)와 해당하는 segcom (select 신호) 설정
  // 비트 반전을 (ex. ~seg1) 하는 이유는 Zynq는 anode형 7-segment를 사용하기 때문에
  // logical 0가 들어와야 해당하는 segment에 불이 켜지는 것이다.
  always @ (com_cnt or seg1 or seg2 or seg3 or seg4 or seg5 or seg6 or seg7 or seg8)
  begin
    case (com_cnt)
      3'd0: begin
        segcom <= 8'b10000000;
        segout <= ~seg1; end
      3'd1: begin
        segcom <= 8'b01000000;
        segout <= ~seg2; end
      3'd2: begin
        segcom <= 8'b00100000;
        segout <= ~seg3; end
      3'd3: begin
        segcom <= 8'b00010000;
        segout <= ~seg4; end
      3'd4: begin
        segcom <= 8'b00001000;
        segout <= ~seg5; end
      3'd5: begin
        segcom <= 8'b00000100;
        segout <= ~seg6; end
      3'd6: begin
        segcom <= 8'b00000010;
        segout <= ~seg7; end
      default: begin
        segcom <= 8'b00000001;
        segout <= ~seg8; end
    endcase
  end

endmodule
