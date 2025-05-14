`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2014/06/05 14:20:03
// Design Name: 
// Module Name: top
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


module top (
    DDR_addr,
    DDR_ba,
    DDR_cas_n,
    DDR_ck_n,
    DDR_ck_p,
    DDR_cke,
    DDR_cs_n,
    DDR_dm,
    DDR_dq,
    DDR_dqs_n,
    DDR_dqs_p,
    DDR_odt,
    DDR_ras_n,
    DDR_reset_n,
    DDR_we_n,
    FIXED_IO_ddr_vrn,
    FIXED_IO_ddr_vrp,
    FIXED_IO_mio,
    FIXED_IO_ps_clk,
    FIXED_IO_ps_porb,
    FIXED_IO_ps_srstb,
    CLK,
    RESETn,
    PushButton,
    DIPSwitch,
    LED
);

  inout [14:0]DDR_addr;     // DDR 메모리 주소
  inout [2:0]DDR_ba;        // DDR 뱅크 주소
  inout DDR_cas_n;          // DDR CAS 신호 (Column Address Strobe), DDR 인터페이스 신호 중 하나이다
  inout DDR_ck_n;           // DDR clock negative
  inout DDR_ck_p;           // DDR clock positive
  inout DDR_cke;            // DDR clock enable
  inout DDR_cs_n;           // DDR chip select
  inout [3:0]DDR_dm;        // DDR data mask
  inout [31:0]DDR_dq;       // DDR data bus
  inout [3:0]DDR_dqs_n;     // DDR data strobe negative
  inout [3:0]DDR_dqs_p;     // DDR data strobe positive
  inout DDR_odt;            // DDR On-Die Termination
  inout DDR_ras_n;          // DDR RAS 신호 (Row Address Strobe)
  inout DDR_reset_n;        // DDR reset 신호 (Active Low)
  inout DDR_we_n;           // DDR Write Enable
  inout FIXED_IO_ddr_vrn;   // DDR 메모리의 전압 reference의 음극
  inout FIXED_IO_ddr_vrp;   // DDR 메모리의 전압 reference의 양극
  inout [53:0]FIXED_IO_mio; // Zynq의 MIO 핀 (Multiplexed I/O)
  inout FIXED_IO_ps_clk;    // Zynq 프로세서 시스템 clock
  inout FIXED_IO_ps_porb;   // Zynq 프로세서 전원 reset
  inout FIXED_IO_ps_srstb;  // Zynq 프로세서 소프트 reset
  input CLK;                // FPGA clock 입력
  input RESETn;             // FPGA reset (Active Low)
  input [2:0] PushButton;   // 3개의 PushButton 입력
  input [7:0] DIPSwitch;    // 8개의 DIPSwitch 입력
  output reg [7:0] LED;     // 8개의 LED 출력

  wire [14:0]DDR_addr;
  wire [2:0]DDR_ba;
  wire DDR_cas_n;
  wire DDR_ck_n;
  wire DDR_ck_p;
  wire DDR_cke;
  wire DDR_cs_n;
  wire [3:0]DDR_dm;
  wire [31:0]DDR_dq;
  wire [3:0]DDR_dqs_n;
  wire [3:0]DDR_dqs_p;
  wire DDR_odt;
  wire DDR_ras_n;
  wire DDR_reset_n;
  wire DDR_we_n;
  wire FIXED_IO_ddr_vrn;
  wire FIXED_IO_ddr_vrp;
  wire [53:0]FIXED_IO_mio;
  wire FIXED_IO_ps_clk;
  wire FIXED_IO_ps_porb;
  wire FIXED_IO_ps_srstb;

  reg [2:0] RegPushButton;

always @ (posedge CLK or negedge RESETn)
begin
  if (!RESETn)
  begin
    RegPushButton <= 3'd0;
    LED <= 8'd0;
  end
  else
  begin
    RegPushButton <= PushButton;                            // RegPusbutton은 입력된 값을 저장할 수 있음 (reg 타입이므로)

/******** 기존 코드 ********/   
//    if ((!PushButton[0]) && (RegPushButton[0]))  
//      LED <= DIPSwitch;                                   // DIPSwitch란 FPGA의 swtich이다. 
//    else if ((!PushButton[1]) && (RegPushButton[1]))  
//      LED <= 8'b10000000; 
/***************************/

    if ((!PushButton[0]) && (RegPushButton[0]))            // Push Button 0번 누르고 뗀 직후 & 누른 값 유지 (Always문)
        LED <= 8'b00000000;                          
    else if ((!PushButton[1]) && (RegPushButton[1]))       // Push Button 1번 누르고 뗀 직후 & 누른 값 유지 (Always문)
        LED <= 8'b10000000;
    else if ((!PushButton[2]) && (RegPushButton[2]))       // Push Button 2번 누르고 뗀 직후 & 누른 값 유지 (Always문)
        if (LED == 8'b00000001) 
            LED <= 8'b10000000;                            // 가장 왼쪽으로 가면 다시 오른쪽으로 복귀
        else
            LED <= LED >> 1;                               // 왼쪽으로 shift
  end
end


system_wrapper system_wrapper_i
       (.DDR_addr(DDR_addr),
        .DDR_ba(DDR_ba),
        .DDR_cas_n(DDR_cas_n),
        .DDR_ck_n(DDR_ck_n),
        .DDR_ck_p(DDR_ck_p),
        .DDR_cke(DDR_cke),
        .DDR_cs_n(DDR_cs_n),
        .DDR_dm(DDR_dm),
        .DDR_dq(DDR_dq),
        .DDR_dqs_n(DDR_dqs_n),
        .DDR_dqs_p(DDR_dqs_p),
        .DDR_odt(DDR_odt),
        .DDR_ras_n(DDR_ras_n),
        .DDR_reset_n(DDR_reset_n),
        .DDR_we_n(DDR_we_n),
        .FIXED_IO_ddr_vrn(FIXED_IO_ddr_vrn),
        .FIXED_IO_ddr_vrp(FIXED_IO_ddr_vrp),
        .FIXED_IO_mio(FIXED_IO_mio),
        .FIXED_IO_ps_clk(FIXED_IO_ps_clk),
        .FIXED_IO_ps_porb(FIXED_IO_ps_porb),
        .FIXED_IO_ps_srstb(FIXED_IO_ps_srstb));

endmodule
