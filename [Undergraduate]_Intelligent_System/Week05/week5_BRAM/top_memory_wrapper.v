`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/28 15:04:56
// Design Name: 
// Module Name: top_memory_wrapper
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


module top_memory_wrapper(
    input wire clk,     // input clk port
    input wire resetn,  // input reset port

    input wire start,   // input start signal port
    output wire done,   // output done signal port

    input wire ext_enb, // test bench에서 들어오는 enable 신호 port
    input wire [7:0] ext_addrb, // test bench에서 들어오는 addrb 신호 port
    output wire [15:0] ext_doutb // test bench에서 들어오는 doutb 신호 port
    );
    
    // Port A는 write 전용
    wire ena;           // Port A의 enable 신호
    wire wea;           // Port A의 write enble 신호
    wire [7:0] addra;   // Port A의 address 신호
    wire [15:0] dina;   // Port A의 data in 신호
    
    // Port B는 read 전용
    wire enb;           // Port B의 enable 신호
    wire [7:0] addrb;   // Port B의 address 신호
    wire [15:0] doutb;  // Port B의 data out 신호

    // memory_ctrlr에서 read 시에 쓰는 enable신호
    wire int_enb;
    // memory_ctrlr에서 read 시에 쓰는 address신호
    wire [7:0] int_addrb;
    // memory_ctrlr에서 read 시에 쓰는 data out신호
    wire [15:0] int_doutb;

    /** done 신호가 뜨면, test bench에서 memory 안의 값을 정답과 비교한다 **/
    /** 이를 위하여 test bench용 신호와 read, write용 신호를 구분한다. **/
    assign enb = (done)? ext_enb: int_enb;
    assign addrb = (done)? ext_addrb: int_addrb;
    assign ext_doutb = doutb;

    // control read & write operation of BRAM
    memory_ctrlr Umemory_ctrlr(
      .clk(clk),
      .resetn(resetn),
      
      .start(start),
      .done(done),

      .ena(ena),
      .wea(wea),
      .addra(addra),
      .dina(dina),
      
      .enb(int_enb),
      .addrb(int_addrb),
      .doutb(doutb)

    );
    
    // BRAM (Simple Dual Port RAM)
    blk_mem_gen_0 UBRAM(
        .clka(clk), 
        .ena(ena), 
        .wea(wea), 
        .addra(addra), 
        .dina(dina), 
        .clkb(clk), 
        .enb(enb), 
        .addrb(addrb), 
        .doutb(doutb)
      );
    
endmodule 
