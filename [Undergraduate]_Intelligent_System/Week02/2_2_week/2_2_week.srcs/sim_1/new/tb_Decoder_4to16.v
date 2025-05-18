`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/12 20:04:08
// Design Name: 
// Module Name: tb_Decoder_4to16
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


module tb_Decoder_4to16;

    reg A, B, C, D;                    // input signals, define as reg
    reg G1A, G1B, G2A, G2B;               // input signals, define as reg
    wire [15:0] Decoder_Y;   // output signals, define as wire

    // intantiating Gate_74LS138 Module, Connect-by-name
    Decoder_4to16 dut_Decoder_4to16 (
        .A(A), .B(B), .C(C), .D(D),
        .G1A(G1A), .G1B(G1B), .G2A(G2A), .G2B(G2B),
        .Y(Decoder_Y)
    );

    // starting simulation
    initial begin

        $display("/--------------------------------/");
        $display("Hello EEE3551");
        $display("Start Simulation for 74LS138 Decoder/Demux");
        $display("/--------------------------------/");

        
        {D, C, B, A} = 4'b1111;         // give input as C, B, A = 111
        {G2A, G2B, G1A, G1B} = 4'b1101;    // give enable signal as G2A, G2B, G1A, G1B = 1101
        #10
        
        #10
        {G2A, G2B, G1A, G1B} = 4'b1011;    // give enable signal as G2A, G2B, G1A, G1B = 1011
        #10
        {G2A, G2B, G1A, G1B} = 4'b1000;    // give enable signal as G2A, G2B, G1A, G1B = 1000

        #10
        {A, B, C, D} = 4'b0000;         // give input as D, C, B, A = 0000
        {G2A, G2B, G1A, G1B} = 4'b0000;    // give enable signal as G2A, G2B, G1 = 001
        #10
        {D, C, B, A} = 4'b0001;         // give input as D, C, B, A = 0001
        #10
        {D, C, B, A} = 4'b0010;         // give input as D, C, B, A = 0010
        #10
        {D, C, B, A} = 4'b0011;         // give input as D, C, B, A = 0011
        #10
        {D, C, B, A} = 4'b0100;         // give input as D, C, B, A = 0100
        #10
        {D, C, B, A} = 4'b0101;         // give input as D, C, B, A = 0101
        #10
        {D, C, B, A} = 4'b0110;         // give input as D, C, B, A = 0110
        #10
        {D, C, B, A} = 4'b0111;         // give input as D, C, B, A = 0111
        #10
        {D, C, B, A} = 4'b1000;         // give input as D, C, B, A = 1000
        #10
        {D, C, B, A} = 4'b1001;         // give input as D, C, B, A = 1001
        #10
        {D, C, B, A} = 4'b1010;         // give input as D, C, B, A = 1010
        #10
        {D, C, B, A} = 4'b1011;         // give input as D, C, B, A = 1011
        #10
        {D, C, B, A} = 4'b1100;         // give input as D, C, B, A = 1100
        #10
        {D, C, B, A} = 4'b1101;         // give input as D, C, B, A = 1101
        #10
        {D, C, B, A} = 4'b1110;         // give input as D, C, B, A = 1110
        #10
        {D, C, B, A} = 4'b1111;         // give input as D, C, B, A = 1111
        #100

        $display("/--------------------------------/");
        $display("This is the end of simulation");
        $display("Good Luck");
        $display("/--------------------------------/");

        $finish();
    end

//    // it monitors change of I/O signal and show the result in console 
//    initial begin
//        $monitor($time, " Change of I/O Signal : G2A = %b G2B = %b G1 = %b C = %b B = %b A = %b | Gate_Y = %b Verilog_Y = %b", G2A, G2B, G1, A, B, C, Gate_Y, Verilog_Y);
//    end

endmodule
