`timescale 1ns / 1ps

module tb_74LS138;

    reg A, B, C;                    // input signals, define as reg
    reg G1, G2A, G2B;               // input signals, define as reg
    wire [7:0] Gate_Y, Verilog_Y, Shift_Y;   // output signals, define as wire
    wire [7:0] Gate_1_8_Y, Verilog_1_8_Y;    // output signals, define as wire

    // intantiating Gate_74LS138 Module, Connect-by-name
    Gate_74LS138 dut_Gate (
        .A(A), .B(B), .C(C),
        .G1(G1), .G2A(G2A), .G2B(G2B),
        .Y(Gate_Y)
    );
    
    // intantiating Verilog_74LS138 Module, Connect-by-name
    Verilog_74LS138 dut_Verilog (
        .A(A), .B(B), .C(C),
        .G1(G1), .G2A(G2A), .G2B(G2B),
        .Y(Verilog_Y)
    );
    
    // intantiating Shift_74LS138 Module, Connect-by-name
    Shift_74LS138 dut_Shift (
        .A(A), .B(B), .C(C),
        .G1(G1), .G2A(G2A), .G2B(G2B),
        .Y(Shift_Y)
    );
    
    // intantiating Shift_74LS138 Module, Connect-by-name
    Gate_1_to_8 dut_Gate_1_to_8 (
        .A(A), .B(B), .C(C),
        .G1(G1),
        .Y(Gate_1_8_Y)
    );
    
    // intantiating Shift_74LS138 Module, Connect-by-name
    Verilog_1_to_8 dut_Verilog_1_to_8 (
        .A(A), .B(B), .C(C),
        .G1(G1),
        .Y(Verilog_1_8_Y)
    );

    // starting simulation
    initial begin

        $display("/--------------------------------/");
        $display("Hello EEE3551");
        $display("Start Simulation for 74LS138 Decoder/Demux");
        $display("/--------------------------------/");

        
        {C, B, A} = 3'b111;         // give input as C, B, A = 111
        {G2A, G2B, G1} = 3'b101;    // give enable signal as G2A, G2B, G1 = 101
        #10
        
        #10
        {G2A, G2B, G1} = 3'b011;    // give enable signal as G2A, G2B, G1 = 011
        #10
        {G2A, G2B, G1} = 3'b000;    // give enable signal as G2A, G2B, G1 = 000

        #10
        {A, B, C} = 3'b000;         // give input as C, B, A = 000
        {G2A, G2B, G1} = 3'b001;    // give enable signal as G2A, G2B, G1 = 001
        #10
        {C, B, A} = 3'b001;         // give input as C, B, A = 001
        #10
        {C, B, A} = 3'b010;         // give input as C, B, A = 010
        #10
        {C, B, A} = 3'b011;         // give input as C, B, A = 011
        #10
        {C, B, A} = 3'b100;         // give input as C, B, A = 100
        #10
        {C, B, A} = 3'b101;         // give input as C, B, A = 101
        #10
        {C, B, A} = 3'b110;         // give input as C, B, A = 110
        #10
        {C, B, A} = 3'b111;         // give input as C, B, A = 111
        #100

        $display("/--------------------------------/");
        $display("This is the end of simulation");
        $display("Good Luck");
        $display("/--------------------------------/");

        $finish();
    end

    // it monitors change of I/O signal and show the result in console 
    initial begin
        $monitor($time, " Change of I/O Signal : G2A = %b G2B = %b G1 = %b C = %b B = %b A = %b | Gate_Y = %b Verilog_Y = %b", G2A, G2B, G1, A, B, C, Gate_Y, Verilog_Y);
    end


endmodule
