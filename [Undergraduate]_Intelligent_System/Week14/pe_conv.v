`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Engineer: 
// Design Name: Fully Parallel Convolutional PE
// Module Name: pe_conv
// Description:
//   Convolutional PE for 3x3 kernel with IN_CH input channels and full OC parallel psum accumulation
//////////////////////////////////////////////////////////////////////////////////

//module pe_conv #(
//    parameter KERNEL_SIZE = 3,
//    parameter KERNEL_SIZE_SQUARE = 9,
//    parameter IN_CH       = 1,
//    parameter OUT_CH      = 8
//)(
//    input  wire                       clk,
//    input  wire                       resetn,
    
//    input wire w1_en,
//    input wire signed [7:0] weight_in,
    
//    input wire signed [7:0] act00, act01, act02, 
//    input wire signed [7:0] act10, act11, act12, 
//    input wire signed [7:0] act20, act21, act22,
    
//    // 9MAC for 8bit data, lets consider IN_CH & OUT_CH in global buffer(accumulator)
//    // (8bit * 8bit) + ... + (8bit * 8bit) = 16bit + log(9) = 21bit (ceiling)
//    output wire signed [20:0] out
//);
//    reg signed [7:0] weight00, weight01, weight02; 
//    reg signed [7:0] weight10, weight11, weight12;
//    reg signed [7:0] weight20, weight21, weight22;
    
//    reg [3:0] weight_align_count;
    
//    /***** weight inputs are 1cycle delayed because it is from BRAM *****/
//    reg w1_en_delay;
//    always @(posedge clk or negedge resetn) begin
//        if(!resetn) begin
//            w1_en_delay <= 0;
//        end
//        else begin
//            w1_en_delay <= w1_en;
//        end
//    end
    
//    always @(posedge clk or negedge resetn) begin
//        if(!resetn) begin
//            weight00 <= 0; weight01 <= 0; weight02 <= 0;
//            weight10 <= 0; weight11 <= 0; weight12 <= 0;
//            weight20 <= 0; weight21 <= 0; weight22 <= 0;
//            weight_align_count <= 0;
//        end
//        else begin
//                if(w1_en_delay) begin
//                    // silmilar with window shift
//                    weight00 <= weight01;
//                    weight01 <= weight02;
//                    weight02 <= weight10;
//                    weight10 <= weight11;
//                    weight11 <= weight12;
//                    weight12 <= weight20;
//                    weight20 <= weight21;
//                    weight21 <= weight22;
//                    weight22 <= weight_in;
//                end
//        end
//    end
    
//    assign out = act00*weight00 + act01*weight01 + act02*weight02 
//                + act10*weight10 + act11*weight11 + act12*weight12 
//                + act20*weight20 + act21*weight21 + act22*weight22;

//endmodule 


module pe_conv #(
    parameter KERNEL_SIZE = 3,
    parameter KERNEL_SIZE_SQUARE = 9,
    parameter IN_CH  = 1,
    parameter OUT_CH = 8
)(
    input  wire clk,
    input  wire resetn,

    input wire w1_en,
    input wire signed [7:0] weight_in,

    input wire signed [7:0] act00, act01, act02, 
    input wire signed [7:0] act10, act11, act12, 
    input wire signed [7:0] act20, act21, act22,

    output wire signed [23:0] out
);

    // --- Weight shift registers
    reg signed [7:0] weight00, weight01, weight02; 
    reg signed [7:0] weight10, weight11, weight12;
    reg signed [7:0] weight20, weight21, weight22;

    reg w1_en_delay;
    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            w1_en_delay <= 0;
        else
            w1_en_delay <= w1_en;
    end

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            weight00 <= 0; weight01 <= 0; weight02 <= 0;
            weight10 <= 0; weight11 <= 0; weight12 <= 0;
            weight20 <= 0; weight21 <= 0; weight22 <= 0;
        end else if (w1_en_delay) begin
            weight00 <= weight01;
            weight01 <= weight02;
            weight02 <= weight10;
            weight10 <= weight11;
            weight11 <= weight12;
            weight12 <= weight20;
            weight20 <= weight21;
            weight21 <= weight22;
            weight22 <= weight_in;
        end
    end

    // --- Zero Skipping MACs
    wire signed [15:0] mac00 = (act00 == 0) ? 0 : act00 * weight00;
    wire signed [15:0] mac01 = (act01 == 0) ? 0 : act01 * weight01;
    wire signed [15:0] mac02 = (act02 == 0) ? 0 : act02 * weight02;
    wire signed [15:0] mac10 = (act10 == 0) ? 0 : act10 * weight10;
    wire signed [15:0] mac11 = (act11 == 0) ? 0 : act11 * weight11;
    wire signed [15:0] mac12 = (act12 == 0) ? 0 : act12 * weight12;
    wire signed [15:0] mac20 = (act20 == 0) ? 0 : act20 * weight20;
    wire signed [15:0] mac21 = (act21 == 0) ? 0 : act21 * weight21;
    wire signed [15:0] mac22 = (act22 == 0) ? 0 : act22 * weight22;

    // --- Binary Adder Tree
//    wire signed [24:0] s0 = mac00 + mac01;
//    wire signed [24:0] s1 = mac02 + mac10;
//    wire signed [24:0] s2 = mac11 + mac12;
//    wire signed [24:0] s3 = mac20 + mac21;

//    wire signed [25:0] t0 = s0 + s1;
//    wire signed [25:0] t1 = s2 + s3;
    
    /////////////////////////////////// FOR TIMING CONSTRAINTS ///////////////////////////////////
    reg signed [16:0] s0_delay, s1_delay, s2_delay, s3_delay;
    reg signed [15:0] mac22_delay;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            mac22_delay <= 0;
            s0_delay <= 0; 
            s1_delay <= 0; 
            s2_delay <= 0;
            s3_delay <= 0;
        end
        else begin
            mac22_delay <= mac22;
            s0_delay <= mac00 + mac01; 
            s1_delay <= mac02 + mac10; 
            s2_delay <= mac11 + mac12;
            s3_delay <= mac20 + mac21;
        end
    end
    
    reg signed [17:0] t0_delay2, t1_delay2;
    reg signed [15:0] mac22_delay2;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            t0_delay2 <= 0;
            t1_delay2 <= 0;
            mac22_delay2 <= 0;
        end 
        else begin
            t0_delay2 <= s0_delay + s1_delay;
            t1_delay2 <= s2_delay + s3_delay;
            mac22_delay2 <= mac22_delay;
        end
    end
    
    reg signed [18:0] u0_delay3;
    reg signed [15:0] mac22_delay3;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            u0_delay3 <= 0;
            mac22_delay3 <= 0;
        end
        else begin
            u0_delay3 <= t0_delay2 + t1_delay2;
            mac22_delay3 <= mac22_delay2;
        end
    end
    ////////////////////////////////////////////////////////////////////////////////////////////////
    
//    wire signed [26:0] u0 = t0 + t1;

//    assign out = u0 + mac22;

    /////////////////////////////////// FOR TIMING CONSTRAINTS ///////////////////////////////////
    reg signed [19:0] out_delay4;
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            out_delay4 <= 0;
        end
        else begin
            out_delay4 <= u0_delay3 +mac22_delay3;
        end
    end
    ////////////////////////////////////////////////////////////////////////////////////////////////
    
    assign out = out_delay4;
endmodule

