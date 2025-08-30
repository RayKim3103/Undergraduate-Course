`timescale 1ns / 1ps

module PE_CONV_3x3 #(
    parameter KERNEL_SIZE = 3,
    parameter KERNEL_SIZE_SQUARE = 9,
    parameter IN_CH  = 1,
    parameter OUT_CH = 8,
    parameter Data_Precision = 32
)(
    input  wire clk,
    input  wire resetn,

    input wire weight_en,
    input wire signed [Data_Precision-1:0] weight_in,

    input wire signed [Data_Precision-1:0] act00, act01, act02, 
    input wire signed [Data_Precision-1:0] act10, act11, act12, 
    input wire signed [Data_Precision-1:0] act20, act21, act22,

    output wire signed [Data_Precision-1:0] out
);
    // --- Weight shift registers
    reg signed [Data_Precision-1:0] weight00, weight01, weight02; 
    reg signed [Data_Precision-1:0] weight10, weight11, weight12;
    reg signed [Data_Precision-1:0] weight20, weight21, weight22;

    reg weight_en_delay;
    always @(posedge clk or negedge resetn) begin
        if (!resetn)
            weight_en_delay <= 0;
        else
            weight_en_delay <= weight_en;
    end

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            weight00 <= 0; weight01 <= 0; weight02 <= 0;
            weight10 <= 0; weight11 <= 0; weight12 <= 0;
            weight20 <= 0; weight21 <= 0; weight22 <= 0;
        end else if (weight_en_delay) begin
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

    // --- MACs (total 9 multiply & add) ---
    wire signed [Data_Precision-1:0] mac00;
    wire signed [Data_Precision-1:0] mac01;
    wire signed [Data_Precision-1:0] mac02;
    wire signed [Data_Precision-1:0] mac10;
    wire signed [Data_Precision-1:0] mac11;
    wire signed [Data_Precision-1:0] mac12;
    wire signed [Data_Precision-1:0] mac20;
    wire signed [Data_Precision-1:0] mac21;
    wire signed [Data_Precision-1:0] mac22;
    
    float32_mul float_mult_0 (
        .clk(clk), .resetn(resetn),
        .out_float(mac00),
        .inA_float(act00),
        .inB_float(weight00)
    );
    float32_mul float_mult_1 (
        .clk(clk), .resetn(resetn),
        .out_float(mac01),
        .inA_float(act01),
        .inB_float(weight01)
    );
    float32_mul float_mult_2 (
        .clk(clk), .resetn(resetn),
        .out_float(mac02),
        .inA_float(act02),
        .inB_float(weight02)
    );
    float32_mul float_mult_3 (
        .clk(clk), .resetn(resetn),
        .out_float(mac10),
        .inA_float(act10),
        .inB_float(weight10)
    );
    float32_mul float_mult_4 (
        .clk(clk), .resetn(resetn),
        .out_float(mac11),
        .inA_float(act11),
        .inB_float(weight11)
    );
    float32_mul float_mult_5 (
        .clk(clk), .resetn(resetn),
        .out_float(mac12),
        .inA_float(act12),
        .inB_float(weight12)
    );
    float32_mul float_mult_6 (
        .clk(clk), .resetn(resetn),
        .out_float(mac20),
        .inA_float(act20),
        .inB_float(weight20)
    );
    float32_mul float_mult_7 (
        .clk(clk), .resetn(resetn),
        .out_float(mac21),
        .inA_float(act21),
        .inB_float(weight21)
    );
    float32_mul float_mult_8 (
        .clk(clk), .resetn(resetn),
        .out_float(mac22),
        .inA_float(act22),
        .inB_float(weight22)
    );
    
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    reg signed [Data_Precision-1:0] mac00_delay, mac01_delay, mac02_delay;
    reg signed [Data_Precision-1:0] mac10_delay, mac11_delay, mac12_delay;
    reg signed [Data_Precision-1:0] mac20_delay, mac21_delay, mac22_delay;
   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            mac00_delay <= 0; mac01_delay <= 0; mac02_delay <= 0;
            mac10_delay <= 0; mac11_delay <= 0; mac12_delay <= 0;
            mac20_delay <= 0; mac21_delay <= 0; mac22_delay <= 0;
        end
        else begin
            mac00_delay <= mac00; mac01_delay <= mac01; mac02_delay <= mac02;
            mac10_delay <= mac10; mac11_delay <= mac11; mac12_delay <= mac12;
            mac20_delay <= mac20; mac21_delay <= mac21; mac22_delay <= mac22;
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////
    
    // --- Binary Adder Tree ---
    wire signed [Data_Precision-1:0] s0;
    wire signed [Data_Precision-1:0] s1;
    wire signed [Data_Precision-1:0] s2;
    wire signed [Data_Precision-1:0] s3;
        
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    reg signed [Data_Precision-1:0] s0_delay, s1_delay, s2_delay, s3_delay;

   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            s0_delay <= 0; s1_delay <= 0; s2_delay <= 0; s3_delay <= 0;
        end
        else begin
            s0_delay <= s0; s1_delay <= s1; 
            s2_delay <= s2; s3_delay <= s3;
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////
    
    float32_add float_add_s0 (
        .clk(clk), .resetn(resetn),
        .out_float(s0),
        .inA_float(mac00_delay),
        .inB_float(mac01_delay)
    );
    float32_add float_add_s1 (
        .clk(clk), .resetn(resetn),
        .out_float(s1),
        .inA_float(mac02_delay),
        .inB_float(mac10_delay)
    );
    float32_add float_add_s2 (
        .clk(clk), .resetn(resetn),
        .out_float(s2),
        .inA_float(mac11_delay),
        .inB_float(mac12_delay)
    );
    float32_add float_add_s3 (
        .clk(clk), .resetn(resetn),
        .out_float(s3),
        .inA_float(mac20_delay),
        .inB_float(mac21_delay)
    );
    
    wire signed [Data_Precision-1:0] t0;
    wire signed [Data_Precision-1:0] t1;
        
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    reg signed [Data_Precision-1:0] t0_delay, t1_delay;

   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            t0_delay <= 0; t1_delay <= 0;
        end
        else begin
            t0_delay <= t0; t1_delay <= t1; 
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////
    
    float32_add float_add_t0 (
        .clk(clk), .resetn(resetn),
        .out_float(t0),
        .inA_float(s0_delay),
        .inB_float(s1_delay)
    );
    float32_add float_add_t1 (
        .clk(clk), .resetn(resetn),
        .out_float(t1),
        .inA_float(s2_delay),
        .inB_float(s3_delay)
    );
    
    wire signed [Data_Precision-1:0] u0;
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    reg signed [Data_Precision-1:0] u0_delay;
    reg signed [Data_Precision-1:0] mac22_delay2, mac22_delay3, mac22_delay4;                           // mac22_delay,
    reg signed [Data_Precision-1:0] mac22_delay5, mac22_delay6, mac22_delay7, mac22_delay8;
    reg signed [Data_Precision-1:0] mac22_delay9, mac22_delay10, mac22_delay11, mac22_delay12;
    reg signed [Data_Precision-1:0] mac22_delay13, mac22_delay14, mac22_delay15, mac22_delay16;

   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            u0_delay <= 0; 
            mac22_delay2  <= 0; mac22_delay3  <= 0; mac22_delay4  <= 0;                                    // mac22_delay <= 0;
            mac22_delay5  <= 0; mac22_delay6  <= 0; mac22_delay7  <= 0; mac22_delay8  <= 0;
            mac22_delay9  <= 0; mac22_delay10 <= 0; mac22_delay11 <= 0; mac22_delay12 <= 0;
            mac22_delay13 <= 0; mac22_delay14 <= 0; mac22_delay15 <= 0; mac22_delay16 <= 0; 
        end
        else begin
            u0_delay <= u0; 
            mac22_delay2  <= mac22_delay;   mac22_delay3  <= mac22_delay2;  mac22_delay4  <= mac22_delay3;    // mac22_delay <= mac22;
            mac22_delay5  <= mac22_delay4;  mac22_delay6  <= mac22_delay5;  mac22_delay7  <= mac22_delay6;  mac22_delay8  <= mac22_delay7;
            mac22_delay9  <= mac22_delay8;  mac22_delay10 <= mac22_delay9;  mac22_delay11 <= mac22_delay10; mac22_delay12 <= mac22_delay11;
            mac22_delay13 <= mac22_delay12; mac22_delay14 <= mac22_delay13; mac22_delay15 <= mac22_delay14; mac22_delay16 <= mac22_delay15; 
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////
    
    float32_add float_add_u0 (
        .clk(clk), .resetn(resetn),
        .out_float(u0),
        .inA_float(t0_delay),
        .inB_float(t1_delay)
    );
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    wire signed [Data_Precision-1:0]    out_direct;
    reg signed [Data_Precision-1:0]     out_reg;
    
   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            out_reg <= 0; 
        end
        else begin
            out_reg <= out_direct; 
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////
    
    assign out = out_reg;
    
    float32_add float_add_out (
        .clk(clk), .resetn(resetn),
        .out_float(out_direct),
        .inA_float(u0_delay),
        .inB_float(mac22_delay16)   // mac22_delay4
    );
    
endmodule

//    wire signed [Data_Precision-1:0] mac00 = (act00 == 0) ? 0 : act00 * weight00;
//    wire signed [Data_Precision-1:0] mac01 = (act01 == 0) ? 0 : act01 * weight01;
//    wire signed [Data_Precision-1:0] mac02 = (act02 == 0) ? 0 : act02 * weight02;
//    wire signed [Data_Precision-1:0] mac10 = (act10 == 0) ? 0 : act10 * weight10;
//    wire signed [Data_Precision-1:0] mac11 = (act11 == 0) ? 0 : act11 * weight11;
//    wire signed [Data_Precision-1:0] mac12 = (act12 == 0) ? 0 : act12 * weight12;
//    wire signed [Data_Precision-1:0] mac20 = (act20 == 0) ? 0 : act20 * weight20;
//    wire signed [Data_Precision-1:0] mac21 = (act21 == 0) ? 0 : act21 * weight21;
//    wire signed [Data_Precision-1:0] mac22 = (act22 == 0) ? 0 : act22 * weight22;

    // --- Binary Adder Tree
//    wire signed [24:0] s0 = mac00 + mac01;
//    wire signed [24:0] s1 = mac02 + mac10;
//    wire signed [24:0] s2 = mac11 + mac12;
//    wire signed [24:0] s3 = mac20 + mac21;

//    wire signed [25:0] t0 = s0 + s1;
//    wire signed [25:0] t1 = s2 + s3;
//    wire signed [26:0] u0 = t0 + t1;

//    assign out = u0 + mac22;
    
//    /////////////////////////////////// FOR TIMING CONSTRAINTS ///////////////////////////////////
//    reg signed [32:0] s0_delay, s1_delay, s2_delay, s3_delay;
//    reg signed [Data_Precision-1:0] mac22_delay;
//    always @(posedge clk or negedge resetn) begin
//        if (!resetn) begin
//            mac22_delay <= 0;
//            s0_delay <= 0; 
//            s1_delay <= 0; 
//            s2_delay <= 0;
//            s3_delay <= 0;
//        end
//        else begin
//            mac22_delay <= mac22;
//            s0_delay <= mac00 + mac01; 
//            s1_delay <= mac02 + mac10; 
//            s2_delay <= mac11 + mac12;
//            s3_delay <= mac20 + mac21;
//        end
//    end
    
//    reg signed [33:0] t0_delay2, t1_delay2;
//    reg signed [Data_Precision-1:0] mac22_delay2;
//    always @(posedge clk or negedge resetn) begin
//        if (!resetn) begin
//            t0_delay2 <= 0;
//            t1_delay2 <= 0;
//            mac22_delay2 <= 0;
//        end 
//        else begin
//            t0_delay2 <= s0_delay + s1_delay;
//            t1_delay2 <= s2_delay + s3_delay;
//            mac22_delay2 <= mac22_delay;
//        end
//    end
    
//    reg signed [34:0] u0_delay3;
//    reg signed [Data_Precision-1:0] mac22_delay3;
//    always @(posedge clk or negedge resetn) begin
//        if (!resetn) begin
//            u0_delay3 <= 0;
//            mac22_delay3 <= 0;
//        end
//        else begin
//            u0_delay3 <= t0_delay2 + t1_delay2;
//            mac22_delay3 <= mac22_delay2;
//        end
//    end
//    ////////////////////////////////////////////////////////////////////////////////////////////////

//    /////////////////////////////////// FOR TIMING CONSTRAINTS ///////////////////////////////////
//    reg signed [35:0] out_delay4;
//    always @(posedge clk or negedge resetn) begin
//        if(!resetn) begin
//            out_delay4 <= 0;
//        end
//        else begin
//            out_delay4 <= u0_delay3 +mac22_delay3;
//        end
//    end
//    ////////////////////////////////////////////////////////////////////////////////////////////////
    
//    assign out = out_delay4;
