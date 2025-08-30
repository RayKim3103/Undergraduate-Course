`timescale 1ns / 1ps

module quantize #(
    parameter INPUT_DATA_WIDTH = 32,
    parameter OUTPUT_DATA_WIDTH = 8
)(
    input                       clk,
    input                       resetn,
    input  signed [INPUT_DATA_WIDTH-1:0] in_data,
    output signed [OUTPUT_DATA_WIDTH-1:0] out_data
);
    wire sign_input;
    wire [7:0] exp_input;
    wire [22:0] frac_input;
    wire [23:0] significand_input;
    wire is_zero;
    wire is_boundary;
    
    // Clamp boundaries
    localparam signed [OUTPUT_DATA_WIDTH-1:0] MAX_VAL =  8'sd127;
    localparam signed [OUTPUT_DATA_WIDTH-1:0] MIN_VAL = -8'sd128;

    localparam SHIFT_AMOUNT_QUANTIZE    = 9;
    localparam EXP_BW                   = 8;
    localparam SIGNIFICAND_BW           = 24;
    localparam BIAS                     = 127;
    
    // float32 value = (-1)^sign * (1.0 + 0.fracation) * 2^exponent
    assign sign_input           = in_data[31];              // 1bit
    assign significand_input    = {1'b1, in_data[22:0]};    // 1bit + 23bit
    assign exp_input            = in_data[30:23];           // 8bit
    
    wire signed [EXP_BW:0] exp_unbiased;
    assign exp_unbiased = $signed({1'b0, exp_input}) - $signed(BIAS);
    
//    assign is_zero     = (in_data == 0) || (exp_unbiased < 0);
    assign is_zero     = (in_data[INPUT_DATA_WIDTH-2:0] == 0);                  // if, data part == 0 (not including sign bit)
    assign is_small    = (exp_unbiased < 0);
    assign is_boundary = (exp_unbiased >= (SHIFT_AMOUNT_QUANTIZE + EXP_BW - 1));
    
//    // (exponent - Bias) < 0
//    assign is_zero = ((in_data == 0) || ((exp_input - BIAS) < 0));
    
//    // (exponent - Bias) >= (7bit + 9bit)
//    assign is_boundary = ((exp_input - BIAS) >= (SHIFT_AMOUNT_QUANTIZE + EXP_BW - 1)) ? 1 : 0;
    
    ////////////////////////////////////////////////////////// float32 -> int ////////////////////////////////////////////////////////// 
    // [significand << exp] = (-1)^sign * (significand << exp) * 2^0            (ex. 1.01110000... * 2^3 = 1011.1... * 2^0)
    // (cf) max_signifinand_bit_width = 24, keep MSB (exponent - bias + 1) bit  (ex. which is 1011.)
    // executed when 0 =< (exponent - Bias) < 16
    wire [15:0] shifted_significand;
    wire [16:0] shifted_significand_remain;
//    wire [31:0] shifted_significand;
    wire is_remained;
    assign shifted_significand          = (is_boundary || is_zero) ?  0 : (significand_input >> (SIGNIFICAND_BW - (exp_input - BIAS) - 1));  // 정수 부분만 살림
    assign is_remained                  = ((significand_input << ((exp_input - BIAS) + 1)) == 0) ? 0 : 1;                                    // 소수 부분만 살림
    assign shifted_significand_remain   = (sign_input && is_remained) ? shifted_significand + 1 : shifted_significand;                       // 나중에 - 붙여 줄거여서 + 1 함 // np.floor() is just doing -1 one more time
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    reg is_zero_reg;
    reg is_small_reg;
    reg is_boundary_reg;
    reg sign_input_reg;
    reg [16:0] shifted_significand_remain_reg;
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            is_zero_reg                     <= 0;
            is_small_reg                    <= 0;
            is_boundary_reg                 <= 0;
            sign_input_reg                  <= 0;
            shifted_significand_remain_reg  <= 0;
        end
        else begin
            is_zero_reg                     <= is_zero;
            is_small_reg                    <= is_small;
            is_boundary_reg                 <= is_boundary;
            sign_input_reg                  <= sign_input;
            shifted_significand_remain_reg  <= shifted_significand_remain;
        end
    end
    ///////////////////////////////////////////////////////////////////////////////////////
    
    
    // Truncate 9 LSBs (maintains sign)
    wire signed [OUTPUT_DATA_WIDTH - 1:0] msb_data;
    assign msb_data = sign_input_reg ? (-shifted_significand_remain_reg >>> 9) : (shifted_significand_remain_reg >>> 9); // 1'b0
//    assign msb_data = sign_input ? (-shifted_significand_remain >>> 9) : (shifted_significand_remain >>> 9); // 1'b0

    // Clamp with assign
    assign out_data = (is_zero_reg)                             ?       0 :
                      (is_small_reg    && (!sign_input_reg))    ?       0 :
                      (is_small_reg    &&    sign_input_reg)    ?      -1 :
                      (is_boundary_reg && (!sign_input_reg))    ? MAX_VAL :
                      (is_boundary_reg &&    sign_input_reg)    ? MIN_VAL :
                      msb_data;
                      
//    // Clamp with assign
//    assign out_data = (is_zero     && (!sign_input))    ?       0 :
//                      (is_zero     &&   sign_input)     ?      -1 :
//                      (is_boundary && (!sign_input))    ? MAX_VAL :
//                      (is_boundary && sign_input)       ? MIN_VAL :
//                      msb_data;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    wire has_remainder = |(shifted_significand[8:0]); // 버려지는 9 LSB 중 하나라도 1인지 체크, 입력 벡터 중 하나라도 1이면 결과가 1 (// = 1 (a[3]|a[2]|a[1]|a[0] = 0|1|0|1 = 1))
//    wire signed [OUTPUT_DATA_WIDTH-1:0] q_trunc = shifted_significand >>> 9;
    
//    assign out_data = (is_zero     && (!sign_input))    ? 0 :
//                      (is_zero     && sign_input)       ? -1 :
//                      (is_boundary && (!sign_input))    ? MAX_VAL :
//                      (is_boundary && sign_input)       ? MIN_VAL :
//                      (sign_input && has_remainder)     ? -q_trunc - 1 :
//                      (sign_input)                      ? -q_trunc :
//                                                           q_trunc;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

endmodule
