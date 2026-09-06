`default_nettype none
`timescale 1ns / 1ps

module mul #(
    WIDTH=25,  // width of numbers in bits (integer and fractional)
    FBITS=20   // integer bit 5, fractional bit 20
    )(
    input wire signed [WIDTH-1:0] a,   // multiplier (factor)
    input wire signed [WIDTH-1:0] b,   // mutiplicand (factor)
    output reg signed [WIDTH-1:0] val  // result value: product
    );
 
    // for selecting result
    localparam IBITS = WIDTH - FBITS;     // 정수 비트 수 계산
    localparam MSB = 2*WIDTH - IBITS - 1; // 곱셈결과의 MSB idx, MSB - {LSB + (FBITS-1)} = IBITS
    localparam LSB = WIDTH - IBITS;       // 곱셈결과의 LSB idx, LSB의 [idx + (FBITS-1)] ~...~ [LSB]의 idx까지가 fraction 
                                          // LSB idx보다 작은 idx는 [FBITS-1] ~...~ [0] idx까지

    // for rounding, 반올림을 위한 상수
    // FBITS만큼의 bit를 생성하고 HALF의 값은 MSB가 1이고 나머지는 0인 binary
    // FBITS에 들어가는 수의 절반을 나타내는 
    localparam HALF = {1'b1, {FBITS-1{1'b0}}}; //반올림 기준 값

    reg sig_diff;  // signs difference of inputs
    reg signed [WIDTH-1:0] a1, b1;  // copy of inputs
    reg signed [WIDTH-1:0] prod_t;  // unrounded, truncated product
    reg signed [2*WIDTH-1:0] prod;  // full product
    reg [FBITS-1:0] rbits;          // rounding bits
    reg round;  // rounding required
    reg even;   // even number
    
    // --------------------------- 수정 -------------------------//
    reg signed [WIDTH-1:0] u_result;    
         

    // calculation state machine
    
    always @(*) begin
        // --------------------------- 수정 -------------------------//
        // 부호가 다르면 둘다 양수로 바꾸고 나중에 음수로 변환
        sig_diff = a[WIDTH-1] ^ b[WIDTH-1];
        // --------------------------- 수정 -------------------------//
        a1 = a[WIDTH-1] ? -a : a;   // 입력 a를 a1에 복사 (음수의 경우에 양수로)
        b1 = b[WIDTH-1] ? -b : b;   // 입력 b를 b1에 복사 (음수의 경우에 양수로)
        
        prod = a1 * b1;  // 곱셈
        prod_t = prod[MSB:LSB]; // 2*width의 결과 중 고정소수점이 유지되도록 MSB:LSB 추출
        rbits  = prod[FBITS-1:0]; // prod[MSB:LSB]에서 추출되지 못한 fraction 부분들
        round  = prod[FBITS-1];   // rbits의 최상위 비트가 1인지 0인지 확인
        even  = ~prod[FBITS];     // prod[FBITS] = prod[LSB]이고, 이 값이 1이면 홀수임, 짝수 반올림 (round half to even) 규칙
                                  // 소수 부분이 정확히 0.5일 때, 반올림된 결과가 짝수가 되야하는 규칙
         // --------------------------- 수정 -------------------------//
         u_result = (round && !(even && rbits == HALF)) ? prod_t + 25'b1 : prod_t; // rbits가 0.5보다 크고, round half to even규칙 성립하면 +1
         
         val = sig_diff ? -u_result : u_result;
        end

endmodule