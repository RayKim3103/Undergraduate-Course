`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/08/16 07:20:33
// Design Name: 
// Module Name: float32_add
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

module float32_add (
    input        clk,
    input        resetn,
    input [31:0] inA_float, // 첫 번째 부동소수점 입력
    input [31:0] inB_float, // 두 번째 부동소수점 입력
    output [31:0] out_float // 덧셈 결과
);

    // 1. 입력 분해
    wire sign_a         = inA_float[31];
    wire sign_b         = inB_float[31];
    wire [7:0] exp_a    = inA_float[30:23];
    wire [7:0] exp_b    = inB_float[30:23];
    wire [22:0] mant_a  = inA_float[22:0];
    wire [22:0] mant_b  = inB_float[22:0];

    // 암묵적 선행 1 추가
    wire [23:0] mant_a_full = (exp_a == 0) ? {1'b0, mant_a} : {1'b1, mant_a};
    wire [23:0] mant_b_full = (exp_b == 0) ? {1'b0, mant_b} : {1'b1, mant_b};

    // 2. 제로 케이스 확인 -------------------------> cycle 1
    wire is_zero_a = (exp_a == 0 && mant_a == 0);
    wire is_zero_b = (exp_b == 0 && mant_b == 0);

    // 제로 케이스 처리
    reg [31:0] result;
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            result <= 0;
        end
        else begin
            if (is_zero_a && is_zero_b) begin
                result <= 32'b0;
            end 
            else if (is_zero_a) begin
                result <= inB_float;
            end 
            else if (is_zero_b) begin
                result <= inA_float;
            end 
            else begin
                result <= 32'b0; // 기본값
            end
        end
    end

    // 3. 지수 정렬 -------------------------> cycle 1
    wire [7:0] exp_diff     = (exp_a >= exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
//    wire exp_a_larger       = (exp_a >= exp_b);
    
    reg exp_a_larger;
    reg [7:0] exp_larger;
    reg [23:0] mant_larger;
    reg [23:0] mant_smaller_shifted;
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            exp_a_larger            <= 0;
            exp_larger              <= 0;
            mant_larger             <= 0;
            mant_smaller_shifted    <= 0;
        end
        else begin
            if (exp_a >= exp_b) begin
                exp_a_larger            <= 1;
                exp_larger              <= exp_a;
                mant_larger             <= mant_a_full;
                mant_smaller_shifted    <= (mant_b_full >> exp_diff);
            end 
            else begin
                exp_a_larger            <= 0;
                exp_larger              <= exp_b;
                mant_larger             <= mant_b_full;
                mant_smaller_shifted    <= (mant_a_full >> exp_diff);
            end
        end
    end 

    // 4. 가수 덧셈 -------------------------> cycle 2
    reg sign_a_reg;
    reg sign_b_reg;
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            sign_a_reg <= 0;
            sign_b_reg <= 0;
        end
        else begin
            sign_a_reg <= sign_a;
            sign_b_reg <= sign_b;
        end
    end 
    reg [24:0]  mant_sum; // 오버플로우를 위해 1비트 추가
    reg         sign_result;
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            mant_sum    <= 0;
            sign_result <= 0;
        end
        else begin
            if (sign_a_reg == sign_b_reg) begin
                mant_sum    <= mant_larger + mant_smaller_shifted;
                sign_result <= sign_a_reg;
            end 
            else begin
                if (mant_larger >= mant_smaller_shifted) begin
                    mant_sum    <= mant_larger - mant_smaller_shifted;
                    if(exp_a_larger)    sign_result <= sign_a_reg;
                    else                sign_result <= sign_b_reg;
                end 
                else begin
                    mant_sum    <= mant_smaller_shifted - mant_larger;
                    if(exp_a_larger)    sign_result <= sign_b_reg;
                    else                sign_result <= sign_a_reg;
                end
            end
        end
    end

    // 5. 정규화 (선행 1 찾기) -------------------------> cycle 3
    reg [4:0] shift_count; // 최대 24비트 쉬프트 (5비트로 표현)
    reg [24:0] temp_mant;
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            shift_count <= 0;
            temp_mant   <= 0;
        end
        else begin
//            if (mant_sum == 0) begin
//                shift_count <= 5'd0;
//                temp_mant   <= mant_sum;
//            end 
//            else if (mant_sum[24]) begin
//                // 오버플로우: 오른쪽 쉬프트, 지수 증가
//                shift_count <= 5'd0;
//                temp_mant   <= mant_sum;
//            end
            // 선행 1 찾기 (상위 비트부터 순차 검사)
            if (mant_sum == 0 || mant_sum[24] || mant_sum[23]) begin
                shift_count <= 5'd0;
                temp_mant   <= mant_sum;
            end 
            else if (mant_sum[22]) begin
                temp_mant   <= mant_sum << 1;
                shift_count <= 5'd1;
            end 
            else if (mant_sum[21]) begin
                temp_mant   <= mant_sum << 2;
                shift_count <= 5'd2;
            end 
            else if (mant_sum[20]) begin
                temp_mant   <= mant_sum << 3;
                shift_count <= 5'd3;
            end 
            else if (mant_sum[19]) begin
                temp_mant   <= mant_sum << 4;
                shift_count <= 5'd4;
            end 
            else if (mant_sum[18]) begin
                temp_mant   <= mant_sum << 5;
                shift_count <= 5'd5;
            end 
            else if (mant_sum[17]) begin
                temp_mant   <= mant_sum << 6;
                shift_count <= 5'd6;
            end 
            else if (mant_sum[16]) begin
                temp_mant   <= mant_sum << 7;
                shift_count <= 5'd7;
            end 
            else if (mant_sum[15]) begin
                temp_mant   <= mant_sum << 8;
                shift_count <= 5'd8;
            end 
            else if (mant_sum[14]) begin
                temp_mant   <= mant_sum << 9;
                shift_count <= 5'd9;
            end 
            else if (mant_sum[13]) begin
                temp_mant   <= mant_sum << 10;
                shift_count <= 5'd10;
            end 
            else if (mant_sum[12]) begin
                temp_mant   <= mant_sum << 11;
                shift_count <= 5'd11;
            end 
            else if (mant_sum[11]) begin
                temp_mant   <= mant_sum << 12;
                shift_count <= 5'd12;
            end 
            else if (mant_sum[10]) begin
                temp_mant   <= mant_sum << 13;
                shift_count <= 5'd13;
            end 
            else if (mant_sum[9]) begin
                temp_mant   <= mant_sum << 14;
                shift_count <= 5'd14;
            end 
            else if (mant_sum[8]) begin
                temp_mant   <= mant_sum << 15;
                shift_count <= 5'd15;
            end 
            else if (mant_sum[7]) begin
                temp_mant   <= mant_sum << 16;
                shift_count <= 5'd16;
            end 
            else if (mant_sum[6]) begin
                temp_mant   <= mant_sum << 17;
                shift_count <= 5'd17;
            end 
            else if (mant_sum[5]) begin
                temp_mant   <= mant_sum << 18;
                shift_count <= 5'd18;
            end 
            else if (mant_sum[4]) begin
                temp_mant   <= mant_sum << 19;
                shift_count <= 5'd19;
            end 
            else if (mant_sum[3]) begin
                temp_mant   <= mant_sum << 20;
                shift_count <= 5'd20;
            end 
            else if (mant_sum[2]) begin
                temp_mant   <= mant_sum << 21;
                shift_count <= 5'd21;
            end 
            else if (mant_sum[1]) begin
                temp_mant   <= mant_sum << 22;
                shift_count <= 5'd22;
            end 
            else if (mant_sum[0]) begin
                temp_mant   <= mant_sum << 23;
                shift_count <= 5'd23;
            end 
            else begin
                temp_mant   <= 25'b0;
                shift_count <= 5'd24;
            end
        end
    end

    // 5. 정규화 -------------------------> cycle 4
    reg [24:0]  mant_sum_delay;
    reg [7:0]   exp_larger_delay,exp_larger_delay2; 
    reg [7:0]   exp_result;
    reg [22:0]  mant_result;
    
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            mant_sum_delay      <= 0;
            exp_larger_delay    <= 0;   exp_larger_delay2      <= 0;
        end
        else begin
            mant_sum_delay      <= mant_sum;
            exp_larger_delay    <= exp_larger;   exp_larger_delay2      <= exp_larger_delay;
        end
    end
    
    always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            mant_result <= 0;
            exp_result  <= 0;
        end
        else begin   
            if (mant_sum_delay== 0) begin
                exp_result  <= 8'b0;
                mant_result <= 23'b0;
            end 
            else if (mant_sum_delay[24]) begin
                // 오버플로우: 오른쪽 쉬프트, 지수 증가
                mant_result <= temp_mant[23:1];
                exp_result  <= exp_larger_delay2 + 1;
            end 
            else begin
                mant_result <= temp_mant[22:0];
                if(exp_larger_delay2 > shift_count)     exp_result  <= exp_larger_delay2 - shift_count;
                else                                    exp_result  <= 8'b0;
//                if (exp_result == 0) begin
//                    mant_result <= 23'b0; // 언더플로우 처리
//                end
            end
        end
    end
    
    wire [22:0] mant_result_2 = (exp_result == 0) ? 23'b0 : mant_result; 

    reg [31:0] result_delay, result_delay2, result_delay3, result_delay4;
    reg        sign_result_delay, sign_result_delay2, sign_result_delay3;
    reg        is_zero_a_delay,is_zero_a_delay2, is_zero_a_delay3, is_zero_a_delay4, is_zero_a_delay5;
    reg        is_zero_b_delay,is_zero_b_delay2, is_zero_b_delay3, is_zero_b_delay4, is_zero_b_delay5; 
   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            result_delay        <= 0; result_delay2         <= 0; result_delay3        <= 0; 
            sign_result_delay   <= 0; sign_result_delay2    <= 0; 
            is_zero_a_delay     <= 0; is_zero_a_delay2      <= 0; is_zero_a_delay3     <= 0; is_zero_a_delay4   <= 0;
            is_zero_b_delay     <= 0; is_zero_b_delay2      <= 0; is_zero_b_delay3     <= 0; is_zero_b_delay4   <= 0;
        end
        else begin
            result_delay  <= result;        result_delay2 <= result_delay;          
            result_delay3 <= result_delay2; 
            
            sign_result_delay <= sign_result; sign_result_delay2 <= sign_result_delay; 
            
            is_zero_a_delay     <= is_zero_a;           is_zero_a_delay2 <= is_zero_a_delay; is_zero_a_delay3 <= is_zero_a_delay2; 
            is_zero_a_delay4    <= is_zero_a_delay3; 
            
            is_zero_b_delay    <= is_zero_b;            is_zero_b_delay2 <= is_zero_b_delay; is_zero_b_delay3 <= is_zero_b_delay2; 
            is_zero_b_delay4   <= is_zero_b_delay3;
        end
    end
    
    // 6. 결과 조립
    assign out_float = (is_zero_a_delay4 || is_zero_b_delay4) ? result_delay3 : {sign_result_delay2, exp_result, mant_result_2};

endmodule









//module float32_add (
//    input [31:0] inA_float, // 첫 번째 부동소수점 입력
//    input [31:0] inB_float, // 두 번째 부동소수점 입력
//    output [31:0] out_float // 덧셈 결과
//);

//    // 1. 입력 분해
//    wire sign_a = inA_float[31];
//    wire sign_b = inB_float[31];
//    wire [7:0] exp_a = inA_float[30:23];
//    wire [7:0] exp_b = inB_float[30:23];
//    wire [22:0] mant_a = inA_float[22:0];
//    wire [22:0] mant_b = inB_float[22:0];

//    // 암묵적 선행 1 추가
//    wire [23:0] mant_a_full = (exp_a == 0) ? {1'b0, mant_a} : {1'b1, mant_a};
//    wire [23:0] mant_b_full = (exp_b == 0) ? {1'b0, mant_b} : {1'b1, mant_b};

//    // 2. 제로 케이스 확인
//    wire is_zero_a = (inA_float == 0);
//    wire is_zero_b = (inB_float == 0);

//    // 제로 케이스 처리
//    reg [31:0] result;
//    always @(*) begin
//        if (is_zero_a && is_zero_b) begin
//            result = 32'b0;
//        end else if (is_zero_a) begin
//            result = inB_float;
//        end else if (is_zero_b) begin
//            result = inA_float;
//        end else begin
//            result = 32'b0; // 기본값
//        end
//    end

//    // 3. 지수 정렬
//    wire [7:0] exp_diff = (exp_a >= exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
//    wire exp_a_larger = (exp_a >= exp_b);
//    wire [7:0] exp_larger = exp_a_larger ? exp_a : exp_b;
//    reg [23:0] mant_smaller_shifted;
//    always @(*) begin
//        if (exp_a_larger) begin
//            mant_smaller_shifted = mant_b_full >> exp_diff;
//        end else begin
//            mant_smaller_shifted = mant_a_full >> exp_diff;
//        end
//    end
//    wire [23:0] mant_larger = exp_a_larger ? mant_a_full : mant_b_full;

//    // 4. 가수 덧셈
//    reg [24:0] mant_sum; // 오버플로우를 위해 1비트 추가
//    reg sign_result;
//    always @(*) begin
//        if (sign_a == sign_b) begin
//            mant_sum = mant_larger + mant_smaller_shifted;
//            sign_result = sign_a;
//        end else begin
//            if (mant_larger >= mant_smaller_shifted) begin
//                mant_sum = mant_larger - mant_smaller_shifted;
//                sign_result = exp_a_larger ? sign_a : sign_b;
//            end else begin
//                mant_sum = mant_smaller_shifted - mant_larger;
//                sign_result = exp_a_larger ? sign_b : sign_a;
//            end
//        end
//    end

//    // 5. 정규화 (조합 논리로 선행 1 찾기)
//    reg [7:0] exp_result;
//    reg [22:0] mant_result;
//    reg [4:0] shift_count; // 최대 24비트 쉬프트 (5비트로 표현)
//    reg [24:0] temp_mant;
//    always @(*) begin
//        temp_mant = mant_sum;
//        shift_count = 5'd0;
//        exp_result = exp_larger;
//        mant_result = temp_mant[22:0];

//        if (mant_sum == 0) begin
//            exp_result = 8'b0;
//            mant_result = 23'b0;
//        end else if (mant_sum[24]) begin
//            // 오버플로우: 오른쪽 쉬프트, 지수 증가
//            mant_result = mant_sum[23:1];
//            exp_result = exp_larger + 1;
//        end else begin
//            // 선행 1 찾기 (상위 비트부터 순차 검사)
//            if (temp_mant[23]) begin
//                shift_count = 5'd0;
//            end else if (temp_mant[22]) begin
//                temp_mant = temp_mant << 1;
//                shift_count = 5'd1;
//            end else if (temp_mant[21]) begin
//                temp_mant = temp_mant << 2;
//                shift_count = 5'd2;
//            end else if (temp_mant[20]) begin
//                temp_mant = temp_mant << 3;
//                shift_count = 5'd3;
//            end else if (temp_mant[19]) begin
//                temp_mant = temp_mant << 4;
//                shift_count = 5'd4;
//            end else if (temp_mant[18]) begin
//                temp_mant = temp_mant << 5;
//                shift_count = 5'd5;
//            end else if (temp_mant[17]) begin
//                temp_mant = temp_mant << 6;
//                shift_count = 5'd6;
//            end else if (temp_mant[16]) begin
//                temp_mant = temp_mant << 7;
//                shift_count = 5'd7;
//            end else if (temp_mant[15]) begin
//                temp_mant = temp_mant << 8;
//                shift_count = 5'd8;
//            end else if (temp_mant[14]) begin
//                temp_mant = temp_mant << 9;
//                shift_count = 5'd9;
//            end else if (temp_mant[13]) begin
//                temp_mant = temp_mant << 10;
//                shift_count = 5'd10;
//            end else if (temp_mant[12]) begin
//                temp_mant = temp_mant << 11;
//                shift_count = 5'd11;
//            end else if (temp_mant[11]) begin
//                temp_mant = temp_mant << 12;
//                shift_count = 5'd12;
//            end else if (temp_mant[10]) begin
//                temp_mant = temp_mant << 13;
//                shift_count = 5'd13;
//            end else if (temp_mant[9]) begin
//                temp_mant = temp_mant << 14;
//                shift_count = 5'd14;
//            end else if (temp_mant[8]) begin
//                temp_mant = temp_mant << 15;
//                shift_count = 5'd15;
//            end else if (temp_mant[7]) begin
//                temp_mant = temp_mant << 16;
//                shift_count = 5'd16;
//            end else if (temp_mant[6]) begin
//                temp_mant = temp_mant << 17;
//                shift_count = 5'd17;
//            end else if (temp_mant[5]) begin
//                temp_mant = temp_mant << 18;
//                shift_count = 5'd18;
//            end else if (temp_mant[4]) begin
//                temp_mant = temp_mant << 19;
//                shift_count = 5'd19;
//            end else if (temp_mant[3]) begin
//                temp_mant = temp_mant << 20;
//                shift_count = 5'd20;
//            end else if (temp_mant[2]) begin
//                temp_mant = temp_mant << 21;
//                shift_count = 5'd21;
//            end else if (temp_mant[1]) begin
//                temp_mant = temp_mant << 22;
//                shift_count = 5'd22;
//            end else if (temp_mant[0]) begin
//                temp_mant = temp_mant << 23;
//                shift_count = 5'd23;
//            end else begin
//                temp_mant = 25'b0;
//                shift_count = 5'd24;
//            end
//            mant_result = temp_mant[22:0];
//            exp_result = (exp_larger > shift_count) ? (exp_larger - shift_count) : 8'b0;
//            if (exp_result == 0) begin
//                mant_result = 23'b0; // 언더플로우 처리
//            end
//        end
//    end

//    // 6. 결과 조립
//    assign out_float = (is_zero_a || is_zero_b) ? result : {sign_result, exp_result, mant_result};

//endmodule
