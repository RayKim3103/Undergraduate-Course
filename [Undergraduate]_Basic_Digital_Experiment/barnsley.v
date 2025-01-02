`default_nettype none
`timescale 1ns / 1ps

module barnsley #(
    FP_WIDTH=25,   // total width of fixed-point number: integer + fractional bits
    FP_INT=5      // integer bits in fixed-point number
    ) (
    input  wire clk,                                    // clock
    input wire reset,                                   // reset
    input wire [19:0] iter_max,                         // iteration max
    input wire iter_change,                             // flag signal when change iteration max value -> �̰� btn�� ����

    output reg done,                                    // calculation complete per iteration(high for one tick)
    output wire complete,                               // all calculation complete
    output reg signed [FP_WIDTH-1:0] xn,                // output x,y position 
    output reg signed [FP_WIDTH-1:0] yn
    
//    output reg [19:0] n
    
//    input wire btn1,                                    // button input for increasing iteration     
//    output wire [32:0] random_out
//    output reg signed [FP_WIDTH-1:0] x,                
//    output reg signed [FP_WIDTH-1:0] y
    );

    // fixed-point multiplication module
    reg signed [FP_WIDTH-1:0] mul_a_1, mul_b_1, mul_a_2, mul_b_2;
    wire signed [FP_WIDTH-1:0] mul_val_1, mul_val_2;
    
    
    // instance fixed-point multiplication module
    mul #(.WIDTH(FP_WIDTH), .FBITS(FP_WIDTH - FP_INT)) mul_inst_0 (
        /* verilator lint_on PINCONNECTEMPTY */
        .a(mul_a_1),
        .b(mul_b_1),
        .val(mul_val_1)
    );
    
        // instance fixed-point multiplication module
    mul #(.WIDTH(FP_WIDTH), .FBITS(FP_WIDTH - FP_INT)) mul_inst_1 (
        /* verilator lint_on PINCONNECTEMPTY */
        .a(mul_a_2),
        .b(mul_b_2),
        .val(mul_val_2)
    );
    
    // instance random generator
    wire [31:0] random_out;
    random random_inst(.out(random_out), .clk(clk), .reset(reset), .request(done) );
    
    

  
   /* assignment 
   //
    mul: 
              input mul_a,   // multiplier (factor)
              input mul_b,   // mutiplicand (factor)
              output mul_val  // result value: product
    
    random : 
              input  clk,  
              input  reset,
              input  done
              output random_out
              
    barnsley:    
            input  wire clk,                                    // clock
            input wire reset,                                   // reset
            input wire [19:0] iter_max,                         // iteration max
            input wire iter_change,                             // flag signal when change iteration max value 
            output reg done,                                    // calculation complete per iteration(high for one tick)
            output wire complete,                               // all calculation complete
            output reg signed [FP_WIDTH-1:0] xn,                // output x,y position 
            output reg signed [FP_WIDTH-1:0] yn             
        
        implement the barnsley fern using multiplication module & random number generator module

   */
   
    reg [19:0] n = 0;
    reg signed [FP_WIDTH-1:0] x = 0;
    reg signed [FP_WIDTH-1:0] y = 0;
    reg [3:0] state = 0; // 3bit�� �ٿ��� ��
    reg [19:0] iter_value = 1000;
    
    // ��ư �Է� �������� �����ϰ� �ϴ� logic -> 5���� �ڵ� ������
    reg iter_reg;
    wire iter_button;
    
    always @(posedge clk) 
    begin
        iter_reg <= iter_change;
    end
    
    assign iter_button = iter_change & ~iter_reg;
    
    
    
    //---------------- step1 & 2---------------------//
    
    
    localparam signed [FP_WIDTH-1:0] FP_0_16 = 25'b0000000101000111101011100; // 0.16 in fixed-point -> 0000000101000111101011100
    localparam signed [FP_WIDTH-1:0] FP_0_15 = 25'b0000000100110011001100110; // 0.15 in fixed-point -> 0000000100110011001100110
    localparam signed [FP_WIDTH-1:0] NFP_0_15 = 25'b1111111011001100110011010; // -0.15 in fixed-point -> 1111111011001100110011010
    localparam signed [FP_WIDTH-1:0] FP_0_85 = 25'b0000011011001100110011001; // 0.85 in fixed-point -> 0000011011001100110011001
    localparam signed [FP_WIDTH-1:0] FP_0_04 = 25'b0000000001010001111010111;  // 0.04 in fixed-point -> 0000000001010001111010111
    localparam signed [FP_WIDTH-1:0] NFP_0_04 = 25'b1111110011001100110011010; // -0.04 
    localparam signed [FP_WIDTH-1:0] FP_0_2  = 25'b0000000110011001100110011;  // 0.2 in fixed-point -> 0000000110011001100110011
    localparam signed [FP_WIDTH-1:0] FP_0_23 = 25'b0000000111010111000010100; // 0.23 in fixed-point -> 0000000111010111000010100
    localparam signed [FP_WIDTH-1:0] FP_0_22 = 25'b0000000111000010100011110; // 0.22 in fixed-point -> 0000000111000010100011110
    localparam signed [FP_WIDTH-1:0] FP_0_24 = 25'b0000000111101011100001010; // 0.24 in fixed-point -> 0000000111101011100001010
    localparam signed [FP_WIDTH-1:0] FP_0_26 = 25'b0000001000010100011110101; // 0.26 in fixed-point
    localparam signed [FP_WIDTH-1:0] NFP_0_26 = 25'b1111110111101011100001011;// -0.26
    localparam signed [FP_WIDTH-1:0] FP_0_28 = 25'b0000001000111101011100001; // 0.28 in fixed-point
    localparam signed [FP_WIDTH-1:0] FP_1_5  = 25'b0000110000000000000000000; // 1.5 in fixed-point
    
    // n=0, �� ���� random_out�� 0���� �ʱ�ȭ �Ǿ� �־� 1%Ȯ���� ��찡 ����
    // ������, random.v�� ���� �ʱ� temp���� 13���� �����Ǿ� �ֱ⿡ 1% Ȯ���� ����ȴ�.
    // ����, n=0�� ���� ���� ������� �ʾƵ� �ڿ������� �����Ѵ�.  
    assign complete = (n >= iter_value);

    always @(posedge clk or posedge reset) 
    begin
        // reset
        if (reset) 
            begin
                x <= 0;  y <= 0;
                xn <= 0; yn <= 0;
                n <= 0;  done <= 0;
                state <= 0;
                iter_value <= 1000;
            end
        // changing iteration 
        else if (iter_button) 
            begin
                if (iter_value < 10000)
                    iter_value <= iter_value + 3000; 
            end
        
        //  while loop in pseudo code [ while n < maximum iterations do ]
        // ��������� cylcle > 4cycle, since random number uses cycle
        else if (!complete) 
            begin
                case (state)
                    // random number ���� ��û -> random number�� Ȯ���ν� ���� 
                    4'b0000: 
                    begin
                        done <= 0;
                        state <= 4'b0001;
                    end
                    // ������ random number�� ������� Barnsley Fern ��ȯ�� ����
                    
                    // --------------------- x�� ���� �� ----------------------------- //
                    4'b0001: 
                    begin
                        done <= 0;
                        // r < 0.01
                        // random_out/(2^32-1) < 0.01 => random_out < 0.01 * (2^32-1) 
                        if (random_out < 32'd21474836)
                                xn <= 0;

                        // r < 0.86
                        // random_out < 0.86 * (2^32-1) 
                        else if (random_out < 32'd1846835936)
                            begin 
                                mul_a_1 <= x;
                                mul_b_1 <= FP_0_85; // 0.85
                                mul_a_2 <= y;
                                mul_b_2 <= FP_0_04; // 0.04
                            end
                        // r < 0.93
                        else if (random_out < 32'd1997159792)
                            begin 
                                mul_a_1 <= x;
                                mul_b_1 <= FP_0_2; // 0.2
                                
                                mul_a_2 <= y;
                                mul_b_2 <= (FP_0_26); // -0.26 => ���߿� calculate���� ���� ó��
                            end
                        else 
                            begin
                                mul_a_1 <= x;
                                mul_b_1 <= (FP_0_15); // -0.15 => ���߿� calculate���� ���� ó��
                                
                                mul_a_2 <= y;
                                mul_b_2 <= FP_0_28; // 0.28
                            end
                        state <= 4'b0010;
                    end
                    // --------------------- x�� ���� �� calculate ----------------------------- //
                    4'b0010: 
                    begin
                        done <= 0;
                        // r < 0.01
                        if (random_out < 32'd21474836)
                                xn <= 0;

                        // r < 0.86
                        else if (random_out < 32'd1846835936)
                            begin
                                // xn = 0.85 * x + 0.04 * y
                                xn <= mul_val_1 + mul_val_2; 
                            end
                            
                        // r < 0.93
                        else if (random_out < 32'd1997159792)
                            begin 
                                // xn = 0.2 * x - 0.26 * y 
                                xn <= mul_val_1 - mul_val_2; 
                            end
                        else 
                            begin
                                // xn = -0.15 * x + 0.28 * y 
                                xn <= -mul_val_1 + mul_val_2;
                            end
                        state <= 4'b0011;
                    end
                    // --------------------- y�� ���� �� ----------------------------- //
                    4'b0011: 
                    begin
                        done <= 0;
                        // r < 0.01
                        if (random_out < 32'd21474836)
                            begin
                                mul_a_1 <= FP_0_16; // 0.16
                                mul_b_1 <= y; 
                            end
                        // r < 0.86
                        else if (random_out < 32'd1846835936)
                            begin 
                                mul_a_1 <= x;
                                mul_b_1 <= (FP_0_04); // -0.04 => ���߿� calculate���� ���� ó��
                                mul_a_2 <= y;
                                mul_b_2 <= FP_0_85; // 0.85
                            end
                        // r < 0.93
                        else if (random_out < 32'd1997159792)
                            begin 
                                mul_a_1 <= x;
                                mul_b_1 <= FP_0_23; // 0.23
                                
                                mul_a_2 <= y;
                                mul_b_2 <= FP_0_22; // 0.22
                            end
                        else 
                            begin
                                mul_a_1 <= x;
                                mul_b_1 <= FP_0_26; // 0.26
                                
                                mul_a_2 <= y;
                                mul_b_2 <= FP_0_24; // 0.24
                            end
                        state <= 4'b0100;
                    end 
                    // --------------------- y�� ���� �� calculate ----------------------------- //
                    4'b0100: 
                    begin
                        done <= 0;
                        // r < 0.01
                        if (random_out < 32'd21474836)
                            begin
                                // yn = 0.16 * y
                                yn <= mul_val_1;
                            end
                        // r < 0.86
                        else if (random_out < 32'd1846835936) 
                            begin 
                                // yn = -0.04 * x + 0.85 * y + 1.5
                                yn <= -mul_val_1 + mul_val_2 + FP_1_5;
                            end
                        // r < 0.93
                        else if (random_out < 32'd1997159792)
                            begin 
                                // yn = 0.23 * x + 0.22 * y + 1.5
                                yn <= mul_val_1 + mul_val_2 + FP_1_5;
                            end
                        else 
                            begin
                                // yn = 0.26 * x + 0.24 * y
                                yn <= mul_val_1 + mul_val_2;
                            end
                        state <= 4'b0101;
                    end                     
                    4'b0101:
                    begin
                        x <= xn;
                        y <= yn;
                        n <= n + 1;
                        done <= 1;
                        state <= 4'b0000;
                    end
                endcase
            end
    end
        
    assign iter_max = iter_value;
endmodule