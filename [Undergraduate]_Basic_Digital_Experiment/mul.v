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
    localparam IBITS = WIDTH - FBITS;     // ���� ��Ʈ �� ���
    localparam MSB = 2*WIDTH - IBITS - 1; // ��������� MSB idx, MSB - {LSB + (FBITS-1)} = IBITS
    localparam LSB = WIDTH - IBITS;       // ��������� LSB idx, LSB�� [idx + (FBITS-1)] ~...~ [LSB]�� idx������ fraction 
                                          // LSB idx���� ���� idx�� [FBITS-1] ~...~ [0] idx����

    // for rounding, �ݿø��� ���� ���
    // FBITS��ŭ�� bit�� �����ϰ� HALF�� ���� MSB�� 1�̰� �������� 0�� binary
    // FBITS�� ���� ���� ������ ��Ÿ���� 
    localparam HALF = {1'b1, {FBITS-1{1'b0}}}; //�ݿø� ���� ��

    reg sig_diff;  // signs difference of inputs
    reg signed [WIDTH-1:0] a1, b1;  // copy of inputs
    reg signed [WIDTH-1:0] prod_t;  // unrounded, truncated product
    reg signed [2*WIDTH-1:0] prod;  // full product
    reg [FBITS-1:0] rbits;          // rounding bits
    reg round;  // rounding required
    reg even;   // even number
    
    // --------------------------- ���� -------------------------//
    reg signed [WIDTH-1:0] u_result;    
         

    // calculation state machine
    
    always @(*) begin
        // --------------------------- ���� -------------------------//
        // ��ȣ�� �ٸ��� �Ѵ� ����� �ٲٰ� ���߿� ������ ��ȯ
        sig_diff = a[WIDTH-1] ^ b[WIDTH-1];
        // --------------------------- ���� -------------------------//
        a1 = a[WIDTH-1] ? -a : a;   // �Է� a�� a1�� ���� (������ ��쿡 �����)
        b1 = b[WIDTH-1] ? -b : b;   // �Է� b�� b1�� ���� (������ ��쿡 �����)
        
        prod = a1 * b1;  // ����
        prod_t = prod[MSB:LSB]; // 2*width�� ��� �� �����Ҽ����� �����ǵ��� MSB:LSB ����
        rbits  = prod[FBITS-1:0]; // prod[MSB:LSB]���� ������� ���� fraction �κе�
        round  = prod[FBITS-1];   // rbits�� �ֻ��� ��Ʈ�� 1���� 0���� Ȯ��
        even  = ~prod[FBITS];     // prod[FBITS] = prod[LSB]�̰�, �� ���� 1�̸� Ȧ����, ¦�� �ݿø� (round half to even) ��Ģ
                                  // �Ҽ� �κ��� ��Ȯ�� 0.5�� ��, �ݿø��� ����� ¦���� �Ǿ��ϴ� ��Ģ
         // --------------------------- ���� -------------------------//
         u_result = (round && !(even && rbits == HALF)) ? prod_t + 25'b1 : prod_t; // rbits�� 0.5���� ũ��, round half to even��Ģ �����ϸ� +1
         
         val = sig_diff ? -u_result : u_result;
        end

endmodule