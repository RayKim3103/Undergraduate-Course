`timescale 1ns / 1ps

module float32_mul#(parameter bias = 8'h7f
)(
    input         clk,
    input         resetn,
    output [31:0] out_float,
    input  [31:0] inA_float, inB_float
);

    wire sign_A, sign_B, sign_Out;
    
    wire [7:0] exp_A, exp_B; 
    wire [7:0] exp_Out;
    wire [9:0] exp_Sum;
    
//    wire [22:0] frac_A, frac_B;
    wire [22:0] frac_Out;
    
    wire [23:0] significand_A, significand_B;
    wire [48-1:0] significand_AxB;
    
    wire is_zero;
    
    assign is_zero = (inA_float == 0) ? 1 : (inB_float == 0) ? 1 : 0; 
    
//    wire infA, infB, zeroA, zeroB, b47;
    
    assign sign_A = inA_float[31];
    assign sign_B = inB_float[31];
    assign sign_Out = sign_A ^ sign_B;
    
    assign significand_A = {1'b1, inA_float[0 +: 23]};
    assign significand_B = {1'b1, inB_float[0 +: 23]};
    assign significand_AxB = is_zero ? 0 : significand_A * significand_B;
    assign frac_Out = significand_AxB[47] ? (significand_AxB[46 -: 23]) : (significand_AxB[45 -: 23]);
    
    //////////////////////////////// FOR TIMING CONSTRAINTS /////////////////////////////////
    assign exp_A    = inA_float[30 -: 8];
    assign exp_B    = inB_float[30 -: 8];
    reg         sign_Out_reg;
    reg [22:0]  frac_Out_reg;
    reg [7:0]   exp_Out_reg;
    reg [9:0]   exp_Sum_reg;
    reg         is_zero_reg;
   always @(posedge clk or negedge resetn) begin
        if(~resetn) begin
            is_zero_reg <= 0;
            sign_Out_reg<= 0;
            frac_Out_reg<= 0;
            exp_Sum_reg <= 0;
        end
        else begin
            is_zero_reg <= is_zero;
            sign_Out_reg<= sign_Out;
            frac_Out_reg<= frac_Out;
            exp_Sum_reg <= exp_A + exp_B - bias + significand_AxB[47];
        end
    end
    assign exp_Sum  = exp_Sum_reg; // if significand has carry -> need to normalize
    assign exp_Out  = exp_Sum[7:0];
    
    assign out_float = is_zero_reg ? 0 : {sign_Out_reg, exp_Out, frac_Out_reg};
    ///////////////////////////////////////////////////////////////////////////////////////

//    assign exp_A    = inA_float[30 -: 8];
//    assign exp_B    = inB_float[30 -: 8];
//    assign exp_Sum  = exp_A + exp_B - bias + significand_AxB[47]; // if significand has carry -> need to normalize
//    assign exp_Out  = exp_Sum[7:0]; 
//    assign out_float = is_zero ? 0 : {sign_Out, exp_Out, frac_Out};

endmodule
