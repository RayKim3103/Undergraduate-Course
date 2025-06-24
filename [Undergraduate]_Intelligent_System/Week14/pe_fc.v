//`timescale 1ns / 1ps
//module pe_fc_multi #(
//    parameter IN_SIZE = 2304,
//    parameter ACC_WIDTH = 24,
//    parameter DATA_WIDTH = 8
//)(
//    input  wire clk,
//    input  wire resetn,
//    //input  wire start,
    
//    input  wire signed [7:0] act_in,
    
//    input  wire signed [7:0] w0_in,
//    input  wire signed [7:0] w1_in,
//    input  wire signed [7:0] w2_in,
//    input  wire signed [7:0] w3_in,
//    input  wire signed [7:0] w4_in,
//    input  wire signed [7:0] w5_in,
//    input  wire signed [7:0] w6_in,
//    input  wire signed [7:0] w7_in,
//    input  wire signed [7:0] w8_in,
//    input  wire signed [7:0] w9_in,
    
//    input  wire valid_in,
    
//    output wire signed [DATA_WIDTH-1 :0] out_0,
//    output wire signed [DATA_WIDTH-1 :0] out_1,
//    output wire signed [DATA_WIDTH-1 :0] out_2,
//    output wire signed [DATA_WIDTH-1 :0] out_3,
//    output wire signed [DATA_WIDTH-1 :0] out_4,
//    output wire signed [DATA_WIDTH-1 :0] out_5,
//    output wire signed [DATA_WIDTH-1 :0] out_6,
//    output wire signed [DATA_WIDTH-1 :0] out_7,
//    output wire signed [DATA_WIDTH-1 :0] out_8,
//    output wire signed [DATA_WIDTH-1 :0] out_9,

//    output reg reg_done
//);
//    reg done;
//    reg [11:0] cnt;
//    //reg active;
//    reg signed [ACC_WIDTH-1 :0] psum_out_0;
//    reg signed [ACC_WIDTH-1 :0] psum_out_1;
//    reg signed [ACC_WIDTH-1 :0] psum_out_2;
//    reg signed [ACC_WIDTH-1 :0] psum_out_3;
//    reg signed [ACC_WIDTH-1 :0] psum_out_4;
//    reg signed [ACC_WIDTH-1 :0] psum_out_5;
//    reg signed [ACC_WIDTH-1 :0] psum_out_6;
//    reg signed [ACC_WIDTH-1 :0] psum_out_7;
//    reg signed [ACC_WIDTH-1 :0] psum_out_8;
//    reg signed [ACC_WIDTH-1 :0] psum_out_9;
    
//    wire signed [DATA_WIDTH-1 :0] psum_quantize_0;
//    wire signed [DATA_WIDTH-1 :0] psum_quantize_1;
//    wire signed [DATA_WIDTH-1 :0] psum_quantize_2;
//    wire signed [DATA_WIDTH-1 :0] psum_quantize_3;
//    wire signed [DATA_WIDTH-1 :0] psum_quantize_4;
//    wire signed [DATA_WIDTH-1 :0] psum_quantize_5;
//    wire signed [DATA_WIDTH-1 :0] psum_quantize_6;
//    wire signed [DATA_WIDTH-1 :0] psum_quantize_7;
//    wire signed [DATA_WIDTH-1 :0] psum_quantize_8;
//    wire signed [DATA_WIDTH-1 :0] psum_quantize_9;
    
//    reg signed [ACC_WIDTH-1:0] acc0, acc1, acc2, acc3, acc4;
//    reg signed [ACC_WIDTH-1:0] acc5, acc6, acc7, acc8, acc9;

////    wire signed [15:0] prod0_wire = act_in * w0_in;
////    wire signed [15:0] prod1_wire = act_in * w1_in;
////    wire signed [15:0] prod2_wire = act_in * w2_in;
////    wire signed [15:0] prod3_wire = act_in * w3_in;
////    wire signed [15:0] prod4_wire = act_in * w4_in;
////    wire signed [15:0] prod5_wire = act_in * w5_in;
////    wire signed [15:0] prod6_wire = act_in * w6_in;
////    wire signed [15:0] prod7_wire = act_in * w7_in;
////    wire signed [15:0] prod8_wire = act_in * w8_in;
////    wire signed [15:0] prod9_wire = act_in * w9_in;

//    reg signed [15:0] prod0_reg, prod1_reg, prod2_reg, prod3_reg, prod4_reg, prod5_reg, prod6_reg, prod7_reg, prod8_reg, prod9_reg;
    
//    wire signed [15:0] prod0 = prod0_reg;
//    wire signed [15:0] prod1 = prod1_reg;
//    wire signed [15:0] prod2 = prod2_reg;
//    wire signed [15:0] prod3 = prod3_reg;
//    wire signed [15:0] prod4 = prod4_reg;
//    wire signed [15:0] prod5 = prod5_reg;
//    wire signed [15:0] prod6 = prod6_reg;
//    wire signed [15:0] prod7 = prod7_reg;
//    wire signed [15:0] prod8 = prod8_reg;
//    wire signed [15:0] prod9 = prod9_reg;
    
//    /***** valid_in delay for synchronizing accumulation with products *****/
//    reg valid_in_delay;
//    always @(posedge clk or negedge resetn) begin
//        if(!resetn) begin
//            valid_in_delay <= 0;
//        end
//        else begin
//            valid_in_delay <= valid_in; 
//        end
//    end
    
//    always @(posedge clk or negedge resetn) begin
//        if (!resetn) begin                          //iteration -> resetn
//            prod0_reg <= 0;
//            prod1_reg <= 0;
//            prod2_reg <= 0;
//            prod3_reg <= 0;
//            prod4_reg <= 0;
//            prod5_reg <= 0;
//            prod6_reg <= 0;
//            prod7_reg <= 0;
//            prod8_reg <= 0;
//            prod9_reg <= 0;
//        end 
//        else begin
//            prod0_reg <= act_in * w0_in;
//            prod1_reg <= act_in * w1_in;
//            prod2_reg <= act_in * w2_in;
//            prod3_reg <= act_in * w3_in;
//            prod4_reg <= act_in * w4_in;
//            prod5_reg <= act_in * w5_in;
//            prod6_reg <= act_in * w6_in;
//            prod7_reg <= act_in * w7_in;
//            prod8_reg <= act_in * w8_in;
//            prod9_reg <= act_in * w9_in;
//        end
//    end
    
    
//    reg [3:0] done_cnt;
//    always @(posedge clk or negedge resetn) begin
//        if (!resetn) begin                          //iteration -> resetn
//            cnt <= 0;
            
//            acc0 <= 0; acc1 <= 0; acc2 <= 0; acc3 <= 0; acc4 <= 0;
//            acc5 <= 0; acc6 <= 0; acc7 <= 0; acc8 <= 0; acc9 <= 0;
            
//            psum_out_0 <= 0; psum_out_1 <= 0; psum_out_2 <= 0; psum_out_3 <= 0; psum_out_4 <= 0;
//            psum_out_5 <= 0; psum_out_6 <= 0; psum_out_7 <= 0; psum_out_8 <= 0; psum_out_9 <= 0;
            
//            done <= 0; done_cnt <= 0;
//            //active <= 0;
//        end 
//        else begin
//            if ((reg_done == 1'b1) && (done ==1'b0)) begin
//                cnt <= 0;       //for iteration
                
//                acc0 <= 0; acc1 <= 0; acc2 <= 0; acc3 <= 0; acc4 <= 0;
//                acc5 <= 0; acc6 <= 0; acc7 <= 0; acc8 <= 0; acc9 <= 0;
                
//                psum_out_0 <= 0; psum_out_1 <= 0; psum_out_2 <= 0; psum_out_3 <= 0; psum_out_4 <= 0;
//                psum_out_5 <= 0; psum_out_6 <= 0; psum_out_7 <= 0; psum_out_8 <= 0; psum_out_9 <= 0;
                
//                done <= 0; done_cnt <= 0;
//            end
//            else if (valid_in_delay) begin // valid_in
//                cnt <= cnt + 1;
                
//                acc0 <= acc0 + prod0;
//                acc1 <= acc1 + prod1;   
//                acc2 <= acc2 + prod2;
//                acc3 <= acc3 + prod3;
//                acc4 <= acc4 + prod4;
//                acc5 <= acc5 + prod5;
//                acc6 <= acc6 + prod6;
//                acc7 <= acc7 + prod7;
//                acc8 <= acc8 + prod8;
//                acc9 <= acc9 + prod9;
                
//                if (cnt == IN_SIZE - 1) begin
//                    psum_out_0 <= acc0;
//                    psum_out_1 <= acc1;
//                    psum_out_2 <= acc2;
//                    psum_out_3 <= acc3;
//                    psum_out_4 <= acc4;
//                    psum_out_5 <= acc5;
//                    psum_out_6 <= acc6;
//                    psum_out_7 <= acc7;
//                    psum_out_8 <= acc8;
//                    psum_out_9 <= acc9;
                    
//                    done <= 1;
//                    //active <= 0;
//                end
//            end 
//            else begin
//                done <= 0;
//            end
//        end
//    end
       
//     always @(posedge clk or negedge resetn) begin
//        if (!resetn) begin
//            reg_done <= 0;
//        end
//        else begin
//            reg_done <= done;
//        end
//     end
    
//   // reg done_hold;

////    always @(posedge clk or negedge resetn) begin
////        if (!resetn) begin
////            done_cnt  <= 0;
////            done_hold <= 0;
////        end
////        else begin
////            if (done == 1 && done_hold == 0) begin
////                // done?? ??? 1?? ??? ????
////                done_hold <= 1;
////                done_cnt  <= 1;
////            end
////            else if (done_hold == 1 && done_cnt < 10) begin
////                done_cnt <= done_cnt + 1;
////            end
////            else if (done_hold == 1 && done_cnt == 10) begin
////                done_hold <= 0;      // 10??? ???? ?? ??????
////                done_cnt  <= 0;
////            end
////        end
////    end


//    quantize #(
//        .INPUT_DATA_WIDTH(ACC_WIDTH),
//        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
//    ) u_quant0 (
//        .in_data(psum_out_0),
//        .out_data(psum_quantize_0)
//    );
//    quantize #(
//        .INPUT_DATA_WIDTH(ACC_WIDTH),
//        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
//    ) u_quant1 (
//        .in_data(psum_out_1),
//        .out_data(psum_quantize_1)
//    );
//    quantize #(
//        .INPUT_DATA_WIDTH(ACC_WIDTH),
//        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
//    ) u_quant2 (
//        .in_data(psum_out_2),
//        .out_data(psum_quantize_2)
//    );
//    quantize #(
//        .INPUT_DATA_WIDTH(ACC_WIDTH),
//        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
//    ) u_quant3 (
//        .in_data(psum_out_3),
//        .out_data(psum_quantize_3)
//    );
//    quantize #(
//        .INPUT_DATA_WIDTH(ACC_WIDTH),
//        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
//    ) u_quant4 (
//        .in_data(psum_out_4),
//        .out_data(psum_quantize_4)
//    );
//    quantize #(
//        .INPUT_DATA_WIDTH(ACC_WIDTH),
//        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
//    ) u_quant5 (
//        .in_data(psum_out_5),
//        .out_data(psum_quantize_5)
//    );
//    quantize #(
//        .INPUT_DATA_WIDTH(ACC_WIDTH),
//        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
//    ) u_quant6 (
//        .in_data(psum_out_6),
//        .out_data(psum_quantize_6)
//    );
//    quantize #(
//        .INPUT_DATA_WIDTH(ACC_WIDTH),
//        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
//    ) u_quant7 (
//        .in_data(psum_out_7),
//        .out_data(psum_quantize_7)
//    );
//    quantize #(
//        .INPUT_DATA_WIDTH(ACC_WIDTH),
//        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
//    ) u_quant8 (
//        .in_data(psum_out_8),
//        .out_data(psum_quantize_8)
//    );
//    quantize #(
//        .INPUT_DATA_WIDTH(ACC_WIDTH),
//        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
//    ) u_quant9 (
//        .in_data(psum_out_9),
//        .out_data(psum_quantize_9)
//    );
    
//    assign out_0 = psum_quantize_0;
//    assign out_1 = psum_quantize_1;
//    assign out_2 = psum_quantize_2;
//    assign out_3 = psum_quantize_3;
//    assign out_4 = psum_quantize_4;
//    assign out_5 = psum_quantize_5;
//    assign out_6 = psum_quantize_6;
//    assign out_7 = psum_quantize_7;
//    assign out_8 = psum_quantize_8;
//    assign out_9 = psum_quantize_9;
    
    
////    // ReLU
////    relu #(
////        .DATA_WIDTH(8)
////    ) u_relu0 (
////        .in_data(psum_quantize_0),
////        .out_data(out_0)
////    );
////    relu #(
////        .DATA_WIDTH(8)
////    ) u_relu1 (
////        .in_data(psum_quantize_1),
////        .out_data(out_1)
////    );
////    relu #(
////        .DATA_WIDTH(8)
////    ) u_relu2 (
////        .in_data(psum_quantize_2),
////        .out_data(out_2)
////    );
////    relu #(
////        .DATA_WIDTH(8)
////    ) u_relu3 (
////        .in_data(psum_quantize_3),
////        .out_data(out_3)
////    );
////    relu #(
////        .DATA_WIDTH(8)
////    ) u_relu4 (
////        .in_data(psum_quantize_4),
////        .out_data(out_4)
////    );
////    relu #(
////        .DATA_WIDTH(8)
////    ) u_relu5 (
////        .in_data(psum_quantize_5),
////        .out_data(out_5)
////    );
////    relu #(
////        .DATA_WIDTH(8)
////    ) u_relu6 (
////        .in_data(psum_quantize_6),
////        .out_data(out_6)
////    );
////    relu #(
////        .DATA_WIDTH(8)
////    ) u_relu7 (
////        .in_data(psum_quantize_7),
////        .out_data(out_7)
////    );
////    relu #(
////        .DATA_WIDTH(8)
////    ) u_relu8 (
////        .in_data(psum_quantize_8),
////        .out_data(out_8)
////    );
////    relu #(
////        .DATA_WIDTH(8)
////    ) u_relu9 (
////        .in_data(psum_quantize_9),
////        .out_data(out_9)
////    );
    
//endmodule

`timescale 1ns / 1ps
module pe_fc_multi #(
    parameter IN_SIZE = 2304,
    parameter ACC_WIDTH = 24,
    parameter DATA_WIDTH = 8
)(
    input  wire clk,
    input  wire resetn,
    //input  wire start,
    
    input  wire signed [7:0] act_in,
    
    input  wire signed [7:0] w0_in,
    input  wire signed [7:0] w1_in,
    input  wire signed [7:0] w2_in,
    input  wire signed [7:0] w3_in,
    input  wire signed [7:0] w4_in,
    input  wire signed [7:0] w5_in,
    input  wire signed [7:0] w6_in,
    input  wire signed [7:0] w7_in,
    input  wire signed [7:0] w8_in,
    input  wire signed [7:0] w9_in,
    
    input  wire valid_in,
    
    output wire signed [DATA_WIDTH-1 :0] out_0,
    output wire signed [DATA_WIDTH-1 :0] out_1,
    output wire signed [DATA_WIDTH-1 :0] out_2,
    output wire signed [DATA_WIDTH-1 :0] out_3,
    output wire signed [DATA_WIDTH-1 :0] out_4,
    output wire signed [DATA_WIDTH-1 :0] out_5,
    output wire signed [DATA_WIDTH-1 :0] out_6,
    output wire signed [DATA_WIDTH-1 :0] out_7,
    output wire signed [DATA_WIDTH-1 :0] out_8,
    output wire signed [DATA_WIDTH-1 :0] out_9,

    output reg reg_done
);
    reg done;
    reg [11:0] cnt;
    //reg active;
    reg signed [ACC_WIDTH-1 :0] psum_out_0;
    reg signed [ACC_WIDTH-1 :0] psum_out_1;
    reg signed [ACC_WIDTH-1 :0] psum_out_2;
    reg signed [ACC_WIDTH-1 :0] psum_out_3;
    reg signed [ACC_WIDTH-1 :0] psum_out_4;
    reg signed [ACC_WIDTH-1 :0] psum_out_5;
    reg signed [ACC_WIDTH-1 :0] psum_out_6;
    reg signed [ACC_WIDTH-1 :0] psum_out_7;
    reg signed [ACC_WIDTH-1 :0] psum_out_8;
    reg signed [ACC_WIDTH-1 :0] psum_out_9;
    
    wire signed [DATA_WIDTH-1 :0] psum_quantize_0;
    wire signed [DATA_WIDTH-1 :0] psum_quantize_1;
    wire signed [DATA_WIDTH-1 :0] psum_quantize_2;
    wire signed [DATA_WIDTH-1 :0] psum_quantize_3;
    wire signed [DATA_WIDTH-1 :0] psum_quantize_4;
    wire signed [DATA_WIDTH-1 :0] psum_quantize_5;
    wire signed [DATA_WIDTH-1 :0] psum_quantize_6;
    wire signed [DATA_WIDTH-1 :0] psum_quantize_7;
    wire signed [DATA_WIDTH-1 :0] psum_quantize_8;
    wire signed [DATA_WIDTH-1 :0] psum_quantize_9;
    
    reg signed [ACC_WIDTH-1:0] acc0, acc1, acc2, acc3, acc4;
    reg signed [ACC_WIDTH-1:0] acc5, acc6, acc7, acc8, acc9;

//    wire signed [15:0] prod0_wire = act_in * w0_in;
//    wire signed [15:0] prod1_wire = act_in * w1_in;
//    wire signed [15:0] prod2_wire = act_in * w2_in;
//    wire signed [15:0] prod3_wire = act_in * w3_in;
//    wire signed [15:0] prod4_wire = act_in * w4_in;
//    wire signed [15:0] prod5_wire = act_in * w5_in;
//    wire signed [15:0] prod6_wire = act_in * w6_in;
//    wire signed [15:0] prod7_wire = act_in * w7_in;
//    wire signed [15:0] prod8_wire = act_in * w8_in;
//    wire signed [15:0] prod9_wire = act_in * w9_in;

    reg signed [15:0] prod0_reg, prod1_reg, prod2_reg, prod3_reg, prod4_reg, prod5_reg, prod6_reg, prod7_reg, prod8_reg, prod9_reg;
    
    wire signed [15:0] prod0 = prod0_reg;
    wire signed [15:0] prod1 = prod1_reg;
    wire signed [15:0] prod2 = prod2_reg;
    wire signed [15:0] prod3 = prod3_reg;
    wire signed [15:0] prod4 = prod4_reg;
    wire signed [15:0] prod5 = prod5_reg;
    wire signed [15:0] prod6 = prod6_reg;
    wire signed [15:0] prod7 = prod7_reg;
    wire signed [15:0] prod8 = prod8_reg;
    wire signed [15:0] prod9 = prod9_reg;
    
    /***** valid_in delay for synchronizing accumulation with products *****/
    reg valid_in_delay;
    always @(posedge clk or negedge resetn) begin
        if(!resetn) begin
            valid_in_delay <= 0;
        end
        else begin
            valid_in_delay <= valid_in; 
        end
    end
    
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin                          //iteration -> resetn
            prod0_reg <= 0;
            prod1_reg <= 0;
            prod2_reg <= 0;
            prod3_reg <= 0;
            prod4_reg <= 0;
            prod5_reg <= 0;
            prod6_reg <= 0;
            prod7_reg <= 0;
            prod8_reg <= 0;
            prod9_reg <= 0;
        end 
        else begin
            prod0_reg <= act_in * w0_in;
            prod1_reg <= act_in * w1_in;
            prod2_reg <= act_in * w2_in;
            prod3_reg <= act_in * w3_in;
            prod4_reg <= act_in * w4_in;
            prod5_reg <= act_in * w5_in;
            prod6_reg <= act_in * w6_in;
            prod7_reg <= act_in * w7_in;
            prod8_reg <= act_in * w8_in;
            prod9_reg <= act_in * w9_in;
        end
    end
    
    
    reg [3:0] done_cnt;
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin                          //iteration -> resetn
            cnt <= 0;
            
            acc0 <= 0; acc1 <= 0; acc2 <= 0; acc3 <= 0; acc4 <= 0;
            acc5 <= 0; acc6 <= 0; acc7 <= 0; acc8 <= 0; acc9 <= 0;
            
            psum_out_0 <= 0; psum_out_1 <= 0; psum_out_2 <= 0; psum_out_3 <= 0; psum_out_4 <= 0;
            psum_out_5 <= 0; psum_out_6 <= 0; psum_out_7 <= 0; psum_out_8 <= 0; psum_out_9 <= 0;
            
            done <= 0; done_cnt <= 0;
            //active <= 0;
        end 
        else begin
            if ((reg_done == 1'b1) && (done ==1'b0)) begin
                cnt <= 0;       //for iteration
                
                acc0 <= 0; acc1 <= 0; acc2 <= 0; acc3 <= 0; acc4 <= 0;
                acc5 <= 0; acc6 <= 0; acc7 <= 0; acc8 <= 0; acc9 <= 0;
                
                psum_out_0 <= 0; psum_out_1 <= 0; psum_out_2 <= 0; psum_out_3 <= 0; psum_out_4 <= 0;
                psum_out_5 <= 0; psum_out_6 <= 0; psum_out_7 <= 0; psum_out_8 <= 0; psum_out_9 <= 0;
                
                done <= 0; done_cnt <= 0;
            end
            else if (valid_in_delay) begin // valid_in
                cnt <= cnt + 1;
                
                acc0 <= acc0 + prod0;
                acc1 <= acc1 + prod1;   
                acc2 <= acc2 + prod2;
                acc3 <= acc3 + prod3;
                acc4 <= acc4 + prod4;
                acc5 <= acc5 + prod5;
                acc6 <= acc6 + prod6;
                acc7 <= acc7 + prod7;
                acc8 <= acc8 + prod8;
                acc9 <= acc9 + prod9;
            end 
            else if (cnt == IN_SIZE) begin
                    psum_out_0 <= acc0;
                    psum_out_1 <= acc1;
                    psum_out_2 <= acc2;
                    psum_out_3 <= acc3;
                    psum_out_4 <= acc4;
                    psum_out_5 <= acc5;
                    psum_out_6 <= acc6;
                    psum_out_7 <= acc7;
                    psum_out_8 <= acc8;
                    psum_out_9 <= acc9;
                    cnt <= 0;
                    done <= 1;
                    //active <= 0;
            end
            else begin
                done <= 0;
            end
        end
    end
       
     always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            reg_done <= 0;
        end
        else begin
            reg_done <= done;
        end
     end
    
   // reg done_hold;

//    always @(posedge clk or negedge resetn) begin
//        if (!resetn) begin
//            done_cnt  <= 0;
//            done_hold <= 0;
//        end
//        else begin
//            if (done == 1 && done_hold == 0) begin
//                // done?? ??? 1?? ??? ????
//                done_hold <= 1;
//                done_cnt  <= 1;
//            end
//            else if (done_hold == 1 && done_cnt < 10) begin
//                done_cnt <= done_cnt + 1;
//            end
//            else if (done_hold == 1 && done_cnt == 10) begin
//                done_hold <= 0;      // 10??? ???? ?? ??????
//                done_cnt  <= 0;
//            end
//        end
//    end


    quantize #(
        .INPUT_DATA_WIDTH(ACC_WIDTH),
        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
    ) u_quant0 (
        .in_data(psum_out_0),
        .out_data(psum_quantize_0)
    );
    quantize #(
        .INPUT_DATA_WIDTH(ACC_WIDTH),
        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
    ) u_quant1 (
        .in_data(psum_out_1),
        .out_data(psum_quantize_1)
    );
    quantize #(
        .INPUT_DATA_WIDTH(ACC_WIDTH),
        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
    ) u_quant2 (
        .in_data(psum_out_2),
        .out_data(psum_quantize_2)
    );
    quantize #(
        .INPUT_DATA_WIDTH(ACC_WIDTH),
        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
    ) u_quant3 (
        .in_data(psum_out_3),
        .out_data(psum_quantize_3)
    );
    quantize #(
        .INPUT_DATA_WIDTH(ACC_WIDTH),
        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
    ) u_quant4 (
        .in_data(psum_out_4),
        .out_data(psum_quantize_4)
    );
    quantize #(
        .INPUT_DATA_WIDTH(ACC_WIDTH),
        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
    ) u_quant5 (
        .in_data(psum_out_5),
        .out_data(psum_quantize_5)
    );
    quantize #(
        .INPUT_DATA_WIDTH(ACC_WIDTH),
        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
    ) u_quant6 (
        .in_data(psum_out_6),
        .out_data(psum_quantize_6)
    );
    quantize #(
        .INPUT_DATA_WIDTH(ACC_WIDTH),
        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
    ) u_quant7 (
        .in_data(psum_out_7),
        .out_data(psum_quantize_7)
    );
    quantize #(
        .INPUT_DATA_WIDTH(ACC_WIDTH),
        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
    ) u_quant8 (
        .in_data(psum_out_8),
        .out_data(psum_quantize_8)
    );
    quantize #(
        .INPUT_DATA_WIDTH(ACC_WIDTH),
        .OUTPUT_DATA_WIDTH(DATA_WIDTH)
    ) u_quant9 (
        .in_data(psum_out_9),
        .out_data(psum_quantize_9)
    );
    
    assign out_0 = psum_quantize_0;
    assign out_1 = psum_quantize_1;
    assign out_2 = psum_quantize_2;
    assign out_3 = psum_quantize_3;
    assign out_4 = psum_quantize_4;
    assign out_5 = psum_quantize_5;
    assign out_6 = psum_quantize_6;
    assign out_7 = psum_quantize_7;
    assign out_8 = psum_quantize_8;
    assign out_9 = psum_quantize_9;
    
    
//    // ReLU
//    relu #(
//        .DATA_WIDTH(8)
//    ) u_relu0 (
//        .in_data(psum_quantize_0),
//        .out_data(out_0)
//    );
//    relu #(
//        .DATA_WIDTH(8)
//    ) u_relu1 (
//        .in_data(psum_quantize_1),
//        .out_data(out_1)
//    );
//    relu #(
//        .DATA_WIDTH(8)
//    ) u_relu2 (
//        .in_data(psum_quantize_2),
//        .out_data(out_2)
//    );
//    relu #(
//        .DATA_WIDTH(8)
//    ) u_relu3 (
//        .in_data(psum_quantize_3),
//        .out_data(out_3)
//    );
//    relu #(
//        .DATA_WIDTH(8)
//    ) u_relu4 (
//        .in_data(psum_quantize_4),
//        .out_data(out_4)
//    );
//    relu #(
//        .DATA_WIDTH(8)
//    ) u_relu5 (
//        .in_data(psum_quantize_5),
//        .out_data(out_5)
//    );
//    relu #(
//        .DATA_WIDTH(8)
//    ) u_relu6 (
//        .in_data(psum_quantize_6),
//        .out_data(out_6)
//    );
//    relu #(
//        .DATA_WIDTH(8)
//    ) u_relu7 (
//        .in_data(psum_quantize_7),
//        .out_data(out_7)
//    );
//    relu #(
//        .DATA_WIDTH(8)
//    ) u_relu8 (
//        .in_data(psum_quantize_8),
//        .out_data(out_8)
//    );
//    relu #(
//        .DATA_WIDTH(8)
//    ) u_relu9 (
//        .in_data(psum_quantize_9),
//        .out_data(out_9)
//    );
    
endmodule
