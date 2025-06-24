`timescale 1ns / 1ps

module quantize #(
    parameter INPUT_DATA_WIDTH = 21,
    parameter OUTPUT_DATA_WIDTH = 8
)(
    input  signed [INPUT_DATA_WIDTH-1:0] in_data,
    output signed [OUTPUT_DATA_WIDTH-1:0] out_data
);
    // Truncate 10 LSBs (maintains sign)
    wire signed [INPUT_DATA_WIDTH - 11:0] msb_data;
    assign msb_data = in_data >>> 10;

    // Clamp boundaries
    localparam signed [OUTPUT_DATA_WIDTH-1:0] MAX_VAL =  8'sd127;
    localparam signed [OUTPUT_DATA_WIDTH-1:0] MIN_VAL = -8'sd128;

    // Clamp with assign
    assign out_data = (msb_data > MAX_VAL) ? MAX_VAL :
                      (msb_data < MIN_VAL) ? MIN_VAL :
                      msb_data[OUTPUT_DATA_WIDTH-1:0];

endmodule
