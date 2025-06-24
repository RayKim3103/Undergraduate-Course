`timescale 1ns / 1ps

module relu #(
    parameter DATA_WIDTH = 8
)(
    input  signed [DATA_WIDTH-1:0] in_data,
    output signed [DATA_WIDTH-1:0] out_data
);

    assign out_data = (in_data > 0) ? in_data : 0;

endmodule
