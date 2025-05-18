`timescale 1ns/1ps
module sobel_pixel (
    input  signed [8:0] p00, p01, p02,
    input  signed [8:0] p10, p11, p12,
    input  signed [8:0] p20, p21, p22,
    output       [7:0]  edge_out      // 8-bit saturate
);
    // compute Gx, Gy
    // Gx = | -1 0 1 |    Gy = | -1 -2  1 |  
    //      | -2 0 2 |         |  0  0  0 |
    //      | -1 0 1 |         |  1  2  1 |
    // 10bit + 11bit + 10bit => 12bit required
    wire signed [11:0] gx =
          (p02 - p00)
        + ((p12 - p10) <<< 1)
        + (p22 - p20);

    wire signed [11:0] gy =
          (p00 - p20)
        + ((p01 - p21) <<< 1)
        + (p02 - p22);

    // |Sx| + |Sy|
    wire [12:0] abs_sum = (gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy);

    // 8-bit precision
    assign edge_out = (abs_sum > 14'd255) ? 8'd255 : abs_sum[7:0];
endmodule
