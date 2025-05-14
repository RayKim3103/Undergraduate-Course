
module debouncer(
    input wire clk,
    input wire in,
    input wire resetn,
    output wire out_pos
//    output wire out_neg
    );

    wire Q, Q2;
    dff Udff0(clk, resetn, in, Q);
    dff Udff1(clk, resetn, Q ,Q2 );

    assign out_pos = ~Q2 & Q;
endmodule


module dff(
    input wire clk, 
    input wire resetn,
    input wire D, 
    output reg Q
    );

    always @ (posedge clk or negedge resetn) begin
        if(!resetn)
            Q <= 0;
        else
            Q <= D;
    end
endmodule
