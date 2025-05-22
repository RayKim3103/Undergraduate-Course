module pb_intr (
    input PCLK,
    input PRESETn,
// APB (with processor system)
    input [31:0] PADDR,
    input PENABLE,
    input PSEL,
    input PWRITE,
    input [31:0] PWDATA,
    output reg [31:0] PRDATA,
    // GPIO (with baseboard)
    input [3:0] PB,
    output [7:0] LED,
        // Interrupt (with processor system)
    output INTR);
    reg [3:0] curr_intr;
    reg [3:0] next_intr;
    reg [7:0] curr_led;
    reg [7:0] next_led;
    reg [3:0] pb_1d;
    reg [3:0] pb_2d;
    integer i;
    assign INTR = curr_intr[3] & curr_intr[2] & curr_intr[1] &
    curr_intr[0];
    assign LED = curr_led;
    always @ (curr_intr or curr_led or pb_1d or pb_2d or PSEL or PWRITE or PENABLE, PADDR, PWDATA)
    begin
        for (i = 0 ; i < 4 ; i = i + 1)
        begin
            if (PSEL && PWRITE && PENABLE && (PADDR[5:2] == 4'b0000) && PWDATA[i])
                next_intr[i] <= 1'b1;
            else if (pb_2d[i] && (!pb_1d[i]))
                next_intr[i] <= 1'b0;
            else
                next_intr[i] <= curr_intr[i];
        end
        if (PSEL && PWRITE && PENABLE && (PADDR[5:2] == 4'b0001))
         next_led <= PWDATA[7:0];
        if (PSEL && (!PWRITE) && PENABLE)
            begin
                case (PADDR[5:2])
                4'b0000: PRDATA <= {28'd0, curr_intr};
                4'b0001: PRDATA <= {24'd0, curr_led};
                default: PRDATA <= 32'd0;
                endcase
             end
        else
            PRDATA <= 32'd0;
    end
    always @ (posedge PCLK or negedge PRESETn)
    begin
        if (!PRESETn)
        begin
            curr_intr <= 4'b1111;
            curr_led <= 8'd0;
            pb_1d <= 4'b1111;
            pb_2d <= 4'b1111;
        end
        else
        begin
            curr_intr <= next_intr;
            curr_led <= next_led;
            pb_1d <= PB;
            pb_2d <= pb_1d;
        end
    end
endmodule