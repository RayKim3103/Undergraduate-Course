`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/27 15:02:20
// Design Name: 
// Module Name: memory_ctrlr
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


module memory_ctrlr #(
    parameter SRAM1_BW = 16,
    parameter SRAM1_AMAX = 256,
    parameter SRAM1_ADR = $clog2(SRAM1_AMAX),
    parameter SRAM2_BW = 32,
    parameter SRAM2_AMAX = 192,
    parameter SRAM2_ADR = $clog2(SRAM2_AMAX)
    )(
    input  wire clk,
    input  wire resetn,
    input  wire start,
    output wire done,

    output wire s1_en,
    output wire s2_en,
    output wire s1_we,
    output wire s2_we,
    output wire [SRAM1_ADR-1:0] s1_addr,
    output wire [SRAM2_ADR-1:0] s2_addr,
    input  wire [SRAM1_BW-1:0] s1_dout,
    output wire [SRAM2_BW-1:0] s2_din
    );


    localparam IDLE = 2'b00;
    localparam RUN1 = 2'b01;
    localparam RUN2 = 2'b10;
    localparam DONE = 2'b11;



    reg [1:0] state, n_state;

    reg [6:0] cnt_run1, cnt_run2;
    wire done_run1, done_run2;

    reg [SRAM1_BW-1:0] s1_dout_buf;
    reg [SRAM2_BW-1:0] s2_din_reg;
    reg s2_enable;

    assign done_run1 = (cnt_run1 == 7'd127);
    assign done_run2 = (cnt_run2 == 7'd127);

    assign s1_en = (state == RUN1) || (state == RUN2);
    assign s1_we = 1'b0;
    assign s1_addr = (state == RUN1)? cnt_run1: 
                     (state == RUN2)? cnt_run2 + 128:
                     {SRAM1_ADR{1'b0}};

    assign s2_en = (state == RUN1)? s2_enable: (state == RUN2)? 1'b1 :1'b0;
    assign s2_we = (state == RUN1)? s2_enable: (state == RUN2)? 1'b1 :1'b0;
    assign s2_addr = (state == RUN1)? 191 - (cnt_run1>>1):
                     (state == RUN2)? cnt_run2:
                     {SRAM2_ADR{1'b0}};

    assign s2_din = s2_din_reg;

    assign done = (state == DONE);

    always @(*) begin
        if (state == RUN1)
            s2_din_reg = {s1_dout_buf, s1_dout};
        else if (state == RUN2) begin
            if(~cnt_run2[0])
                s2_din_reg = {16'd0 ,s1_dout};
            else
                s2_din_reg = {s1_dout, 16'd0};
        end
        else s2_din_reg = {SRAM2_BW{1'b0}};
    end


    always @(*) begin
        case (state)
            IDLE : begin 
                if(start) n_state = RUN1;
                else n_state = IDLE;
            end            
            RUN1 : begin
                if(done_run1) n_state = RUN2;
                else n_state = RUN1;
            end
            RUN2 : begin
                if(done_run2) n_state = DONE;
                else n_state = RUN2;
            end
            DONE : begin
                if(~resetn) n_state = IDLE;
                else n_state = DONE;
            end
            default :  n_state = IDLE;
        endcase
    end


    always @(posedge clk or negedge resetn) begin
        if(~resetn) s2_enable <= 1'b0;
        else begin
            if (state ==RUN1) begin
                if(cnt_run1[0])
                    s2_enable <= 1'b0;
                else
                    s2_enable <= s1_en;
            end
            else begin
                s2_enable <= 1'b0;
            end
        end
    end


    always @(posedge clk or negedge resetn) begin
        if(~resetn) s1_dout_buf <= 16'd0;
        else begin
            if(state == RUN1) 
                s1_dout_buf <= s1_dout;
            else
                s1_dout_buf <= 16'd0;
        end
    end

    always @(posedge clk or negedge resetn) begin
        if(~resetn) cnt_run1 <= 7'd0;
        else begin
            if(state == RUN1) 
                cnt_run1 <= cnt_run1 + 7'd1;
            else
                cnt_run1 <= 7'd0;
        end
    end

    always @(posedge clk or negedge resetn) begin
        if(~resetn) cnt_run2 <= 7'd0;
        else begin
            if(state == RUN2) 
                cnt_run2 <= cnt_run2 + 7'd1;
            else
                cnt_run2 <= 7'd0;
        end
    end

    always @(posedge clk or negedge resetn) begin
        if(~resetn) state <= IDLE;
        else state <= n_state;
    end



endmodule
