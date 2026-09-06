`timescale 1ns / 1ps


module gfx(
    input wire clk,
    input wire reset,
    input wire [15:0] i_x,
    input wire [15:0] i_y,
    input wire i_v_sync,
    input wire [2:0] btn,
    output reg [7:0] o_red,
    output reg [7:0] o_green,
    output reg [7:0] o_blue

    );
    wire bg_hit, sprite_hit;
    wire [7:0] bg_red;
    wire [7:0] bg_green;
    wire [7:0] bg_blue;
    wire [7:0] sprite_red;
    wire [7:0] sprite_green;
    wire [7:0] sprite_blue;
    
    wire [24:0] x_start, y_start, step, change;
   
   ctrl_barnsley #(
    .FB_WIDTH(800),   // framebuffer width in pixels
    .FB_HEIGHT(600),  // framebuffer height in pixels
    .FP_WIDTH(25),    // total width of fixed-point number: integer + fractional bits
    .FP_INT(5)       // integer bits in fixed-point number
    ) 
    barnsley_inst(
        .clk(clk),                            // clock
        .reset(reset),
        .sx(i_x),
        .sy(i_y),
        .btn(btn),
        .red(bg_red),
        .green(bg_green),
        .blue(bg_blue)
    ); 
    
    // sprite_compositor 인스턴스 추가
    sprite_compositor #(
        .H_RES(800),    // horizontal resolution
        .V_RES(600)
        ) sprite_compositor_1 (
        .i_x        (i_x),
        .i_y        (i_y),
        .i_v_sync   (i_v_sync),
        .o_red      (sprite_red),
        .o_green    (sprite_green),
        .o_blue     (sprite_blue),
        .o_sprite_hit   (sprite_hit),
        .move_btn (btn[1])
    );
    
    // 색 결정 알고리즘
    always@(*) begin

        if(sprite_hit==1) 
            begin
                if (bg_red == 8'd0)
                    begin
                        o_red=8'd0;
                        o_green=8'd0;
                        o_blue=8'd255;
                    end
                else
                    begin
                        o_red=sprite_red;
                        o_green=sprite_green;
                        o_blue=sprite_blue;
                    end
                end
            else 
                begin
                    o_red=bg_red;
                    o_green=bg_green;
                    o_blue=bg_blue;
                end

    end
  
//always@(*) begin
//    o_red=bg_red;
//    o_green=bg_green;
//    o_blue=bg_blue;
//end
    

    
endmodule