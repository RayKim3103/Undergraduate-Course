`timescale 1ns / 1ps



module sprite_compositor #(
    parameter H_RES = 800,  // Horizontal resolution
    parameter V_RES = 600   // Vertical resolution
    )(
    input wire [15:0] i_x,
    input wire [15:0] i_y,
    input wire i_v_sync,
    output wire [7:0] o_red,
    output wire [7:0] o_green,
    output wire [7:0] o_blue,
    output wire o_sprite_hit,
    
    // btn�� �Է��� ����
    input wire move_btn   // btn�� �Է��� ���� �� �ְ� �Ͽ�, �ʱ� ��ġ�� ����
    );
    
    // sprite�� ����� ���� ����
    reg [15:0] sprite_x     = 16'd00;
    reg [15:0] sprite_y     = 16'd00; 
    reg sprite_x_direction  = 1;
    reg sprite_y_direction  = 1;
    reg sprite_flip         = 0;
    wire sprite_hit_x, sprite_hit_y;

    
    // Calculate the circle equation
    localparam RADIUS = 80; 
    wire circle_inside;
    
    wire signed [15:0] dx_signed, dy_signed; // ��ǥ ���� ���̴� ������ �� �� �����Ƿ� signed ���
    wire [15:0] dx, dy;                       // unsigned�� �ٲ���
    
    assign dx_signed = (RADIUS + sprite_x) - i_x; // sprite_x
    assign dy_signed = (RADIUS + sprite_y) - i_y; // sprite_y
    assign dx = (dx_signed[15]) ? -dx_signed : dx_signed;
    assign dy = (dy_signed[15]) ? -dy_signed : dy_signed;
       
    wire [31:0] dx_squared = dx * dx; // 16bit�� ���� �ִ� 32bit
    wire [31:0] dy_squared = dy * dy; // 16bit�� ���� �ִ� 32bit
    wire [31:0] radius_squared = RADIUS * RADIUS;
    
    assign circle_inside = (dx_squared + dy_squared) <= radius_squared;

    // sprite ���� ����: ���� ���� 160�ȼ�
    assign sprite_hit_x = (i_x >= sprite_x) && (i_x < sprite_x + 160);
    assign sprite_hit_y = (i_y >= sprite_y) && (i_y < sprite_y + 160);

    // i_v_sync�� positive edgea���� ���� (i_v_syncv�� display timing���� v_sync�� �ǹ� -> ���ο� frame ������ �� ����)
     always @(posedge i_v_sync ) 
     begin
        // iteration �ٲ� ���� ���� btn logic�� �ᵵ ������ �ʿ� X
        if(move_btn == 1)
            begin
                sprite_x <= 0;
                sprite_y <= 0;
                sprite_flip <= 0;
            end
        else
            begin
            // sprite_y == V_RES-160�� ���: sprite image�� y�� �Ʒ��� �� ����
            // �̶�, sprite_y <= sprite_y + 1;�� ������ ������ ���� posedge i_v_sync�� �ٽ� sprite_y == V_RES-160�̱⿡
            // �� �ڵ带 ����� sprite_flip�� ����� �۵��Ѵ�
            if (sprite_y >= V_RES-160)
                 begin
                    sprite_y_direction <= 0;
                    sprite_y <= sprite_y - 1;
                    sprite_flip <= ~sprite_flip;
                 end            
            // sprite_y <= 1�� ���: sprite image�� y�� ���� �� ����
             else if (sprite_y <= 1)
                 begin
                    sprite_y_direction <= 1;
                    sprite_y <= sprite_y + 1;
                    sprite_flip <= ~sprite_flip;
                 end
             else
                begin
                    sprite_x <= sprite_x + (sprite_x_direction ? 1 : -1);
                    sprite_y <= sprite_y + (sprite_y_direction ? 1 : -1);
                end
            // sprite_x == H_RES-160�� ���: sprite image�� x�� ������ �� ����
             if (sprite_x >= H_RES-160) 
                 begin
                    sprite_x_direction <= 0;
                    sprite_x <= sprite_x - 1;
                    sprite_flip <= ~sprite_flip;
                 end
            // sprite_x <= 1�� ���: sprite image�� x�� ���� �� ����
             else if (sprite_x <= 1) 
                 begin
                    sprite_x_direction <= 1;
                    sprite_x <= sprite_x + 1;
                    sprite_flip <= ~sprite_flip;
                 end
             else
                begin
                    sprite_x <= sprite_x + (sprite_x_direction ? 1 : -1);
                    sprite_y <= sprite_y + (sprite_y_direction ? 1 : -1);
                end
            end
        
     end 
     
// RED �� BLACK ���� ���� logic, 12���� �ڵ�� ����
    wire [7:0] temp_red;
    wire [7:0] temp_green;
    wire [7:0] temp_blue;
    
    assign temp_red = (sprite_flip == 0) ? 8'd0 : 8'd255;
    assign temp_green = 8'd0;
    assign temp_blue = 8'd0;
    
    
    assign o_red    = (sprite_hit_x && sprite_hit_y) ? temp_red : 8'hXX;
    assign o_green  = (sprite_hit_x && sprite_hit_y) ? temp_green : 8'hXX;
    assign o_blue   = (sprite_hit_x && sprite_hit_y) ? temp_blue : 8'hXX;
    assign o_sprite_hit = (sprite_hit_y & sprite_hit_x) && (circle_inside != 1'd0);
    
endmodule