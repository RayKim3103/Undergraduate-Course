`default_nettype none
`timescale 1ns / 1ps

module ctrl_barnsley #(
    FB_WIDTH=800,   // framebuffer width in pixels
    FB_HEIGHT=600,  // framebuffer height in pixels
    FP_WIDTH=25,    // total width of fixed-point number: integer + fractional bits
    FP_INT=5       // integer bits in fixed-point number
    ) (
    input  wire clk,                            // clock
    input wire reset,                           // reset
    input wire [15:0] sx,                       // x position
    input wire [15:0] sy,                       // y position
    input wire [2:0] btn,                       // pushbutton 
    output wire [7:0] red,                      // R
    output wire [7:0] green,                    // G
    output wire [7:0] blue                      // B
    );

    // instance barnsley module; add the input/output signal to empty port
    wire complete;
    wire done;
    wire [19:0] iter_max;
    wire [FP_WIDTH-1:0] xn;
    wire [FP_WIDTH-1:0] yn;
    
    barnsley #(
        .FP_WIDTH(FP_WIDTH),
        .FP_INT(FP_INT)
    ) barnsley_inst_00 (
        .clk(clk),
        .reset(reset),
        .iter_max(iter_max),
        .iter_change(btn[0]),
        .done(done),
        .complete(complete),
        .xn(xn),
        .yn(yn)
    );

    // instance BRAM;add the input/output signal to empty port if you need
        
    //wire [18:0] addr; // BRAM�� �ּ�, �ִ� 480000, 19bit
    wire [4:0] out; // BRAM���� ���� ������ ���, 5bit
    
        // ENABLE �ؾ��ϴ� ��Ʈ ENABLE��
        blk_mem_gen_0 inst_0(
        // port a is used to write
        .clka(clk),
        .ena(1'b1),
        .wea(1'b1),
        .addra(hcnt + vcnt),            // A ��Ʈ �ּ� 
        .dina(5'b11111),
        .douta(),
        // port b is used to read
        .clkb(clk),
        .enb(1'b1),
        .web(1'b0),
        .addrb(addr),
        .dinb(5'd0),
        .doutb(out)
        );
  
  
  
   /* assignment 1
    caculate the address using xn,yn of barnsley module
   */
    


   /* assignment 2
    change the iteration max value when the puhbutton is on
   */
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // X��� Y���� �ּ� ���� = 2^-6, 5bit integer, 6bit fraction
    wire signed [10:0] x_pos = ({xn[FP_WIDTH-1:20], xn[19:14]}); 
    wire signed [10:0] y_pos = ({yn[FP_WIDTH-1:20], yn[19:14]}); 
    
    
    reg [18:0] hcnt = 0; // ���� cnt
    reg [18:0] vcnt = 0; // ���� cnt
    
    // x_pos, y_pos ����� ������ ������ �� 
    always @(posedge done) 
        begin
            if (done == 1)
            begin
                    hcnt <= x_pos + 400; 
                    vcnt <= (y_pos) * 800; //*19'd800       
            end
        end
    
    // create the address control signal using input sx, sy
        wire [18:0] px, py;
        wire [18:0] addr;
        
        assign px = {3'b000,sx}; // x ��ǥ Ȯ��    
        assign py = {3'b000,sy};
        assign addr = 480000 - (-px + py*19'd800); 
       
        reg [7:0] temp_red; // ������ ��
        reg [7:0] temp_green; // �ʷϻ� ��
        reg [7:0] temp_blue; // �Ķ��� ��
        
        // 12���� �ڵ�� ����, BRAM �����Ϳ� ���� ��� ����
        always @(*) 
        begin 
            case(out) // ��µ� �����Ϳ� ���� ���� ����
                5'b00000 : begin temp_red <= 8'd255; temp_green <= 8'd255; temp_blue <= 8'd255; end
                5'b11111 : begin temp_red <= 8'd0; temp_green <= 8'd255; temp_blue <= 8'd0; end  

               default : begin temp_red <= 8'd255; temp_green <= 8'd255; temp_blue <= 8'd255; end
            endcase
       end
        

    // change the R,G,B value using BRAM output
       assign red[7:0] = temp_red;
       assign green[7:0] = temp_green;
       assign blue[7:0] = temp_blue;
       
       

endmodule