module CHAR_TOP (
  // Example Port
  input SYSTEM_CLOCK,    // 100MHz LVTTL SYSTEM CLOCK
  input nSYSPOR,
  input [2:0] SW,        // switches
  output reg [7:0] LED,  // User LEDs
  output VGA_HSYNCH,     // horizontal sync for the VGA output connector
  output VGA_VSYNCH,     // vertical sync for the VGA output connector
  output [7:3] LCD_R,    // RED DAC data
  output [7:2] LCD_G,    // GREEN DAC data
  output [7:3] LCD_B,    // BLUE DAC data
  output VGA_CLK,
  output VGA_EN,         // VGA Enable
  output lcd_blk_on
  // Processor System Port
  // inout [14:0]DDR_addr,
  // inout [2:0]DDR_ba,
  // inout DDR_cas_n,
  // inout DDR_ck_n,
  // inout DDR_ck_p,
  // inout DDR_cke,
  // inout DDR_cs_n,
  // inout [3:0]DDR_dm,
  // inout [31:0]DDR_dq,
  // inout [3:0]DDR_dqs_n,
  // inout [3:0]DDR_dqs_p,
  // inout DDR_odt,
  // inout DDR_ras_n,
  // inout DDR_reset_n,
  // inout DDR_we_n,
  // inout FIXED_IO_ddr_vrn,
  // inout FIXED_IO_ddr_vrp,
  // inout [53:0]FIXED_IO_mio,
  // inout FIXED_IO_ps_clk,
  // inout FIXED_IO_ps_porb,
  // inout FIXED_IO_ps_srstb
);

//wire VGA_HSYNCH;     // horizontal sync for the VGA output connector
//wire VGA_VSYNCH;     // vertical sync for the VGA output connector
wire system_clock_buffered;  // buffered SYSTEM CLOCK
reg pixel_clock;            // generated from SYSTEM CLOCK
wire reset;                  // reset asserted when DCMs are NOT LOCKED

wire vga_red_data;           // red video data
wire vga_green_data;         // green video data
wire vga_blue_data;          // blue video data

// internal video timing signals
wire h_synch; // horizontal synch for VGA connector
wire v_synch; // vertical synch for VGA connector
wire blank;   // composite blanking
wire [10:0] pixel_count;    // bit mapped pixel position within the line
wire [9:0] line_count;      // bit mapped line number in a frame lines within the frame
wire [2:0] subchar_pixel;   // pixel position within the character
wire [2:0] subchar_line;    // identifies the line number within a character block
wire [6:0] char_column;     // character number on the current line
wire [6:0] char_line;       // line number on the screen

wire TEMP_HSYNCH;
wire TEMP_VSYNCH;
wire temp_reset;

wire VGA_OUT_RED;
wire VGA_OUT_GREEN;
wire VGA_OUT_BLUE;
/*
system_wrapper system_wrapper_i (
  .DDR_addr(DDR_addr),
  .DDR_ba(DDR_ba),
  .DDR_cas_n(DDR_cas_n),
  .DDR_ck_n(DDR_ck_n),
  .DDR_ck_p(DDR_ck_p),
  .DDR_cke(DDR_cke),
  .DDR_cs_n(DDR_cs_n),
  .DDR_dm(DDR_dm),
  .DDR_dq(DDR_dq),
  .DDR_dqs_n(DDR_dqs_n),
  .DDR_dqs_p(DDR_dqs_p),
  .DDR_odt(DDR_odt),
  .DDR_ras_n(DDR_ras_n),
   .DDR_reset_n(DDR_reset_n),
   .DDR_we_n(DDR_we_n),
   .FIXED_IO_ddr_vrn(FIXED_IO_ddr_vrn),
   .FIXED_IO_ddr_vrp(FIXED_IO_ddr_vrp),
   .FIXED_IO_mio(FIXED_IO_mio),
   .FIXED_IO_ps_clk(FIXED_IO_ps_clk),
   .FIXED_IO_ps_porb(FIXED_IO_ps_porb),
   .FIXED_IO_ps_srstb(FIXED_IO_ps_srstb)
 );
 */
 assign reset = ~nSYSPOR;
 
 assign LCD_R[7:3] = (VGA_OUT_RED) ? 5'b11111 : 0;
 assign LCD_G[7:2] = (VGA_OUT_GREEN) ? 6'b111111 : 0;
 assign LCD_B[7:3] = (VGA_OUT_BLUE) ? 5'b11111 : 0;
 
 always @ (posedge pixel_clock or negedge nSYSPOR) begin
   if (!nSYSPOR)
   begin
     LED <= 8'd0;
   end
   else
   begin
     LED[0] <= SW[0]; // Turn LED 0 on when switch 0 is on
     LED[1] <= SW[1]; // Turn LED 1 on when switch 1 is on
     LED[2] <= SW[2]; // Turn LED 2 on when switch 2 is on
     LED[3] <= SW[0]; // Turn LED 3 on when switch 3 is on
     LED[4] <= SW[1]; // Turn LED 4 on when switch 4 is on
     LED[5] <= SW[2]; // Turn LED 5 on when switch 5 is on
     LED[6] <= SW[0]; // Turn LED 6 on when switch 6 is on
     LED[7] <= SW[1]; // Turn LED 7 on when switch 7 is on
   end
 end
 
 // instantiate the character generator
 CHAR_DISPLAY CHAR_DISPLAY (
   .char_column(char_column),     // character number on the current line
   .char_line(char_line),         // line number on the screen
   .subchar_line(subchar_line),   // the line number within a character block 0-8
   .subchar_pixel(subchar_pixel), // the pixel number within a character block 0-8
   .pixel_clock(pixel_clock),
   .reset(reset),
   .vga_red_data(vga_red_data),
     .vga_green_data(vga_green_data),
     .vga_blue_data(vga_blue_data)
   );
   
   // instantiate the video timing generator
   SVGA_TIMING_GENERATION SVGA_TIMING_GENERATION(
     .pixel_clock(pixel_clock),     // pixel clock
     .reset(reset),                 // reset
     .h_synch(h_synch),             // horizontal synch for VGA connector
     .v_synch(v_synch),             // vertical synch for VGA connector
     .blank(blank),                 // composite blanking
     .pixel_count(pixel_count),     // counts the pixels in a line
     .line_count(line_count),       // counts the display lines
     .subchar_pixel(subchar_pixel), // pixel position within the character
     .subchar_line(subchar_line),   // identifies the line number within a character block
     .char_column(char_column),     // character number on the current line
     .char_line(char_line)          // line number on the screen
   );
   
   // instantiate the video output mux
   VIDEO_OUT VIDEO_OUT (
     .pixel_clock(pixel_clock),
     .reset(reset),
     .vga_red_data(vga_red_data),
     .vga_green_data(vga_green_data),
     .vga_blue_data(vga_blue_data),
     .h_synch(h_synch),
     .v_synch(v_synch),
     .blank(blank),
     .VGA_HSYNCH(TEMP_HSYNCH),
     .VGA_VSYNCH(TEMP_VSYNCH),
     .VGA_OUT_RED(VGA_OUT_RED),
     .VGA_OUT_GREEN(VGA_OUT_GREEN),
     .VGA_OUT_BLUE(VGA_OUT_BLUE)
   );
   
   always @ (posedge SYSTEM_CLOCK or negedge nSYSPOR)
   begin
     if (!nSYSPOR)
       pixel_clock = 1'b0;
     else
      pixel_clock = ~pixel_clock;
    end
    
    assign VGA_HSYNCH = ~TEMP_HSYNCH;
    assign VGA_VSYNCH = ~TEMP_VSYNCH;
    assign VGA_EN = 1'b1;
    assign VGA_CLK = ~pixel_clock;
    assign lcd_blk_on = 1'b1;
    
    endmodule // MAIN