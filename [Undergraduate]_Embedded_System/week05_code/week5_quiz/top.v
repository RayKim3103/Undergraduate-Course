module top (
  // Processor System
  inout [14:0]DDR_addr,
  inout [2:0]DDR_ba,
  inout DDR_cas_n,
  inout DDR_ck_n,
  inout DDR_ck_p,
  inout DDR_cke,
  inout DDR_cs_n,
  inout [3:0]DDR_dm,
  inout [31:0]DDR_dq,
  inout [3:0]DDR_dqs_n,
  inout [3:0]DDR_dqs_p,
  inout DDR_odt,
  inout DDR_ras_n,
  inout DDR_reset_n,
  inout DDR_we_n,
  inout FIXED_IO_ddr_vrn,
  inout FIXED_IO_ddr_vrp,
  inout [53:0]FIXED_IO_mio,
  inout FIXED_IO_ps_clk,
  inout FIXED_IO_ps_porb,
  inout FIXED_IO_ps_srstb,
  input TFTLCD_CLK,
  input TFTLCD_nRESET,
  output TFTLCD_TCLK,	        // TFT-LCD Clock
  output wire TFTLCD_Hsync,	  // TFT-LCD HSYNC
  output wire TFTLCD_Vsync,	  // TFT-LCD VSYNC
  output wire TFTLCD_DE_out,	// TFT-LCD Data enable
  output [7:3] TFTLCD_R,      // TFT-LCD Red signal 
  output [7:2] TFTLCD_G,      // TFT-LCD Green signal
  output [7:3] TFTLCD_B,      // TFT-LCD Blue signal
  output TFTLCD_Tpower,       // TFT-LCD Backlight On signal

  // YOUR CODE GOES IN HERE////////////////////////////////////////
  input [3:0] TFTLCD_SW,
  input [2:0] PushButton,
  // YOUR CODE GOES IN HERE////////////////////////////////////////

  output [1:0] LED
);

  wire BRAMCLK;             //BRAM Clock
  wire [16:0] BRAMADDRA;    //BRAM Address
  wire [15:0] BRAMDATA;     //BRAM Data 16bits

  wire [31:0]M_AHB_1_haddr;   // AHB Master 1 Address
  wire [2:0]M_AHB_1_hburst;   // AHB Master 1 Burst
  wire [3:0]M_AHB_1_hprot;    // AHB Master 1 Protection
  wire [31:0]M_AHB_1_hrdata;  // AHB Master 1 Read Data
  wire M_AHB_1_hready;        // AHB Master 1 Ready
  wire M_AHB_1_hresp;         // AHB Master 1 Response
  wire [2:0]M_AHB_1_hsize;    // AHB Master 1 Size
  wire [1:0]M_AHB_1_htrans;   // AHB Master 1 Transfer
  wire [31:0]M_AHB_1_hwdata;  // AHB Master 1 Write Data
  wire M_AHB_1_hwrite;        // AHB Master 1 Write 
  wire [31:0]M_AHB_haddr;     // AHB Master Address
  wire [2:0]M_AHB_hburst;     // AHB Master Burst
  wire [3:0]M_AHB_hprot;      // AHB Master Protection
  wire [31:0]M_AHB_hrdata;    // AHB Master Read Data
  wire M_AHB_hready;          // AHB Master Ready
  wire M_AHB_hresp;           // AHB Master Response
  wire [2:0]M_AHB_hsize;      // AHB Master Size
  wire [1:0]M_AHB_htrans;     // AHB Master Transfer
  wire [31:0]M_AHB_hwdata;    // AHB Master Write Data
  wire M_AHB_hwrite;          // AHB Master Write 
  wire m_ahb_hclk;            // AHB Clock  
  wire m_ahb_hclk_1;          // AHB Clock 1  
  wire m_ahb_hmastlock;       // AHB Master Lock  
  wire m_ahb_hmastlock_1;     // AHB Master Lock 1
  wire m_ahb_hresetn;         // AHB Resetn
  wire m_ahb_hresetn_1;       // AHB Resetn 1
  wire [1:0] register_set_0_hresp;  // AHB Slave 0 Response
  wire [1:0] register_set_1_hresp;  // AHB Slave 1 Response
  wire [31:0] register_set_1_reg0;  // AHB Slave 1 Register 0
  wire [31:0] register_set_1_reg1;  // AHB Slave 1 Register 1
  /////////////////////////////////////////////////
  wire [31:0] register_set_1_reg2;  // AHB Slave 1 Register 2
  wire [31:0] register_set_1_reg3;  // AHB Slave 1 Register 3
  /////////////////////////////////////////////////
 
  // YOUR CODE GOES IN HERE
  wire [3:0] TFTLCD_SW;     // TFT-LCD Switches
  wire [2:0] PushButton;    // Push Buttons
  // YOUR CODE GOES IN HERE

  TFTLCDCtrl TFTLCDCtrl_i (
      .CLK(TFTLCD_CLK),       
      .nRESET(TFTLCD_nRESET),
      .TCLK(TFTLCD_TCLK),
      .Hsync(TFTLCD_Hsync),
      .Vsync(TFTLCD_Vsync),
      .DE_out(TFTLCD_DE_out),
      .R(TFTLCD_R),
      .G(TFTLCD_G),
      .B(TFTLCD_B),
      .Tpower(TFTLCD_Tpower),
      
      // YOUR CODE GOES IN HERE////////////////////////////////////////
      .SW(TFTLCD_SW),       // Switches for reverse, changing picture
      .BTN(PushButton[0]),  // Button 0 for color inversion
      // YOUR CODE GOES IN HERE////////////////////////////////////////
      
      .BRAMCLK(BRAMCLK),
      .BRAMADDR(BRAMADDRA),
      .BRAMDATA(BRAMDATA)
  );
  
  // BRAM for TFT-LCD
  bufferram
    bufferram_i (
      .clka( TFTLCD_CLK ),
      .ena(1'b1),
      .wea( 1'b0 ),
      .addra( BRAMADDRA ),
      .dina( 16'd0 ),
      .douta( BRAMDATA ));

assign TFTLCD_SW[0] = register_set_1_reg0[0];
assign TFTLCD_SW[1] = register_set_1_reg1[0];
/////////////////////////////////////////////////
assign TFTLCD_SW[2] = register_set_1_reg2[0]; 
assign TFTLCD_SW[3] = register_set_1_reg3[0]; 
/////////////////////////////////////////////////

assign LED[0] = register_set_1_reg0[0];
assign LED[1] = register_set_1_reg1[0];

endmodule
