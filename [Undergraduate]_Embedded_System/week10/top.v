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
  output TFTLCD_TCLK,   // TFT-LCD Clock
  output wire TFTLCD_Hsync,   // TFT-LCD HSYNC
  output wire TFTLCD_Vsync,   // TFT-LCD VSYNC
  output wire TFTLCD_DE_out,   // TFT-LCD Data enable
  output [7:3] TFTLCD_R, // TFT-LCD Red signal 
  output [7:2] TFTLCD_G, // TFT-LCD Green signal
  output [7:3] TFTLCD_B, // TFT-LCD Blue signal
  output TFTLCD_Tpower,  // TFT-LCD Backlight On signal
//  input [1:0] TFTLCD_SW,
  output [1:0] LED
);

  wire BRAMCLK;
  wire [16:0] BRAMADDRA;
  wire [15:0] BRAMDATA;

  wire [31:0]M_AHB_1_haddr;
  wire [2:0]M_AHB_1_hburst;
  wire [3:0]M_AHB_1_hprot;
  wire [31:0]M_AHB_1_hrdata;
  wire M_AHB_1_hready;
  wire M_AHB_1_hresp;
  wire [2:0]M_AHB_1_hsize;
  wire [1:0]M_AHB_1_htrans;
  wire [31:0]M_AHB_1_hwdata;
  wire M_AHB_1_hwrite;
  wire [31:0]M_AHB_haddr;
  wire [2:0]M_AHB_hburst;
  wire [3:0]M_AHB_hprot;
  wire [31:0]M_AHB_hrdata;
  wire M_AHB_hready;
  wire M_AHB_hresp;
  wire [2:0]M_AHB_hsize;
  wire [1:0]M_AHB_htrans;
  wire [31:0]M_AHB_hwdata;
  wire M_AHB_hwrite;
  wire m_ahb_hclk;
  //wire m_ahb_hclk_1;
  //wire m_ahb_hmastlock;
 // wire m_ahb_hmastlock_1;
  wire m_ahb_hresetn;
 // wire m_ahb_hresetn_1;
  wire [1:0] register_set_0_hresp;
  wire [1:0] register_set_1_hresp;
  wire [31:0] register_set_1_reg0;
  wire [31:0] register_set_1_reg1;
  wire [1:0] TFTLCD_SW;

  system_wrapper usystem_wrapper (
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
    .FIXED_IO_ps_srstb(FIXED_IO_ps_srstb),
    .M_AHB_1_haddr(M_AHB_1_haddr),
    .M_AHB_1_hburst(M_AHB_1_hburst),
    .M_AHB_1_hprot(M_AHB_1_hprot),
    .M_AHB_1_hrdata(M_AHB_1_hrdata),
    .M_AHB_1_hready(M_AHB_1_hready),
    .M_AHB_1_hresp(M_AHB_1_hresp),
    .M_AHB_1_hsize(M_AHB_1_hsize),
    .M_AHB_1_htrans(M_AHB_1_htrans),
    .M_AHB_1_hwdata(M_AHB_1_hwdata),
    .M_AHB_1_hwrite(M_AHB_1_hwrite),
    .M_AHB_haddr(M_AHB_haddr),
    .M_AHB_hburst(M_AHB_hburst),
    .M_AHB_hprot(M_AHB_hprot),
    .M_AHB_hrdata(M_AHB_hrdata),
    .M_AHB_hready(M_AHB_hready),
    .M_AHB_hresp(M_AHB_hresp),
    .M_AHB_hsize(M_AHB_hsize),
    .M_AHB_htrans(M_AHB_htrans),
    .M_AHB_hwdata(M_AHB_hwdata),
    .M_AHB_hwrite(M_AHB_hwrite),
    .m_ahb_hclk(m_ahb_hclk),
    //.m_ahb_hclk_1(m_ahb_hclk_1),
    //.m_ahb_hmastlock(m_ahb_hmastlock),
    //.m_ahb_hmastlock_1(m_ahb_hmastlock_1),
    .m_ahb_hresetn(m_ahb_hresetn)
    //.m_ahb_hresetn_1(m_ahb_hresetn_1)
  );

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
      .SW(TFTLCD_SW),
      .BRAMCLK(BRAMCLK),
      .BRAMADDR(BRAMADDRA),
      .BRAMDATA(BRAMDATA)
  );
 /* 
  bufferram
    bufferram_i (
      .clka( TFTLCD_CLK ),
      .ena(1'b1),
      .wea( 1'b0 ),
      .addra( BRAMADDRA ),
      .dina( 16'd0 ),
      .douta( BRAMDATA ));
*/

assign M_AHB_hresp = (register_set_0_hresp == 2'b00) ? 1'b0 : 1'b1;

  AHB2PORT1RAM AHB2PORT1RAM_i (
    .HRESETn ( m_ahb_hresetn ),
    .BIGEND ( 1'b0 ),
    .PORT1HCLK ( m_ahb_hclk ),
    .PORT1HSEL ( 1'b1 ),
    .PORT1HREADYIN ( M_AHB_hready ),
    .PORT1HTRANS ( M_AHB_htrans ),
    .PORT1HSIZE ( M_AHB_hsize[1:0] ),
    .PORT1HWRITE ( M_AHB_hwrite ),
    .PORT1HWDATA ( M_AHB_hwdata ),
    .PORT1HRDATA ( M_AHB_hrdata ),
    .PORT1HADDR ( M_AHB_haddr ),
    .PORT1HREADYOUT ( M_AHB_hready ),
    .PORT1HRESP (register_set_0_hresp ),
    .PORT2HCLK ( BRAMCLK ),
    .port2addr ( BRAMADDRA ),
    .port2di ( 32'd0 ),
    .port2cs ( 1'b1 ),
    .port2bwe ( 2'b0000 ),
    .port2do ( BRAMDATA ));

assign M_AHB_1_hresp = (register_set_1_hresp == 2'b00) ? 1'b0 : 1'b1;

register_set register_set_1 (
  .HCLK(m_ahb_hclk),
  .HRESETn(m_ahb_hresetn),
  .HSEL(1'b1),
  .HADDR(M_AHB_1_haddr),
  .HWRITE(M_AHB_1_hwrite),
  .HTRANS(M_AHB_1_htrans),
  .HSIZE(M_AHB_1_hsize),
  .HWDATA(M_AHB_1_hwdata),
  .HREADY(M_AHB_1_hready),
  .HREADYin(M_AHB_1_hready),
  .HRESP(register_set_1_hresp),
  .HRDATA(M_AHB_1_hrdata),
  .REG0(register_set_1_reg0),
  .REG1(register_set_1_reg1)
);

assign TFTLCD_SW[0] = register_set_1_reg0[0];
assign TFTLCD_SW[1] = register_set_1_reg1[0];

assign LED[0] = register_set_1_reg0[0];
assign LED[1] = register_set_1_reg1[0];

endmodule