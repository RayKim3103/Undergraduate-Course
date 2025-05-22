module top(
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
input [3:0] PB,
output [7:0] LED
);
// AHB Slave
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
wire m_ahb_hmastlock;
wire m_ahb_hresetn;
// APB Slave
wire [31:0] PADDR;
wire PENABLE;
wire PSEL;
wire PWRITE;
wire [31:0] PWDATA;
wire [31:0] PRDATA;
wire Core0_nIRQ;
system_wrapper system_wrapper_i (
.Core0_nIRQ(~Core0_nIRQ),
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
.m_ahb_hresetn(m_ahb_hresetn)
);
ahb2apb_zynq ahb2apb_zynq_i (
.m_ahb_hclk(m_ahb_hclk),
.m_ahb_hresetn(m_ahb_hresetn),
.M_AHB_hwrite(M_AHB_hwrite),
.M_AHB_hsize(M_AHB_hsize),
.M_AHB_htrans(M_AHB_htrans),
.M_AHB_hwdata(M_AHB_hwdata),
.M_AHB_hprot(M_AHB_hprot),
.M_AHB_hburst(M_AHB_hburst),
.M_AHB_haddr(M_AHB_haddr),
.M_AHB_hready(M_AHB_hready),
.M_AHB_hresp(M_AHB_hresp),
.M_AHB_hrdata(M_AHB_hrdata),
.PADDR(PADDR),
.PENABLE(PENABLE),
.PSEL(PSEL),
.PWRITE(PWRITE),
.PWDATA(PWDATA),
.PRDATA(PRDATA)
);
pb_intr pb_intr_i(
.PCLK(m_ahb_hclk),
.PRESETn(m_ahb_hresetn),
.PADDR(PADDR),
.PENABLE(PENABLE),
.PSEL(PSEL),
.PWRITE(PWRITE),
.PWDATA(PWDATA),
.PRDATA(PRDATA),
.PB(PB),
.LED(LED),
.INTR(Core0_nIRQ)
);
endmodule