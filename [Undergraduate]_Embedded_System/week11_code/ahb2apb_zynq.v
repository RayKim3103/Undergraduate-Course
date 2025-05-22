module ahb2apb_zynq (
    input m_ahb_hclk,
    input m_ahb_hresetn,
    input M_AHB_hwrite,
    input [2:0] M_AHB_hsize,
    input [1:0] M_AHB_htrans,
    input [31:0] M_AHB_hwdata,
    input [3:0] M_AHB_hprot,
    input [2:0] M_AHB_hburst,
    input [31:0] M_AHB_haddr,
    output M_AHB_hready,
    output M_AHB_hresp,
    output [31:0] M_AHB_hrdata,
    output [31:0] PADDR,
    output reg PENABLE,
    output reg PSEL,
    output reg PWRITE,
    output [31:0] PWDATA,
    input [31:0] PRDATA
    );
    reg [1:0] st_ctrl;
    reg [31:0] reg_addr;
    assign M_AHB_hready = (st_ctrl == 2'b00) ? 1'b1 : 1'b0;
    assign M_AHB_hresp = 1'b0;
    assign PADDR = reg_addr;
    assign PWDATA = M_AHB_hwdata;
    assign M_AHB_hrdata = PRDATA;
    always @ (posedge m_ahb_hclk or negedge m_ahb_hresetn)
    begin
        if (!m_ahb_hresetn)
            st_ctrl <= 2'b00;
        else
            begin
                case (st_ctrl)
                2'b00: begin
                    if (M_AHB_htrans[1])
                        st_ctrl <= 2'b01;
                end
                default: st_ctrl <= 2'b00;
                endcase
            end
    end
    always @ (posedge m_ahb_hclk or negedge m_ahb_hresetn)
        begin
        if (!m_ahb_hresetn)
            begin
                reg_addr <= 31'd0;
                PENABLE <= 1'b0;
                PSEL <= 1'b0;
                PWRITE <= 1'b0;
            end
        else
            begin
            if ((st_ctrl == 2'b00) && M_AHB_htrans[1])
                reg_addr <= M_AHB_haddr;
            if ( ((st_ctrl == 2'b00) && M_AHB_htrans[1]) || (st_ctrl == 2'b01))
                PSEL <= 1'b1;
            else
                PSEL <= 1'b0;
            if ((st_ctrl == 2'b00) && M_AHB_htrans[1])
                PWRITE <= M_AHB_hwrite;
            if (st_ctrl == 2'b01)
                PENABLE <= 1'b1;
            else
                PENABLE <= 1'b0;
            end
    end
endmodule