
module fifo #(
    parameter DATA_WIDTH = 8, // Bit width (8bits)
    parameter FIFO_DEPTH = 8  // FIFO depth (2^n)
    )(
    input   wire                        clk,
    input   wire                        rst_n,
    input   wire                        wren_i, // write enable
    input   wire                        rden_i, // read enable
    input   wire    [DATA_WIDTH-1:0]    wdata_i,// write data (8bits)
    output  wire    [DATA_WIDTH-1:0]    rdata_o,// read data (8bits)
    output  wire                        full_o, // when FIFO is full->High
    output  wire                        empty_o // when FIFO is empty->High
    
//    output wire wrptr_out,
//    output wire rdptr_out
    );
    
    // address width
    localparam ADDR_W = $clog2(FIFO_DEPTH);
    
    //----------------------------------------------------------
    // Write / Read Pointers  (Gray-code ?????, ???? ???)
    //----------------------------------------------------------
    reg [ADDR_W:0] wrptr;   // +1 ??? ?? full ????
    reg [ADDR_W:0] rdptr;
 
    // Full & empty check
    assign empty_o = (wrptr == rdptr);
    assign full_o  = (wrptr[ADDR_W-1:0] == rdptr[ADDR_W-1:0]) &&
                     (wrptr[ADDR_W]     != rdptr[ADDR_W]);

    // Write pointer counter seq logic
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            // set 0 with FIFO_DEPTH_LG2+1 bits
            wrptr <= {(ADDR_W+1){1'b0}};
        end 
        else if (wren_i) begin
            wrptr <= wrptr + 'd1;
        end
        else begin
            wrptr <= wrptr;
        end
    end

    // Read pointer counter seq logic   
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            // set 0 with FIFO_DEPTH_LG2+1 bits
            rdptr <= {(ADDR_W+1){1'b0}};
        end 
        else if (rden_i) begin
            rdptr <= rdptr + 'd1;
        end
        else begin
            rdptr <= rdptr;
        end
    end
    
    fifo_mem UMEM(
        .clka(clk), // write clock
        .ena(wren_i), // write enable
        .wea(wren_i), // write enable
        .addra(wrptr[ADDR_W-1:0]), // write address
        .dina(wdata_i), // write data

        .clkb(clk), // read clock
        .enb(rden_i), // read enable
        .addrb(rdptr[ADDR_W-1:0]), // read address
        .doutb(rdata_o)// read data
    );

endmodule