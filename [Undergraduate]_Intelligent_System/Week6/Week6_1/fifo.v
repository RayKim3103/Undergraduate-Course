
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
    );
    
    // bits to address the FIFO
    localparam FIFO_DEPTH_LG2 = $clog2(FIFO_DEPTH);
    
    reg [FIFO_DEPTH_LG2:0] wrptr; // FIFO write pointer
    reg [FIFO_DEPTH_LG2:0] rdptr; // FIFO read pointer
 
    // Full & empty check
    assign empty_o  =   (wrptr==rdptr);
    assign full_o   =   (wrptr[FIFO_DEPTH_LG2-1:0]==rdptr[FIFO_DEPTH_LG2-1:0]) & 
                        (wrptr[FIFO_DEPTH_LG2] != rdptr[FIFO_DEPTH_LG2]);

    // Write pointer counter seq logic
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            // set 0 with FIFO_DEPTH_LG2+1 bits
            wrptr <= {(FIFO_DEPTH_LG2+1){1'b0}};
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
            rdptr <= {(FIFO_DEPTH_LG2+1){1'b0}};
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
        .addra(wrptr[FIFO_DEPTH_LG2-1:0]), // write address
        .dina(wdata_i), // write data

        .clkb(clk), // read clock
        .enb(rden_i), // read enable
        .addrb(rdptr[FIFO_DEPTH_LG2-1:0]), // read address
        .doutb(rdata_o)// read data
    );

endmodule