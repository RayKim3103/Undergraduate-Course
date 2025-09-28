
`timescale 1ns / 1ps

module dense_core#(
    // SRAM address widths (enough to cover depth)
    parameter ADDR_IN           = 20,           // 2^20 = 1,048,576 > 34x34x512 = 591,872 (HWC)
    parameter ADDR_W            = 18,           // 2^18 = 262,144 > 3x3x512x32 = 147,456 (KH, KW, IC, OC_tile)
    parameter ADDR_PSUM         = 15,           // 2^15 = 32,768 > 32x32x32 = 32,768
    parameter INPUT_BW          = 8,            // 8bit Data comes from AXI interface
    parameter PSUM_BW           = 32,           // 8bit Data goes to AXI interface (after Quantization)
    parameter ACT_PER_CORE      = 11,
    parameter WEIGHT_PER_CORE   = 9,
    parameter NUM_ROWS          = 32,
    parameter NUM_IA_ROW_MEM    = 96,
    parameter NUM_WEIGHT_ROW_MEM= 3
    )(
    input wire clk,
    input wire resetn,
    // ------------------------------------------------------------------------
    // Configurable Data
    // ------------------------------------------------------------------------
    input wire core_start,
    input  wire [2:0] K,        // Max: 3
    input  wire [9:0] TOTAL_IC, // Max: 512
    input  wire [5:0] IMG_H,    // Tile_Size_oc_H = 32
    input  wire [5:0] IMG_W,    // Tile_Size_oc_W = 32
    input  wire [7:0] OC,       // Tile_Size_oc = 32
    input  wire [2:0] STRIDE,
    output wire core_done,
    // ------------------------------------------------------------------------
    // act_row_mem outputs & write
    // ------------------------------------------------------------------------
    input wire signed [INPUT_BW-1:0]   act_row_mem_data,
    input wire [ACT_PER_CORE-1:0]      act_row_mem_addr,
    // ------------------------------------------------------------------------
    // weight_row_mem outputs & write
    // ------------------------------------------------------------------------
    input wire signed [INPUT_BW-1:0]       weight_row_mem_data,
    input wire [WEIGHT_PER_CORE-1:0]       weight_row_mem_addr,
    // ------------------------------------------------------------------------
    // psum from core 32bit x 32rows (2-D shape array like wire or reg is not supported in verilog)
    // ------------------------------------------------------------------------
    output wire [PSUM_BW*NUM_ROWS-1:0] psum_rows
    );
    
    localparam IA_ROW_MEM_ADDR      = 6;
    localparam WEIGHT_ROW_MEM_ADDR  = 7;
    
    // ------------------------------------------------------------------------
    // data_2_row_mem IP : transfer data comes from act_n_weight_ctrlr to corresponding row_mems
    // ------------------------------------------------------------------------
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    /***** IA_ROW_MEMs Signals *****/
    // each row mem contains maximum 34 datas (K=3: 34x8bit / K=1: 32x8bit)
    // PE_ARRAY have total 3x32 PEs -> Thus, total 96 row mems are needed (for K=1, every PE needs row mem)
//    wire [95:0]                             ia_row_mem_ena;         // 0 to 95
//    wire [95:0]                             ia_row_mem_wea;
//    wire [(96 * IA_ROW_MEM_ADDR) - 1:0]     ia_row_mem_addra_flat;  // flattened address
//    wire [(96 * INPUT_BW) - 1:0]            ia_row_mem_dina_flat;   // flattened data
    
//    wire [95:0]                             ia_row_mem_enb;
//    wire [(96 * IA_ROW_MEM_ADDR) - 1:0]     ia_row_mem_addrb_flat;
//    wire [(96 * INPUT_BW) - 1:0]            ia_row_mem_doutb_flat;
    
    wire [NUM_IA_ROW_MEM-1:0]               ia_row_mem_ena;         // 0 to 95
    wire [NUM_IA_ROW_MEM-1:0]               ia_row_mem_wea;
    wire [IA_ROW_MEM_ADDR-1:0]              ia_row_mem_addra [NUM_IA_ROW_MEM-1:0];
    wire [INPUT_BW-1:0]                     ia_row_mem_dina  [NUM_IA_ROW_MEM-1:0];
    
    wire [NUM_IA_ROW_MEM-1:0]               ia_row_mem_enb;
    wire [IA_ROW_MEM_ADDR-1:0]              ia_row_mem_addrb [NUM_IA_ROW_MEM-1:0];
    wire [INPUT_BW-1:0]                     ia_row_mem_doutb [NUM_IA_ROW_MEM-1:0];
   
    /***** Total 96 IA_ROW_MEMs *****/
    genvar i;
    generate
        for (i = 0; i < 96; i = i + 1) begin : gen_ia_row_mem
            IA_ROW_MEM IA_ROW_MEM_inst (
                .clka (clk),
                .ena (ia_row_mem_ena[i]),
                .wea (ia_row_mem_wea[i]),
                .addra(ia_row_mem_addra[i]),
                .dina (ia_row_mem_dina[i]),
                .clkb (clk),
                .enb (ia_row_mem_enb[i]),
                .addrb(ia_row_mem_addrb[i]),
                .doutb(ia_row_mem_doutb[i])
            );
        end
    endgenerate
    
    /***** WEIGHT_ROW_MEMs Signals *****/
    wire [2:0]                      weight_row_mem_ena;
    wire [2:0]                      weight_row_mem_wea;
    wire [WEIGHT_ROW_MEM_ADDR-1:0]  weight_row_mem_addra [NUM_WEIGHT_ROW_MEM-1:0];
    wire [INPUT_BW-1:0]             weight_row_mem_dina  [NUM_WEIGHT_ROW_MEM-1:0];
    wire [2:0]                      weight_row_mem_enb;
    wire [WEIGHT_ROW_MEM_ADDR-1:0]  weight_row_mem_addrb [NUM_WEIGHT_ROW_MEM-1:0];
    wire [INPUT_BW-1:0]             weight_row_mem_doutb [NUM_WEIGHT_ROW_MEM-1:0];
    
    /***** Total 3 WEIGHT_ROW_MEMs *****/
    generate
        for (i = 0; i < 3; i = i + 1) begin : gen_weight_row_mem
            WEIGHT_ROW_MEM WEIGHT_ROW_MEM_inst (
                .clka (clk),
                .ena (weight_row_mem_ena[i]),
                .wea (weight_row_mem_wea[i]),
                .addra(weight_row_mem_addra[i]), // bit slicing
                .dina (weight_row_mem_dina[i]), // bit slicing
                .clkb (clk),
                .enb (weight_row_mem_enb[i]),
                .addrb(weight_row_mem_addrb[i]),
                .doutb(weight_row_mem_doutb[i])
            );
        end
    endgenerate
    
//    IA_ROW_MEM IA_ROW_MEM_0 (
//      .clka (clk),
//      .ena  (ia_row_mem_0_ena),
//      .wea  (ia_row_mem_0_wea),
//      .addra(ia_row_mem_0_addra),
//      .dina (ia_row_mem_0_dina),
 
//      .clkb (clk),
//      .enb  (ia_row_mem_0_enb),
//      .addrb(ia_row_mem_0_addrb),
//      .doutb(ia_row_mem_0_doutb)
//    );
    

endmodule
