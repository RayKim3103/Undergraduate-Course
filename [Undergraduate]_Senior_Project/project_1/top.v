
`timescale 1ns / 1ps

module top#(
    // SRAM address widths (enough to cover depth)
    parameter ADDR_IN       = 20,           // 2^20 = 1,048,576 > 34x34x512 = 591,872 (HWC)
    parameter ADDR_W        = 18,           // 2^18 = 262,144 > 3x3x512x32 = 147,456 (KH, KW, IC, OC_tile)
    parameter ADDR_OUT      = 15,           // 2^15 = 32,768 > 32x32x32 = 32,768
    parameter INPUT_BW      = 8,            // 8bit Data comes from AXI interface
    parameter OUTPUT_BW     = 8             // 8bit Data goes to AXI interface (after Fixed point Quantization)
    )(
    input wire clk,
    input wire resetn,
    // ------------------------------------------------------------------------
    // Configurable Data
    // ------------------------------------------------------------------------
    input  wire start,
    input  wire [2:0] K,        // Max: 3
    input  wire [9:0] IC,       // Max: 512
    input  wire [5:0] IMG_H,    // Tile_Size_oc_H = 32
    input  wire [5:0] IMG_W,    // Tile_Size_oc_W = 32
    input  wire [7:0] OC,       // Tile_Size_oc = 32
    input  wire [3:0] shift_n,  // Max: 7 or 8
    input  wire [2:0] STRIDE,
//    output wire ready,
    output wire done,
    // ------------------------------------------------------------------------
    // input image memory [port A (PS writes image in advance), port B (PL reads)]
    // ------------------------------------------------------------------------
    input wire                          input_clka,
    input wire                          input_ena,
    input wire                          input_wea,
    input wire [ADDR_IN-1:0]            input_addra, 
    input wire signed [INPUT_BW-1:0]    input_dina,
    output wire signed [INPUT_BW-1:0]   input_douta,
    // ------------------------------------------------------------------------
    // Store Total Weights
    // ------------------------------------------------------------------------
    input  wire                             weight_clka,
    input  wire                             weight_ena,
    input  wire                             weight_wea,
    input  wire [ADDR_W-1:0]                weight_addra,
    input  wire signed [INPUT_BW-1:0]       weight_dina,
    output wire signed [INPUT_BW-1:0]       weight_douta,
    // ------------------------------------------------------------------------
    // Store Output Logits
    // ------------------------------------------------------------------------
    input  wire                             out_mem_clkb,
    input  wire                             out_mem_enb,
    input  wire                             out_mem_web,
    input  wire [ADDR_OUT-1:0]              out_mem_addrb,
    input  wire signed [OUTPUT_BW-1:0]      out_mem_dinb,
    output wire signed [OUTPUT_BW-1:0]      out_mem_doutb
    );
    // ------------------------------------------------------------------------
    // IF MAP & TOTAL WEIGHT MEM B-ports (READ), OUT MEM A-ports (WRITE)
    // ------------------------------------------------------------------------
    wire                            input_enb;
    wire                            input_web;
    wire [ADDR_IN-1:0]              input_addrb;
    wire signed [INPUT_BW-1:0]      input_dinb;
    wire signed [INPUT_BW-1:0]      input_doutb;
    
    wire                            weight_enb;
    wire                            weight_web;
    wire [ADDR_W-1:0]               weight_addrb;
    wire signed [INPUT_BW-1:0]      weight_dinb;
    wire signed [INPUT_BW-1:0]      weight_doutb;
    
    wire                            out_mem_ena;
    wire                            out_mem_wea;
    wire [ADDR_OUT-1:0]             out_mem_addra;
    wire signed [OUTPUT_BW-1:0]     out_mem_dina;
    wire signed [OUTPUT_BW-1:0]     out_mem_douta;
    
    // ------------------------------------------------------------------------
    // act_n_weight_ctrlr IP : locate activations & weights to appropriate core
    // ------------------------------------------------------------------------
    localparam  ACT_PER_CORE      = 11;
    localparam  WEIGHT_PER_CORE   = 9;
    localparam  NUM_DENSE_CORE    = 8;
    localparam  NUM_SPARSE_CORE   = 4;
    
    wire [INPUT_BW-1:0]  input_mem_data;
    wire [ADDR_IN-1:0]   input_mem_addr;
    wire                 input_mem_en;
    assign input_mem_data  = input_doutb;
    assign input_addrb  = input_mem_addr;
    assign input_enb    = input_mem_en;
    
    wire [INPUT_BW-1:0]  weight_mem_data;
    wire [ADDR_W-1:0]    weight_mem_addr;
    wire                 weight_mem_en;
    assign weight_mem_data = weight_doutb;
    assign weight_addrb = weight_mem_addr;
    assign weight_enb   = weight_mem_en;
    
    wire signed [INPUT_BW-1:0]      act_row_mem_data;
    wire [ACT_PER_CORE-1:0]         act_row_mem_addr;

    wire signed [INPUT_BW-1:0]      weight_row_mem_data;
    wire [WEIGHT_PER_CORE-1:0]      weight_row_mem_addr;
    
    wire [NUM_DENSE_CORE-1:0]     which_dense_core_done;
    wire [NUM_SPARSE_CORE-1:0]    which_sparse_core_done;
    
    wire [NUM_DENSE_CORE-1:0]     which_dense_core_start;
    wire [NUM_SPARSE_CORE-1:0]    which_sparse_core_start;
    
    act_n_weight_ctrlr act_n_weight_ctrlr (   .clk(clk), .resetn(resetn),
                                              .start(start), .done(done),
                                              .TOTAL_IC(IC), .OC(OC), .IMG_H(IMG_H), .IMG_W(IMG_W), .K(K),
                                              
                                              .input_mem_data(input_mem_data), .input_mem_addr(input_mem_addr), .input_mem_en(input_mem_en),
                                              .weight_mem_data(weight_mem_data), .weight_mem_addr(weight_mem_addr), .weight_mem_en(weight_mem_en),
                                              
                                              .act_row_mem_data(act_row_mem_data), .act_row_mem_addr(act_row_mem_addr),
                                              .weight_row_mem_data(weight_row_mem_data), .weight_row_mem_addr(weight_row_mem_addr),
                                              
                                              .which_dense_core_done(which_dense_core_done), .which_sparse_core_done(which_sparse_core_done),
                                              .which_dense_core_start(which_dense_core_start), .which_sparse_core_start(which_sparse_core_start)
                                          );

    // ------------------------------------------------------------------------
    // activations & weights from act_n_weight_ctrlr IP goes equally to Total COREs
    // fan out = Num of COREs = (8+4) -> "may need register insertions to avoid Timing violation"
    // ------------------------------------------------------------------------
    // signals for total dense_core x8
    wire signed [INPUT_BW-1:0]  core_act_row_mem_data;
    wire [ACT_PER_CORE-1:0]     core_act_row_mem_addr;
    
    assign core_act_row_mem_data    = act_row_mem_data;
    assign core_act_row_mem_addr    = act_row_mem_addr;
    
    wire signed [INPUT_BW-1:0]  core_weight_row_mem_data;
    wire [WEIGHT_PER_CORE-1:0]  core_weight_row_mem_addr;
    
    assign core_weight_row_mem_data = weight_row_mem_data;
    assign core_weight_row_mem_addr = weight_row_mem_addr;
    // ------------------------------------------------------------------------
    // total 8 dense_core IP (doesn't support Zero-Skip): 
    // 1) get IA & Weight Data of corresponding IC (K=3: 1 IC / K=1: 3 IC)
    // 2) transfer datas to corresponding IA_ROW_MEMs & WEIGHT_ROW_MEMs
    // 3) compute MAC operation (need pe control & pe net control)
    // 4) PSUMs are accumulated and goes to 32 different output psum ports (in implementation psums ports are flattened)
    // ------------------------------------------------------------------------
    localparam PSUM_BW           = 32;
    localparam NUM_ROWS          = 32;
    
    wire [PSUM_BW*NUM_ROWS-1:0] dense_core_psum_rows [7:0];
    
    wire [7:0] dense_core_done ;
    wire [7:0] dense_core_start;
    assign which_dense_core_done    = dense_core_done;
    assign dense_core_start         = which_dense_core_start;
    
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : gen_dense_core
            dense_core dense_core_inst (
                .clk(clk),
                .resetn(resetn),
                
                .core_start(dense_core_start[i]),
                .core_done(dense_core_done[i]),
                
                .TOTAL_IC(IC),
                .OC(OC),
                .IMG_H(IMG_H),
                .IMG_W(IMG_W),
                .K(K),
                .STRIDE(STRIDE),
                
                .act_row_mem_data(core_act_row_mem_data),
                .act_row_mem_addr(core_act_row_mem_addr),
                .weight_row_mem_data(core_weight_row_mem_data),
                .weight_row_mem_addr(dcore_weight_row_mem_addr),
                
                .psum_rows(dense_core_psum_rows[i])
            );
        end
    endgenerate   

    // ------------------------------------------------------------------------
    // sparse_core IP (support Zero-Skip): 
    // 1) get IA & Weight Data of corresponding IC (K=3: 1 IC / K=1: 3 IC)
    // 2) transfer datas to corresponding IA_ROW_MEMs & WEIGHT_ROW_MEMs
    // 3) compute MAC operation (need pe control & pe net control & zero-skip control)
    // 4) PSUMs are accumulated and goes to 32 different output psum ports (in implementation psums ports are flattened)
    // ------------------------------------------------------------------------



    
    // ------------------------------------------------------------------------
    // PSUM MEM & PSUM_MEM_ACC Ports A-ports (WRITE), B-ports (READ)
    // ------------------------------------------------------------------------
//    wire                            psum_mem_ena;
//    wire                            psum_mem_wea;
//    wire [ADDR_OUT-1:0]             psum_mem_addra;
//    wire signed [INPUT_BW-1:0]      psum_mem_dina;
 
//    wire                            psum_mem_enb;
//    wire [ADDR_OUT-1:0]             psum_mem_addrb;
//    wire signed [INPUT_BW-1:0]      psum_mem_doutb;
    
//    wire                            psum_mem_acc_ena;
//    wire                            psum_mem_acc_wea;
//    wire [ADDR_OUT-1:0]             psum_mem_acc_addra;
//    wire signed [INPUT_BW-1:0]      psum_mem_acc_dina;
 
//    wire                            psum_mem_acc_enb;
//    wire [ADDR_OUT-1:0]             psum_mem_acc_addrb;
//    wire signed [INPUT_BW-1:0]      psum_mem_acc_doutb;
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    INPUT_MEM INPUT_MEM (
      .clka (input_clka),
      .ena  (input_ena),
      .wea  (input_wea),
      .addra(input_addra),
      .dina (input_dina),
      .douta(input_douta),
      .clkb (clk),
      .enb  (input_enb),
      .web  (input_web),
      .addrb(input_addrb),
      .dinb (input_dinb),
      .doutb(input_doutb)
    );
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    WEIGHT_MEM WEIGHT_MEM (
      .clka (weight_clka),
      .ena  (weight_ena),
      .wea  (weight_wea),
      .addra(weight_addra), 
      .dina (weight_dina),
      .douta(weight_douta), 
      .clkb (clk),
      .enb  (weight_enb),
      .web  (weight_web),         
      .addrb(weight_addrb),     
      .dinb (weight_dinb),        
      .doutb(weight_doutb)   
    );
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
endmodule







////////////////////////////////////////////////////////////////////////////////
//    wire signed [INPUT_BW-1:0]   core_0_act_row_mem_data;
//    wire [ACT_PER_CORE-1:0]      core_0_act_row_mem_addr;

//    wire signed [INPUT_BW-1:0]   core_0_weight_row_mem_data;
//    wire [WEIGHT_PER_CORE-1:0]   core_0_weight_row_mem_addr;

//    wire [PSUM_BW*NUM_ROWS-1:0] core_0_psum_rows;

//    wire core_0_done;
//    wire core_0_start;
    
//    dense_core dense_core_0 (   .clk(clk), .resetn(resetn),
//                                .core_start(), .core_done(),
//                                .TOTAL_IC(IC), .OC(OC), .IMG_H(IMG_H), .IMG_W(IMG_W), .K(K), .STRIDE(STRIDE),
                              
//                                .act_row_mem_data(), .act_row_mem_addr(),
                                
//                                .weight_row_mem_data(), .weight_row_mem_addr(),
                                
//                                .psum_rows()
//                            );
////////////////////////////////////////////////////////////////////////////////




