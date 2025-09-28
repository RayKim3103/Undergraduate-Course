`timescale 1ns / 1ps

module top_tb;

    reg clk;
    initial clk = 1'b0;
    always #5 clk <= ~clk;

    reg resetn;
    reg start;
    wire done;
    reg [2:0] K;
    reg [13:0] IC;
    reg [5:0] IMG_H;
    reg [5:0] IMG_W;
    reg [7:0] OC;
    reg [2:0] stride;
    reg [3:0] shift_n;
    
//    wire real_done;
//    wire conv_done;

    reg             out_mem_ena;
    reg     [15:0]  out_mem_addra, out_mem_addra_buf;
    wire    [31:0]  out_mem_douta;
    reg     [31:0]  out_mem_douta_buf;
    reg             out_mem_wea;
    
    reg     [31:0]  answer_mem [0:1000];
    
    initial begin
        $display("Welcom EE3551_Practice12!");
        resetn <= 1'b1;
        start <= 1'b0;
        out_mem_ena <= 1'b0;
        out_mem_wea <= 1'b0;
        out_mem_addra <= 14'd0;
        out_mem_addra_buf <= 14'd0;
        out_mem_douta_buf <= 7'd0;
        
//////////////////////////////
//        IC <= 14'd16;
//        K <= 3'd1;
//        IMG_H <= 6'd32;
//        IMG_W <= 6'd32;
//        OC <= 8'd32;
//        stride <= 3'd1;
//        shift_n <= 4'd8;
//////////////////////////////        
        IC <= 14'd8;
        K <= 3'd3;
        IMG_H <= 6'd32;
        IMG_W <= 6'd32;
        OC <= 8'd16;
        stride <= 3'd1;
        shift_n <= 4'd8;    
//////////////////////////////

        #200
        start <= 1'b1;
        #10
        start <= 1'b0;
        
        #302
        resetn <= 1'b0;
        #50
        resetn <= 1'b1;
        #50
        start <= 1'b1;
        #10
        start <= 1'b0;
        
        @(posedge done) begin
//            #55
//            start <= 1'b1;
//            #10
//            start <= 1'b0;
            
            $display("Done Done Done Start Start Start");
            $display("DUT Finishs Operation!");
            #10
            compare_memory();
            
            #100
            $finish();
        end
    end

    initial begin
      #3000000
        $display("Error: Hit safety net @ %8dns", $time);
        $finish();
    end
    
    integer i;
    task compare_memory;
        begin
            $readmemh("C:/minsung/6_ISL_Intern/ISL_NPU/VIVADO/project_2_lite_v2_no_quant/conv4_tile_layer_output_hex_1d.hex", answer_mem);
            out_mem_ena <= 1'b1;
            #20
            for(i=0; i<1000; i=i+1) begin // 0 ~ 10404
                out_mem_addra <= i;
                out_mem_addra_buf <= out_mem_addra;
                #10
//                if(i%100 == 0)  $display ("By 100th: [%d] IIDEAL : %h DUT : %h", out_mem_addra_buf, answer_mem[out_mem_addra_buf],out_mem_douta);
                if(i >= 0) begin
                    if(answer_mem[out_mem_addra] != out_mem_douta) begin                                                            // if(answer_mem[out_mem_addra_buf] != out_mem_douta) begin
                        $display("Error: memory comparison failed @ %8dns", $time);
                        $display ("[%d] IIDEAL : %h DUT : %h", out_mem_addra, answer_mem[out_mem_addra],out_mem_douta );    // $display ("[%d] IIDEAL : %h DUT : %h", out_mem_addra_buf, answer_mem[out_mem_addra_buf],out_mem_douta );
                        $finish;
                    end
                end
            end
            $display("PASS: memory comparison succeed @ %8dns", $time);
        end
    endtask
    
    top uut (
        .clk(clk), .resetn(resetn), .start(start),
        .done(done),
        
        .K(K),
        .IC(IC),
        .IMG_H(IMG_H),
        .IMG_W(IMG_W),
        .OC(OC),
        .shift_n(shift_n),
        .STRIDE(stride),

        // INPUT A-port
        .input_clka(clk), .input_ena(input_ena), .input_wea(input_wea),
        .input_addra(input_addra), 
        .input_dina(input_dina), 

        // WEIGHT A-port
        .weight_clka(clk), .weight_ena(tot_weight_ena), .weight_wea(tot_weight_wea),
        .weight_addra(tot_weight_addra), 
        .weight_dina(tot_weight_dina), 
        
        // OUTPUT B-port
        .out_mem_clkb(clk), .out_mem_enb(out_mem_ena), .out_mem_web(out_mem_wea),
        .out_mem_addrb(out_mem_addra), 
        .out_mem_doutb(out_mem_douta)
        
    );

endmodule