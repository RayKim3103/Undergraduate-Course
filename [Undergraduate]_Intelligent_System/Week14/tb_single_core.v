`timescale 1ns / 1ps

module tb_single_core;

    reg clk;
    initial clk = 1'b0;
    always #5 clk <= ~clk;

    reg resetn;
    reg start;
    wire done;
//    wire real_done;
//    wire conv_done;

    reg out_mem_ena;
    reg [3:0] out_mem_addra, out_mem_addra_buf;
    wire [7:0] out_mem_douta;
    reg [7:0] answer_mem [0:9];
    
    initial begin
        $display("Welcom EE3551_Practice12!");
        resetn <= 1'b1;
        start <= 1'b0;
        out_mem_ena <= 1'b0;
        out_mem_addra <= 14'd0;
        out_mem_addra_buf <= 14'd0;
        
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
            #50
            start <= 1'b1;
            #10
            start <= 1'b0;
//            $display("DUT Finishs Operation!");
//            #10
//            compare_memory();
//            #10
//            $finish();
        end
    end

    initial begin
      #1500000
        $display("Error: Hit safety net @ %8dns", $time);
        $finish();
    end
    
    integer i;
    task compare_memory;
        begin
            $readmemh("C:/2025exp_accelerator/week12_Assignment4_combined/final_project_ip.xpr/final_project_ip/output_hex_file/output_01.hex", answer_mem);
            out_mem_ena <= 1'b1;
            for(i=0; i<10000; i=i+1) begin // 0 ~ 10404
                out_mem_addra <= i;
                out_mem_addra_buf <= out_mem_addra; #10
                if(i > 0) begin
                    if(answer_mem[out_mem_addra_buf] != out_mem_douta) begin
                        $display("Error: memory comparison failed @ %8dns", $time);
                        $display ("[%d] IIDEAL : %h DUT : %h", out_mem_addra_buf, answer_mem[out_mem_addra_buf],out_mem_douta );
//                        $finish;
                    end
                end
            end
            $display("PASS: memory comparison succeed @ %8dns", $time);
        end
    endtask
    top uut (
        .clk(clk), .resetn(resetn), .start(start),
        .done(done), 
//        .real_done(real_done),
//        .conv_done(conv_done),
//        .done_led(done_led),

        // INPUT A-port
        .input_clka(clk), .input_ena(input_ena), .input_wea(input_wea),
        .input_addra(input_addra), 
        .input_dina(input_dina), 
//        .input_douta(input_douta),

        // WEIGHT A-port
        .tot_weight_clka(clk), .tot_weight_ena(tot_weight_ena), .tot_weight_wea(tot_weight_wea),
        .tot_weight_addra(tot_weight_addra), 
        .tot_weight_dina(tot_weight_dina), 
//        .tot_weight_douta(tot_weight_douta),
        
        // OUTPUT B-port
        .out_mem_0_clkb(clk), .out_mem_0_enb(out_mem_0_enb), .out_mem_0_web(out_mem_0_web),
        .out_mem_0_addrb(out_mem_0_addrb), 
        .out_mem_0_doutb(out_mem_0_doutb)
        
//        // OUTPUT B-port
//        .out_mem_1_clkb(clk), .out_mem_1_enb(out_mem_0_enb), .out_mem_1_web(out_mem_0_web),
//        .out_mem_1_addrb(out_mem_0_addrb), 
//        .out_mem_1_doutb(out_mem_0_doutb)
    );

endmodule
