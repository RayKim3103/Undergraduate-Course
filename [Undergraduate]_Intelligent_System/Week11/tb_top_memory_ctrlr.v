`timescale 1ns / 1ps

module tb_top_memory_ctrlr;

    reg clk;
    initial clk = 1'b0;
    always #5 clk <= ~clk;

    reg resetn;
    reg start;
    wire done;

    reg s2_ena;
    reg [13:0] s2_addra, s2_addra_buf;
    wire [7:0] s2_douta;
    reg [7:0] answer_mem [0:10403];
    
    initial begin
        $display("Welcom EE3551_Practice10!");
        resetn <= 1'b1;
        start <= 1'b0;
        s2_ena <= 1'b0;
        s2_addra <= 14'd0;
        s2_addra_buf <= 14'd0;
        
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

            $display("DUT Finishs Operation!");
            #10
            compare_memory();
            #10
            $finish();
        end
    end

    initial begin
      #300000
        $display("Error: Hit safety net @ %8dns", $time);
        $finish();
    end
    
    integer i;
    task compare_memory;
        begin
            $readmemh("C:/2025exp_accelerator/week10_AS3/0_results_compare/answer_memory.hex", answer_mem);
            s2_ena <= 1'b1;
            for(i=0; i<10000; i=i+1) begin // 0 ~ 10404
                s2_addra <= i;
                s2_addra_buf <= s2_addra; #10
                if(i > 0) begin
                    if(answer_mem[s2_addra_buf] != s2_douta) begin
                        $display("Error: memory comparison failed @ %8dns", $time);
                        $display ("[%d] IIDEAL : %h DUT : %h", s2_addra_buf, answer_mem[s2_addra_buf],s2_douta );
//                        $finish;
                    end
                end
            end
            $display("PASS: memory comparison succeed @ %8dns", $time);
        end
    endtask

    top_memory_ctrlr uut (
        .clk(clk), .resetn(resetn), .start(start),
        .done(done), .done_led(done_led),

        // SRAM1 A-port
        .s1_clka(clk), .s1_ena(s1_ena), .s1_wea(s1_wea),
        .s1_addra(s1_addra), .s1_dina(s1_dina), .s1_douta(s1_douta),

        // SRAM2 A-port
        .s2_clka(clk), .s2_ena(s2_ena), .s2_wea(s2_wea),
        .s2_addra(s2_addra), .s2_dina(s2_dina), .s2_douta(s2_douta)
    );

endmodule


