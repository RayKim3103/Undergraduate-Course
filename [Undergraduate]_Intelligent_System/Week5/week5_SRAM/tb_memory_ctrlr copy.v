`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/27 16:25:11
// Design Name: 
// Module Name: tb_memory_ctrlr
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module tb_memory_ctrlr();

    reg clk;
    initial clk = 1'b0;
    always #5 clk <= ~clk;

    reg resetn;
    reg start;

    reg [31:0] SRAM2_compare [0:191];

    wire done;

    initial begin
//        $readmemh("YOUR PROJECT PATH/init_memory.hex", Utop_memory_ctrlr.SRAM1.mem);
        $readmemh("C:/2025exp_accelerator/week5_1/init_memory.hex", Utop_memory_ctrlr.SRAM1.mem);
    end


    initial begin
        $display("Welcom EE3551_Practice5_1!");
        resetn <= 1'b1;
        start <= 1'b0;
        #302
        resetn <= 1'b0;
        #50
        resetn <= 1'b1;
        #50
        start <= 1'b1;
        #10
        start <= 1'b0;
        // done signal의 posedge에서 시작
        @(posedge done) begin
            // done signal의 posedge라는 것은 
            // read & write operation이 끝났다는 것
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
//            $readmemh("YOUR PROJECT PATH/answer_memory.hex", SRAM2_compare);
            // .hex파일을 통해 정답 데이터를 불러옴
            $readmemh("C:/2025exp_accelerator/week5_1/answer_memory.hex", SRAM2_compare);
            for(i=0; i<192; i=i+1) begin
                // 정답 메모리 값과 DUT에서 읽어온 값이 다르면
                // 메모리 비교 실패 메시지 출력
                if(Utop_memory_ctrlr.SRAM2.mem[i] != SRAM2_compare[i])
                begin
                    $display("Error: memory comparison failed @ %8dns", $time);
                    $display("[%d] IDEAL : %h DUT : %h", i, SRAM2_compare[i], Utop_memory_ctrlr.SRAM2.mem[i]);
                    $writememh("C:/2025exp_accelerator/week5_1/result_memory.hex", Utop_memory_ctrlr.SRAM2.mem);
//                    $writememh("YOUR PROJECT PATH/result_memory.hex", Utop_memory_ctrlr.SRAM2.mem);
                    $finish;
                end
                if(i<10)
                    $display("[%d] IDEAL : %h DUT : %h", i, SRAM2_compare[i], Utop_memory_ctrlr.SRAM2.mem[i]);

            end
            $display("PASS: memory comparison succeed @ %8dns", $time);
//            $writememh("YOUR PROJECT PATH/result_memory.hex", Utop_memory_ctrlr.SRAM2.mem);
            $writememh("C:/2025exp_accelerator/week5_1/result_memory.hex", Utop_memory_ctrlr.SRAM2.mem);
        end
    endtask

    top_memory_ctrlr Utop_memory_ctrlr(
        .clk(clk),
        .resetn(resetn),

        .start(start),
        .done(done)
    );

endmodule
