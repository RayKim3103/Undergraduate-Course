`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/28 16:29:51
// Design Name: 
// Module Name: tb_top_memory_wrapper
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


module tb_top_memory_wrapper;

    reg clk;
    initial clk = 1'b0;
    always #5 clk <= ~clk;

    reg resetn;
    reg start;
    wire done;

    reg enb;
    reg [7:0] addrb, addrb_buf;
    wire [15:0] doutb;
    reg [15:0] answer_mem [0:255];
    
    initial begin
        $display("Welcom EE3551_Practice5_2!");
        resetn <= 1'b1;
        start <= 1'b0;
        enb <= 1'b0;
        addrb <= 8'd0;
        addrb_buf <= 8'd0;
        
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
            compare_memory();   // 메모리 비교 함수 호출
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
            $readmemh("C:/2025exp_accelerator/week5_2/answer_memory.hex", answer_mem);
            enb <= 1'b1;    // enable 활성화
            for(i=0; i<257; i=i+1) begin // 0 ~ 256
                // 주소 reg에 i값 저장
                addrb <= i;
                // 주소값을 버퍼에 저장하고 10 ns 기다림
                addrb_buf <= addrb; #10
                if(i > 0) begin
                    if(answer_mem[addrb_buf] != doutb) begin
                        // 예상 메모리 값과 DUT에서 읽어온 값이 다르면
                        // 메모리 비교 실패 메시지 출력
                        $display("Error: memory comparison failed @ %8dns", $time);
                        $display ("[%d] IIDEAL : %h DUT : %h", addrb_buf, answer_mem[addrb_buf],doutb );
                        $finish;
                    end
                end
            end
            $display("PASS: memory comparison succeed @ %8dns", $time);
        end
    endtask

    top_memory_wrapper Utop_memory_wrapper(
        .clk(clk),
        .resetn(resetn),

        .start(start),
        .done(done),

        .ext_enb(enb),
        .ext_addrb(addrb),
        .ext_doutb(doutb)
    );
    
  

endmodule
