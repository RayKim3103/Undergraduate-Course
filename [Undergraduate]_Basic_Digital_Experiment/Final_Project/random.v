`timescale 1ns / 1ps

 module random(
  output wire [31:0] out,
  input wire clk,  
  input wire reset,
  input wire request        
  ); 
  
  // random generator parameter
  parameter [31:0] mod = 32'b1000_0000_0000_0000_0000_0000_0000_0000; // 32bit modulous연산을 위한 값
  parameter [31:0] a = 32'd22695477;   // LCG의 곱셈 상수
  parameter [31:0] c = 32'b1;           // LCG의 더하기 상수
  
  //registor for temporal value
  reg [31:0] temp = 32'd13;    // 현재 난수 값을 저장하는 reg, 초기값은 decimal num: 13 [크기: 32비트]
  reg [63:0] temp1;            // 중간 계산값을 저장하는 64비트 reg
  reg [31:0] temp2;            // 모듈러 연산 후 값을 저장하는 32비트 reg
  reg [31:0] temp_out = 8'd0;  // 최종 난수 출력값을 저장하는 32비트 reg
  reg [1:0] state = 2'b0;      // FSM의 state 저장

  //FSM for generator random number
always @(posedge clk) begin
        // state = 00
        if(state == 2'b00) begin 
            temp1 <= temp * a + c;  // temp1 ← temp × a + c : 난수 생성
            state <= 2'b01;
        end
        // state = 01
        else if(state == 2'b01) begin
            temp2 <= temp1 % mod;  // temp2 ← temp1 % mod : 난수를 32bit만큼 남도록 나머지 연산
            state <= 2'b10;
        end
        // state = 10
        else begin
            temp <= temp2;        // temp ← temp2
            if(request) 
                state <= 2'b00;   // request 신호가 1이면 state = 00
        end
end

// output value when receive the request
always @(posedge request) begin
        temp_out <= temp2;      // temp_out ← temp2
end

assign out = temp_out;


 endmodule 
