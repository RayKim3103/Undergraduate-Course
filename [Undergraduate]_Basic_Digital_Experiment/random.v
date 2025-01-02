`timescale 1ns / 1ps

 module random(
  output wire [31:0] out,
  input wire clk,  
  input wire reset,
  input wire request        
  ); 
  
  // random generator parameter
  parameter [31:0] mod = 32'b1000_0000_0000_0000_0000_0000_0000_0000; // 32bit modulous������ ���� ��
  parameter [31:0] a = 32'd22695477;   // LCG�� ���� ���
  parameter [31:0] c = 32'b1;           // LCG�� ���ϱ� ���
  
  //registor for temporal value
  reg [31:0] temp = 32'd13;    // ���� ���� ���� �����ϴ� reg, �ʱⰪ�� decimal num: 13 [ũ��: 32��Ʈ]
  reg [63:0] temp1;            // �߰� ��갪�� �����ϴ� 64��Ʈ reg
  reg [31:0] temp2;            // ��ⷯ ���� �� ���� �����ϴ� 32��Ʈ reg
  reg [31:0] temp_out = 8'd0;  // ���� ���� ��°��� �����ϴ� 32��Ʈ reg
  reg [1:0] state = 2'b0;      // FSM�� state ����

  //FSM for generator random number
always @(posedge clk) begin
        // state = 00
        if(state == 2'b00) begin 
            temp1 <= temp * a + c;  // temp1 �� temp �� a + c : ���� ����
            state <= 2'b01;
        end
        // state = 01
        else if(state == 2'b01) begin
            temp2 <= temp1 % mod;  // temp2 �� temp1 % mod : ������ 32bit��ŭ ������ ������ ����
            state <= 2'b10;
        end
        // state = 10
        else begin
            temp <= temp2;        // temp �� temp2
            if(request) 
                state <= 2'b00;   // request ��ȣ�� 1�̸� state = 00
        end
end

// output value when receive the request
always @(posedge request) begin
        temp_out <= temp2;      // temp_out �� temp2
end

assign out = temp_out;


 endmodule 
