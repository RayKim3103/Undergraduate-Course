//
// TFT-LCD Clock ����
// �Է� Clock�� 2���� ��Ŵ
//

// input CLK를 기준으로 toggle되는 출력 신호 UP_CLK를 생성하는 clk divider
// frequency = CLK / 2
module g2m(
  input CLK,
  input RESET,
  output reg UP_CLK);
    
  always@(posedge RESET or posedge CLK)
  begin
    if(RESET == 1)
        UP_CLK = 0;
    else
        UP_CLK = ~UP_CLK;
  end
    
endmodule