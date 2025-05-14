`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: HUINS
// Engineer:
//
// Create Date: 2012/11/28 13:46:07
// Design Name: segment
// Module Name: clock
// Project Name: segment
// Target Devices: xc7z020clg484-1
// Tool Versions: Xilinx PlanAhead 14.3
// Description:
//
// Dependencies:
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
//
//////////////////////////////////////////////////////////////////////////////////


module clock(
  input clk_in,                 // input clock
  input resetn,                 // reset 신호 (active low)
  output reg [31:0] segdata);   // 32비트 segment 데이터 출력

  reg temp_clk;                 // 필요 없음
  reg onesec_clk;               // 주기가 1초인 clock
  reg tensec_clk;               // 주기가 10초인 clock 
  reg onemin_clk;               // 주기가 1분인 clock
  reg tenmin_clk;               // 주기가 10분인 clock
  reg hour_clk;                 // 주기가 1시간인 clock
  reg [3:0] onesec_cnt;         // 1초가 몇 번 지났는 지 저장
  reg [3:0] tensec_cnt;         // 10초가 몇 번 지났는 지 저장
  reg [3:0] onemin_cnt;         // 1분이 몇 번 지났는 지 저장  
  reg [3:0] tenmin_cnt;         // 10분이 몇 번 지났는 지 저장
  reg [3:0] hour_cnt;           // 1시간이 몇 번 지났는 지 저장
  reg [32:0] cnt;               // input clk이 몇 번 지났는지 저장 (25M hz clock의 cycle 수 저장) 

  // 디스플레이에 12-00-00을 나타낼 경우
  // reset시 segdata의 "-"가 있는 자리를 0으로 초기화
  // 그 외에는 그 자리에 "-"를 보여주도록 설정
  always @ (negedge resetn or posedge clk_in)
  begin
    if (!resetn)
    begin
      segdata[11:8] <= 4'b0000;
      segdata[23:20] <= 4'b0000;
    end
    else
    begin
      segdata[11:8] <= 4'b1010;
      segdata[23:20] <= 4'b1010;
    end
  end

  // reset시 초기화 하고 input clk 들어 올 때마다 cycle수를 세어 1초짜리 clock을 만듬
  always @ (negedge resetn or posedge clk_in)
  begin                     
    if (!resetn)                            // reset 시 1초를 측정하기 위한 counter와 1Hz clock을 초기화
    begin
      cnt <= 33'd0;
      onesec_clk <= 1'b0;
    end
    else
    begin
      if (cnt < 33'd12499999)               // cnt가 12,499,999미만이면 cnt += 1을 하여 clock cycle을 센다.
            cnt <= cnt + 33'd1;
      else cnt <= 33'd0;
      if (cnt == 33'd12499999)              // input clock은 25MHz이므로, 12,499,999마다 토글해주면 1Hz clock이 생성됨 
            onesec_clk <= ~onesec_clk;
    end
  end
  
  // 먼저, segdata[3:0]은 reset 시 0초~9초를 나타내는 자리의 binary data
  // onesec_clk은 1초 주기의 clock이므로 이 always문은 1초마다 작동한다.또는 reset 시에 작동
  // onesec_clk의 posedge를 감지하기 때문
  always @ (negedge resetn or posedge  onesec_clk)
  begin
    if (!resetn)
    begin
    /***** 기본 코드 *****/
//      onesec_cnt <= 4'b0001;                    // 1초 주기의 Clock의 clock cycle 횟수를 1로 초기화 
//                                                 // -> 다음 clk의 posedge에서는 패널에 1을 디스플레이 해야 하기 때문
//      tensec_clk <= 1'b0;                       // 10초 주기의 clock을 초기화
//      segdata[3:0] <= 4'b0000;                  // segdata를 0으로 초기화 -> 디스플레이 패널에 숫자 0이 표기됨
    /*********************/
      onesec_cnt <= 4'b1000;                   // 1초 주기의 Clock의 clock cycle 횟수를 8로 초기화
                                                // -> 다음 clk의 posedge에 패널에 8을 디스플레이 해야하기 때문
      tensec_clk <= 1'b0;                      // 10초 주기의 clock을 초기화
      segdata[3:0] <= 4'b0111;                 // segdata를 7으로 초기화 -> 디스플레이 패널에 숫자 7이 표기됨
    end
    else
    begin
      if (onesec_cnt < 4'b1001)                // 9초까지는 onesec_cnt += 1 (패널이 0~9까지 디스플레이)
        onesec_cnt <= onesec_cnt + 4'b0001;
      else                                     // 9초 초과는 0으로 초기화 되어야 함
        onesec_cnt <= 4'b0000;
      if (onesec_cnt == 4'b0000)               // onesec_cnt = 0이라는 것은 10초가 되었다는 것이다.
        tensec_clk <= 1'b1;                     // 10초 마다, tensec_clk을 1로 만듬 
      else                                       
        tensec_clk <= 1'b0;                     // 그 외의 0~9초동안은 tensec_clk은 0

      segdata[3:0] <= onesec_cnt;               // onesec_cnt의 수가 그동안 몇초가 흘렀는지 센것이므로 그 숫자를 segdata에 할당
    end
  end

  // 먼저, segdata[7:4]은 reset 시 10초~59초를 나타내는 자리의 binary data
  // tensec_clk은 10초 주기의 clock이므로 이 always문은 10초마다 작동한다.또는 reset 시에 작동
  // tensec_clk의 posedge를 감지하기 때문
  always @  (negedge resetn or posedge tensec_clk)
  begin
    if (!resetn)
    begin
    /***** 기본 코드 *****/
//      tensec_cnt <= 4'b0001;      // 10초가 몇번 흘렀는지 저장하는 cnt를 1로 초기화
//                                   // -> 다음 tensec_clk에 패널에 1을 표시해야하기 때문
//      onemin_clk <= 1'b0;         // 60초 주기의 clock을 0으로 초기화
//      segdata[7:4] <= 4'b0000;    // 10~59초를 나타내는 자리를 0으로 초기화 -> 디스플레이 패널에 숫자 0이 표기됨
    /*********************/
      tensec_cnt <= 4'b1001;        // 10초가 몇번 흘렀는지 저장하는 cnt를 9로 초기화
                                    // -> 다음 tensec_clk에 패널에 9를 표시해야하기 때문
      onemin_clk <= 1'b0;           // 60초 주기의 clock을 0으로 초기화
      segdata[7:4] <= 4'b1000;      // 10~59초를 나타내는 자리를 8으로 초기화 -> 디스플레이 패널에 숫자 8이 표기됨
    end
    else
    // Quiz 1의 경우 reset 시 tensec_cnt = 9가 된다.
    // 따라서, 기존 코드인 tensec_cnt < 4'b0101을 바꿔야 하는가에 대해 생각을 해보면,
    // 9 -> 10이 될 때 결국 tensec_cnt를 0으로 초기화해야 한다.
    // 따라서, 굳이 바꿀 필요는 없다.
    // 하지만, 안정성을 위해서 if 조건을 하나 추가해도 된다.
    // 질문이 생길 수 있다. if문에서 tensec_cnt = 9일 때, 0으로 바뀌었으니 패널에 8 다음에 0이 나오는 것 아닐까?
    // 하지만, 8 다음에 9는 잘 표시 된다. 이 이유는 non-blocking으로 값을 전해주었기 때문이다.
    /***** 가능한 코드 1*****/ 
//    begin
//      if (tensec_cnt < 4'b0101)                 // tenseec_cnt는 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0 반복 (0초 ~ 59초까지 10초 자리에는 0~5의 숫자가 들어온다) 
//        tensec_cnt <= tensec_cnt + 4'b0001;
//      else                                      // 60초가 되는 순간 tensec_cnt를 0으로 초기화 (디스플레이에 0이 표기되어야 함)
//         tensec_cnt <= 4'b0000;
//      if (tensec_cnt == 4'b0000)                // 60초가 되는 순간 onemin_clk을 1로 설정, 1분 주기의 신호 생성
//        onemin_clk <= 1'b1;
//      else                                     
//        onemin_clk <= 1'b0;                      // 그 외에는 onemin_clk은 0이다.
//      segdata[7:4] <= tensec_cnt;                // 10~59초를 나타내는 자리를 tensec_cnt가 저장하는 값을 디스플레이
//    end
    /*************************/
    /***** 가능한 코드 2*****/
    begin
      if (tensec_cnt >= 4'b1001)                  // 9 이상이면 cnt가 0으로 가야하기에 tensec_cnt를 초기화 한다.
        tensec_cnt <= 4'b0000;
      else if (tensec_cnt < 4'b0101)             // tenseec_cnt는 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0 반복 (0초 ~ 59초까지 10초 자리에는 0~5의 숫자가 들어온다)
        tensec_cnt <= tensec_cnt + 4'b0001;
      else                                        // 60초가 되는 순간 tensec_cnt를 0으로 초기화 (디스플레이에 0이 표기되어야 함)
        tensec_cnt <= 4'b0000;
      
      if (tensec_cnt == 4'b0000)                  // 60초 또는 9이상에서 초기화가 되는 순간 onemin_clk을 1로 설정, 1분 자리의 디스플레이 변화가 가능하도록 함 
        onemin_clk <= 1'b1;
      else                                        // 그 외에는 onemin_clk은 0이다.
        onemin_clk <= 1'b0;
      segdata[7:4] <= tensec_cnt;                  // 10~59초를 나타내는 자리를 tensec_cnt가 저장하는 값을 디스플레이
    end
    /*************************/
  end

  always @ (negedge resetn or posedge onemin_clk)
  begin
    if (!resetn)
    begin
    /***** 기본 코드 *****/
//      onemin_cnt <= 4'b0001;
//      tenmin_clk <= 1'b0;
//      segdata[15:12] <= 4'b0000;
    /*********************/
      onemin_cnt <= 4'b0001;            // 1분이 몇번 흘렀는지 저장하는 cnt를 1로 초기화
                                        // -> 다음 onemin_clk에 패널에 1를 표시해야하기 때문
      tenmin_clk <= 1'b0;               // 10분 주기의 clock을 0으로 초기화
      segdata[15:12] <= 4'b0000;        // 0~9분을 나타내는 자리를 0으로 초기화 -> 디스플레이 패널에 숫자 0이 표기됨      
    end
    else
    begin
      if (onemin_cnt < 4'b1001)
        onemin_cnt <= onemin_cnt + 4'b0001;
      else
        onemin_cnt <= 4'b0000;
      if (onemin_cnt == 4'b0000)
        tenmin_clk <= 1'b1;
      else
        tenmin_clk <= 1'b0;
        segdata[15:12] <= onemin_cnt;
    end
  end

  always @ (negedge resetn or posedge tenmin_clk)
  begin
    if (!resetn)
    begin
    /***** 기본 코드 *****/
//      tenmin_cnt <= 4'b0001;
//      hour_clk <= 1'b0;
//      segdata[19:16] <= 4'b0000;
    /*********************/
      tenmin_cnt <= 4'b0011;            // 10분이 몇번 흘렀는지 저장하는 cnt를 3으로 초기화
                                         // -> 다음 onemin_clk에 패널에 1를 표시해야하기 때문
      hour_clk <= 1'b0;                  // 1시간 주기의 clock을 0으로 초기화
      segdata[19:16] <= 4'b0010;        // 10~59분을 나타내는 자리를 2로 초기화 -> 디스플레이 패널에 숫자 2가 표기됨
    end
    else
    begin
      if (tenmin_cnt < 4'b0101)
        tenmin_cnt <= tenmin_cnt + 4'b0001;
      else
        tenmin_cnt <= 4'b0000;
      if (tenmin_cnt == 4'b0000)
        hour_clk <= 1'b1;
      else
        hour_clk <= 1'b0;
      segdata[19:16] <= tenmin_cnt;
    end
  end

  always @ (negedge resetn or posedge hour_clk)
  begin
    if (!resetn)                        // reset 시
    begin
      hour_cnt <= 4'b0001;              // 1시간이 몇번 흘렀는지 저장하는 cnt를 1로 초기화
                                        // -> 다음 hour_clk에 패널에 1을 표시해야하기 때문
      segdata[31:24] <= 8'h12;          // [31:24]는 8bit 신호이므로 8'h12를 하여 시간을 나타내는 자리 디스플레이 패널 2개에 12를 표기 
    end
    else
    begin
      if (hour_cnt < 4'b1011)           // 시계에서 시간은 총 1~12까지 표기 되므로, hour_cnt는 decimal로 0~11의 값을 저장
        hour_cnt <= hour_cnt + 4'b0001; // hour_cnd가 11보다 작으면 hour_cnt += 1 
      else
        hour_cnt <= 4'b0000;            // hour_cnt가 11이 되면 다시 0으로 초기화
      if (hour_cnt == 4'b0000)          // hour_cnt가 0일 떄는 12시를 의미하므로 시간을 나타내는 segdata에 8'h12를 할당
        segdata[31:24] <= 8'h12;
      else if (hour_cnt == 4'b1010)     // hour_cnt가 4'd10일 떄는 10시를 의미하므로 시간을 나타내는 segdata에 8'h10를 할당
        segdata[31:24] <= 8'h10;
      else if (hour_cnt == 4'b1011)     // hour_cnt가 4'd11일 떄는 11시를 의미하므로 시간을 나타내는 segdata에 8'h11를 할당
        segdata[31:24] <= 8'h11;
      else
        segdata[31:24] <= {4'b0000, hour_cnt};      // 10,11,12를 제외한 시간은 10의 자리가 0이므로 앞의 4bit는 0000을 사용하고 뒤의 4bit는 hour_cnt값을 할당
    end
  end

endmodule