`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2014/07/28 14:57:38
// Design Name: 
// Module Name: textlcd
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


// 각 문자를 나타내는 8bit 값 선언
`define lcd_blank 8'b00100000
`define lcd_dash 8'b00101101
`define lcd_colon 8'b00111010
`define lcd_comma 8'b00101100
`define lcd_Dot 8'b00101110
`define lcd_0 8'b00110000
`define lcd_1 8'b00110001
`define lcd_2 8'b00110010
`define lcd_3 8'b00110011
`define lcd_4 8'b00110100
`define lcd_5 8'b00110101
`define lcd_6 8'b00110110
`define lcd_7 8'b00110111
`define lcd_8 8'b00111000
`define lcd_9 8'b00111001
`define lcd_a 8'b01000001
`define lcd_b 8'b01000010
`define lcd_c 8'b01000011
`define lcd_d 8'b01000100
`define lcd_e 8'b01000101
`define lcd_f 8'b01000110
`define lcd_g 8'b01000111
`define lcd_h 8'b01001000
`define lcd_i 8'b01001001
`define lcd_j 8'b01001010
`define lcd_k 8'b01001011
`define lcd_l 8'b01001100
`define lcd_m 8'b01001101
`define lcd_n 8'b01001110
`define lcd_o 8'b01001111
`define lcd_p 8'b01010000
`define lcd_q 8'b01010001
`define lcd_r 8'b01010010
`define lcd_s 8'b01010011
`define lcd_t 8'b01010100
`define lcd_u 8'b01010101
`define lcd_v 8'b01010110
`define lcd_w 8'b01010111
`define lcd_x 8'b01011000
`define lcd_y 8'b01011001
`define lcd_z 8'b01011010
`define lcd_under 8'b01011111
`define lcd_s_a 8'b01100001
`define lcd_s_b 8'b01100010
`define lcd_s_c 8'b01100011
`define lcd_s_d 8'b01100100
`define lcd_s_e 8'b01100101
`define lcd_s_f 8'b01100110
`define lcd_s_g 8'b01100111
`define lcd_s_h 8'b01101000
`define lcd_s_i 8'b01101001
`define lcd_s_j 8'b01101010
`define lcd_s_k 8'b01101011
`define lcd_s_l 8'b01101100
`define lcd_s_m 8'b01101101
`define lcd_s_n 8'b01101110
`define lcd_s_o 8'b01101111
`define lcd_s_p 8'b01110000
`define lcd_s_q 8'b01110001
`define lcd_s_r 8'b01110010
`define lcd_s_s 8'b01110011
`define lcd_s_t 8'b01110100
`define lcd_s_u 8'b01110101
`define lcd_s_v 8'b01110110
`define lcd_s_w 8'b01110111
`define lcd_s_x 8'b01111000
`define lcd_s_y 8'b01111001
`define lcd_s_z 8'b01111010
`define add_line_1 8'b10000000
`define add_line_2 8'b11000000
`define add_line_3 8'b10010100
`define add_line_4 8'b11010100
module textlcd(
	resetn,
	lcdclk,
	lcd_rs,
	lcd_rw,
	lcd_en,
	lcd_data);

input  resetn;              // 리셋 신호
input  lcdclk;              // LCD 디스플레이의 클록 신호
output lcd_rs;              // 레지스터 선택
output lcd_rw;              // 읽기/쓰기 모드
output lcd_en;              // LCD 활성화 신호
output [7:0] lcd_data;      // LCD로 전송될 8비트 데이터

wire [31:0] reg_a;            // 1행에 표시할 1 번째 4문자 데이터 (32bit)
wire [31:0] reg_b;            // 1행에 표시할 2 번째 4문자 데이터 (32bit)
wire [31:0] reg_c;            // 1행에 표시할 3 번째 4문자 데이터 (32bit)
wire [31:0] reg_d;            // 1행에 표시할 4 번째 4문자 데이터 (32bit)
wire [31:0] reg_e;            // 2행에 표시할 1 번째 4문자 데이터 (32bit)
wire [31:0] reg_f;            // 2행에 표시할 2 번째 4문자 데이터 (32bit)
wire [31:0] reg_g;            // 2행에 표시할 3 번째 4문자 데이터 (32bit)
wire [31:0] reg_h;            // 2행에 표시할 4 번째 4문자 데이터 (32bit)
reg [23:0] delay_lcdclk;      // clock
reg [15:0] count_lcd;         // LCD 상태 전환을 위한 16bit counter
reg lcd_en;                   // Text-LCD Device enable signal
reg  [4:0] lcd_mode = 0;      // 현재 LCD 동작 모드를 저장하는 5bit register
wire [4:0] mode_pwron = 1 ;   // power on
wire [4:0] mode_fnset = 2 ;   // function set
wire [4:0] mode_onoff = 3 ;   // display on/off Control
wire [4:0] mode_entr1 = 4 ;   //
wire [4:0] mode_entr2 = 5 ;   //
wire [4:0] mode_entr3 = 6 ;   //
wire [4:0] mode_seta1 = 7 ;   // set addr 1st line
wire [4:0] mode_wr1st = 8 ;   // Write 1st line
wire [4:0] mode_seta2 = 9 ;   // set addr 2nd line
wire [4:0] mode_wr2nd = 10;   // Write 2nd line
wire [4:0] mode_delay = 11;   // dealy
wire [4:0] mode_actcm = 12;   // user command
reg [9:0] set_data;           // rs, rw, DB0 ~ 7신호



assign reg_a = {`lcd_r,   `lcd_p, 	`lcd_s,     `lcd_dash	} ;                 // "RPS-" (1행 1~4번째 문자)
assign reg_b = {`lcd_z,   `lcd_7,      	`lcd_0, 	    `lcd_2	} ;             // "Z702" (1행 5~8번째 문자)
assign reg_c = {`lcd_0,   `lcd_dash,         `lcd_t, 	    `lcd_k	} ;         // "0-TK" (1행 9~12번째 문자)
assign reg_d = {`lcd_blank,   `lcd_b,        `lcd_d, 	    `lcd_Dot	} ;     // " BD." (1행 13~16번째 문자)
assign reg_e = {`lcd_h,   `lcd_s_u,      `lcd_s_i, 	    `lcd_s_n	} ;         // "hsui" (2행 1~4번째 문자, 소문자)
assign reg_f = {`lcd_s_s,  `lcd_blank,    `lcd_c, 	    `lcd_s_o	} ;         // "s co" (2행 5~8번째 문자, 소문자)
assign reg_g = {`lcd_Dot,  `lcd_comma,  `lcd_blank,   `lcd_s_l	} ;             // "., l" (2행 9~12번째 문자, 소문자)
assign reg_h = {`lcd_s_t,  `lcd_s_d,     `lcd_Dot, 	   `lcd_blank	} ;         // "td. " (2행 13~16번째 문자, 소문자)

// 이 always문에서는 lcdclk에 따라 delay_lcdclk와 count_lcd를 업데이트하고, 
// 또한, delay_lcdclk를 통하여 lcd_en를 제어함
// 이는 LCD 데이터 전송 타이밍을 맞추기 위한 logic

// Counterx2000 & //Counterx40
always @(posedge lcdclk)
begin
	if (resetn == 0) begin
            delay_lcdclk <= 0;      // delay 카운터를 0으로 초기화
            count_lcd <= 0;         // mode를 결정하는 카운터를 0으로 초기화
            lcd_en <= 1'b0;         // lcd enable 신호를 0으로 설정 (비활성)
    end
    else begin
        // Counterx2000
        if (delay_lcdclk < 1999)                // delay 카운터가 1999 미만이면
            delay_lcdclk <=  delay_lcdclk + 1;  // delay 카운터 증가
        else                                    // 1999에 도달하면
            delay_lcdclk <= 0;                  // delay 카운터를 0으로 리셋
        // Counterx40 -> 계속 반복되며 count_lcd를 증가 및 6으로 초기화
        if (delay_lcdclk == 0) begin            // delay 카운터가 0일 때마다
            if (count_lcd < 40)                 // mode를 결정하는 카운터가 40 미만이면
                count_lcd <= count_lcd + 1;     // mode를 결정하는 카운터 증가
            else                                // 40에 도달하면
                count_lcd <= 6;                 // mode를 결정하는 카운터를 6으로 설정
        end
        if (delay_lcdclk == 200)                // delay 카운터가 200일 때
            lcd_en <= 1'b1;                     // lcd enable 신호를 1로 설정 (활성화)
        else if (delay_lcdclk == 1800)          // delay 카운터가 1800일 때
            lcd_en <= 1'b0;                     // lcd enable 신호를 0으로 설정 (비활성화) 
    end
end

    // count_lcd에 따라 lcd_mode를 전환 
    // LCD 초기화와 데이터 쓰기를 순차적으로 수행
    
    // decoder switch
    always @(posedge lcdclk)
    begin
        if (resetn == 0)
            lcd_mode <= mode_pwron;                 // LCD 모드를 전원 켜기 상태로 초기화
        else 
            begin
                case (count_lcd)                    // 상태 카운터 값에 따라 모드 전환
                    0  : lcd_mode <= mode_pwron ;  // 0: 전원 켜기
                    1  : lcd_mode <= mode_fnset ;   // 1: 기능 설정
                    2  : lcd_mode <= mode_onoff ;   // 2: 디스플레이 온/오프
                    3  : lcd_mode <= mode_entr1 ;   // 3: entr 모드 1
                    4  : lcd_mode <= mode_entr2 ;   // 4: entr 모드 2
                    5  : lcd_mode <= mode_entr3 ;   // 5: entr 모드 3
                    6  : lcd_mode <= mode_seta1 ;   // 6: 1행 주소 설정
                    7  : lcd_mode <= mode_wr1st ;   // 7: 1행 데이터 쓰기
                    23 : lcd_mode <= mode_seta2 ;   // 23: 2행 주소 설정
                    24 : lcd_mode <= mode_wr2nd ;   // 24: 2행 데이터 쓰기
                    40 : lcd_mode <= mode_delay ;   // 40: 지연 상태
                    41 : lcd_mode <= mode_actcm ;   // 41: 사용자 명령
                    default : begin end             // 그 외: 아무 동작 없음
                endcase    
            end
    end
    
    // set_data 레지스터의 값을 기반으로 LCD 출력 신호(lcd_rs, lcd_rw, lcd_data)를 할당
    assign lcd_rs = set_data[9];        // set_data[9]를 LCD 레지스터 선택 신호에 연결
    assign lcd_rw = set_data[8];        // set_data[8]를 LCD 읽기/쓰기 신호에 연결
    assign lcd_data = set_data[7:0];    // set_data[7:0]를 LCD 데이터 버스에 연결

    // 각 LCD 모드에 따라 set_data 값을 설정하여 LCD에 보낼 명령어 또는 데이터를 결정 
    // 초기화 명령어와 문자열 데이터를 순차적으로 처리

    // decoder output
    always @(lcdclk or lcd_mode or count_lcd)           // clock, mode, count_lcd 변경 시 동작
    begin 
        if (resetn == 0)                                // 리셋 신호가 0일 때       
            set_data <= 10'b0000000000;                 // set_data를 모두 0으로 초기화
        else
            begin
            case (lcd_mode)                             // 현재 LCD 모드에 따라 데이터 설정
                mode_pwron : set_data <= {2'b00, 8'h38};// 전원 켜기
                mode_fnset : set_data <= {2'b00, 8'h38};// 기능 설정
                mode_onoff : set_data <= {2'b00, 8'h0e};// 디스플레이 온
                mode_entr1 : set_data <= {2'b00, 8'h06};// entry 모드: 오른쪽으로 커서 이동
                mode_entr2 : set_data <= {2'b00, 8'h02};// entry 모드: 커서를 홈으로 이동
                mode_entr3 : set_data <= {2'b00, 8'h01};// entry 모드: 디스플레이 클리어         
                mode_seta1 : set_data <= {2'b00, 8'h80};// DDRAM의 1행 첫 번째 주소 설정
                mode_wr1st :                            // CGRAM나 DDRAM write 
                begin 
                case (count_lcd)
                 7 : set_data <= {1'b1, 1'b0, reg_a[31:24]};    // "R"
                 8 : set_data <= {1'b1, 1'b0, reg_a[23:16]};    // "P"
                 9 : set_data <= {1'b1, 1'b0, reg_a[15: 8]};    // "S"
                10 : set_data <= {1'b1, 1'b0, reg_a[7 : 0]};    // "-"
                11 : set_data <= {1'b1, 1'b0, reg_b[31:24]};    // "Z"
                12 : set_data <= {1'b1, 1'b0, reg_b[23:16]};    // "7"
                13 : set_data <= {1'b1, 1'b0, reg_b[15: 8]};    // "0"
                14 : set_data <= {1'b1, 1'b0, reg_b[7 : 0]};    // "2"
                15 : set_data <= {1'b1, 1'b0, reg_c[31:24]};    // "0"
                16 : set_data <= {1'b1, 1'b0, reg_c[23:16]};    // "-"
                17 : set_data <= {1'b1, 1'b0, reg_c[15: 8]};    // "T"
                18 : set_data <= {1'b1, 1'b0, reg_c[7 : 0]};    // "K"
                19 : set_data <= {1'b1, 1'b0, reg_d[31:24]};    // 공백
                20 : set_data <= {1'b1, 1'b0, reg_d[23:16]};    // "B"
                21 : set_data <= {1'b1, 1'b0, reg_d[15: 8]};    // "D"
                22 : set_data <= {1'b1, 1'b0, reg_d[7 : 0]};    // "."
                endcase
            end
               mode_seta2 : set_data <= {2'b00, 8'hc0};
               mode_wr2nd : 
                begin
                case (count_lcd)
                24 : set_data <= {1'b1, 1'b0, reg_e[31:24]};    // "h"
                25 : set_data <= {1'b1, 1'b0, reg_e[23:16]};    // "u"
                26 : set_data <= {1'b1, 1'b0, reg_e[15: 8]};    // "i"
                27 : set_data <= {1'b1, 1'b0, reg_e[7 : 0]};    // "n"
                28 : set_data <= {1'b1, 1'b0, reg_f[31:24]};    // "s"
                29 : set_data <= {1'b1, 1'b0, reg_f[23:16]};    // 공백
                30 : set_data <= {1'b1, 1'b0, reg_f[15: 8]};    // "c"
                31 : set_data <= {1'b1, 1'b0, reg_f[7 : 0]};    // "o"
                32 : set_data <= {1'b1, 1'b0, reg_g[31:24]};    // "."
                33 : set_data <= {1'b1, 1'b0, reg_g[23:16]};    // ","
                34 : set_data <= {1'b1, 1'b0, reg_g[15: 8]};    // 공백
                35 : set_data <= {1'b1, 1'b0, reg_g[7 : 0]};    // "I"
                36 : set_data <= {1'b1, 1'b0, reg_h[31:24]};    // "t"
                37 : set_data <= {1'b1, 1'b0, reg_h[23:16]};    // "d"
                38 : set_data <= {1'b1, 1'b0, reg_h[15: 8]};    // "."
                39 : set_data <= {1'b1, 1'b0, reg_h[7 : 0]};    // 공백
                endcase
                end
                mode_delay : set_data <= {2'b00, 8'h02};
                mode_actcm : set_data <= {2'b00, 8'h02};
            default : begin end
            endcase
        end
    end

endmodule