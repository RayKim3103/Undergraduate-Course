//`timescale 1ns / 1ps
////////////////////////////////////////////////////////////////////////////////////
//// Company:
//// Engineer:
////
//// Create Date: 2014/07/28 14:57:38
//// Design Name:
//// Module Name: textlcd
//// Project Name:
//// Target Devices:
//// Tool Versions:
//// Description:
////
//// Dependencies:
////
//// Revision:
//// Revision 0.01 - File Created
//// Additional Comments:
////
////////////////////////////////////////////////////////////////////////////////////

//`define lcd_blank 8'b00100000
//`define lcd_dash 8'b00101101
//`define lcd_colon 8'b00111010
//`define lcd_comma 8'b00101100
//`define lcd_Dot 8'b00101110
//`define lcd_0 8'b00110000
//`define lcd_1 8'b00110001
//`define lcd_2 8'b00110010
//`define lcd_3 8'b00110011
//`define lcd_4 8'b00110100
//`define lcd_5 8'b00110101
//`define lcd_6 8'b00110110
//`define lcd_7 8'b00110111
//`define lcd_8 8'b00111000
//`define lcd_9 8'b00111001
//`define lcd_a 8'b01000001
//`define lcd_b 8'b01000010
//`define lcd_c 8'b01000011
//`define lcd_d 8'b01000100
//`define lcd_e 8'b01000101
//`define lcd_f 8'b01000110
//`define lcd_g 8'b01000111
//`define lcd_h 8'b01001000
//`define lcd_i 8'b01001001
//`define lcd_j 8'b01001010
//`define lcd_k 8'b01001011
//`define lcd_l 8'b01001100
//`define lcd_m 8'b01001101
//`define lcd_n 8'b01001110
//`define lcd_o 8'b01001111
//`define lcd_p 8'b01010000
//`define lcd_q 8'b01010001
//`define lcd_r 8'b01010010
//`define lcd_s 8'b01010011
//`define lcd_t 8'b01010100
//`define lcd_u 8'b01010101
//`define lcd_v 8'b01010110
//`define lcd_w 8'b01010111
//`define lcd_x 8'b01011000
//`define lcd_y 8'b01011001
//`define lcd_z 8'b01011010
//`define lcd_under 8'b01011111
//`define lcd_s_a 8'b01100001
//`define lcd_s_b 8'b01100010
//`define lcd_s_c 8'b01100011
//`define lcd_s_d 8'b01100100
//`define lcd_s_e 8'b01100101
//`define lcd_s_f 8'b01100110
//`define lcd_s_g 8'b01100111
//`define lcd_s_h 8'b01101000
//`define lcd_s_i 8'b01101001
//`define lcd_s_j 8'b01101010
//`define lcd_s_k 8'b01101011
//`define lcd_s_l 8'b01101100
//`define lcd_s_m 8'b01101101
//`define lcd_s_n 8'b01101110
//`define lcd_s_o 8'b01101111
//`define lcd_s_p 8'b01110000
//`define lcd_s_q 8'b01110001
//`define lcd_s_r 8'b01110010
//`define lcd_s_s 8'b01110011
//`define lcd_s_t 8'b01110100
//`define lcd_s_u 8'b01110101
//`define lcd_s_v 8'b01110110
//`define lcd_s_w 8'b01110111
//`define lcd_s_x 8'b01111000
//`define lcd_s_y 8'b01111001
//`define lcd_s_z 8'b01111010
//`define add_line_1 8'b10000000
//`define add_line_2 8'b11000000
//`define add_line_3 8'b10010100
//`define add_line_4 8'b11010100
//module textlcd(
//resetn,
//PushButton,
//lcdclk,
//lcd_rs,
//lcd_rw,
//lcd_en,
//lcd_data);

//    input [2:0] PushButton;
//    input  resetn;
//    input  lcdclk;
//    output lcd_rs;
//    output lcd_rw;
//    output lcd_en;
//    output [7:0] lcd_data;
    
//    wire [2:0] PushButton;
    
    
//    reg [31:0] reg_a;
//    reg [31:0] reg_b;
//    reg [31:0] reg_c;
//    reg [31:0] reg_d;
//    reg [31:0] reg_e;
//    reg [31:0] reg_f;
//    reg [31:0] reg_g;
//    reg [31:0] reg_h;
    
    
//    //reg [31:0] reg_a_2;
//    //reg [31:0] reg_b_2;
//    //reg [31:0] reg_c_2;
//    //reg [31:0] reg_d_2;
//    //reg [31:0] reg_e_2;
//    //reg [31:0] reg_f_2;
//    //reg [31:0] reg_g_2;
//    //reg [31:0] reg_h_2;
    
    
//    reg [23:0] delay_lcdclk; //clock
//    reg [15:0] count_lcd;
//    reg lcd_en;       // Text-LCD Device enable signal
//    reg  [4:0] lcd_mode = 0;
//    wire [4:0] mode_pwron = 1 ;  // power on
//    wire [4:0] mode_fnset = 2 ;  // function set
//    wire [4:0] mode_onoff = 3 ;  // display on/off Control
//    wire [4:0] mode_entr1 = 4 ;  //
//    wire [4:0] mode_entr2 = 5 ;  //
//    wire [4:0] mode_entr3 = 6 ;  //
//    wire [4:0] mode_seta1 = 7 ;  // set addr 1st line
//    wire [4:0] mode_wr1st = 8 ;  // Write 1st line
//    wire [4:0] mode_seta2 = 9 ;  // set addr 2nd line
//    wire [4:0] mode_wr2nd = 10;  // Write 2nd line
//    wire [4:0] mode_delay = 11;  // dealy
//    wire [4:0] mode_actcm = 12;  // user command
//    reg [9:0] set_data;
    
    
//    //reg_a = {`lcd_2,   `lcd_0, `lcd_2,     `lcd_5 } ;
//    //reg_b = {`lcd_blank,   `lcd_e,       `lcd_m,    `lcd_b } ;
//    //reg_c = {`lcd_e,   `lcd_d,         `lcd_d,    `lcd_e } ;
//    //reg_d = {`lcd_d,   `lcd_blank,        `lcd_blank,    `lcd_blank } ;
    
    
//    //reg_e = {`lcd_s,   `lcd_y,      `lcd_s,    `lcd_t } ;
//    //reg_f = {`lcd_e,  `lcd_m,    `lcd_blank,    `lcd_l } ;
//    //reg_g = {`lcd_a,  `lcd_b,  `lcd_blank,   `lcd_blank } ;
//    //reg_h = {`lcd_blank,  `lcd_blank,     `lcd_blank,   `lcd_blank } ;


////////////////////////////////////////////////////////////////////////////////////
////    // Counterx2000 & //Counterx40
////    always @(posedge lcdclk)
////    begin
////        if (resetn == 0)
////        begin
////            delay_lcdclk <= 0;
////            count_lcd <= 0;
////            lcd_en <= 1'b0;
////        end
////        else
////        begin
////            // Counterx2000
////            if (delay_lcdclk < 5) ///// Simulation /////
////                delay_lcdclk <=  delay_lcdclk + 1;
////            else
////                delay_lcdclk <= 0;
   
////            // Counterx40
////            if (delay_lcdclk == 0)
////            begin
////                if (count_lcd < 40)
////                    count_lcd <= count_lcd + 1;
////                else
////                    count_lcd <= 6;
////            end
       
////            if (delay_lcdclk == 1) ///// Simulation /////
////                lcd_en <= 1'b1;
////            else if (delay_lcdclk == 5) ///// Simulation /////
////                lcd_en <= 1'b0;
               
////        end
////    end
////////////////////////////////////////////////////////////////////////////////////


//    // Counterx2000 & //Counterx40
//    always @(posedge lcdclk)
//    begin
//        if (resetn == 0)
//        begin
//            delay_lcdclk <= 0;
//            count_lcd <= 0;
//            lcd_en <= 1'b0;
//        end
//        else
//        begin
//            // Counterx2000
//            if (delay_lcdclk < 1999)
//                delay_lcdclk <=  delay_lcdclk + 1;
//            else
//                delay_lcdclk <= 0;
   
//            // Counterx40
//            if (delay_lcdclk == 0)
//            begin
//                if (count_lcd < 40)
//                    count_lcd <= count_lcd + 1;
//                else
//                    count_lcd <= 6;
//            end
       
//            if (delay_lcdclk == 200)
//                lcd_en <= 1'b1;
//            else if (delay_lcdclk == 1800)
//                lcd_en <= 1'b0;
               
//        end
//    end
   
//    // decoder switch
//    always @(posedge lcdclk)
//    begin
//        if (resetn == 0)
//            lcd_mode <= mode_pwron;
//        else
//            begin
//                case (count_lcd)
//                    0  : lcd_mode <= mode_pwron ;
//                    1  : lcd_mode <= mode_fnset ;
//                    2  : lcd_mode <= mode_onoff ;
//                    3  : lcd_mode <= mode_entr1 ;
//                    4  : lcd_mode <= mode_entr2 ;
//                    5  : lcd_mode <= mode_entr3 ;
//                    6  : lcd_mode <= mode_seta1 ;
//                    7  : lcd_mode <= mode_wr1st ;
//                    23 : lcd_mode <= mode_seta2 ;
//                    24 : lcd_mode <= mode_wr2nd ;
//                    40 : lcd_mode <= mode_delay ;
//                    41 : lcd_mode <= mode_actcm ;
//                    default : begin end
//                endcase    
//            end
//    end
   
//    assign lcd_rs = set_data[9];
//    assign lcd_rw = set_data[8];
//    assign lcd_data = set_data[7:0];
   
//    // decoder output
//    always @(lcdclk or lcd_mode or count_lcd)
//    begin
//        if (resetn == 0)
//            set_data <= 10'b0000000000;
//        else
//            begin
//            case (lcd_mode)
//                mode_pwron : set_data <= {2'b00, 8'h38};
//                mode_fnset : set_data <= {2'b00, 8'h38};
//                mode_onoff : set_data <= {2'b00, 8'h0e};
//                mode_entr1 : set_data <= {2'b00, 8'h06};
//                mode_entr2 : set_data <= {2'b00, 8'h02};
//                mode_entr3 : set_data <= {2'b00, 8'h01};                
//                mode_seta1 : set_data <= {2'b00, 8'h80};
//                mode_wr1st :
//                begin
//                case (count_lcd)
//                 7 : set_data <= {1'b1, 1'b0, reg_a[31:24]};
//                 8 : set_data <= {1'b1, 1'b0, reg_a[23:16]};
//                 9 : set_data <= {1'b1, 1'b0, reg_a[15: 8]};
//                10 : set_data <= {1'b1, 1'b0, reg_a[7 : 0]};
//                11 : set_data <= {1'b1, 1'b0, reg_b[31:24]};
//                12 : set_data <= {1'b1, 1'b0, reg_b[23:16]};
//                13 : set_data <= {1'b1, 1'b0, reg_b[15: 8]};
//                14 : set_data <= {1'b1, 1'b0, reg_b[7 : 0]};
//                15 : set_data <= {1'b1, 1'b0, reg_c[31:24]};
//                16 : set_data <= {1'b1, 1'b0, reg_c[23:16]};
//                17 : set_data <= {1'b1, 1'b0, reg_c[15: 8]};
//                18 : set_data <= {1'b1, 1'b0, reg_c[7 : 0]};
//                19 : set_data <= {1'b1, 1'b0, reg_d[31:24]};
//                20 : set_data <= {1'b1, 1'b0, reg_d[23:16]};
//                21 : set_data <= {1'b1, 1'b0, reg_d[15: 8]};
//                22 : set_data <= {1'b1, 1'b0, reg_d[7 : 0]};
//                endcase
//            end
//               mode_seta2 : set_data <= {2'b00, 8'hc0};
//               mode_wr2nd :
//                begin
//                case (count_lcd)
//                24 : set_data <= {1'b1, 1'b0, reg_e[31:24]};
//                25 : set_data <= {1'b1, 1'b0, reg_e[23:16]};
//                26 : set_data <= {1'b1, 1'b0, reg_e[15: 8]};
//                27 : set_data <= {1'b1, 1'b0, reg_e[7 : 0]};
//                28 : set_data <= {1'b1, 1'b0, reg_f[31:24]};
//                29 : set_data <= {1'b1, 1'b0, reg_f[23:16]};
//                30 : set_data <= {1'b1, 1'b0, reg_f[15: 8]};
//                31 : set_data <= {1'b1, 1'b0, reg_f[7 : 0]};
//                32 : set_data <= {1'b1, 1'b0, reg_g[31:24]};
//                33 : set_data <= {1'b1, 1'b0, reg_g[23:16]};
//                34 : set_data <= {1'b1, 1'b0, reg_g[15: 8]};
//                35 : set_data <= {1'b1, 1'b0, reg_g[7 : 0]};
//                36 : set_data <= {1'b1, 1'b0, reg_h[31:24]};
//                37 : set_data <= {1'b1, 1'b0, reg_h[23:16]};
//                38 : set_data <= {1'b1, 1'b0, reg_h[15: 8]};
//                39 : set_data <= {1'b1, 1'b0, reg_h[7 : 0]};
//                endcase
//            end
//                mode_delay : set_data <= {2'b00, 8'h02};
//                mode_actcm : set_data <= {2'b00, 8'h02};
//                default : begin end
//            endcase
//        end
//    end



////    reg [2:0] RegPushButton;

//    always @ (posedge lcdclk or negedge resetn) begin
//          if (!resetn) begin
////            RegPushButton <= 3'd0;
//            reg_a <= {`lcd_2,   `lcd_0, `lcd_2,     `lcd_5 } ;
//            reg_b <= {`lcd_blank,   `lcd_e,          `lcd_m,         `lcd_b    } ;
//            reg_c <= {`lcd_e,   `lcd_d,         `lcd_d,         `lcd_e    } ;
//            reg_d <= {`lcd_d,   `lcd_blank,        `lcd_blank,         `lcd_blank    } ;

//            reg_e <= {`lcd_s,   `lcd_y,      `lcd_s,         `lcd_t    } ;
//            reg_f <= {`lcd_e,  `lcd_m,    `lcd_blank,         `lcd_l    } ;
//            reg_g <= {`lcd_a,  `lcd_b,  `lcd_blank,   `lcd_blank    } ;
//            reg_h <= {`lcd_blank,  `lcd_blank,     `lcd_blank,        `lcd_blank    } ;
           
////            reg_d <= {`lcd_2,   `lcd_0, `lcd_2,     `lcd_5 } ;
////            reg_a <= {`lcd_blank,   `lcd_e,          `lcd_m,         `lcd_b    } ;
////            reg_b <= {`lcd_e,   `lcd_d,         `lcd_d,         `lcd_e    } ;
////            reg_c <= {`lcd_d,   `lcd_blank,        `lcd_blank,         `lcd_blank    } ;
           
           
////            reg_h <= {`lcd_s,   `lcd_y,      `lcd_s,         `lcd_t    } ;
////            reg_e <= {`lcd_e,  `lcd_m,    `lcd_blank,         `lcd_l    } ;
////            reg_f <= {`lcd_a,  `lcd_b,  `lcd_blank,   `lcd_blank    } ;
////            reg_g <= {`lcd_blank,  `lcd_blank,     `lcd_blank,        `lcd_blank    } ;

//          end
//          else begin
////            RegPushButton <= PushButton;
////            if ((!PushButton[0]) && (RegPushButton[0]))  begin
//            if (PushButton[0]) begin
//                    reg_a <= reg_d;
//                    reg_b <= reg_a;
//                    reg_c <= reg_b ;
//                    reg_d <= reg_c ;
                   
                   
//                    reg_e <= reg_h ;
//                    reg_f <= reg_e ;
//                    reg_g <= reg_f;
//                    reg_h <= reg_g;
//                end
////                else if ((!PushButton[1]) && (RegPushButton[1]))  begin                                          
//            else if(PushButton[1]) begin
//                    reg_a <= reg_b;
//                    reg_b <= reg_c;
//                    reg_c <= reg_d ;
//                    reg_d <= reg_a ;
                   
                   
//                    reg_e <= reg_f ;
//                    reg_f <= reg_g ;
//                    reg_g <= reg_h;
//                    reg_h <= reg_e;
//            end
////                else if ((!PushButton[2]) && (RegPushButton[2]))  begin
//            else if (PushButton[2]) begin
//                    reg_a <= {`lcd_m,   `lcd_e, `lcd_s,     `lcd_s } ;
//                    reg_b <= {`lcd_a,   `lcd_g,          `lcd_e,         `lcd_blank    } ;
//                    reg_c <= {`lcd_blank,   `lcd_blank,         `lcd_blank,         `lcd_blank    } ;
//                    reg_d <= {`lcd_blank,   `lcd_blank,        `lcd_blank,         `lcd_blank    } ;
                   
                   
//                    reg_e <= {`lcd_r,   `lcd_o,      `lcd_t,         `lcd_a    } ;
//                    reg_f <= {`lcd_t,  `lcd_i,    `lcd_o,         `lcd_n    } ;
//                    reg_g <= {`lcd_blank,   `lcd_blank,         `lcd_blank,         `lcd_blank} ;
//                    reg_h <= {`lcd_blank,  `lcd_blank,     `lcd_blank,        `lcd_blank    } ;
//            end
//            else begin
//                    reg_a <= reg_a;
//                    reg_b <= reg_b;
//                    reg_c <= reg_c ;
//                    reg_d <= reg_d ;
                   
                   
//                    reg_e <= reg_e ;
//                    reg_f <= reg_f ;
//                    reg_g <= reg_g;
//                    reg_h <= reg_h;
//            end
//        end
//    end
   
////assign reg_a = reg_a_2;
//    //assign reg_b = reg_b_2;
//    //assign reg_c = reg_c_2;
//    //assign reg_d = reg_d_2;
//    //assign reg_e = reg_e_2;
//    //assign reg_f = reg_f_2;
//    //assign reg_g = reg_g_2;
//    //assign reg_h = reg_h_2;
   
   
//    //    reg [2:0] RegPushButton;
   
//    //    always @ (posedge lcdclk or negedge resetn)
//    //        begin
//    //          if (!resetn)
//    //          begin
//    //            RegPushButton <= 3'd0;
//    //            reg_a <= {`lcd_2,   `lcd_0, `lcd_2,     `lcd_5 } ;
//    //            reg_b <= {`lcd_blank,   `lcd_e,          `lcd_m,         `lcd_b    } ;
//    //            reg_c <= {`lcd_e,   `lcd_d,         `lcd_d,         `lcd_e    } ;
//    //            reg_d <= {`lcd_d,   `lcd_blank,        `lcd_blank,         `lcd_blank    } ;
               
               
//    //            reg_e <= {`lcd_s,   `lcd_y,      `lcd_s,         `lcd_t    } ;
//    //            reg_f <= {`lcd_e,  `lcd_m,    `lcd_blank,         `lcd_l    } ;
//    //            reg_g <= {`lcd_a,  `lcd_b,  `lcd_blank,   `lcd_blank    } ;
//    //            reg_h <= {`lcd_blank,  `lcd_blank,     `lcd_blank,        `lcd_blank    } ;
   
//    //          end
//    //          else
//    //          begin
//    //            RegPushButton <= PushButton;
//    //            if ((!PushButton[0]) && (RegPushButton[0]))  begin
//    //                reg_a <= reg_d;
//    //                reg_b <= reg_a;
//    //                reg_c <= reg_b ;
//    //                reg_d <= reg_c ;
                   
                   
//    //                reg_e <= reg_h ;
//    //                reg_f <= reg_e ;
//    //                reg_g <= reg_f;
//    //                reg_h <= reg_g;
//    //            end
                                           
//    //            else if ((!PushButton[1]) && (RegPushButton[1]))      begin
//    //                reg_a <= reg_b;
//    //                reg_b <= reg_c;
//    //                reg_c <= reg_d ;
//    //                reg_d <= reg_a ;
                   
                   
//    //                reg_e <= reg_f ;
//    //                reg_f <= reg_g ;
//    //                reg_g <= reg_h;
//    //                reg_h <= reg_e;
//    //            end
                   
//    //            else if ((!PushButton[2]) && (RegPushButton[2]))      begin
//    //                reg_a <= {`lcd_m,   `lcd_e, `lcd_s,     `lcd_s } ;
//    //                reg_b <= {`lcd_a,   `lcd_g,          `lcd_e,         `lcd_blank    } ;
//    //                reg_c <= {`lcd_blank,   `lcd_blank,         `lcd_blank,         `lcd_blank    } ;
//    //                reg_d <= {`lcd_blank,   `lcd_blank,        `lcd_blank,         `lcd_blank    } ;
                   
                   
//    //                reg_e <= {`lcd_r,   `lcd_o,      `lcd_t,         `lcd_a    } ;
//    //                reg_f <= {`lcd_t,  `lcd_i,    `lcd_o,         `lcd_n    } ;
//    //                reg_g <= {`lcd_blank,   `lcd_blank,         `lcd_blank,         `lcd_blank} ;
//    //                reg_h <= {`lcd_blank,  `lcd_blank,     `lcd_blank,        `lcd_blank    } ;
//    //            end
//    //            else begin
//    //                reg_a <= reg_a;
//    //                reg_b <= reg_b;
//    //                reg_c <= reg_c ;
//    //                reg_d <= reg_d ;
                   
                   
//    //                reg_e <= reg_e ;
//    //                reg_f <= reg_f ;
//    //                reg_g <= reg_g;
//    //                reg_h <= reg_h;
//    //            end
   
//    //          end
//    //    end
   
//    // reg [2:0] RegPushButton;
   
//    //    always @ (posedge lcdclk or negedge resetn)
//    //        begin
//    //          if (!resetn)
//    //          begin
//    ////            RegPushButton <= 3'd0;
//    ////            reg_a_2 <= {`lcd_2,   `lcd_0, `lcd_2,     `lcd_5 } ;
//    ////            reg_b_2 <= {`lcd_blank,   `lcd_e,          `lcd_m,         `lcd_b    } ;
//    ////            reg_c_2 <= {`lcd_e,   `lcd_d,         `lcd_d,         `lcd_e    } ;
//    ////            reg_d_2 <= {`lcd_d,   `lcd_blank,        `lcd_blank,         `lcd_blank    } ;
   
//    //            reg_d_2 <= {`lcd_2,   `lcd_0, `lcd_2,     `lcd_5 } ;
//    //            reg_a_2 <= {`lcd_blank,   `lcd_e,          `lcd_m,         `lcd_b    } ;
//    //            reg_b_2 <= {`lcd_e,   `lcd_d,         `lcd_d,         `lcd_e    } ;
//    //            reg_c_2 <= {`lcd_d,   `lcd_blank,        `lcd_blank,         `lcd_blank    } ;
               
               
//    //            reg_h_2 <= {`lcd_s,   `lcd_y,      `lcd_s,         `lcd_t    } ;
//    //            reg_e_2 <= {`lcd_e,  `lcd_m,    `lcd_blank,         `lcd_l    } ;
//    //            reg_f_2 <= {`lcd_a,  `lcd_b,  `lcd_blank,   `lcd_blank    } ;
//    //            reg_g_2 <= {`lcd_blank,  `lcd_blank,     `lcd_blank,        `lcd_blank    } ;
   
//    //          end
//    //          else
//    //          begin
//    ////            RegPushButton <= PushButton;
//    ////            if ((!PushButton[0]) && (RegPushButton[0]))  begin
//    //            if (PushButton[0])  begin
//    //                reg_a_2 <= reg_d_2;
//    //                reg_b_2 <= reg_a_2;
//    //                reg_c_2 <= reg_b_2 ;
//    //                reg_d_2 <= reg_c_2 ;
                   
                   
//    //                reg_e_2 <= reg_h_2 ;
//    //                reg_f_2 <= reg_e_2 ;
//    //                reg_g_2 <= reg_f_2;
//    //                reg_h_2 <= reg_g_2;
//    //            end
                                           
//    //            else if(PushButton[1])  begin
//    //                reg_a_2 <= reg_b_2;
//    //                reg_b_2 <= reg_c_2;
//    //                reg_c_2 <= reg_d_2 ;
//    //                reg_d_2 <= reg_a_2 ;
                   
                   
//    //                reg_e_2 <= reg_f_2 ;
//    //                reg_f_2 <= reg_g_2 ;
//    //                reg_g_2 <= reg_h_2;
//    //                reg_h_2 <= reg_e_2;
//    //            end
                   
//    //            else if (PushButton[2])      begin
//    //                reg_a_2 <= {`lcd_m,   `lcd_e, `lcd_s,     `lcd_s } ;
//    //                reg_b_2 <= {`lcd_a,   `lcd_g,          `lcd_e,         `lcd_blank    } ;
//    //                reg_c_2 <= {`lcd_blank,   `lcd_blank,         `lcd_blank,         `lcd_blank    } ;
//    //                reg_d_2 <= {`lcd_blank,   `lcd_blank,        `lcd_blank,         `lcd_blank    } ;
                   
                   
//    //                reg_e_2 <= {`lcd_r,   `lcd_o,      `lcd_t,         `lcd_a    } ;
//    //                reg_f_2 <= {`lcd_t,  `lcd_i,    `lcd_o,         `lcd_n    } ;
//    //                reg_g_2 <= {`lcd_blank,   `lcd_blank,         `lcd_blank,         `lcd_blank} ;
//    //                reg_h_2 <= {`lcd_blank,  `lcd_blank,     `lcd_blank,        `lcd_blank    } ;
//    //            end
//    //            else begin
//    //                reg_a_2 <= reg_a_2;
//    //                reg_b_2 <= reg_b_2;
//    //                reg_c_2 <= reg_c_2 ;
//    //                reg_d_2 <= reg_d_2 ;
                   
                   
//    //                reg_e_2 <= reg_e_2 ;
//    //                reg_f_2 <= reg_f_2 ;
//    //                reg_g_2 <= reg_g_2;
//    //                reg_h_2 <= reg_h_2;
//    //            end
   
//    //          end
//    //    end

//endmodule

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
PushButton,
lcdclk,
lcd_rs,
lcd_rw,
lcd_en,
lcd_data);


input [2:0] PushButton;     // 버튼 입력 신호
input  resetn;              // 리셋 신호
input  lcdclk;              // LCD 디스플레이의 클록 신호
output lcd_rs;              // 레지스터 선택
output lcd_rw;              // 읽기/쓰기 모드
output lcd_en;              // LCD 활성화 신호
output [7:0] lcd_data;      // LCD로 전송될 8비트 데이터

wire [2:0] PushButton;      // 버튼 입력 신호
reg [31:0] reg_a;            // 1행에 표시할 1 번째 4문자 데이터 (32bit)
reg [31:0] reg_b;            // 1행에 표시할 2 번째 4문자 데이터 (32bit)
reg [31:0] reg_c;            // 1행에 표시할 3 번째 4문자 데이터 (32bit)
reg [31:0] reg_d;            // 1행에 표시할 4 번째 4문자 데이터 (32bit)
reg [31:0] reg_e;            // 2행에 표시할 1 번째 4문자 데이터 (32bit)
reg [31:0] reg_f;            // 2행에 표시할 2 번째 4문자 데이터 (32bit)
reg [31:0] reg_g;            // 2행에 표시할 3 번째 4문자 데이터 (32bit)
reg [31:0] reg_h;            // 2행에 표시할 4 번째 4문자 데이터 (32bit)
reg [23:0] delay_lcdclk;     // clock
reg [15:0] count_lcd;        // LCD 상태 전환을 위한 16bit counter
reg lcd_en;                  // Text-LCD Device enable signal
reg  [4:0] lcd_mode = 0;     // 현재 LCD 동작 모드를 저장하는 5bit register
wire [4:0] mode_pwron = 1 ;  // power on
wire [4:0] mode_fnset = 2 ;  // function set
wire [4:0] mode_onoff = 3 ;  // display on/off Control
wire [4:0] mode_entr1 = 4 ;  //
wire [4:0] mode_entr2 = 5 ;  //
wire [4:0] mode_entr3 = 6 ;  //
wire [4:0] mode_seta1 = 7 ;  // set addr 1st line
wire [4:0] mode_wr1st = 8 ;  // Write 1st line
wire [4:0] mode_seta2 = 9 ;  // set addr 2nd line
wire [4:0] mode_wr2nd = 10;  // Write 2nd line
wire [4:0] mode_delay = 11;  // dealy
wire [4:0] mode_actcm = 12;  // user command
reg [9:0] set_data;          // rs, rw, DB0 ~ 7신호


//reg_a = {`lcd_2,   `lcd_0, `lcd_2,     `lcd_5 } ;
//reg_b = {`lcd_blank,   `lcd_e,       `lcd_m,    `lcd_b } ;
//reg_c = {`lcd_e,   `lcd_d,         `lcd_d,    `lcd_e } ;
//reg_d = {`lcd_d,   `lcd_blank,        `lcd_blank,    `lcd_blank } ;


//reg_e = {`lcd_s,   `lcd_y,      `lcd_s,    `lcd_t } ;
//reg_f = {`lcd_e,  `lcd_m,    `lcd_blank,    `lcd_l } ;
//reg_g = {`lcd_a,  `lcd_b,  `lcd_blank,   `lcd_blank } ;
//reg_h = {`lcd_blank,  `lcd_blank,     `lcd_blank,   `lcd_blank } ;

////////////////////////////////////////////////////////////////////////////////////
//    // Counterx2000 & //Counterx40
//    always @(posedge lcdclk)
//    begin
//        if (resetn == 0)
//        begin
//            delay_lcdclk <= 0;
//            count_lcd <= 0;
//            lcd_en <= 1'b0;
//        end
//        else
//        begin
//            // Counterx2000
//            if (delay_lcdclk < 5) ///// Simulation /////
//                delay_lcdclk <=  delay_lcdclk + 1;
//            else
//                delay_lcdclk <= 0;
   
//            // Counterx40
//            if (delay_lcdclk == 0)
//            begin
//                if (count_lcd < 40)
//                    count_lcd <= count_lcd + 1;
//                else
//                    count_lcd <= 6;
//            end
       
//            if (delay_lcdclk == 1) ///// Simulation /////
//                lcd_en <= 1'b1;
//            else if (delay_lcdclk == 5) ///// Simulation /////
//                lcd_en <= 1'b0;
               
//        end
//    end
////////////////////////////////////////////////////////////////////////////////////

// Counterx2000 & //Counterx40
always @(posedge lcdclk)
begin
if (resetn == 0)
begin
            delay_lcdclk <= 0;
            count_lcd <= 0;
            lcd_en <= 1'b0;
        end
        else
        begin
            // Counterx2000
            if (delay_lcdclk < 1999)
                delay_lcdclk <=  delay_lcdclk + 1;
            else
                delay_lcdclk <= 0;
   
            // Counterx40
            if (delay_lcdclk == 0)
            begin
                if (count_lcd < 40)
                    count_lcd <= count_lcd + 1;
                else
                    count_lcd <= 6;
                end
       
            if (delay_lcdclk == 200)
                lcd_en <= 1'b1;
            else if (delay_lcdclk == 1800)
                lcd_en <= 1'b0;
               
            end
    end
   
    // decoder switch
    always @(posedge lcdclk)
    begin
        if (resetn == 0)
            lcd_mode <= mode_pwron;
        else
            begin
                case (count_lcd)
                    0  : lcd_mode <= mode_pwron ;
                    1  : lcd_mode <= mode_fnset ;
                    2  : lcd_mode <= mode_onoff ;
                    3  : lcd_mode <= mode_entr1 ;
                    4  : lcd_mode <= mode_entr2 ;
                    5  : lcd_mode <= mode_entr3 ;
                    6  : lcd_mode <= mode_seta1 ;
                    7  : lcd_mode <= mode_wr1st ;
                    23 : lcd_mode <= mode_seta2 ;
                    24 : lcd_mode <= mode_wr2nd ;
                    40 : lcd_mode <= mode_delay ;
                    41 : lcd_mode <= mode_actcm ;
                    default : begin end
                endcase    
            end
    end
   
    assign lcd_rs = set_data[9];
    assign lcd_rw = set_data[8];
    assign lcd_data = set_data[7:0];
   
    // decoder output
    always @(lcdclk or lcd_mode or count_lcd)
    begin
        if (resetn == 0)
            set_data <= 10'b0000000000;
        else
            begin
            case (lcd_mode)
                mode_pwron : set_data <= {2'b00, 8'h38};
                mode_fnset : set_data <= {2'b00, 8'h38};
                mode_onoff : set_data <= {2'b00, 8'h0e};
                mode_entr1 : set_data <= {2'b00, 8'h06};
                mode_entr2 : set_data <= {2'b00, 8'h02};
                mode_entr3 : set_data <= {2'b00, 8'h01};                
                mode_seta1 : set_data <= {2'b00, 8'h80};
                mode_wr1st :
                begin
                case (count_lcd)
                 7 : set_data <= {1'b1, 1'b0, reg_a[31:24]};
                 8 : set_data <= {1'b1, 1'b0, reg_a[23:16]};
                 9 : set_data <= {1'b1, 1'b0, reg_a[15: 8]};
                10 : set_data <= {1'b1, 1'b0, reg_a[7 : 0]};
                11 : set_data <= {1'b1, 1'b0, reg_b[31:24]};
                12 : set_data <= {1'b1, 1'b0, reg_b[23:16]};
                13 : set_data <= {1'b1, 1'b0, reg_b[15: 8]};
                14 : set_data <= {1'b1, 1'b0, reg_b[7 : 0]};
                15 : set_data <= {1'b1, 1'b0, reg_c[31:24]};
                16 : set_data <= {1'b1, 1'b0, reg_c[23:16]};
                17 : set_data <= {1'b1, 1'b0, reg_c[15: 8]};
                18 : set_data <= {1'b1, 1'b0, reg_c[7 : 0]};
                19 : set_data <= {1'b1, 1'b0, reg_d[31:24]};
                20 : set_data <= {1'b1, 1'b0, reg_d[23:16]};
                21 : set_data <= {1'b1, 1'b0, reg_d[15: 8]};
                22 : set_data <= {1'b1, 1'b0, reg_d[7 : 0]};
                endcase
            end
               mode_seta2 : set_data <= {2'b00, 8'hc0};
               mode_wr2nd :
                begin
                case (count_lcd)
                24 : set_data <= {1'b1, 1'b0, reg_e[31:24]};
                25 : set_data <= {1'b1, 1'b0, reg_e[23:16]};
                26 : set_data <= {1'b1, 1'b0, reg_e[15: 8]};
                27 : set_data <= {1'b1, 1'b0, reg_e[7 : 0]};
                28 : set_data <= {1'b1, 1'b0, reg_f[31:24]};
                29 : set_data <= {1'b1, 1'b0, reg_f[23:16]};
                30 : set_data <= {1'b1, 1'b0, reg_f[15: 8]};
                31 : set_data <= {1'b1, 1'b0, reg_f[7 : 0]};
                32 : set_data <= {1'b1, 1'b0, reg_g[31:24]};
                33 : set_data <= {1'b1, 1'b0, reg_g[23:16]};
                34 : set_data <= {1'b1, 1'b0, reg_g[15: 8]};
                35 : set_data <= {1'b1, 1'b0, reg_g[7 : 0]};
                36 : set_data <= {1'b1, 1'b0, reg_h[31:24]};
                37 : set_data <= {1'b1, 1'b0, reg_h[23:16]};
                38 : set_data <= {1'b1, 1'b0, reg_h[15: 8]};
                39 : set_data <= {1'b1, 1'b0, reg_h[7 : 0]};
                endcase
            end
                mode_delay : set_data <= {2'b00, 8'h02};
                mode_actcm : set_data <= {2'b00, 8'h02};
                default : begin end
            endcase
        end
    end

    reg [2:0] RegPushButton; // 1cycle 동안 Button 입력 저장하는 reg, Button을 누르고 뗀 그 순간에만 조건문을 작동할 수 있게 함
    always @ (posedge lcdclk or negedge resetn) begin
          if (!resetn) begin                                            // Active Low reset 시 
            RegPushButton <= 3'd0;                                      // RegPushButton 초기화
            reg_a <= {`lcd_2, `lcd_0, `lcd_2, `lcd_5 };                 // reg_a = "2025"
            reg_b <= {`lcd_blank, `lcd_e, `lcd_m, `lcd_b };             // reg_b = " emb"
            reg_c <= {`lcd_e, `lcd_d, `lcd_d, `lcd_e };                 // reg_c = "edde"
            reg_d <= {`lcd_d, `lcd_blank, `lcd_blank, `lcd_blank };     // reg_d = "d   "
           
            reg_e <= {`lcd_s, `lcd_y, `lcd_s, `lcd_t };                 // reg_e = "syst"
            reg_f <= {`lcd_e,  `lcd_m,    `lcd_blank, `lcd_l };         // reg_f = "em l"
            reg_g <= {`lcd_a, `lcd_b, `lcd_blank, `lcd_blank };         // reg_g = "ab  "
            reg_h <= {`lcd_blank, `lcd_blank, `lcd_blank, `lcd_blank};  // reg_h = "    "
          end
          else begin
            RegPushButton <= PushButton;                                // 1cycle 동안 Button 입력 저장하는 reg에 현재 입력 저장 
            if ((!PushButton[0]) && (RegPushButton[0]))  begin          // Button 0을 누르고 뗀 직후
                reg_a <= reg_d;                                         // reg_a ~ reg_d를 right shift
                reg_b <= reg_a;
                reg_c <= reg_b ;
                reg_d <= reg_c ;
               
               
                reg_e <= reg_h ;                                        // reg_e ~ reg_h를 right shift 
                reg_f <= reg_e ;
                reg_g <= reg_f;
                reg_h <= reg_g;
            end
                                       
            else if ((!PushButton[1]) && (RegPushButton[1])) begin      // Button 1을 누르고 뗀 직후
                reg_a <= reg_b;                                         // reg_a ~ reg_d를 left shift
                reg_b <= reg_c;
                reg_c <= reg_d ;
                reg_d <= reg_a ;
               
               
                reg_e <= reg_f ;                                        // reg_f ~ reg_h를 left shift
                reg_f <= reg_g ;
                reg_g <= reg_h;
                reg_h <= reg_e;
            end
            else if ((!PushButton[2]) && (RegPushButton[2])) begin          // Button 2를 누르고 뗀 직후, reg에 저장된 값 변경 
                reg_a <= {`lcd_m, `lcd_e, `lcd_s, `lcd_s };                 // reg_a = "mess"
                reg_b <= {`lcd_a, `lcd_g, `lcd_e, `lcd_blank };             // reg_b = "age "
                reg_c <= {`lcd_blank, `lcd_blank, `lcd_blank, `lcd_blank }; // reg_c = "    "
                reg_d <= {`lcd_blank, `lcd_blank, `lcd_blank, `lcd_blank} ; // reg_d = "    "
               
                reg_e <= {`lcd_r, `lcd_o, `lcd_t, `lcd_a };                 // reg_e = "rota"
                reg_f <= {`lcd_t, `lcd_i, `lcd_o, `lcd_n };                 // reg_f = "tion"
                reg_g <= {`lcd_blank, `lcd_blank, `lcd_blank, `lcd_blank }; // reg_g = "    "
                reg_h <= {`lcd_blank, `lcd_blank, `lcd_blank, `lcd_blank }; // reg_h = "    "
            end
            else begin              // 별다른 입력 없으면 reg_a~reg_h 값 유지
                reg_a <= reg_a;
                reg_b <= reg_b;
                reg_c <= reg_c ;
                reg_d <= reg_d ;
               
               
                reg_e <= reg_e ;
                reg_f <= reg_f ;
                reg_g <= reg_g;
                reg_h <= reg_h;
            end
          end
        end
endmodule