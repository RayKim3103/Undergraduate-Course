`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/27 15:02:20
// Design Name: 
// Module Name: memory_ctrlr
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


module memory_ctrlr #(
    // BW: Bit Width
    // AMAX: Max number of Address line (memory depth¸¦ ÀÇ¹Ì)
    // ADR: Number of bits to represent address
    parameter SRAM1_BW = 16,
    parameter SRAM1_AMAX = 256,
    parameter SRAM1_ADR = $clog2(SRAM1_AMAX),
    parameter SRAM2_BW = 32,
    parameter SRAM2_AMAX = 192,
    parameter SRAM2_ADR = $clog2(SRAM2_AMAX)
    )(
    input  wire clk,
    input  wire resetn,
    input  wire start,
    output wire done,

    output wire s1_en,  // SRAM 1 enable signal port
    output wire s2_en,  // SRAM 2 enable signal port
    output wire s1_we,  // SRAM 1 write enable signal port
    output wire s2_we,  // SRAM 2 write enable signal port
    output wire [SRAM1_ADR-1:0] s1_addr, // SRAM 1 read address
    output wire [SRAM2_ADR-1:0] s2_addr, // SRAM 2 write address
    input  wire [SRAM1_BW-1:0] s1_dout,  // SRAM 1 read data
    output wire [SRAM2_BW-1:0] s2_din    // SRAM 2 write data
    );


    //To Do
    /*** Read / Write / IDLE state ***/
    localparam IDLE  = 2'b00,       // IDLE: initial state
               READnWRITE = 2'b01,  // SRAM 1 read & SRAM 2 write concurrently
               WRITEnDONE = 2'b10,  // SRAM 2 write & make DONE flag High
               DONE  = 2'b11;       // If DONE flag is high, initialize ctrl signals
    /*** Read Delay: 1 cycle ***/
    localparam ReadDelay    = 1'd1;
    
    /*** registers for state transition ***/
    reg [1:0] state, next_state;    // current state & next state
    reg [7:0] cnt;                  // address counter
    
    /*** register for storing data ***/
    reg [SRAM1_BW-1:0] temp_data;   // temporary data for Concatenating data 

    /*** registers which will be assign to output signal port (wire) ***/
// when SRAM read / write opeartion is done: High
    reg reg_done;
// store SRAM 1 & SRAM 2 enable and write enable signals                                   
    reg reg_s1_en, reg_s2_en, reg_s1_we, reg_s2_we;
// Store SRAM 1 address (need for reading)
    reg [SRAM1_ADR-1:0] reg_s1_addr;
// Store SRAM 2 address (need for writing)
    reg [SRAM2_ADR-1:0] reg_s2_addr;
// The data read from SRAM 1 and give it to SRAM 2
    reg [SRAM2_BW-1:0] reg_s2_din;

    /*** assign output signals to corressponding registers ***/
    assign done    = reg_done;
    assign s1_en   = reg_s1_en;
    assign s2_en   = reg_s2_en;
    assign s1_we   = reg_s1_we;
    assign s2_we   = reg_s2_we;
    assign s1_addr = reg_s1_addr;
    assign s2_addr = reg_s2_addr;
    assign s2_din  = reg_s2_din;

    /*** FSM state transition ***/
    always @(posedge clk or negedge resetn) begin
        if (!resetn)    // when reset state goes to IDLE
            state <= IDLE;
        else            // when posedge clk, state goes to next state
            state <= next_state;
    end

    /*** Determine next state corresponding to current state ***/
    always @(*) begin
        case (state)
// IDLE goes to READnWRITE state when start signal goes High
            IDLE:           next_state = (start) ? READnWRITE : IDLE;
// READnWRITE state is kept until It reads the last address of SRAM 1
            READnWRITE:     next_state = (cnt == SRAM1_AMAX - 1) ? WRITEnDONE : READnWRITE;
// WRITEnDONE state write the last address data of SRAM 1 to SRAM 2, and make DONE flag High
            WRITEnDONE:     next_state = DONE;
            default:        next_state = IDLE;
        endcase
    end

    // Operations corresponding to it's state
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin      // when reset, initialize all the regs
            cnt         <= 0;
            temp_data   <= 0;
            reg_done      <= 0;
            reg_s1_en     <= 0;
            reg_s2_en     <= 0;
            reg_s1_we     <= 0;
            reg_s2_we     <= 0;
            reg_s1_addr   <= 0;
            reg_s2_addr   <= 0;
            reg_s2_din    <= 0;
        end 
        else begin
            case (state)
                IDLE: begin
// when start is High, SRAM 1 read data from address 0 (cnt: 0)
                    if(start == 1) begin
                        reg_s1_en <= 1;
                        reg_s1_we <= 0;
                        reg_s1_addr <= cnt;
                        cnt <= cnt + 1;
                    end
// when start is Low, SRAM ctrlr does not operate 
                    else begin
                        reg_s1_en <= 0;
                        reg_s2_en <= 0;
                        reg_s1_we <= 0;
                        reg_s2_we <= 0;
                    end
                end
                READnWRITE: begin
// address 0x00 ~ 0x7F ¡æ concatenate 16bit * 2 = 32bit
                    reg_s1_en <= 1;     // SRAM 1 should operate
                    reg_s2_en <= 1;     // SRAM 2 should operate
                    
                    reg_s1_we <= 0;     // needs to read from SRAM 1
                    reg_s2_we <= 1;     // needs to write SRAM 2
                    
                    reg_s1_addr <= cnt; // SRAM 1 read address = address counter
                    
// address 0 ~127 of SRAM 1 should be concatenate and store in address 0xBF~0x80 of SRAM 2 
                    if (cnt < 128) begin
// when cnt is odd number than we are reading odd address of SRAM1
// so we need to store the previous read value in temp for concatenation
                        if (cnt[0] == 1) begin
                            temp_data <= s1_dout;   // store previous read data
                            cnt <= cnt + 1;         // Increase address counter
                        end 
                        else begin
// store the read data from Highest address of SRAM 2
// As SRAM 2 BitWidth = 32, SRAM 1 BitWidth = 16, (cnt - ReadDelay) should be divided by 2
                            reg_s2_addr <= 191 - ((cnt - ReadDelay) >> 1); // set address to be written 
                            reg_s2_din <= {temp_data, s1_dout};  // set data to be written
                            cnt <= cnt + 1;                      // increase address counter
                        end
                    end 
                    else begin
// address 128 ~255 of SRAM 1 should use Zero padding 
// -> even address: 16bit from MSB = Zero
// -> odd address: 16bit from LSB = Zero 
// stored in address 0x80 ~ 0x00 of SRAM 2
                        reg_s2_addr <= (cnt - ReadDelay) - 128;
                        if (cnt[0] == 1)
                            reg_s2_din <= {16'd0, s1_dout}; // even ¡æ Zero Padditng from MSB
                        else
                            reg_s2_din <= {s1_dout, 16'd0}; // odd ¡æ Zero Padditng from LSB
                        cnt <= cnt + 1;                     // Increase address counter
                    end
                end
                
                WRITEnDONE: begin
                    
                    reg_s1_en <= 0; // SRAM 1 don't have to operate
                    reg_s2_en <= 1; // SRAM 2 should operate
                    reg_s2_we <= 1; // needs to write SRAM 2
                    
                    reg_s2_addr <= (cnt - ReadDelay) - 128;  // stored in address 0x80 in this simulation 
                    
                    if (cnt[0] == 1)
                        reg_s2_din <= {16'd0, s1_dout}; // even ¡æ Zero Padditng from MSB
                    else
                        reg_s2_din <= {s1_dout, 16'd0}; // odd ¡æ Zero Padditng from LSB
                    cnt <= cnt + 1;                     // Increase address counter
                    reg_done <= 1;                      // make DONE flag High
                end
                
                DONE: begin
// initialize Done Flag & control signals of SRAM 1, 2
                    reg_done      <= 0;
                    reg_s1_en     <= 0;
                    reg_s2_en     <= 0;
                    reg_s1_we     <= 0;
                    reg_s2_we     <= 0;
                end
                
            endcase
        end
    end
endmodule
