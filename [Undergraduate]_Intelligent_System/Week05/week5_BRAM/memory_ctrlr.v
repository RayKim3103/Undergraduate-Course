`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/03/28 16:17:49
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


module memory_ctrlr (
  input wire clk,           // input clk port
  input wire resetn,        // input reset port

  input wire start,         // input start signal port
  output wire done,         // output done signal port
  
  // Port A로 내보내는 signal, 이를 통해 BRAM에 write
  output wire ena,          // Port A enable signal
  output wire wea,          // Port A write enable signal
  output wire [7:0] addra,  // Port A address
  output wire [15:0] dina,  // Port A input data
    
  // Port B로 내보내는 signal, 이를 통해 BRAM에서 read
  output wire enb,          // Port B enable signal
  output wire [7:0] addrb,  // Port B address
  input wire [15:0] doutb   // Port B output data
);

//To Do
    /*** READnWRITE / WRITE / IDLE / DONE state ***/
    localparam IDLE         = 2'b00,        // IDLE: initial state
               READnWRITE   = 2'b01,        // BRAM read & write concurrently
               WRITE        = 2'b10,        // BRAM write
               DONE         = 2'b11;        // Make DONE flag high, initialize ctrl signals

    /*** Read Delay: 2 cycle, Accumulating: 1cycle => total 3cycle delay ***/
    localparam ReadDelay    = 2'd3;
    
    /*** MaxAddress of BRAM: 255 (since, bit width:16 & mem depth: 256) ***/
    localparam MaxAddr      = 8'd255;

    /*** registers for state transition ***/
    reg [1:0] state, next_state;    // current state & next state
    reg [8:0] cnt;                  // address counter
    
    /*** register for storing data ***/
    reg [15:0] accum;               // store accumulated data

    /*** registers which will be assign to output signal port (wire) ***/
    reg reg_done;
    reg reg_ena, reg_enb, reg_wea;
    reg [7:0] reg_addra;
    reg [7:0] reg_addrb;
    
    assign done = reg_done;
    assign ena = reg_ena;
    assign enb = reg_enb;
    assign wea = reg_wea;
    assign addra = reg_addra;
    assign addrb = reg_addrb;
    assign dina = accum;
    
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
            IDLE:           next_state = (start) ? READnWRITE : IDLE;
            READnWRITE:     next_state = (cnt == MaxAddr) ? WRITE : READnWRITE;
            WRITE:          next_state = (cnt- ReadDelay == MaxAddr) ? DONE : WRITE;
            default:        next_state = IDLE;
        endcase
    end
    
    reg enb_d1;
    reg read_valid;
    
    // 2-cycle read latency 시 read valid flag 생성
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            enb_d1 <= 0;
            read_valid <= 0;
        end else begin
            enb_d1 <= reg_enb;       // enb: read request
            read_valid <= enb_d1;    // read_valid: doutb가 출력되었을 때 1됨
        end
    end

    
    /*** accumulating doutb values every cycle ***/
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin      // when reset, initialize all the regs
            accum <= 0;
        end
        else if (read_valid)        // read를 했을 때만, 지연을 계산하여 accumulate
            accum <= accum + doutb;
        else 
            accum <= accum;
    end
    
    // Operations corresponding to it's state
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin      // when reset, initialize all the regs
            cnt <= 0;
            reg_done <= 0;
            reg_ena <= 0;
            reg_enb <= 0;
            reg_wea <= 0;
            reg_addra <= 0;
            reg_addrb <= 0;
            reg_done <= 0;
        end 
        else begin
            case (state)
                IDLE: begin
// when start is High, BRAM Port B read data from address 0 (cnt: 0)
                    if(start == 1) begin
                        reg_enb <= 1;
                        reg_wea <= 0;
                        reg_addrb <= cnt;
                        cnt <= cnt + 1;
                    end
// when start is Low, BRAM ctrlr does not operate 
                    else begin
                        reg_enb <= 0;
                        reg_ena <= 0;
                        reg_wea <= 0;
                    end
                end
                READnWRITE: begin
                    reg_enb <= 1;     // BRAM Port B should read
                    reg_addrb <= cnt; // BRAM read address = address counter
                    
                    // after ReadDelay, Port A writes & PortB reads concurrently
                    if(cnt - ReadDelay < MaxAddr)  begin
                        reg_ena <= 1;     // BRAM Port A should write
                        reg_wea <= 1;     // port A needs to write BRAM
                        // address must start from 0 (subtract ReadDelay)   
                        reg_addra <= cnt - ReadDelay;  
                    end
                    
                    cnt <= cnt+1;   // Increase address counter
                end
                
                WRITE: begin
                    reg_enb <= 0;   // BRAM port B don't have to read (already read all)
                    if(cnt - ReadDelay <= MaxAddr)  begin
                        reg_ena <= 1;     // BRAM Port A should write
                        reg_wea <= 1;     // port A needs to write BRAM
                        // address must start from 0 (subtract ReadDelay
                        reg_addra <= cnt - ReadDelay;
                        cnt <= cnt+1;   // Increase address counter
                    end
                end
                
                DONE: begin
                // Raise Done Flag & initialize control signals of BRAM port A, B
                    reg_done <= 1; // make DONE flag High
                    reg_ena     <= 0;
                    reg_enb     <= 0;
                    reg_wea     <= 0;
                end
                
            endcase
        end
    end
endmodule