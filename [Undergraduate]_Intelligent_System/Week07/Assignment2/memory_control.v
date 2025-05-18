`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/11/11 00:04:01
// Design Name: 
// Module Name: fsm_control
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


module memory_control#(
    parameter MEMORY_DEPTH = 100
)
(

    input   wire    rst,
    input   wire    clk,
    input   wire    rx_switch,
    input   wire    tx_switch,
    output  wire    led,
    
    //uart_rx signal
    input   wire    [7:0]   uart_rx_data,
    input   wire    uart_rx_ready,
    
    //uart_tx signal
    output  reg     [7:0]   uart_tx_data,
    input   wire    uart_tx_pop,
    output  wire    uart_tx_on,    
    
    //memory signal
    output  wire    memory_en,
    output  wire    memory_write_en,
    output  reg     [14:0]  memory_addr,
    output  wire    [7:0]   memory_data_in,
    input   wire    [7:0]   memory_data_out
    );
    

    //Write Down Your Code
    // FSM State
    localparam 
        IDLE  = 2'b00,
        RX    = 2'b01,
        WAIT  = 2'b10,
        TX    = 2'b11;

    // registers for FSM state
    reg [1:0] state;                // FSM state
    reg [14:0] addr_counter;        // BRAM address counter, 2^15 = 32768

    reg reg_led;                    // LED control
    reg reg_uart_tx_on;             // UART TX control

    reg reg_memory_en;              // BRAM enable signal
    reg reg_memory_write_en;        // BRAM write enable signal
    reg [7:0] reg_memory_data_in;   // BRAM data input
    
    wire uart_rx_ready_edge;
    wire uart_tx_pop_edge;

    assign led = reg_led;           
    assign uart_tx_on = reg_uart_tx_on;

    assign memory_en = reg_memory_en;
    assign memory_write_en = reg_memory_write_en;
    assign memory_data_in = reg_memory_data_in;

    // fsm
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            reg_led <= 1'b0;
            reg_uart_tx_on <= 1'b0;
        end
        else begin
            case (state)
                IDLE: begin
                    reg_led <= 1'b0;
                    reg_uart_tx_on <= 1'b0;
                    if (rx_switch) begin
                        state <= RX;
                    end
                end
                
                RX: begin
                    // when the memory is full, wait for the tx_switch to be pressed
                    if (addr_counter == MEMORY_DEPTH) begin //  && uart_rx_ready
                        state <= WAIT;
                    end
                end
                
                WAIT: begin
                    reg_led <= 1'b1;
                    if (tx_switch) begin        // tx_switch pressed
                        state <= TX;            // start transmitting
                    end
                end
                
                TX: begin
                    reg_uart_tx_on <= 1'b1; // turn on the UART TX
                    // when the memory is empty, go back to IDLE
                    if (addr_counter == MEMORY_DEPTH && uart_tx_pop) begin
                        state <= IDLE;
                        reg_uart_tx_on <= 1'b0; // turn off the UART TX
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end

    // "address counter"
    // increment the address counter when receiving data or transmitting data
    // when the state is RX, increment the address counter when uart_rx_ready is high
    // when the state is TX, increment the address counter when uart_tx_pop is high
    // else, reset the address counter to 0, it is important!!
    // since, TX needs to read from address 0
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            addr_counter <= 15'b0;
        end
        else begin
            case (state)
                RX: begin
                    if (uart_rx_ready_edge) begin
                        addr_counter <= addr_counter + 1;
                    end
                end
                
                TX: begin
                    if (uart_tx_pop_edge) begin
                        addr_counter <= addr_counter + 1;
                    end
                end              
                default: addr_counter <= 15'b0;
            endcase
        end
    end

    // BRAM control, "address"
    // when the state is RX, write data to the BRAM
    // when the state is TX, read data from the BRAM
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            reg_memory_en <= 1'b0;
            reg_memory_write_en <= 1'b0;
            memory_addr <= 15'b0;
            reg_memory_data_in <= 8'b0;
        end
        else begin
            reg_memory_en <= 1'b1;  // BRAM enable signal always high
            case (state)
                RX: begin
                    reg_memory_write_en <= uart_rx_ready_edge;  // write enable signal
                    memory_addr <= addr_counter;                // BRAM address
                    if(uart_rx_ready_edge)
                        reg_memory_data_in <= uart_rx_data;     // BRAM data input
                end
                
                TX: begin
                    reg_memory_write_en <= 1'b0;    // write enable signal, TX reads data
                    memory_addr <= addr_counter;    // BRAM address
                end
                
                default: begin
//                    reg_memory_write_en <= 1'b0;
//                    memory_addr <= 15'b0;
                    reg_memory_write_en <= reg_memory_write_en;
                    memory_addr <= memory_addr;
                end
            endcase
        end
    end

    // UART_TX "data"
    // when the state is TX, send data from the BRAM to the UART
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            uart_tx_data <= 8'b0;
        end
        else if (state == TX) begin
            uart_tx_data <= memory_data_out;  // BRAM data output
        end
        else begin
            uart_tx_data <= 8'b0;
        end
    end
    
    // use posedge_detector to detect the rising edge of uart_rx_ready and uart_tx_pop
    posedge_detector Uposedge_detector_0 (
    .clk(clk),
    .rst(rst),
    .in(uart_rx_ready),
    .out(uart_rx_ready_edge)
    );
    
    posedge_detector Uposedge_detector_1 (
    .clk(clk),
    .rst(rst),
    .in(uart_tx_pop),
    .out(uart_tx_pop_edge)
    );
    /////////////////////////////////////////////////////////
    
endmodule
