`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/11/11 02:38:11
// Design Name: 
// Module Name: tester_loopback
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


//module tester_loopback(
//    input   clk,
//    input   rst,
//    input   rx_switch,
//    input   tx_switch,
//    input   uart_rx,
//    output  uart_tx
//    );
    
//    parameter   BAUD_RATE = 115_200;
//    parameter   CLOCK_RATE = 50_000_000;
//    parameter   MEMORY_DEPTH = 16384;
    
//    wire rst_sync;
//    //************************************************************
//    //read mem dump
//    reg [7:0] read_mem [MEMORY_DEPTH-1 : 0];
//    integer i;
//    initial begin
//        for(i=0; i<MEMORY_DEPTH ; i =i+1) begin
//            read_mem[i] = i+1;
//        end
//        //readmemb("YOUR Lenna.txt PATH", read_mem); //use for fill image translation
//    end
    
//    wire uart_tx_pop;
//    wire [7:0] uart_tx_data ;
//    reg [13:0] read_addr;
    
//    always @(posedge clk, negedge rst) begin
//        if(rst) read_addr = 14'd0;
//        else begin
//            if(tx_switch) begin
//                if(uart_tx_pop && read_addr <MEMORY_DEPTH) begin 
//                    read_addr <= read_addr+1;
                   
//                end
//                else read_addr <= read_addr;
//            end
//            else read_addr <= read_addr;
//        end
//    end
    
//    always @(read_addr) begin
//        $display("Tester transmit %h to on-Chip memory", uart_tx_data);
//    end
    
//    assign uart_tx_data = read_mem[read_addr];
//    //************************************************************
//    //write mem dump
//    reg [7:0] write_mem [MEMORY_DEPTH-1 : 0];
//    reg [13:0] write_addr;
//    wire uart_rx_ready;
//    wire rx_ready_edge;
//    wire [7:0] uart_rx_data;
    
//    always @(posedge clk, negedge rst) begin
//        if(rst) write_addr = 14'd0;
//        else begin
//            if(rx_switch) begin
//                if(rx_ready_edge && uart_rx_data != 8'd0 &&  write_addr <MEMORY_DEPTH+1) write_addr <= write_addr+1;
//                else write_addr <= write_addr;
//            end
//            else write_addr = 14'd0;
//        end
//    end 
    
//    always @(posedge clk) begin
//        if(rx_switch) begin
//            if(rx_ready_edge && uart_rx_data != 8'd0 &&  write_addr <MEMORY_DEPTH+1) begin
//                $display("Tester Reciesive %h from on-chip Memory", uart_rx_data );
//                if(uart_rx_data == 10) $finish;
//            end
//        end
//    end 
    
//    negedge_detector nd(
//    .clk(clk),
//    .rst(rst),
//    .in(uart_rx_ready),
//    .out(rx_ready_edge)
//    );
//   //************************************************************
    
//    reset_bridge rst_bridge_0(
//        .clk_dst(clk),      // Destination clock
//        .rst_in(rst),       // Asynchronous reset signal
//        .rst_dst(rst_sync)       // Synchronized reset signal
//    );
    
//    uart_rx #(
//        .BAUD_RATE(BAUD_RATE),
//        .CLOCK_RATE(CLOCK_RATE)
//    )   uart_rx_0(
//        .clk_rx(clk),       // Clock input
//        .rst_clk_rx(rst_sync),   // Active HIGH reset - synchronous to clk_rx
//        .rxd_i(uart_rx),        // RS232 RXD pin - Directly from pad
//        .rx_data(uart_rx_data),      // 8 bit data output
//        .rx_data_rdy(uart_rx_ready)  // Ready signal for rx_data
//    );
    
//    uart_tx #(
//        .BAUD_RATE(BAUD_RATE),
//        .CLOCK_RATE(CLOCK_RATE)
//    )   uart_tx_0(
//        .clk_tx(clk),          // Clock input
//        .rst_clk_tx(rst_sync),      // Active HIGH reset - synchronous to clk_tx
       
//        .char_fifo_empty(1'b0), // Empty signal from char FIFO (FWFT)
//        .char_fifo_dout(uart_tx_data),  // Data from the char FIFO
//        .char_fifo_rd_en(uart_tx_pop), // Pop signal to the char FIFO
       
//        .txd_tx(uart_tx)           // The transmit serial signal
//    );    
//endmodule

module tester_loopback(
    input   clk,
    input   rst,
    input   rx_switch,
    input   tx_switch,
    input   uart_rx,
    output  uart_tx
    );
    
    parameter   BAUD_RATE = 115_200;
    parameter   CLOCK_RATE = 50_000_000;
    parameter   MEMORY_DEPTH = 16384;
    
    wire rst_sync;
    ///////////////////////////////////
    wire clk_50MHz;
    ///////////////////////////////////
    //************************************************************
    //read mem dump
    reg [7:0] read_mem [MEMORY_DEPTH-1 : 0];
    integer i;
    initial begin
        for(i=0; i<MEMORY_DEPTH ; i =i+1) begin
            read_mem[i] = i+1;
        end
        //readmemb("YOUR Lenna.txt PATH", read_mem); //use for fill image translation
    end
    
    wire uart_tx_pop;
    wire [7:0] uart_tx_data ;
    reg [13:0] read_addr;
    
    always @(posedge clk_50MHz, posedge rst) begin
        if(rst) read_addr <= 14'd0;
        else begin
            if(tx_switch) begin
                if(uart_tx_pop && read_addr <MEMORY_DEPTH) begin 
                    read_addr <= read_addr+1;
                   
                end
                else read_addr <= read_addr;
            end
            else read_addr <= read_addr;
        end
    end
    
    always @(read_addr) begin
        $display("Tester transmit %h to on-Chip memory", uart_tx_data);
    end
    
    assign uart_tx_data = read_mem[read_addr];
    //************************************************************
    //write mem dump
    reg [7:0] write_mem [MEMORY_DEPTH-1 : 0];
    reg [13:0] write_addr;
    wire uart_rx_ready;
    wire rx_ready_edge;
    wire [7:0] uart_rx_data;
    
    always @(posedge clk_50MHz, posedge rst) begin
        if(rst) write_addr <= 14'd0;
        else begin
            if(rx_switch) begin
                if(rx_ready_edge && uart_rx_data != 8'd0 &&  write_addr <MEMORY_DEPTH+1) write_addr <= write_addr+1;
                else write_addr <= write_addr;
            end
            else write_addr <= 14'd0;
        end
    end 
    
    always @(posedge clk_50MHz) begin
        if(rx_switch) begin
            if(rx_ready_edge && uart_rx_data != 8'd0 &&  write_addr <MEMORY_DEPTH+1) begin
                $display("Tester Reciesive %h from on-chip Memory", uart_rx_data );
                if(uart_rx_data == 10) $finish;
            end
        end
    end 
    
    negedge_detector nd(
    .clk(clk_50MHz),
    .rst(rst),
    .in(uart_rx_ready),
    .out(rx_ready_edge)
    );
   //************************************************************
    
    reset_bridge rst_bridge_0(
        .clk_dst(clk_50MHz),      // Destination clock
        .rst_in(rst),       // Asynchronous reset signal
        .rst_dst(rst_sync)       // Synchronized reset signal
    );
    
    uart_rx #(
        .BAUD_RATE(BAUD_RATE),
        .CLOCK_RATE(CLOCK_RATE)
    )   uart_rx_0(
        .clk_rx(clk_50MHz),       // Clock input
        .rst_clk_rx(rst_sync),   // Active HIGH reset - synchronous to clk_rx
        .rxd_i(uart_rx),        // RS232 RXD pin - Directly from pad
        .rx_data(uart_rx_data),      // 8 bit data output
        .rx_data_rdy(uart_rx_ready)  // Ready signal for rx_data
    );
    
    uart_tx #(
        .BAUD_RATE(BAUD_RATE),
        .CLOCK_RATE(CLOCK_RATE)
    )   uart_tx_0(
        .clk_tx(clk_50MHz),          // Clock input
        .rst_clk_tx(rst_sync),      // Active HIGH reset - synchronous to clk_tx
       
        .char_fifo_empty(1'b0), // Empty signal from char FIFO (FWFT)
        .char_fifo_dout(uart_tx_data),  // Data from the char FIFO
        .char_fifo_rd_en(uart_tx_pop), // Pop signal to the char FIFO
       
        .txd_tx(uart_tx)           // The transmit serial signal
    );    
    
    /////////////////////////////////////////////////////
    //Clock Conversion Module is added
    //Clock frequency Requirement for UART is 50 MHz 
    //100MHz clock generated by FPGA is converted to 50MHz, here. 
    ClockDivider#(
        .divide_rate(2)
    )UClockDivider(
        .clk_in(clk),
        .rst(rst),
        .clk_out(clk_50MHz)
    );
    /////////////////////////////////////////////////////
endmodule


module negedge_detector(
    input   clk,
    input   rst,
    input   in,
    output   out
    );
    
    reg d_ff;
    always @(posedge clk, posedge rst) begin
        if(rst) d_ff <= 1'b0;
        else d_ff <=in;
    end
    
    assign out = (~in)&&(d_ff^in);
endmodule

