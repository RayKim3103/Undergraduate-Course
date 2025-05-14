module horizontal(CLK,UP_CLKa,H_COUNT,Hsync,hDE,RESET);
    
    input CLK;
    input RESET;

    // CLK pulse generated at the end of the horizontal line 
    // (used for vertical timing) 
    output reg UP_CLKa;         
    
    output reg [9:0] H_COUNT;   // Horizontal Counter
    output reg Hsync;           // Horizontal Sync Pulse  
    output reg hDE;             // Horizontal Display Enable
    
    always@(posedge CLK or posedge RESET)
    begin
        if(RESET)       // initialization
        begin
            Hsync <= 1'b0;
            H_COUNT <= 10'd0;
            hDE <= 1'b0;
            UP_CLKa <= 1'b0;
        end
        else
        begin
            UP_CLKa <= ~Hsync;       
            // Sync Pulse (Active LOW, for 41 cycles) 
            if (H_COUNT <= 10'd40)       
            begin
                Hsync <= 1'b0;
                hDE <= 1'b0;
            end
            // Hsync Back porch (for 2 cycles)
            else if ((H_COUNT > 10'd40) && (H_COUNT <= 10'd42))     
            begin
                Hsync <= 1'b1;
                hDE <= 1'b0;
            end
            // Active Video (for 480 cycles)
            else if ((H_COUNT > 10'd42) && (H_COUNT <= 10'd522))    
            begin
                Hsync <= 1'b1;
                hDE <= 1'b1;
            end
            // Hsync Front porch (for 2 cycles)
            else if ((H_COUNT > 10'd522) && (H_COUNT <= 10'd524))   
            begin
                Hsync <= 1'b1;
                hDE <= 1'b0;
            end
            if (H_COUNT < 10'd524)  // Hsync counter (for 524 cycles)
                H_COUNT <= H_COUNT + 10'd1;
            else
                H_COUNT <= 10'd0;
        end
    end

endmodule