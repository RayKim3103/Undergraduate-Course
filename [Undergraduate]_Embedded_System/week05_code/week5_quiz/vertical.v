module vertical(CLK,V_COUNT,Vsync,vDE,RESET);
    
    input CLK;
    input RESET;
    
    output reg [9:0] V_COUNT;   // Vertical Counter
    output reg Vsync;           // Vertical Sync Pulse
    output reg vDE;             // Vertical Display Enable
    
    always@(posedge CLK or posedge RESET)
    begin
        if(RESET)   // initialization
        begin
            V_COUNT <= 10'd0;
            Vsync <= 1'b0;
            vDE <= 1'b0;
        end
        else
        begin
            // Sync Pulse (Active LOW, for 10 cycles)
            if (V_COUNT <= 9)   
            begin
                Vsync <= 1'b0;
                vDE <= 1'b0;
            end
            // Vsync Back porch (for 2 cycles)
            else if ((V_COUNT > 9) && (V_COUNT <= 11))  
            begin
                Vsync <= 1'b1;
                vDE <= 1'b0;
            end
            // Active Video (for 272 cycles)
            else if ((V_COUNT > 11) && (V_COUNT <= 283))    
            begin
                Vsync <= 1'b1;
                vDE <= 1'b1;
            end
            // Vsync Front porch (for 2 cycles)
            else if ((V_COUNT > 283) && (V_COUNT <= 285))   
            begin
                Vsync <= 1'b1;
                vDE <= 1'b0;
            end
            // Vsync counter (for 285 cycles)
            if (V_COUNT < 285)  
                V_COUNT <= V_COUNT + 10'b1;
            else
              V_COUNT <= 10'd0;
        end
    end
    
endmodule