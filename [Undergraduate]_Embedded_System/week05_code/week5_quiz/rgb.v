module rgb(R,G,B,H_COUNT,V_COUNT,DE,RESET);
    
    input [9:0] H_COUNT;    // horizontal pixel position
    input [9:0] V_COUNT;    // vertical line position
    input DE;               // display enable signal
    input RESET;            // reset signal
    
    output reg [7:3] R;     // red ouput (5 bits)
    output reg [7:2] G;     // green output (6 bits)
    output reg [7:3] B;     // blue output (5 bits)
    

    always@(DE,H_COUNT,V_COUNT,RESET) begin
        if(RESET == 1) begin    // initialization, color: black
            R = 0;
            G = 0;
            B = 0;
        end

        // only update colors during active display area
        else if(DE == 1) begin
            if(H_COUNT >= 10'd1) begin
                // white section: makes first white ROW of LCD
                if ((V_COUNT > 10'd11) && 
                    (V_COUNT <= 10'd45)) begin	/* white */ 
                    R = 5'b11111;
                    G = 6'b111111;
                    B = 5'b11111;
                end
                // yellow section: makes second yellow ROW of LCD
                else if ((V_COUNT > 10'd45) && 
                        (V_COUNT <= 10'd79)) begin 	/* yellow */                
                    R = 5'b11111;
                    G = 6'b111111;
                    B = 0;
                end
                // cyan section: makes third cyan ROW of LCD
                else if ((V_COUNT > 10'd79) && 
                        (V_COUNT <= 10'd113)) begin 	/* cyan */
                    R = 0;
                    G = 6'b111111;
                    B = 5'b11111;
                end
                // green section: makes fourth green ROW of LCD
                else if ((V_COUNT > 10'd113) && 
                        (V_COUNT <= 10'd147)) begin	/* green */
                    R = 0;
                    G = 6'b111111;
                    B = 0;
                end
                // purple section: makes fifth purple ROW of LCD
                else if ((V_COUNT > 10'd147) && 
                        (V_COUNT <= 10'd181)) begin    /* purple */
                    R = 5'b11111;
                    G = 0;
                    B = 5'b11111;
                end
                // red section: makes sixth red ROW of LCD
                else if ((V_COUNT > 10'd181) && 
                        (V_COUNT <= 10'd215)) begin   /* red */
                    R = 5'b11111;
                    G = 0;
                    B = 0;
                end  
                // blue section: makes seventh blue ROW of LCD
                else if ((V_COUNT > 10'd215) && 
                        (V_COUNT <= 10'd249)) begin     /* blue */
                    R = 0;
                    G = 0;
                    B = 5'b11111;
                end
                // white section: makes eighth white ROW of LCD
                else if ((V_COUNT > 10'd249) && 
                        (V_COUNT <= 10'd283)) begin    /* white */
                    R = 5'b11111;
                    G = 6'b111111;
                    B = 5'b11111;
                end
                // Default: black
                else begin
                    R = 0;
                    G = 0;
                    B = 0;
                end
            end
        end
        
        else begin
            R = 0;
            G = 0;
            B = 0;
        end
                       
    end
                   
endmodule