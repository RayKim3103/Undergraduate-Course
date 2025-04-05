//
// TFT-LCD�� Color Test Pattern�� display�ϱ� ���� coding
//

module TFTLCDCtrl (
    input CLK,
    input nRESET,
    output TCLK,	    // TFT-LCD Clock output
    output reg Hsync,	// TFT-LCD HSYNC
    output reg Vsync,	// TFT-LCD VSYNC
    output DE_out,	    // TFT-LCD Data enable
    output [7:3] R,     // TFT-LCD Red signal 
    output [7:2] G,     // TFT-LCD Green signal
    output [7:3] B,     // TFT-LCD Blue signal
    output Tpower,      // TFT-LCD Backlight On signal

    ////////////////////////////  YOUR CODE  /////////////////////////////
    /********** Extra input signals for the TFT-LCD controller **********/
    input [3:0] SW,     // TFT-LCD DIP Switches
    input BTN,          // Push Button
    //////////////////////////////////////////////////////////////////////

    output BRAMCLK,         //BRAM Clock
    output [17:0] BRAMADDR, //BRAM Address
    input [15:0] BRAMDATA); //BRAM Data 16bits
    
    wire g2mclk;            // g2m output Clock
    wire hclk;              // horizontal output Clock
    wire [9:0] H_COUNT;     // horizontal counter (0~479)
    wire [9:0] V_COUNT;     // vertical counter (0~271)
    wire hDE;               // horizontal display enable
    wire vDE;               // vertical display enable
    wire DEimage;	        // display enable (valid pixel area)
	wire RESET;             // reset signal   
	wire Hsyncimage;	    // TFT-LCD HSYNC
	wire Vsyncimage;	    // TFT-LCD VSYNC
    wire [7:3] BRAM_R;      // BRAM Red signal
    wire [7:2] BRAM_G;      // BRAM Green signal
    wire [7:3] BRAM_B;      // BRAM Blue signal
    wire [7:3] BAR_R;       // Color Bar Red signal (rgb.v)
    wire [7:2] BAR_G;       // Color Bar Green signal (rgb.v)
    wire [7:3] BAR_B;       // Color Bar Blue signal (rgb.v)
    ////////////////////////////  YOUR CODE  /////////////////////////////
    /**** Extra internal wires for assigning Quiz Image RGB signals *****/
    wire [7:3] H_R;         // H_half.v Red signal
    wire [7:2] H_G;         // H_half.v Green signal
    wire [7:3] H_B;         // H_half.v Blue signal
    wire [7:3] V_R;         // V_half.v Red signal
    wire [7:2] V_G;         // V_half.v Green signal
    wire [7:3] V_B;         // V_half.v Blue signal
    wire [7:3] G_R;         // Grad.v Red signal
    wire [7:2] G_G;         // Grad.v Green signal
    wire [7:3] G_B;         // Grad.v Blue signal
    /////////////////////////////////////////////////////////////////////


    // YOUR NEW CODE GOES BELOW
    // NOTICE
    // WIRES FOR EACH IMAGE WILL MAKE THE PROBLEM EASY
    // DECLARING A REGISTER FOR THE FINAL RESULT AND APPLYING MUX FOR IT WILL MAKE THE PROBLEM EASY

    assign RESET = ~nRESET;     // toggle active low reset signal, nRESET to RESET
    assign Tpower = 1;          // TFT-LCD Backlight On signal (always on)
    assign TCLK = g2mclk;       // TFT-LCD Clock output (g2mclk)
    assign DE_out = 1'b1;       // TFT-LCD Data enable (always on)
    assign DEimage = hDE & vDE; // display enable (valid pixel area)
    
    // set the Hsync and Vsync signals to 0 when RESET is high
    // set the Hsync and Vsync signals to the generated signals when RESET is low
    always @ (posedge g2mclk or posedge RESET) begin
        if (RESET) begin
            Vsync <= 1'b0;
            Hsync <= 1'b0;
        end
        
        else begin
            Vsync <= Vsyncimage;
            Hsync <= Hsyncimage;
        end
    end 
    
    // TFT-LCD CLOCK divider
    g2m a_g2m
    (
        .CLK        (CLK),
        .UP_CLK        (g2mclk),
        .RESET        (RESET)
    );

    // HSYNC & hDE generator
    horizontal b_horizontal
    (
        .CLK        (g2mclk),
        .UP_CLKa    (hclk),
        .H_COUNT     (H_COUNT),
        .Hsync        (Hsyncimage),
        .hDE        (hDE),
        .RESET        (RESET)
    );


    // VSYNC & vDE generator
    vertical c_vertical
    (
        .CLK        (hclk),
        .V_COUNT    (V_COUNT),
        .Vsync		(Vsyncimage),
        .vDE        (vDE),
        .RESET        (RESET)
    );
            
    /********** original code **********/        
        // TFT-LCD R/G/B Data (Color Bar) ����
    //    rgb e_rgb
    //    (
    //        .R            (BAR_R),
    //        .G            (BAR_G),
    //        .B            (BAR_B),
    //        .H_COUNT    (H_COUNT),
    //        .V_COUNT    (V_COUNT),
    //        .DE            (DEimage),
    //        .RESET        (RESET)
    //    );
    /***********************************/
    
    ////////////////////////////  YOUR CODE  /////////////////////////////
    /*********** Extra instances for making Image RGB signals ***********/
    // H_half
    H_half eH_half
    (
        .R            (H_R),
        .G            (H_G),
        .B            (H_B),
        .H_COUNT    (H_COUNT),
        .V_COUNT    (V_COUNT),
        .DE            (DEimage),
        .RESET        (RESET)
    );
    
    // V_half
    V_half eV_half
    (
        .R            (V_R),
        .G            (V_G),
        .B            (V_B),
        .H_COUNT    (H_COUNT),
        .V_COUNT    (V_COUNT),
        .DE            (DEimage),
        .RESET        (RESET)
    );
    
    // V_half
    Grad eGrad
    (
        .R            (G_R),
        .G            (G_G),
        .B            (G_B),
        .H_COUNT    (H_COUNT),
        .V_COUNT    (V_COUNT),
        .DE            (DEimage),
        .RESET        (RESET)
    );
    /////////////////////////////////////////////////////////////////////
    
    // BRAM Controller
    BRAMCtrl f_BRAMCtrl
    (
        .CLK(g2mclk),
        .RESET(RESET),
        .Vsync(Vsyncimage),
        .Hsync(Hsyncimage),
        .DE(DEimage),
        .BRAMCLK(BRAMCLK),
        .BRAMADDR(BRAMADDR),
        .BRAMDATA(BRAMDATA),
        .R(BRAM_R),
        .G(BRAM_G),
        .B(BRAM_B),
        .Reverse_SW(SW[0])
    );
    ////////////////////////////  YOUR CODE  /////////////////////////////
    /******** Extra wires & mux for assigning Image RGB signals *********/
    wire [4:0] wire_R;
    wire [5:0] wire_G;
    wire [4:0] wire_B;
    assign wire_R = (SW[1]) ? BRAM_R : (SW[2]) ? H_R : (SW[3]) ? V_R : G_R;
    assign wire_G = (SW[1]) ? BRAM_G : (SW[2]) ? H_G : (SW[3]) ? V_G : G_G;
    assign wire_B = (SW[1]) ? BRAM_B : (SW[2]) ? H_B : (SW[3]) ? V_B : G_B;
    /////////////////////////////////////////////////////////////////////
    
    
    ////////////////////////////  YOUR CODE  /////////////////////////////
    /********* Extra registers for assigning Image RGB signals **********/
    reg [4:0] reg_R;
    reg [5:0] reg_G;
    reg [4:0] reg_B;
    
    /******* BTN debounce using Reg_BTN & isBTN Hold BTN value  ********/
    reg Reg_BTN;
    reg isBTN;
    always @ (posedge CLK or posedge RESET) begin
        if (RESET) begin
            Reg_BTN <= 0;
            isBTN <= 0;
        end
        else  begin
            Reg_BTN <= BTN;
            if(!BTN && Reg_BTN)
                isBTN <= ~isBTN;
            end
        
    end
    /////////////////////////////////////////////////////////////////////
    
    ////////////////////////////  YOUR CODE  /////////////////////////////
    /********** Inversion of RGB signals based on isBTN value ***********/
    always @(*) begin
            if(isBTN) begin
                reg_R <= 5'b11111 - wire_R;
                reg_G <= 6'b111111- wire_G;
                reg_B <= 5'b11111 - wire_B;
            end
            else begin
                reg_R <= wire_R;
                reg_G <= wire_G;
                reg_B <= wire_B;
            end
        end
    
    assign R = reg_R;
    assign G = reg_G;
    assign B = reg_B;
    /////////////////////////////////////////////////////////////////////
endmodule