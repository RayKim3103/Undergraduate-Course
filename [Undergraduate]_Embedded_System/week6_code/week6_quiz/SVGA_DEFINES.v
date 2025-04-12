/*
---------------------------------------------------------------------------------
To select a resolution and refresh rate, remove the comments around the desired
block in this file. The pixel clock output by the DCM module should approximately
equal the rate specified above the timing block that is uncommented.
---------------------------------------------------------------------------------
*/

// DEFINE THE VARIOUS PIPELINE DELAYS

`define CHARACTER_DECODE_DELAY  4

//  480 X 272 @ 60Hz with a 25.175MHz pixel clock
`define H_ACTIVE 480       // pixels
`define H_FRONT_PORCH 16   // pixels
`define H_BACK_PORCH 48    // pixels
`define H_BACK_PORCH 16    // pixels
`define H_TOTAL 560        // pixels

`define V_ACTIVE 272       // lines
`define V_FRONT_PORCH 11   // lines
`define V_SYNCH 2          // lines
`define V_BACK_PORCH 11    // lines
`define V_TOTAL 296        // lines

`define CLK_MULTIPLY 2		// 50 * 2/4 = 25.000 MHz
`define CLK_DIVIDE 4
