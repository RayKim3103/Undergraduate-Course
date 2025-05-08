/*
 * Copyright (c) 2009-2012 Xilinx, Inc.  All rights reserved.
 *
 * Xilinx, Inc.
 * XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS" AS A
 * COURTESY TO YOU.  BY PROVIDING THIS DESIGN, CODE, OR INFORMATION AS
 * ONE POSSIBLE   IMPLEMENTATION OF THIS FEATURE, APPLICATION OR
 * STANDARD, XILINX IS MAKING NO REPRESENTATION THAT THIS IMPLEMENTATION
 * IS FREE FROM ANY CLAIMS OF INFRINGEMENT, AND YOU ARE RESPONSIBLE
 * FOR OBTAINING ANY RIGHTS YOU MAY REQUIRE FOR YOUR IMPLEMENTATION.
 * XILINX EXPRESSLY DISCLAIMS ANY WARRANTY WHATSOEVER WITH RESPECT TO
 * THE ADEQUACY OF THE IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO
 * ANY WARRANTIES OR REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE
 * FROM CLAIMS OF INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <stdio.h>

#include "xparameters.h"
#include "textlcd.h"
#include "xil_io.h"

//void print(char *str);

int main()
{
int i;
char up_line[20];
char down_line[20];

char reg_up_line[20];
char reg_down_line[20];


printf(" Tex LCD SDK Application \n");
int count = 0;
while (1)
{
// sprintf (up_line, "RPS-Z7020-TK BD.");
// sprintf (down_line, "Huins Co,. Ltd. ");

sprintf (up_line, "Count: %02d       ", count);
sprintf (down_line, "KMS PHH         ");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}

// sprintf (up_line,   "Chapter9 Example");
// sprintf (down_line, "TextLCD Done...!");
//
// for (i=0; i<4;i++)
// {
// TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
// up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
// TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
// down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
// }
//
// for (i=0; i<100000000;i++){}

sprintf (up_line, " Count: %02d      ", count);
sprintf (down_line, " KMS PHH        ");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}
sprintf (up_line, "  Count: %02d     ", count);
sprintf (down_line, "  KMS PHH       ");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}
sprintf (up_line, "   Count: %02d    ", count);
sprintf (down_line, "   KMS PHH      ");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

///////////////////////////////////////////////////////

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}
sprintf (up_line, "    Count: %02d   ", count);
sprintf (down_line, "    KMS PHH     ");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}
sprintf (up_line, "     Count: %02d  ", count);
sprintf (down_line, "     KMS PHH    ");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}
sprintf (up_line, "      Count: %02d ", count);
sprintf (down_line, "      KMS PHH   ");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}
sprintf (up_line, "       Count: %02d", count);
sprintf (down_line, "       KMS PHH  ");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}

////////////////////////////////////////////

sprintf (up_line, "%01d       Count: %01d", count%10, count/10);
sprintf (down_line, "        KMS PHH ");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}
sprintf (up_line, "%02d       Count: ", count);
sprintf (down_line, "         KMS PHH");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}
sprintf (up_line, " %02d       Count:", count);
sprintf (down_line, "H         KMS PH");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}

sprintf (up_line, ": %02d       Count", count);
sprintf (down_line, "HH         KMS P");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

//////////////////////////////////////////////////////////

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}
sprintf (up_line, "t: %02d       Coun", count);
sprintf (down_line, "PHH         KMS ");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}
sprintf (up_line, "nt: %02d       Cou", count);
sprintf (down_line, " PHH         KMS");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}
sprintf (up_line, "unt: %02d       Co", count);
sprintf (down_line, "S PHH         KM");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}

sprintf (up_line, "ount: %02d       C", count);
sprintf (down_line, "MS PHH         K");

for (i=0; i<4;i++)
{
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4,
up_line[i*4+3] + (up_line[i*4+2]<<8) + (up_line[i*4+1]<<16) + (up_line[i*4]<<24));
TEXTLCD_mWriteReg(XPAR_TEXTLCD_0_S00_AXI_BASEADDR, i*4+16,
down_line[i*4+3] + (down_line[i*4+2]<<8) + (down_line[i*4+1]<<16) + (down_line[i*4]<<24));
}

// for (i=0; i<100000000;i++){}
for (i=0; i<20000000;i++){}



count++;
}

return 0;
}