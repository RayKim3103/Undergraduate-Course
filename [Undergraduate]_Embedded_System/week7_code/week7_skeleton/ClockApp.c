/******************************************************************************
*
* (c) Copyright 2010-13 Xilinx, Inc. All rights reserved.
*
* This file contains confidential and proprietary information of Xilinx, Inc.
* and is protected under U.S. and international copyright and other
* intellectual property laws.
*
* DISCLAIMER
* This disclaimer is not a license and does not grant any rights to the
* materials distributed herewith. Except as otherwise provided in a valid
* license issued to you by Xilinx, and to the maximum extent permitted by
* applicable law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL
* FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS,
* IMPLIED, OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF
* MERCHANTABILITY, NON-INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE;
* and (2) Xilinx shall not be liable (whether in contract or tort, including
* negligence, or under any other theory of liability) for any loss or damage
* of any kind or nature related to, arising under or in connection with these
* materials, including for any direct, or any indirect, special, incidental,
* or consequential loss or damage (including loss of data, profits, goodwill,
* or any type of loss or damage suffered as a result of any action brought by
* a third party) even if such damage or loss was reasonably foreseeable or
* Xilinx had been advised of the possibility of the same.
*
* CRITICAL APPLICATIONS
* Xilinx products are not designed or intended to be fail-safe, or for use in
* any application requiring fail-safe performance, such as life-support or
* safety devices or systems, Class III medical devices, nuclear facilities,
* applications related to the deployment of airbags, or any other applications
* that could lead to death, personal injury, or severe property or
* environmental damage (individually and collectively, "Critical
* Applications"). Customer assumes the sole risk and liability of any use of
* Xilinx products in Critical Applications, subject only to applicable laws
* and regulations governing limitations on product liability.
*
* THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE
* AT ALL TIMES.
*
******************************************************************************/
/*****************************************************************************/
/**
* @file xiicps_polled_master_example.c
*
* A design example of using the device in polled mode as master.
*
* The example uses buffer size 132. Please set the send buffer of the
* Aardvark device to be continuous 64 bytes from 0x00 to 0x3F.
 *
 * <pre> MODIFICATION HISTORY:
 *
 * Ver   Who Date     Changes
 * ----- --- -------- -----------------------------------------------
 * 1.00a jz  01/30/10 First release
 *
 * </pre>
 *
 ****************************************************************************/

/***************************** Include Files **********************************/
#include "xparameters.h"
#include "xiicps.h"
#include "xil_printf.h"
#include "sevenseg.h"

/************************** Constant Definitions ******************************/

/*
 * The following constants map to the XPAR parameters created in the
 * xparameters.h file. They are defined here such that a user can easily
 * change all the needed parameters in one place.
 */
#define IIC_DEVICE_ID		XPAR_XIICPS_0_DEVICE_ID

/*
 * The slave address to send to and receive from.
 */
#define IIC_SLAVE_ADDR		0x51
#define IIC_SCLK_RATE		100000
//#define IIC_SCLK_RATE		50000

/*
 * The following constant controls the length of the buffers to be sent
 * and received with the IIC.
 */
//#define TEST_BUFFER_SIZE	132

/**************************** Type Definitions ********************************/


/************************** Function Prototypes *******************************/

int ReadRTC(void);
/************************** Variable Definitions ******************************/

XIicPs Iic;		/**< Instance of the IIC Device */

/*
 * The following buffers are used in this example to send and receive data
 * with the IIC.
 */
u8 SendBuffer[4];    /**< Buffer for Transmitting Data */
u8 RecvBuffer[3];    /**< Buffer for Receiving Data */


/******************************************************************************/
/**
*
* Main function to call the polled master example.
*
* @param	None.
*
* @return	XST_SUCCESS if successful, XST_FAILURE if unsuccessful.
*
* @note		None.
*
*******************************************************************************/
int main(void)
{
	int Status;
	u16 wait;
	int segdata;

//	xil_printf("IIC Master Polled Example Test \r\n");
	/*
	 * Run the Iic polled example in master mode, specify the Device
	 * ID that is specified in xparameters.h.
	 */
    while (1) {
    	Status = ReadRTC();
    	segdata = ((RecvBuffer[2] & 0x3f)<<24) + (0xA << 20) + ((RecvBuffer[1] & 0x7f)<<12) + (0xA << 8) + (RecvBuffer[0] & 0x7f);
    	SEVENSEG_mWriteReg (XPAR_SEVENSEG_0_S00_AXI_BASEADDR, 0, segdata);
    	for (wait=0;wait<3000;wait++);
    }
	if (Status != XST_SUCCESS) {
		xil_printf("IIC Master Polled Example Test Failed\r\n");
		return XST_FAILURE;
	}
	return XST_SUCCESS;
}

int ReadRTC(void)
{
	int Status;
	XIicPs_Config *Config;

	/*
	 * Initialize the IIC driver so that it's ready to use
	 * Look up the configuration in the config table,
	 * then initialize it.
	 */
	Config = XIicPs_LookupConfig(IIC_DEVICE_ID);
	if (NULL == Config) {
		return XST_FAILURE;
	}

	Status = XIicPs_CfgInitialize(&Iic, Config, Config->BaseAddress);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	/*
	 * Perform a self-test to ensure that the hardware was built correctly.
	 */
	Status = XIicPs_SelfTest(&Iic);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	/*
	 * Set the IIC serial clock rate.
	 */
	XIicPs_SetSClk(&Iic, IIC_SCLK_RATE);

	/*
	 * Initialize the send buffer bytes with a pattern to send and the
	 * the receive buffer bytes to zero to allow the receive data to be
	 * verified.
	 */
//	for (Index = 0; Index < TEST_BUFFER_SIZE; Index++) {
//		SendBuffer[Index] = (Index % TEST_BUFFER_SIZE);
//		RecvBuffer[Index] = 0;
//	}

	SendBuffer[0] = 0x02;
	RecvBuffer[0] = 0x0;
	RecvBuffer[1] = 0x0;
	RecvBuffer[2] = 0x0;

	/*
	 * Send the buffer using the IIC and ignore the number of bytes sent
	 * as the return value since we are using it in interrupt mode.
	 */
	Status = XIicPs_MasterSendPolled(&Iic, SendBuffer, 1, IIC_SLAVE_ADDR);

	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	/*
	 * Wait until bus is idle to start another transfer.
	 */
	while (XIicPs_BusIsBusy(&Iic)) {
		/* NOP */
	}

	Status = XIicPs_MasterRecvPolled(&Iic, RecvBuffer, 3, IIC_SLAVE_ADDR);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	/*
	 * Verify received data is correct.
	 */
//	 for(Index = 0; Index < TEST_BUFFER_SIZE; Index ++) {
//
//		/* Aardvark as slave can only set 64 bytes for output */
//		if (RecvBuffer[Index] != Index % 64) {
//			return XST_FAILURE;
//		}
//	}

	return XST_SUCCESS;
}


