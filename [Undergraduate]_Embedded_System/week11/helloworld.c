/******************************************************************************
*
* Copyright (C) 2010 - 2015 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* Use of the Software is limited solely to applications:
* (a) running on a Xilinx device, or
* (b) that interact with a Xilinx device through a bus or interconnect.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/
/******************************************************************************/
/**
*
* @file xscugic_example.c
*
* This file contains a design example using the Interrupt Controller driver
* (XScuGic) and hardware device. Please reference other device driver examples
* to see more examples of how the intc and interrupts can be used by a software
* application.
*
* @note
*
* None
*
* <pre>
*
* MODIFICATION HISTORY:
* Ver   Who  Date     Changes
* ----- ---- -------- ----------------------------------------------------
* 1.00a drg  01/18/10 First release
* </pre>
******************************************************************************/

/***************************** Include Files *********************************/

#include <stdio.h>
#include <stdlib.h>
#include "xil_io.h"
#include "xil_exception.h"
#include "xparameters.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xil_types.h"
#include "xscugic.h"


#include "xiicps.h"
/************************** Constant Definitions *****************************/

/*
 * The following constants map to the XPAR parameters created in the
 * xparameters.h file. They are defined here such that a user can easily
 * change all the needed parameters in one place.
 */
#define IIC_DEVICE_ID XPAR_XIICPS_0_DEVICE_ID

/*
 * The slave address to send to and receive from.
 */
#define IIC_SLAVE_ADDR 0x51
#define IIC_SCLK_RATE 100000
//#define IIC_SCLK_RATE 50000

/*
 * The following constant controls the length of the buffers to be sent
 * and received with the IIC.
 */
//#define TEST_BUFFER_SIZE 132

//////////////////////////////////////////////////////////////////////////////////

/*
 * The following constants map to the XPAR parameters created in the
 * xparameters.h file. They are defined here such that a user can easily
 * change all the needed parameters in one place.
 */
#define INTC_DEVICE_ID XPAR_SCUGIC_0_DEVICE_ID
#define INTC_DEVICE_INT_ID 31

/**************************** Type Definitions *******************************/
#define AXI2AHBLite XPAR_M_AHB_BASEADDR
/***************** Macros (Inline Functions) Definitions *********************/

/************************** Function Prototypes ******************************/
int ScuGicExample(u16 DeviceId);
int SetUpInterruptSystem(XScuGic *XScuGicInstancePtr);
void DeviceDriverHandler(void *CallbackRef);

//////////////////////////////////////////////////////////////////////////////
int ReadRTC(void);
int states;
/************************** Variable Definitions *****************************/

XScuGic InterruptController;     /* Instance of the Interrupt Controller */
static XScuGic_Config *GicConfig;    /* The configuration parameters of the
                                       controller */

//////////////////////////////////////////////////////////////////////////////
XIicPs Iic; /**< Instance of the IIC Device */

/*
 * The following buffers are used in this example to send and receive data
 * with the IIC.
 */
u8 SendBuffer[4];    /**< Buffer for Transmitting Data */
u8 RecvBuffer[3];    /**< Buffer for Receiving Data */
//////////////////////////////////////////////////////////////////////////////

/*
 * Create a shared variable to be used by the main thread of processing and
 * the interrupt processing
 */
volatile static int InterruptProcessed = FALSE;

static void AssertPrint(const char8 *FilenamePtr, s32 LineNumber){
xil_printf("ASSERT: File Name: %s ", FilenamePtr);
xil_printf("Line Number: %d\r\n",LineNumber);
}

/*****************************************************************************/
/**
*
* This is the main function for the Interrupt Controller example.
*
* @param None.
*
* @return XST_SUCCESS to indicate success, otherwise XST_FAILURE.
*
* @note None.
*
****************************************************************************/
int main(void)
{
  int Status;


  /*
  * Setup an assert call back to get some info if we assert.
  */
  Xil_Out32LE(AXI2AHBLite, 0xf);

  Xil_AssertSetCallback(AssertPrint);

  xil_printf("GIC Example Test\r\n");

  /*
  *  Run the Gic example , specify the Device ID generated in xparameters.h
  */
  Status = ScuGicExample(INTC_DEVICE_ID);

  if (Status != XST_SUCCESS) {
  xil_printf("GIC Example Test Failed\r\n");
  return XST_FAILURE;
  }

  xil_printf("Successfully ran GIC Example Test\r\n");
  return XST_SUCCESS;


}

/*****************************************************************************/
/**
*
* This function is an example of how to use the interrupt controller driver
* (XScuGic) and the hardware device.  This function is designed to
* work without any hardware devices to cause interrupts. It may not return
* if the interrupt controller is not properly connected to the processor in
* either software or hardware.
*
* This function relies on the fact that the interrupt controller hardware
* has come out of the reset state such that it will allow interrupts to be
* simulated by the software.
*
* @param DeviceId is Device ID of the Interrupt Controller Device,
* typically XPAR_<INTC_instance>_DEVICE_ID value from
* xparameters.h
*
* @return XST_SUCCESS to indicate success, otherwise XST_FAILURE
*
* @note None.
*
******************************************************************************/
int ScuGicExample(u16 DeviceId)
{
  int Status;

  /*
  * Initialize the interrupt controller driver so that it is ready to
  * use.
  */
  GicConfig = XScuGic_LookupConfig(DeviceId);
  if (NULL == GicConfig) {
    return XST_FAILURE;
  }

  Status = XScuGic_CfgInitialize(&InterruptController, GicConfig,
  GicConfig->CpuBaseAddress);

  if (Status != XST_SUCCESS) {
    return XST_FAILURE;
  }


  /*
  * Perform a self-test to ensure that the hardware was built
  * correctly
  */
  Status = XScuGic_SelfTest(&InterruptController);
  if (Status != XST_SUCCESS) {
    return XST_FAILURE;
  }


  /*
  * Setup the Interrupt System
  */
  Status = SetUpInterruptSystem(&InterruptController);
  if (Status != XST_SUCCESS) {
    return XST_FAILURE;
  }

  /*
  * Connect a device driver handler that will be called when an
  * interrupt for the device occurs, the device driver handler performs
  * the specific interrupt processing for the device
  */
  Status = XScuGic_Connect(&InterruptController, INTC_DEVICE_INT_ID,
                          (Xil_ExceptionHandler)DeviceDriverHandler,
                          (void *)&InterruptController);

  if (Status != XST_SUCCESS) {
    return XST_FAILURE;
  }

  /*
  * Enable the interrupt for the device and then cause (simulate) an
  * interrupt so the handlers will be called
  */
  XScuGic_Enable(&InterruptController, INTC_DEVICE_INT_ID);

  /*
  *  Simulate the Interrupt
  */
  // Status = XScuGic_SoftwareIntr(&InterruptController,
  // INTC_DEVICE_INT_ID,
  // XSCUGIC_SPI_CPU0_MASK);
  // if (Status != XST_SUCCESS) {
  // return XST_FAILURE;
  // }

  /*
  * Wait for the interrupt to be processed, if the interrupt does not
  * occur this loop will wait forever
  */
  while (1) {
    /*
    * If the interrupt occurred which is indicated by the global
    * variable which is set in the device driver handler, then
    * stop waiting
    */
    if (InterruptProcessed) {
      break;
    }
  }

  return XST_SUCCESS;
}

/******************************************************************************/
/**
*
* This function connects the interrupt handler of the interrupt controller to
* the processor.  This function is seperate to allow it to be customized for
* each application.  Each processor or RTOS may require unique processing to
* connect the interrupt handler.
*
* @param XScuGicInstancePtr is the instance of the interrupt controller
* that needs to be worked on.
*
* @return None.
*
* @note None.
*
****************************************************************************/
int SetUpInterruptSystem(XScuGic *XScuGicInstancePtr)
{
    
  /*
  * Connect the interrupt controller interrupt handler to the hardware
  * interrupt handling logic in the ARM processor.
  */
  Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,
                              (Xil_ExceptionHandler) XScuGic_InterruptHandler,
                              XScuGicInstancePtr);

  /*
  * Enable interrupts in the ARM
  */
  Xil_ExceptionEnable();

  return XST_SUCCESS;
}

/******************************************************************************/
/**
*
* This function is designed to look like an interrupt handler in a device
* driver. This is typically a 2nd level handler that is called from the
* interrupt controller interrupt handler.  This handler would typically
* perform device specific processing such as reading and writing the registers
* of the device to clear the interrupt condition and pass any data to an
* application using the device driver.  Many drivers already provide this
* handler and the user is not required to create it.
*
* @param CallbackRef is passed back to the device driver's interrupt
* handler by the XScuGic driver.  It was given to the XScuGic
* driver in the XScuGic_Connect() function call.  It is typically
* a pointer to the device driver instance variable.
* In this example, we do not care about the callback
* reference, so we passed it a 0 when connecting the handler to
* the XScuGic driver and we make no use of it here.
*
* @return None.
*
* @note None.
*
****************************************************************************/
void DeviceDriverHandler(void *CallbackRef)
{
    /*
     * Indicate the interrupt has been processed using a shared variable
     */

    ////////////////////// QUIZ code ////////////////////////////
    // xil_printf("XScuGic_Device Driver Start\r\n");
    // int Status_RTC;
    // Status_RTC = ReadRTC();

      char pb;
      int Status_RTC;

      pb = Xil_In32LE(AXI2AHBLite);
      if((pb & 1) == 0){
        while(1) {
          pb = Xil_In32LE(AXI2AHBLite);
          Xil_Out32LE(AXI2AHBLite, pb | 1);

          if(((pb & 2) == 0)||((pb & 4) == 0)||((pb & 8) == 0)) break;


          for(int i =0; i<1600000; i++){
            pb = Xil_In32LE(AXI2AHBLite);
            if(((pb & 2) == 0)||((pb & 4) == 0)||((pb & 8) == 0)) break;
            Xil_Out32(AXI2AHBLite + 4, (1 << 7));
          }

          for(int i =0; i<1600000; i++){ // 10000000
            pb = Xil_In32LE(AXI2AHBLite);
            if(((pb & 2) == 0)||((pb & 4) == 0)||((pb & 8) == 0)) break;
            Xil_Out32(AXI2AHBLite + 4, (1 << 6));
          }
          ///////////////////////////////////////////////////////////////////////////////

          pb = Xil_In32LE(AXI2AHBLite);
          Xil_Out32LE(AXI2AHBLite, pb | 1);
          Status_RTC = ReadRTC();

          xil_printf("Hello: %d\n", RecvBuffer[0]);
          xil_printf("Hello_pb: %d\n", pb);

          if(RecvBuffer[0]%2 == 0){
            Xil_Out32(AXI2AHBLite + 4, (1 << 7));
          }
          if(RecvBuffer[0]%2 == 1){
            Xil_Out32(AXI2AHBLite + 4, (1 << 6));
          }

          xil_printf("Byeee_pb: %d\n", pb);
          xil_printf("Byeee: %d\n", RecvBuffer[0]);

          pb = Xil_In32LE(AXI2AHBLite);
          Status_RTC = ReadRTC();

          if(((pb & 2) == 0)||((pb & 4) == 0)||((pb & 8) == 0)) break;
          ///////////////////////////////////////////////////////////////////////////////

          // pb = Xil_In32LE(AXI2AHBLite);
          
          // Xil_Out32LE(AXI2AHBLite, pb | 1);
          
          // Xil_Out32(AXI2AHBLite + 4, (1 << 7));
          // sleep(1);

          // Xil_Out32(AXI2AHBLite + 4, (1 << 6));
          // sleep(1);
          
          // if(((pb & 1) == 0)||((pb & 2) == 0)||((pb & 8) == 0)) break;
          ///////////////////////////////////////////////////////////////////////////////

        }
      }
      else if((pb & 2) == 0){
        while(1) {
          pb = Xil_In32LE(AXI2AHBLite);
          Xil_Out32LE(AXI2AHBLite, pb | 2);

          if(((pb & 1) == 0)||((pb & 4) == 0)||((pb & 8) == 0)) break;


          for(int i =0; i<1600000; i++){
            pb = Xil_In32LE(AXI2AHBLite);
            if(((pb & 1) == 0)||((pb & 4) == 0)||((pb & 8) == 0)) break;
            Xil_Out32(AXI2AHBLite + 4, (1 << 5));
          }

          for(int i =0; i<1600000; i++){ // 10000000
            pb = Xil_In32LE(AXI2AHBLite);
            if(((pb & 1) == 0)||((pb & 4) == 0)||((pb & 8) == 0)) break;
            Xil_Out32(AXI2AHBLite + 4, (1 << 4));
          }

          ///////////////////////////////////////////////////////////////////////////////

          // pb = Xil_In32LE(AXI2AHBLite);
          // Xil_Out32LE(AXI2AHBLite, pb | 2);
          // Status_RTC = ReadRTC();

          // xil_printf("Hello: %d\n", RecvBuffer[0]);
          // xil_printf("Hello_pb: %d\n", pb);

          // if(RecvBuffer[0]%2 == 0){
          //   Xil_Out32(AXI2AHBLite + 4, (1 << 5));
          // }

          // if(RecvBuffer[0]%2 == 1){ // for(int i =0; i<1000000; i++)
          //   Xil_Out32(AXI2AHBLite + 4, (1 << 4));
          // }

          // xil_printf("Byeee_pb: %d\n", pb);
          // xil_printf("Byeee: %d\n", RecvBuffer[0]);

          // pb = Xil_In32LE(AXI2AHBLite);
          // Status_RTC = ReadRTC();

          // if(((pb & 1) == 0)||((pb & 4) == 0)||((pb & 8) == 0)) break;
          ///////////////////////////////////////////////////////////////////////////////

          // pb = Xil_In32LE(AXI2AHBLite);
          // Status_RTC = ReadRTC();
          
          // Xil_Out32LE(AXI2AHBLite, pb | 2);
          
          // Xil_Out32(AXI2AHBLite + 4, (1 << 5));
          // sleep(1);

          // Xil_Out32(AXI2AHBLite + 4, (1 << 4));
          // sleep(1);
          
          // if(((pb & 1) == 0)||((pb & 4) == 0)||((pb & 8) == 0)) break;
        }
      }
      else if((pb & 4) == 0){
        while(1) {
          pb = Xil_In32LE(AXI2AHBLite);
          Xil_Out32LE(AXI2AHBLite, pb | 4);

          if(((pb & 1) == 0)||((pb & 2) == 0)||((pb & 8) == 0)) break;


          for(int i =0; i<1600000; i++){
            pb = Xil_In32LE(AXI2AHBLite);
            if(((pb & 1) == 0)||((pb & 2) == 0)||((pb & 8) == 0)) break;
            Xil_Out32(AXI2AHBLite + 4, (1 << 3));
          }

          for(int i =0; i<1600000; i++){ // 10000000
            pb = Xil_In32LE(AXI2AHBLite);
            if(((pb & 1) == 0)||((pb & 2) == 0)||((pb & 8) == 0)) break;
            Xil_Out32(AXI2AHBLite + 4, (1 << 2));
          }

          ///////////////////////////////////////////////////////////////////////////////
          // pb = Xil_In32LE(AXI2AHBLite);
          // Xil_Out32LE(AXI2AHBLite, pb | 4);
          // Status_RTC = ReadRTC();

          // xil_printf("Hello: %d\n", RecvBuffer[0]);
          // xil_printf("Hello_pb: %d\n", pb);

          // if(RecvBuffer[0]%2 == 0){
          //   Xil_Out32(AXI2AHBLite + 4, (1 << 3));
          // }
          // if(RecvBuffer[0]%2 == 1){ // for(int i =0; i<1000000; i++)
          //   Xil_Out32(AXI2AHBLite + 4, (1 << 2));
          // }

          // xil_printf("Byeee_pb: %d\n", pb);
          // xil_printf("Byeee: %d\n", RecvBuffer[0]);

          // pb = Xil_In32LE(AXI2AHBLite);
          // Status_RTC = ReadRTC();

          // if(((pb & 1) == 0)||((pb & 2) == 0)||((pb & 8) == 0)) break;

          ///////////////////////////////////////////////////////////////////////////////
          // pb = Xil_In32LE(AXI2AHBLite);
          // Status_RTC = ReadRTC();
          
          // Xil_Out32LE(AXI2AHBLite, pb | 4);
          
          // Xil_Out32(AXI2AHBLite + 4, (1 << 3));
          // sleep(1);
          
          // Xil_Out32(AXI2AHBLite + 4, (1 << 2));
          // sleep(1);
          
          // if(((pb & 1) == 0)||((pb & 2) == 0)||((pb & 8) == 0)) break;
        }
      }
      else if((pb & 8) == 0){
        while(1) {
          pb = Xil_In32LE(AXI2AHBLite);
          Xil_Out32LE(AXI2AHBLite, pb | 8);

          if(((pb & 1) == 0)||((pb & 2) == 0)||((pb & 4) == 0)) break;


          for(int i =0; i<1600000; i++){
            pb = Xil_In32LE(AXI2AHBLite);
            if(((pb & 1) == 0)||((pb & 2) == 0)||((pb & 4) == 0)) break;
            Xil_Out32(AXI2AHBLite + 4, (1 << 1));
          }

          for(int i =0; i<1600000; i++){ // 10000000
            pb = Xil_In32LE(AXI2AHBLite);
            if(((pb & 1) == 0)||((pb & 2) == 0)||((pb & 4) == 0)) break;
            Xil_Out32(AXI2AHBLite + 4, (1 << 0));
          }

          ///////////////////////////////////////////////////////////////////////////////
          // pb = Xil_In32LE(AXI2AHBLite);
          // Xil_Out32LE(AXI2AHBLite, pb | 8);
          // Status_RTC = ReadRTC();

          // xil_printf("Hello: %d\n", RecvBuffer[0]);
          // xil_printf("Hello_pb: %d\n", pb);

          // if(RecvBuffer[0]%2 == 0){
          //   Xil_Out32(AXI2AHBLite + 4, (1 << 1));
          // }
          // if(RecvBuffer[0]%2 == 1){ // for(int i =0; i<1000000; i++)
          //   Xil_Out32(AXI2AHBLite + 4, (1 << 0));
          // }

          // xil_printf("Byeee_pb: %d\n", pb);
          // xil_printf("Byeee: %d\n", RecvBuffer[0]);

          // pb = Xil_In32LE(AXI2AHBLite);
          // Status_RTC = ReadRTC();

          // if(((pb & 1) == 0)||((pb & 2) == 0)||((pb & 4) == 0)) break;

          ///////////////////////////////////////////////////////////////////////////////
          // pb = Xil_In32LE(AXI2AHBLite);
          // Status_RTC = ReadRTC();
          
          // Xil_Out32LE(AXI2AHBLite, pb | 8);
          
          // Xil_Out32(AXI2AHBLite + 4, (1 << 1));
          // sleep(1);

          // Xil_Out32(AXI2AHBLite + 4, (1 << 0));
          // sleep(1);
          
          // if(((pb & 1) == 0)||((pb & 2) == 0)||((pb & 4) == 0)) break;
        }
      }
    /////////////////////////////////////////////////////////////
    //    char pb;
    //    pb = Xil_In32LE(AXI2AHBLite);
    //    if((pb & 1) == 0){
    //        Xil_Out32LE(AXI2AHBLite, pb | 1);
    //        Xil_Out32(AXI2AHBLite + 4, (1 << 7) | (1 << 6));
    //    }
    //    else if((pb & 2) == 0){
    //        Xil_Out32LE(AXI2AHBLite, pb | 2);
    //        Xil_Out32(AXI2AHBLite + 4, (1 << 5) | (1 << 4));
    //    }
    //    else if((pb & 4) == 0){
    //        Xil_Out32LE(AXI2AHBLite, pb | 4);
    //        Xil_Out32(AXI2AHBLite + 4, (1 << 3) | (1 << 2));
    //    }
    //    else if((pb & 8) == 0){
    //        Xil_Out32LE(AXI2AHBLite, pb | 8);
    //        Xil_Out32(AXI2AHBLite + 4, (1 << 1) | (1 << 0));
    //    }
    //
    //    InterruptProcessed = TRUE;
}


////////////////////////////////////////////////////////////////////////////////
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
  // for (Index = 0; Index < TEST_BUFFER_SIZE; Index++) {
  // SendBuffer[Index] = (Index % TEST_BUFFER_SIZE);
  // RecvBuffer[Index] = 0;
  // }

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
  // for(Index = 0; Index < TEST_BUFFER_SIZE; Index ++) {
  //
  // /* Aardvark as slave can only set 64 bytes for output */
  // if (RecvBuffer[Index] != Index % 64) {
  // return XST_FAILURE;
  // }
  // }

  return XST_SUCCESS;
}
////////////////////////////////////////////////////////////////////////////////