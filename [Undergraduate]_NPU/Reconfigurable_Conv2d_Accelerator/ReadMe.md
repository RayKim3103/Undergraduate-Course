# ISL NPU Project #2: 3x3 Conv2d Accelerator User Manual

This README summarizes the content of the User Manual for the ISL NPU Project #2, which implements a reconfigurable 3x3 Conv2d accelerator on FPGA for MNIST classification. The system uses Xilinx Arty Z7-20 and supports neural network acceleration via Python drivers on Pynq. The document is structured based on the original manual's sections.

## Project Information
- **Project Name**: FPGA를 이용한 MNIST classification
- **Overall Description**: 이 문서는 최종결과물의 사용자메뉴얼을 담고 있습니다. (This document contains the user manual for the final product.)
- **Team Leader**: Jueun Jung
- **Revision History**:
  | Date          | Version | Description          |
  |---------------|---------|----------------------|
  | August 29, 2025 | 1.0    | Final Submission Version |

## Table of Contents
1. [Overview](#1-overview)
   - 1.1 [Board Specification](#11-board-specification)
   - 1.2 [프로젝트 Block 명칭과 기능](#12-프로젝트-block-명칭과-기능)
     - 1.2.1 [전체 Reconfigurable Conv2d System 구조](#121-전체-reconfigurable-conv2d-system-구조)
     - 1.2.2 [Zynq Processor](#122-zynq-processor)
     - 1.2.3 [Block Memory Blocks](#123-block-memory-blocks)
     - 1.2.4 [신경망가속기 Control Register](#124-신경망가속기-control-register)
     - 1.2.5 [신경망가속기](#125-신경망가속기)
2. [Tutorial](#2-tutorial)
   - 2.1 [Python 패키지](#21-python-패키지)

## 1. Overview
이 장에서는 최종결과물의 주요한 기능과 소개하며, 설계한 시스템의 구성요소 및 사양에 대해서 설명한다. (This chapter introduces the main functions of the final product and explains the system's components and specifications.)

이 장은 다음과 같은 내용으로 구성되어 있다: (This chapter consists of the following:)
1. Board Specification
2. 프로젝트 전체구조 (Project overall structure)
3. 프로젝트 block 명칭과 기능 (Project block names and functions)

### 1.1 Board Specification
최종결과물은 Xilinx사의 Arty Z7-20 FPGA 디바이스를 사용하고 있으며, 주요 사양은 다음과 같다: (The final product uses Xilinx's Arty Z7-20 FPGA device with the following key specifications:)

- Logic Slices: 13,300
- 6-input LUTs: 53,200
- Flip-Flops: 106,400
- Total Block RAM: 630 kB
- DSP Slices: 220
- SDRAM (DDR3): 512 MB w/ 16-bit bus @ 1050 Mbps
- USB & Ethernet: Gigabit Ethernet PHY
- CPU: 650 MHz ARM® Cortex®-A9 dual-core processor

[그림 1] Arty Z7-20 (Image of the board is referenced in the original document.)

### 1.2 프로젝트 Block 명칭과 기능
#### 1.2.1 전체 Reconfigurable Conv2d System 구조
전체 시스템 개요도는 [그림 2]와 같으며, 세부 block 은 하위섹션에서 설명한다. 전체 시스템 제어를 위한 Xilinx사의 Zynq를 중심으로 메모리 Block, 신경망가속 및 제어 block, Configuration block으로 나누어진다. 각 하위 Block은 AXI Interface로 연결되어, Zynq를 통해서 제어 및 실행된다. (The overall system diagram is as in [Figure 2]. The system is centered on Xilinx Zynq for control, divided into memory blocks, neural network acceleration and control blocks, and configuration blocks. All sub-blocks are connected via AXI interface and controlled/executed through Zynq.)

[그림 2] 최종결과물 전체 시스템 개요도 (Overall system overview diagram.)

#### 1.2.2 Zynq Processor
##### 1.2.2.1 ARM Cortex-A9 dual-core processor
해당 프로젝트의 CPU로 Zynq의 ARM Core를 사용한다. 해당 CPU를 이용해, Python Jupiter Notebook 환경에 접속할 수 있으며, Python을 이용하여, Zynq의 DRAM과 BRAM간 Data 통신을 제어할 수 있다. 통신 규약의 경우 AXI4 Lite 인터페이스를 사용하여 통신한다. (Uses Zynq's ARM Core as the CPU. Allows access to Python Jupyter Notebook environment and controls data communication between Zynq's DRAM and BRAM via Python. Communication uses AXI4 Lite interface.)

[그림 3] ZTNQ7 Processing System IP Block (Zynq Processing System IP block diagram.)

#### 1.2.3 Block Memory Blocks
##### 1.2.3.1 Input Feature Memory & Memory Controller (INPUT MEM)
INPUT_MEM은 신경망가속기의 Input Activation data를 저장하기 위한 메모리 및 Zynq와 통신하기 위한, AXI BRAM Controller Block이다. Input Memory는 True Dual Port 타입의 Block Memory (BRAM)을 사용하여 약 65 kB의 메모리를 사용한다. (Stores input activation data for the neural network accelerator and communicates with Zynq via AXI BRAM Controller. Uses True Dual Port BRAM, approximately 65 kB.)

[그림 6] INPUT_MEM IP Block 및 구성 (INPUT_MEM IP block and configuration.)

##### 1.2.3.2 Weight Memory (WEIGHT_MEM)
WEIGHT_MEM은 신경망가속기의 가중치를 저장하기 위한 메모리 및 Zynq와 통신하기 위한, AXI BRAM Controller Block이다. True Dual Port 타입의 약 65kB의 BRAM을 사용한다. (Stores weights for the neural network accelerator and communicates with Zynq via AXI BRAM Controller. Uses True Dual Port BRAM, approximately 65 kB.)

[그림 7] WEIGHT_MEM IP Block 및 구성 (WEIGHT_MEM IP block and configuration.)

##### 1.2.3.3 Output Feature Memory (OUT_MEM)
OUT_MEM은 신경망가속기의 각 계층에서 연산이 완료된 출력활성화를 저장하기 위한 메모리 및 Zynq와 통신하기 위한, AXI BRAM Controller Block이다. True Dual Port 타입의 130 kB의 BRAM을 사용한다. (Stores output activations from each layer of the neural network accelerator and communicates with Zynq via AXI BRAM Controller. Uses True Dual Port BRAM, 130 kB.)

[그림 9] OUT_MEM IP Block 및 구성 (OUT_MEM IP block and configuration.)

#### 1.2.4 신경망가속기 Control Register
본 프로젝트의 신경망가속기는 3x3 Filter Size와 다양한 Input Feature Map Size의 Convolution연산을 지원한다. 제어레지스터는 신경망가속기에 명령어 전달을 위해 사용되며, User는 Input Channel와 Output Channel에 대한, Configuration Data를 Python을 통해서 정해줄 수 있으며, 이 Data들과 더불어, data shape에 대한 정보가 AXI4-lite interface를 통해서 신경망 계층마다 재구성하여 사용한다. (Supports 3x3 filter size and various input feature map sizes for convolution. Control registers transmit commands to the accelerator. Users configure input/output channels via Python, and data shape info is reconfigured per layer via AXI4-lite.)

- **1.2.4.1 CORE DONE: Address 0x00 (Read Only)**: Core status signal (1-bit). '0' = operating, '1' = idle.
- **1.2.4.2 layer_start: Address 0x04 (Write Only)**: Core activation signal (1-bit). '0→1' = start operation.
- **1.2.4.3 Information of Kernel Size, Image Size, Channels: Address 0x08~0x1C (Write Only)**: Details on kernel/image sizes and channels.

#### 1.2.5 신경망가속기
설계된 시스템은 [그림 10]과 같이 Configuration Data 및 Input Feature, Weight Data를 받아서, 2D Convolution연산을 진행한다. (The system receives configuration data, input features, and weights to perform 2D convolution as in [Figure 10].)

시스템의 흐름 (System flow):
1. Input Feature Map 분배 (Distribute input feature maps to 8 units for parallel processing of 8 input channels).
2. Input Window 형성 및 Weight 분배 후 2D Convolution 진행 (Form 3-row windows and perform 2D convolution with 3x3 filters).
3. Input Channel 들의 연산 결과를 Accumulate (Accumulate results from input channels to form partial sums).
4. 모든 Input Channel 들을 처리할 때까지 Stage 1 ~ Stage 3 을 Iterate (Iterate stages until all channels are processed).
5. 번외 (Optional: Quantize FP32 results to INT8 in hardware and store in OUT_MEM).

[그림 10] 신경망가속기 IP Block 및 Control Path 구성 (Neural network accelerator IP block and control path.)

## 2. Tutorial
이 장에서는 최종결과물의 제어방법과 Pynq의 Python을 이용한 신경망가속 사용법을 숙지한다. (This chapter covers control methods and usage of the neural network accelerator via Pynq Python.)

### 2.1 Python 패키지
#### 2.1.1 개요
본 프로젝트는 Vivado를 통해 구현되었고, Bitstream 파일을 Pynq의 Overlay를 통해 읽어 FPGA를 Programming하는 식으로 Hardware가 작동한다. 드라이버는 python언어를 기반으로 작성되었고, 메모리 제어를 위해 각각의 BRAM AXI Interface에 할당하는 Address Space는 아래 그림과 같다. 드라이버는 Python 패키지로 구성되며, import *를 통해서 User는 드라이버를 사용할 수 있다. (Implemented in Vivado; hardware runs by loading bitstream via Pynq Overlay. Driver is Python-based, with BRAM AXI interfaces assigned address spaces. Users import the package to use the driver.)

[그림 11] Example: How to use Driver  
[그림 10] BRAM AXI Interface Address Space & Driver Packager Hierachy (Note: Figure numbers may overlap in original.)

#### 2.1.2 사용 라이브러리
- **numpy**: For matrix/tensor creation, append, byte conversion (e.g., np.zeros, np.zeros_like, np.view, np.reshape, np.pad, np.tobytes).
- **struct**: For converting float32 to hex for hardware transmission (e.g., struct.unpack).
- **Overlay**: For controlling IP blocks in Python (e.g., Overlay(bitfile_path), self.hw.{IP_NAME}.mmio.array).

#### 2.1.3 신경망가속기 Configuration 제어
Configuration is handled via `class LayerConfig` for each layer before computation.

```python
class LayerConfig:
    def __init__(self, input_w, input_h, kernel_w, kernel_h, input_ch, output_ch):
        self.input_w = input_w
        self.input_h = input_h
        self.kernel_w = kernel_w
        self.kernel_h = kernel_h
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.stride = 1
```

#### 2.1.4 Input Feature 및 Weight 데이터 전송 및 로직 활성화
Data is 32-bit between Zynq/AXI and Conv2d system. Pad with zeros if data depth < BRAM depth.

Example padding code:
```python
padded_in_hex_int = in_hex_int[:input_data_size] + [0] * (TOTAL_MEM_SIZE - input_data_size)
self.imem[0:TOTAL_MEM_SIZE] = padded_in_hex_int
# Similar for weights
```

#### 2.1.5 Output Feature 데이터 수신
For hardware quantization: Concatenate 24-bit zeros to 8-bit data. Without quantization: Direct 32-bit transfer.

#### 2.1.6 신경망 연산 튜토리얼
##### 2.1.6.1 driver.py
- Conv2d Loop: Defines tile sizes and performs tiled Conv2d via `run_conv_2d` method.
- FC Loop: Reinterprets FC as 2D convolution for reuse (`run_fc_2d` method).

Code snippets for `start_npu`, `run_conv_2d`, and `run_fc_2d` are provided in the original (pages 14-16).

##### 2.1.6.2 nn.py
- **class Module**: Loads bitstream and shares NPUDriver.
- **class Conv2d**: Initializes channels, sets weights, forwards input.
- **class Linear**: Similar for FC layers.
- **class ReLU**: Applies ReLU.
- **class LeakyReLU**: Applies LeakyReLU with alpha.
- **class MaxPool2d**: Applies max pooling.

#### 2.1.7 신경망 연산 Driver 사용 예시 영상
https://youtu.be/Ji6pQB_KKWo