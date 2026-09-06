---
title: "10주차 결과 - AHB TFT-LCD 이미지 출력"
course: "Embedded System"
week: 10
type: "result"
tags:
  - embedded-system
  - ahb
  - tft-lcd
  - memory-mapped-io
---

# 10주차 결과 - AHB TFT-LCD 이미지 출력

이전: [[09주차 결과 - AXI Text-LCD PS PL 연동]]  
다음: [[11주차 예비 - Push Button Interrupt System]]

## 핵심 요약

이번 실습에서는 PS가 C 코드에서 image 배열을 읽고, memory mapped I/O 방식으로 PL의 BRAM에 image data를 전송하여 TFT-LCD에 출력한다. Zynq PS는 AXI 기반이지만 실습용 TFT-LCD 시스템은 AHB interface를 사용하므로 AXI-AHB Lite bridge가 필요하다.

## 전체 데이터 흐름

```text
C image array in PS
-> Xil_Out32()
-> AXI GP master
-> AXI Interconnect
-> AXI AHBLite Bridge
-> AHB2PORT1RAM
-> dual-port BRAM
-> TFTLCDCtrl / BRAMCtrl
-> RGB565 TFT-LCD output
```

## 주요 블록

| 블록 | 역할 |
|---|---|
| Zynq7 PS | image data 생성 및 register write |
| AXI Interconnect | AXI master/slave 연결과 address decoding |
| Processor System Reset | reset 안정화 |
| AXI AHBLite Bridge | AXI protocol을 AHB-Lite protocol로 변환 |
| `AHB2PORT1RAM` | AHB master와 dual-port BRAM 연결 |
| `register_set` | control register 제공 |
| `TFTLCDCtrl` | BRAM image 또는 color bar를 TFT-LCD에 출력 |

## 왜 AXI-AHB Bridge가 필요한가

Zynq PS의 GP master는 AXI interface를 사용한다. 하지만 이번 TFT-LCD용 PL 시스템은 AHB-Lite 기반으로 작성되어 있다. 두 protocol은 transaction 방식이 다르므로 bridge가 address, data, write/read control을 변환해야 한다.

## `top.v`의 변화

이전 TFT-LCD 실습과 비교해 `top.v`에는 AHB 관련 port와 모듈이 추가된다.

- `M_AHB_haddr`, `M_AHB_hwdata`, `M_AHB_hrdata`
- `M_AHB_hwrite`, `M_AHB_htrans`, `M_AHB_hsize`
- `AHB2PORT1RAM` 인스턴스
- `register_set` 인스턴스
- `TFTLCD_SW[0]`, `TFTLCD_SW[1]`를 control register bit에 연결

이전에는 `.coe`로 BRAM 초기값을 넣었지만, 이번에는 PS가 runtime에 image data를 BRAM에 write한다.

## `AHB2PORT1RAM`

이 모듈은 AHB-Lite transaction을 dual-port BRAM 접근으로 바꾼다.

| 기능 | 설명 |
|---|---|
| Address/Data phase 분리 | AHB write의 address phase와 data phase를 register로 맞춤 |
| `REQ[1]` 생성 | valid read/write transaction일 때 BRAM enable |
| `P1HADDRMUX` | write는 저장된 address, read는 현재 address 사용 |
| `HREADYOUT` | write 후 read 전환 시 충돌 방지 |
| Endian 처리 | `BIGEND`에 따라 하위 address bit 반전 |
| Byte write enable | byte/halfword/word 크기에 맞춰 `BWE1[3:0]` 생성 |

BRAM은 port B를 PS/AHB write용으로, port A를 TFT-LCD read용으로 사용한다.

## `register_set`

`register_set`은 AHB address에 따라 control register를 read/write한다.

| Address slice | register | 역할 |
|---|---|---|
| `HADDR[5:2] == 0` | `REG0` | image source 선택 |
| `HADDR[5:2] == 1` | `REG1` | image direction 선택 |

`HSIZE`에 따라 byte, halfword, word 접근을 처리한다. AXI/AHB data bus가 32비트이므로 작은 단위 접근은 address 하위 bit를 이용해 필요한 byte lane을 선택한다.

## 기본 C 코드

기본 application은 `hex`, `image0`, `image1` 세 이미지를 차례로 출력한다.

중요한 address 계산:

```c
XPAR_M_AHB_BASEADDR + y * 960 + x * 4
```

이유:

- TFT-LCD 가로 480 pixel
- RGB565 pixel 1개 = 16bit
- `Xil_Out32` 한 번에 32bit, 즉 2pixel write
- 한 줄은 480pixel = 240번 write
- 240번 x 4byte = 960byte

따라서 `x` loop는 0~239까지만 돈다.

## Quiz 이미지 구성

Quiz에서는 5가지 출력 상태를 구현했다.

| 번호 | 출력 | 구현 |
|---|---|---|
| 1 | RGB color bar | control register에 0 write |
| 2 | jet/village stripe 역방향 | 34 line마다 image source 교차, direction bit 설정 |
| 3 | 사람 이미지 좌측 절반 | x 0~119 image, 120~239 black |
| 4 | 사람 이미지 우측 절반 | x 0~119 black, 120~239 image |
| 5 | 사람 이미지 RGB inversion | `0xffffffff - packed_pixel` |

stripe는 272 line을 8개 band로 나누기 위해 `34 = 272/8`을 사용했다.

## Memory Mapped I/O

AHB는 memory mapped 방식이다. Processor는 I/O device와 memory를 별도 명령으로 구분하지 않고, 특정 address에 read/write하여 device register나 BRAM에 접근한다.

## 정리

10주차 결과의 핵심은 PS software가 `Xil_Out32`로 image data를 쓰면, interconnect와 bridge를 지나 PL BRAM에 저장되고, TFT-LCD controller가 그 BRAM을 읽어 화면으로 출력한다는 점이다. 이 구조는 embedded system에서 CPU, bus, memory, custom hardware가 협력하는 전형적인 예시다.
