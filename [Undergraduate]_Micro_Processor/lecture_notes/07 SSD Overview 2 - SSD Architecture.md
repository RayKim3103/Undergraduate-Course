# SSD Overview 2 - SSD Architecture

tags: #micro-processor #ssd #architecture #ssd-controller #sram #dram #nand-flash #ahb

관련 노트: [[06 SSD Overview 1 - Solid State Disk Basics]], [[08 SSD Hardware 4 - NAND Flash Memory]]

## 핵심 요약

이 자료는 SSD 내부 architecture를 설명한다. SSD는 host interface, SSD controller, microcontroller, internal bus, SRAM/DRAM buffer, NAND flash controller, NAND flash memory chips로 구성된다. 핵심은 SSD controller가 host의 block I/O 요청을 NAND flash의 page/block 동작으로 변환한다는 점이다.

## SSD 전체 구조

SSD는 크게 다음으로 구성된다.

- external host interface
- SSD controller
- microcontroller
- internal bus
- SRAM
- DRAM cache buffer
- NAND flash controller
- NAND flash memory chips

Host는 SSD를 block device로 보지만, 내부에서는 flash page read/program과 block erase 단위로 동작한다.

## SSD Controller

SSD controller는 SSD subsystem의 중심이다.

역할:

- host command 처리
- address translation
- garbage collection
- wear-leveling
- bad block management
- ECC
- NAND channel scheduling
- cache/buffer 관리

## Microcontroller

Microcontroller는 SSD firmware를 실행하는 두뇌 역할을 한다. 자료에서는 ARM7, ARM9 같은 embedded processor 예를 든다.

Microcontroller는 internal bus를 통해 SRAM, DRAM controller, NAND controller, host interface를 제어한다.

## Internal Bus

SSD controller 내부 bus로 AMBA AHB 같은 high-performance bus를 사용할 수 있다.

Internal bus가 연결하는 대상:

- microcontroller
- SRAM controller
- DRAM controller
- NAND flash controller
- host interface block

## SRAM과 SRAM Controller

SRAM은 가장 빠른 on-chip memory로, 작은 table이나 firmware working data를 저장하는 데 쓰인다.

특징:

- 빠른 access
- refresh 불필요
- 면적이 큼
- 용량은 제한적

SRAM controller는 bus transaction을 SRAM read/write로 변환한다.

## External Interface

SSD는 host와 ATA, SATA, USB, PCIe 같은 interface로 통신한다.

| Interface | 특징 |
|---|---|
| PATA | parallel ATA, 구형 |
| SATA | serial ATA, HDD/SSD에서 널리 사용 |
| USB | 범용 외부 storage |
| PCIe | 고속, 낮은 latency |

고속 interface는 NAND 내부 parallelism과 controller 성능을 충분히 끌어낼 수 있어야 한다.

## Cache Buffer와 DRAM

DRAM은 cache buffer와 mapping table 저장에 사용된다.

역할:

- host read/write data buffering
- write coalescing
- mapping table cache
- garbage collection 중 임시 data 저장

DRAM controller는 refresh, timing, burst transfer 등을 관리한다.

## NAND Flash Memory

NAND flash는 SSD의 실제 non-volatile storage이다.

특징:

- page 단위 read/program
- block 단위 erase
- erase-before-write
- SLC, MLC 등 cell당 저장 bit 수에 따라 성능/수명 차이

## NAND Flash Controller

NAND controller는 NAND command, address, data timing을 생성한다.

주요 기능:

- page read
- page program
- block erase
- ECC encode/decode
- bad block table 관리
- multiple chip/channel control

## SLC와 MLC

| 종류 | Cell당 bit | 장점 | 단점 |
|---|---:|---|---|
| SLC | 1 bit | 빠르고 endurance 높음 | bit당 비용 높음 |
| MLC | 2 bit 이상 | 저장 밀도 높음 | 속도와 endurance 불리 |

## 시험ㆍ복습 체크포인트

- SSD controller가 host block I/O와 NAND flash operation 사이에서 하는 일을 설명할 수 있어야 한다.
- SRAM과 DRAM이 SSD controller 안에서 각각 왜 필요한지 말할 수 있어야 한다.
- NAND flash의 page/program과 block/erase 단위 차이를 이해해야 한다.
- SLC와 MLC의 성능/수명/비용 tradeoff를 설명할 수 있어야 한다.

