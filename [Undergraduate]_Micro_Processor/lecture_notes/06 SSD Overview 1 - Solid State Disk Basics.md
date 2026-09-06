# SSD Overview 1 - Solid State Disk Basics

tags: #micro-processor #ssd #flash-memory #storage #hdd #sata #pcie

관련 노트: [[05 Computer Organization Chapter 7 - Multiprocessors]], [[07 SSD Overview 2 - SSD Architecture]]

## 핵심 요약

이 자료는 SSD의 정의, HDD와의 차이, interface, hybrid SSD, SSD 내부 구조 개요, 성능 비교, NAND flash 시장 흐름을 소개한다. 핵심은 SSD가 moving part가 없는 non-volatile memory 기반 저장장치라서 random access latency가 HDD보다 훨씬 낮다는 점이다.

## SSD 정의

SSD는 Solid-State Disk 또는 Solid-State Drive의 약자로, magnetic disk가 아니라 non-volatile memory chip을 사용해 data를 저장하는 장치이다.

Solid-state의 의미:

- vacuum tube나 gas-discharge tube가 아님
- relay, switch처럼 움직이는 electro-mechanical device가 아님
- transistor, microprocessor, DRAM, flash memory처럼 고체 물질 기반

## SSD 특징

장점:

- random access latency가 낮음
- mechanical seek가 없음
- 충격에 강함
- 소음이 없음
- 전력 소모가 낮을 수 있음

한계:

- flash cell의 program/erase cycle 수명 제한
- erase-before-write 특성
- write amplification과 garbage collection overhead
- controller와 FTL 설계가 성능에 큰 영향

## SSD Interface

대표 interface:

- SATA
- PCI Express
- rackmount storage interface

SATA는 HDD와 호환되는 storage interface로 출발했고, PCIe 기반 SSD는 더 높은 bandwidth와 낮은 latency를 제공한다.

## Hybrid SSD

Hybrid SSD는 DRAM, NAND flash, magnetic disk 또는 다른 storage 계층을 조합해 성능과 비용을 절충한다. 자주 접근하는 data는 빠른 memory에 두고, 대용량 data는 비용이 낮은 media에 둔다.

## SSD와 HDD 내부 차이

| 항목 | SSD | HDD |
|---|---|---|
| 저장 매체 | NAND flash memory | magnetic platter |
| 움직이는 부품 | 없음 | spindle, head 존재 |
| Random access | 빠름 | seek와 회전 대기 필요 |
| Sequential access | interface/controller 영향 큼 | platter transfer rate 영향 |
| 내구성 | write endurance 관리 필요 | mechanical failure 가능 |

## 성능 비교

### Access Time

HDD는 head 이동과 rotational latency가 필요하다. SSD는 전기적으로 cell과 page에 접근하므로 random access time이 훨씬 작다.

### Sequential Read/Write

Sequential access에서는 interface bandwidth와 내부 parallelism이 중요하다. HDD도 sequential transfer에서는 비교적 강하지만, SSD는 여러 NAND channel을 병렬로 사용해 높은 throughput을 낼 수 있다.

### Random Read/Write

SSD는 random read에서 매우 강하다. Random write는 erase-before-write, garbage collection, write amplification 때문에 controller와 FTL 설계에 크게 좌우된다.

## NAND Flash 시장

자료는 NAND flash가 storage 시장에서 지배적 위치를 차지하게 된 흐름과 SSD 가격이 HDD 가격에 가까워지는 추세를 소개한다. Flash density 증가와 가격 하락은 SSD 보급의 핵심 배경이다.

## War of the Disks

SSD와 HDD는 성능, 가격, 용량, 신뢰성에서 서로 다른 장단점을 가진다. SSD는 latency와 random I/O가 강하고, HDD는 대용량당 비용에서 강점이 있었다. 시간이 지나며 SSD 가격이 낮아지면서 적용 범위가 넓어졌다.

## 시험ㆍ복습 체크포인트

- SSD가 HDD보다 random access에 강한 이유를 설명할 수 있어야 한다.
- SSD의 장점과 flash memory 기반 한계를 함께 말할 수 있어야 한다.
- SATA와 PCIe interface의 성능 관점 차이를 이해해야 한다.
- Random write 성능이 FTL과 garbage collection에 좌우되는 이유를 설명할 수 있어야 한다.

