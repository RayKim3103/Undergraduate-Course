# SSD Software 1 - FTL Overview, BAST, FAST

tags: #micro-processor #ssd #ftl #address-translation #garbage-collection #bast #fast

관련 노트: [[08 SSD Hardware 4 - NAND Flash Memory]], [[10 SSD Software 2 - Superblock LAST and FTL Functions]]

## 핵심 요약

이 자료는 SSD software의 핵심인 Flash Translation Layer를 설명한다. FTL은 host가 보는 logical block interface를 NAND flash의 physical page/block operation으로 변환한다. Address translation, garbage collection, bad block management, wear-leveling을 다루고, block-mapped FTL의 대표 방식인 BAST와 FAST를 비교한다.

## FTL의 역할

Flash Translation Layer는 host의 LBA(Logical Block Address)를 NAND flash의 physical address로 변환한다.

FTL이 필요한 이유:

- NAND는 erase-before-write 특성을 가진다.
- Program은 page 단위, erase는 block 단위이다.
- Bad block이 존재한다.
- Cell endurance가 제한되어 wear-leveling이 필요하다.
- Legacy file system은 HDD와 같은 block device interface를 기대한다.

## FTL과 File System

Legacy file system은 logical sector를 overwrite할 수 있다고 가정한다. FTL은 실제 flash에서는 새 physical page에 out-of-place update를 수행하고 mapping table을 갱신한다.

```text
File system LBA -> FTL mapping -> physical flash page/block
```

## Address Translation

### Page-Mapped FTL

Page 단위 mapping을 사용한다.

장점:

- update flexibility가 높다.
- random write 처리에 강하다.
- garbage collection 비용을 줄일 수 있다.

단점:

- mapping table이 매우 크다.

예를 들어 64 GB SSD에서 4 KB page마다 4 B mapping entry를 두면 table 크기가 커진다.

### Block-Mapped FTL

Block 단위 mapping을 사용한다.

장점:

- mapping table이 작다.
- SRAM/DRAM 요구량이 줄어든다.

단점:

- page-level update flexibility가 낮다.
- random write에서 merge 비용이 커질 수 있다.

자료에서는 block-mapped FTL을 중심으로 다룬다.

## Garbage Collection

Out-of-place update가 반복되면 invalid page가 쌓이고 free block이 부족해진다. Garbage collection은 victim block을 선택하고 valid page를 복사한 뒤 block을 erase해 free block으로 만든다.

GC 비용은 FTL 성능에 큰 영향을 준다.

## Merge Operation

Block-mapped FTL에서는 data block과 log block을 합치는 merge가 중요하다.

| Merge | 비용 | 설명 |
|---|---|---|
| Switch merge | 가장 빠름 | log block이 새 data block으로 바로 전환 가능 |
| Partial merge | 중간 | 일부 valid page만 복사 |
| Full merge | 가장 느림 | data block과 log block 모두에서 valid page를 모아 새 block 생성 |

Full merge는 erase와 copy가 많아 성능 저하가 크다.

## Bad Block Management

FTL은 어떤 block이 bad block인지 파악하고, bad block을 mapping 대상에서 제외해야 한다. Factory bad block과 runtime bad block 모두 관리 대상이다.

## Wear-Leveling

Flash cell은 program/erase 횟수 수명이 제한되어 있다. 특정 logical range만 자주 write되면 일부 block이 빨리 마모된다.

Wear-leveling은 erase count가 고르게 분포되도록 physical block을 재배치한다.

## BAST

BAST는 Block-level Associative Sector Translation이다.

특징:

- block-mapped FTL
- 각 data block에 log block을 associatively 연결
- incoming LBA의 LSB를 page offset으로 사용
- MSB를 block-level mapping table index로 사용
- switch merge와 full merge를 지원

### BAST의 문제

Random update가 여러 data block에 흩어지면 log block이 부족해지고 full merge가 자주 발생한다. Thrashing과 frequent full merge가 성능 문제이다.

## FAST

FAST는 Fully Associative Sector Translation이다.

목표:

- BAST의 log block utilization 문제 완화
- switch/partial merge 기회 증가
- log block을 더 유연하게 사용

FAST는 sequential log block과 random log block을 구분한다.

### Sequential Log Block

Sequential write pattern이 조건을 만족하면 switch merge가 가능하다. 조건이 부족하면 partial merge가 필요할 수 있다.

### Random Log Block

Random update를 fully associative하게 수용한다. 하지만 random log block merge 시 여러 data block과 관련된 page를 처리해야 하므로 full merge가 여러 번 발생할 수 있다.

자료의 예에서는 1 page update 때문에 3번 full merge가 필요한 상황을 보여주며, 이것이 FAST의 큰 overhead가 될 수 있음을 설명한다.

## BAST와 FAST 비교

| 항목 | BAST | FAST |
|---|---|---|
| Log block 연결 | data block과 associative | fully associative |
| 장점 | 구조 단순 | log block utilization 개선 |
| 약점 | thrashing, full merge 빈번 | random log merge 비용 큼 |
| Merge | switch/full 중심 | sequential/random log에 따라 다양 |

## 시험ㆍ복습 체크포인트

- FTL이 erase-before-write 문제를 어떻게 숨기는지 설명할 수 있어야 한다.
- Page mapping과 block mapping의 memory/performance tradeoff를 비교해야 한다.
- Switch, partial, full merge의 비용 차이를 이해해야 한다.
- BAST에서 full merge가 자주 발생하는 이유를 말할 수 있어야 한다.
- FAST가 BAST의 어떤 문제를 해결하려고 하는지 설명할 수 있어야 한다.

