# SSD Software 2 - Superblock, LAST, FTL Functions

tags: #micro-processor #ssd #ftl #superblock #last #bad-block-management #wear-leveling

관련 노트: [[09 SSD Software 1 - FTL Overview BAST FAST]], [[10A SSD Software 2 - Superblock LAST Annotated Image Copy]]

## 핵심 요약

이 자료는 block-mapped FTL의 대표 방식 중 Superblock FTL과 LAST를 설명하고, bad block management와 wear-leveling을 자세히 다룬다. Superblock FTL은 인접 logical block을 묶어 merge 비용을 줄이고, LAST는 access locality를 이용해 hot/cold data를 나누어 full merge를 줄인다.

## Superblock FTL 개요

Superblock은 인접한 여러 logical block을 하나의 큰 group으로 묶은 것이다.

도입 이유:

- BAST는 large block NAND 특성 때문에 full merge가 많아질 수 있다.
- FAST는 single log block을 여러 data block이 공유하면서 full merge overhead가 생길 수 있다.
- Superblock은 block-level mapping table과 비슷한 크기의 mapping table을 유지하면서 더 유연한 page 배치를 제공한다.

## D-Block과 U-Block

Superblock FTL은 D-block과 U-block을 사용한다.

| 블록 | 의미 |
|---|---|
| D-block | data block 역할 |
| U-block | update/log block 역할 |

Update data는 U-block에 기록되고, mapping table이 logical page의 실제 위치를 추적한다.

## Three-Level Mapping

Superblock FTL은 3-level mapping table을 사용한다.

개념적으로:

1. Logical superblock index를 찾는다.
2. Superblock 내부 logical block/page offset을 계산한다.
3. PBMT 같은 physical block mapping table로 실제 physical block을 찾는다.

이 구조는 mapping table 크기를 block-level 방식에 가깝게 유지하면서 page 배치 유연성을 높인다.

## PBMT

PBMT(Physical Block Mapping Table)는 physical block mapping 정보를 담는다. NAND flash는 느리므로 mapping information을 매번 flash에서 읽으면 성능이 떨어진다. 따라서 controller는 mapping table cache와 SRAM/DRAM 사용을 함께 고려한다.

## Superblock Garbage Collection

Superblock FTL의 GC는 U-block과 D-block을 고려한다.

대표 흐름:

1. Free block이 부족하면 victim superblock을 찾는다.
2. Invalid page가 많거나 merge cost가 낮은 superblock을 선택한다.
3. U-block과 D-block의 valid page를 정리한다.
4. Partial merge 또는 full merge를 수행한다.

자료는 Superblock이 BAST/FAST 대비 약 30-32% 성능 향상을 보이며, switch merge 비중이 증가한다고 설명한다.

## Parameter Sensitivity

U-block 수가 많을수록 update를 흡수할 여지가 늘어 성능이 좋아질 수 있다. 하지만 DRAM/cache hit rate가 높거나 workload locality가 강하면 parameter 변화의 영향이 줄어들 수 있다.

## LAST 개요

LAST는 Locality-Aware Sector Translation이다. Access pattern의 locality를 관찰해 hot data와 cold data를 구분하고, 각각 다른 log block partition에서 관리한다.

## Hot Data와 Cold Data

| 분류 | 의미 |
|---|---|
| Hot data | 자주 update되는 data |
| Cold data | 드물게 update되는 data |

Hot data가 cold data와 섞이면 cold valid page까지 merge에 끌려 들어가 full merge 비용이 커진다. LAST는 hot/cold를 분리해 dead block을 더 많이 만들고 full merge를 줄인다.

## LAST Scheme

LAST는 random log block을 hot partition과 cold partition으로 나눈다.

핵심 구성:

- locality detector
- hot/cold classifier
- hot log blocks
- cold log blocks
- adaptive partition control

Locality threshold와 partition size는 workload에 따라 조정된다.

## LAST Garbage Collection

LAST의 GC는 두 단계로 볼 수 있다.

1. Candidate partition이나 block group을 정한다.
2. Victim block을 선택하고 merge를 수행한다.

Hot partition에서는 invalid page 비율이 높아 full merge 비용이 줄어들 가능성이 크다.

## Adaptiveness

LAST는 hot/cold partition size와 locality threshold를 workload에 맞게 조정한다. Hot data가 많으면 hot partition을 늘리고, cold data가 많으면 cold partition을 늘리는 식이다.

## Bad Block Management

NAND flash에는 bad block이 존재하므로 FTL은 bad block을 free block pool에서 제외하거나 replacement block으로 교체해야 한다.

기본 처리:

1. 특정 page에서 error가 발생한다.
2. 해당 block의 정상 page를 buffer 또는 다른 block으로 복사한다.
3. bad block을 더 이상 사용하지 않도록 표시한다.
4. mapping table을 갱신한다.

Bad block management는 hardware와 software가 함께 담당할 수 있지만, FTL은 free block allocation을 관리하므로 bad block 회피에 중요한 역할을 한다.

## Wear-Leveling

Wear-leveling은 block erase count를 고르게 만들어 flash 수명을 늘리는 기법이다.

### Dynamic Wear-Leveling

자주 write되는 data를 erase count가 낮은 block에 배치해 hot block 마모를 완화한다.

### Static Wear-Leveling

거의 변하지 않는 cold data가 낮은 erase count block을 오래 점유하면 다른 block만 계속 마모된다. Static wear-leveling은 cold data도 가끔 이동시켜 전체 erase count를 균등하게 만든다.

## 정리

FTL의 주요 기능:

- address translation
- garbage collection
- bad block management
- wear-leveling

자료의 결론은 BAST, FAST, Superblock, LAST 중 LAST가 locality를 활용해 가장 좋은 성능을 보인다는 것이다.

## 시험ㆍ복습 체크포인트

- Superblock이 BAST/FAST의 full merge 문제를 어떻게 줄이는지 설명할 수 있어야 한다.
- PBMT와 3-level mapping의 역할을 이해해야 한다.
- LAST에서 hot/cold data를 나누는 이유를 말할 수 있어야 한다.
- Bad block management 절차를 순서대로 설명할 수 있어야 한다.
- Dynamic wear-leveling과 static wear-leveling을 구분할 수 있어야 한다.

