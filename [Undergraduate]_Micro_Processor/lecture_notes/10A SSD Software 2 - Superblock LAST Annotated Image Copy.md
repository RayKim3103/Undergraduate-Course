# SSD Software 2 - Superblock LAST Annotated Image Copy

tags: #micro-processor #ssd #ftl #superblock #last #annotated-copy

관련 노트: [[10 SSD Software 2 - Superblock LAST and FTL Functions]]

## 핵심 요약

이 파일은 `SSD Software Details 2`의 image/annotated copy이다. 텍스트 추출은 제한적이지만, 렌더링 확인 결과 Superblock FTL과 LAST, bad block management, wear-leveling을 다루는 같은 강의자료 계열이다. 필기 흔적이 있는 페이지에서는 LAST의 locality-aware sector translation, hot/cold partition, full merge 감소가 강조되어 있다.

## 자료 성격

이 사본은 텍스트 기반 PDF인 [[10 SSD Software 2 - Superblock LAST and FTL Functions]]와 같은 53쪽 강의의 이미지 기반 버전으로 보인다. 따라서 개념 흐름은 10번 노트와 동일하게 잡고, 필기/주석이 있는 복습용 사본으로 활용하면 좋다.

## Superblock FTL 복습

Superblock FTL은 인접한 logical block을 하나의 superblock으로 묶어 update block 사용을 더 유연하게 만든다.

핵심:

- D-block과 U-block 사용
- three-level mapping
- PBMT 활용
- partial/switch merge 비중 증가
- BAST/FAST보다 full merge overhead 감소

## LAST 복습

LAST는 Locality-Aware Sector Translation이다. 접근 locality를 관찰해 random log block을 hot/cold partition으로 나눈다.

필기 확인상 강조된 내용:

- locality를 이용한 sector translation
- hot data와 cold data 분리
- hot partition은 dead block을 더 많이 만들어 full merge 감소
- cold data가 hot update와 섞이면 merge cost가 커짐

## Hot/Cold 분리의 의미

Hot data는 자주 갱신되어 invalid page가 빨리 생긴다. Cold data는 오래 valid 상태로 남는다. 둘이 같은 log block에 섞이면 garbage collection 시 cold valid page를 계속 복사해야 한다.

```text
hot/cold 분리 -> valid page copy 감소 -> full merge cost 감소
```

## Bad Block Management 복습

Bad block은 초기 제조 시점부터 있거나 사용 중 발생할 수 있다. FTL은 bad block을 사용하지 않도록 mapping table과 free block pool을 관리한다.

처리 흐름:

1. Error 발생 block 감지
2. 정상 page를 다른 block으로 이동
3. Bad block 표시
4. Mapping 갱신

## Wear-Leveling 복습

Wear-leveling은 모든 block의 erase count를 균등하게 만드는 작업이다.

- Dynamic wear-leveling: update되는 data 중심으로 마모 분산
- Static wear-leveling: 오래 움직이지 않는 cold data도 이동해 전체 block을 고르게 사용

## 이 사본으로 공부할 때 볼 것

- 주석이 있는 page에서 교수자가 강조한 단어와 그림을 먼저 본다.
- 텍스트 검색은 제한적이므로 개념 검색은 10번 노트를 사용한다.
- Superblock과 LAST의 공통 목표가 merge 비용 감소라는 점을 중심으로 연결한다.
- LAST의 차별점은 locality detection과 hot/cold partition이다.

## 시험ㆍ복습 체크포인트

- LAST의 약자를 풀어 쓰고 핵심 아이디어를 설명할 수 있어야 한다.
- Hot/cold partition이 full merge를 줄이는 이유를 말할 수 있어야 한다.
- Superblock FTL과 LAST의 차이를 비교할 수 있어야 한다.
- 이 사본은 필기/이미지 확인용이고, 상세 텍스트 정리는 10번 노트와 함께 보는 것이 좋다.

