---
title: "03주차 예비 - LED와 7-Segment 디지털 시계"
course: "Embedded System"
week: 3
type: "pre"
tags:
  - embedded-system
  - seven-segment
  - led
  - clock
---

# 03주차 예비 - LED와 7-Segment 디지털 시계

이전: [[02주차 결과 - Vivado PS PL LED 실습]]  
다음: [[03주차 결과 - 7-Segment 시계와 전광판 구현]]

## 핵심 요약

이번 예비보고서는 LED와 7-segment를 이용해 디지털 시계를 설계하기 위한 배경을 정리한다. 핵심은 25MHz clock을 분주하여 1초 단위 시간을 만들고, 8개의 7-segment 자리를 빠르게 순환 구동하여 사람이 동시에 켜진 것처럼 보게 만드는 것이다.

## 목표

- LED와 7-segment의 구조와 구동 방식을 이해한다.
- Anode형과 Cathode형 7-segment의 차이를 구분한다.
- 25MHz clock 기반 시계 회로의 모듈 구조를 이해한다.
- Vivado와 ModelSim을 이용한 설계 및 검증 흐름을 준비한다.

## LED

LED는 전류가 흐를 때 빛을 방출하는 PN 접합 반도체 소자이다. 전자와 정공이 결합하면서 에너지가 광자로 방출된다. 보드의 LED는 FPGA의 PL I/O 핀과 연결되어 있고, RTL에서 출력한 bit 값에 따라 on/off된다.

## 7-Segment

7-segment는 `A~G` 7개 segment와 `DP` decimal point로 구성된다. 조합에 따라 숫자 0~9, 일부 문자, 16진수 A~F 등을 표시할 수 있다.

| 구조 | 공통 단자 | 켜지는 조건 |
|---|---|---|
| Common Anode | VCC | segment 입력이 0일 때 |
| Common Cathode | GND | segment 입력이 1일 때 |

RPS-Z7020-TK 보드는 Anode형 7-segment를 사용한다. 따라서 cathode형 기준으로 만든 segment pattern은 실제 출력에서 bit 반전이 필요할 수 있다.

## 다중 자리 표시 원리

8개의 7-segment 자리를 모두 동시에 독립 구동하는 대신, 한 번에 한 자리씩 빠르게 켜고 끄는 방식을 사용한다.

1. 첫 번째 digit 선택, 해당 segment data 출력
2. 두 번째 digit 선택, 해당 segment data 출력
3. 같은 방식으로 모든 digit 순환
4. 충분히 빠르게 반복하면 사람 눈에는 모든 자리가 동시에 켜진 것처럼 보인다.

이 방식은 multiplexing이며, `segcom`이 digit 선택을, `segout`이 segment pattern을 담당한다.

## 디지털 시계 블록

| 모듈 | 역할 |
|---|---|
| `bin2seg.v` | 4비트 숫자를 7-segment 8비트 패턴으로 변환 |
| `seven_seg.v` | 32비트 표시 데이터를 8개 digit으로 나누어 순환 출력 |
| `clock.v` | 25MHz clock을 이용해 초, 분, 시간 값을 생성 |
| `top.v` | 하위 모듈을 연결하는 최상위 모듈 |

## 데이터 구조

7-segment 1자리는 4비트 숫자 하나로 표현된다. 8자리이므로 전체 시간 데이터는 32비트가 된다.

예시:

| 자리 | 의미 | `segdata` 범위 |
|---|---|---|
| 1 | 10시간 | `[31:28]` |
| 2 | 1시간 | `[27:24]` |
| 3 | 구분자 | `[23:20]` |
| 4 | 10분 | `[19:16]` |
| 5 | 1분 | `[15:12]` |
| 6 | 구분자 | `[11:8]` |
| 7 | 10초 | `[7:4]` |
| 8 | 1초 | `[3:0]` |

## 정리

3주차 예비의 핵심은 시계 회로를 단순한 소프트웨어 카운터가 아니라, clock divider, 자리별 counter, binary-to-segment decoder, digit multiplexing으로 나누어 보는 것이다. 이 구조를 이해하면 초기 시각 변경이나 전광판 구현도 `segdata`를 어떻게 업데이트하느냐의 문제로 정리된다.
