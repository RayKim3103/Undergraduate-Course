---
title: "06주차 예비 - TFT-LCD ASCII 문자 출력"
course: "Embedded System"
week: 6
type: "pre"
tags:
  - embedded-system
  - ascii
  - tft-lcd
  - bram
---

# 06주차 예비 - TFT-LCD ASCII 문자 출력

이전: [[05주차 결과 - TFT-LCD 타이밍 BRAM 색상 패턴]]  
다음: [[06주차 결과 - ASCII 문자 생성기와 화면 표시]]

## 핵심 요약

이번 예비보고서는 ASCII 문자 데이터를 TFT-LCD 화면에 출력하기 위한 구조를 정리한다. 핵심은 문자열 자체를 곧바로 화면에 보내는 것이 아니라, ASCII 코드, 문자 ROM, 문자 위치 좌표, 8x8 bitmap pixel 변환 과정을 거쳐 RGB video data로 출력한다는 점이다.

## 목표

- ASCII 코드의 의미와 구조를 이해한다.
- TFT-LCD 문자 출력용 주요 Verilog 모듈의 역할을 파악한다.
- Dual Port BRAM이 문자 데이터 저장에 사용되는 이유를 이해한다.
- JTAG 기반 보드 검증 조건을 확인한다.

## ASCII

ASCII는 7비트 문자 인코딩 표준이며 총 128개 코드를 정의한다.

| 구분 | 내용 |
|---|---|
| 제어 문자 | 통신/제어용, 출력되지 않는 문자 |
| 출력 문자 | 알파벳, 숫자, 특수문자, 공백 |
| 예시 | `A=65`, `B=66`, 공백 `0x20` |

실습에서는 각 문자를 8비트 값으로 저장하여 BRAM과 ROM에서 처리한다.

## 주요 모듈

| 모듈 | 역할 |
|---|---|
| `SVGA_DEFINES.v` | 해상도, porch, sync timing 등 상수 정의 |
| `SVGA_TIMING_GENERATION.v` | `HSYNC`, `VSYNC`, pixel/character 좌표 생성 |
| `CHAR_DISPLAY.v` | 화면에 표시할 문자열과 색상 결정 |
| `CHAR_GEN.v` | ASCII code를 문자 bitmap pixel로 변환 |
| `CHAR_GEN_ROM.v` | 각 ASCII 문자의 8x8 bitmap data 저장 |
| `CHAR_DPRAM.xci` | 출력할 ASCII 문자 code를 저장하는 dual-port BRAM |
| `VIDEO_OUT.v` | 최종 RGB와 sync 신호 출력 |

## BRAM 구조

실습에서 사용하는 `CHAR_DPRAM.xci`는 True Dual Port RAM이다.

- Port A: 문자 데이터를 write하는 용도
- Port B: 현재 화면 좌표에 대응하는 ASCII를 read하는 용도
- Width: 8bit
- Depth: 16383
- 초기값: remaining memory location을 `0x20`으로 설정하여 공백 문자로 초기화

Dual port 구조를 사용하면 한쪽 포트에서 화면에 표시할 문자를 계속 쓰는 동안, 다른 포트에서 문자 데이터를 읽어 bitmap 변환을 수행할 수 있다.

## 화면 문자 좌표

TFT-LCD는 480x272 해상도이고, 한 문자가 8x8 pixel block을 사용한다. 이론적으로는 가로 60문자, 세로 34문자 정도가 가능하지만, 실제 skeleton code에서는 안정적인 표시 범위를 위해 더 작은 문자 배열을 사용한다.

## 검증 조건

5주차와 같이 Cascaded JTAG 통신을 사용한다. BOOT_MODE J19가 2-3번 핀으로 연결되어 있는지 확인하고, 보드 실습을 진행한다.

## 정리

6주차 예비의 핵심은 문자 출력이 `문자열 -> ASCII -> 문자 ROM address -> 8x8 bitmap -> pixel_on -> RGB`의 흐름이라는 점이다. 특히 timing generator가 pixel 좌표뿐 아니라 현재 문자 line/column까지 제공한다는 점이 중요하다.
