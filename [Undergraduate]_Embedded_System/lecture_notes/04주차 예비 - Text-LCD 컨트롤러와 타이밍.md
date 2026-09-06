---
title: "04주차 예비 - Text-LCD 컨트롤러와 타이밍"
course: "Embedded System"
week: 4
type: "pre"
tags:
  - embedded-system
  - text-lcd
  - timing
  - controller
---

# 04주차 예비 - Text-LCD 컨트롤러와 타이밍

이전: [[03주차 결과 - 7-Segment 시계와 전광판 구현]]  
다음: [[04주차 결과 - Text-LCD 문자열 출력과 버튼 회전]]

## 핵심 요약

이번 예비보고서는 Text-LCD 컨트롤러 설계를 준비한다. 핵심은 LCD 모듈이 단순한 LED 배열이 아니라 내부 controller와 DDRAM/CGRAM을 가진 장치이며, `RS`, `R/W`, `E`, `DB[7:0]` 신호를 데이터시트의 timing constraint에 맞춰 제어해야 한다는 점이다.

## 목표

- LCD와 Text-LCD의 구조를 이해한다.
- LCD 제어 코드와 DDRAM address 구조를 이해한다.
- Character LCD write/read timing diagram을 해석한다.
- RPS-Z7020-TK 보드에서 JTAG 모드로 PL 회로를 검증하는 절차를 확인한다.

## LCD와 Text-LCD

LCD는 전압에 따라 액정 분자의 배열이 변하고, 그 결과 빛의 투과가 달라지는 원리를 이용한다. Text-LCD는 LCD 패널과 제어기가 결합된 모듈이므로, 설계자는 픽셀을 직접 제어하기보다 controller에 명령어와 문자 데이터를 전송한다.

## LCD 인터페이스 신호

| 신호 | 역할 |
|---|---|
| `DB0~DB7` | 8비트 데이터 버스 |
| `E` | Enable, 데이터 전송 타이밍 제어 |
| `R/W` | 읽기/쓰기 선택 |
| `RS` | 명령 register와 data memory 선택 |
| `Vdd`, `Vss` | 전원 |

`RS=0`이면 명령어 또는 busy flag/address 관련 접근이고, `RS=1`이면 DDRAM/CGRAM 데이터 접근이다. `R/W=0`은 write, `R/W=1`은 read를 의미한다.

## 주요 LCD 명령

| 명령 범주 | 기능 |
|---|---|
| Entry mode set | 문자 입력 후 address 증가/감소, display shift 여부 설정 |
| Display on/off | 화면, 커서, blink on/off |
| Cursor/display shift | 커서 또는 화면을 좌우 이동 |
| Function set | 8비트/4비트, 1행/2행, font 크기 설정 |
| CGRAM address set | 사용자 문자 생성용 메모리 주소 설정 |
| DDRAM address set | 화면에 표시될 문자 메모리 주소 설정 |
| Data write/read | CGRAM 또는 DDRAM에 문자 데이터 접근 |

## Text-LCD 동작 타이밍

LCD는 write/read 동작에서 setup time, hold time, enable pulse width 등을 만족해야 한다. 따라서 Verilog 코드에서는 단순히 값을 바로 바꾸는 것이 아니라 counter를 이용해 충분한 delay를 두고 `lcd_en`을 High/Low로 전환해야 한다.

## 구동 회로 블록

Text-LCD 구동 회로는 다음 흐름으로 구성된다.

1. 입력 clock을 counter x2000에 넣어 LCD enable timing을 만든다.
2. counter x40이 LCD 동작 단계, 즉 mode를 진행시킨다.
3. mode decoder가 전원 설정, function set, address set, write 동작을 순서대로 선택한다.
4. decoder output이 `RS`, `R/W`, `DB[7:0]`로 나간다.

## 보드 검증 조건

Zynq 보드는 Boot/JTAG mode switch로 부팅 방식을 선택한다. 이번 실습에서는 T-flash를 사용하지 않고 Cascaded JTAG로 PL을 다운로드한다.

확인 사항:

- BOOT_MODE[3]에 해당하는 J19 점퍼를 2-3 연결
- T-flash 제거
- JTAG 연결 후 bitstream 다운로드

## 정리

4주차 예비의 핵심은 Text-LCD가 내부 controller를 가진 장치이므로, 문자 데이터뿐 아니라 **명령 순서와 timing** 이 설계의 중심이라는 점이다. 다음 결과 실습에서는 이 개념이 `lcd_mode`, `count_lcd`, `set_data`로 구현된다.
