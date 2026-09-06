---
title: "04주차 결과 - Text-LCD 문자열 출력과 버튼 회전"
course: "Embedded System"
week: 4
type: "result"
tags:
  - embedded-system
  - text-lcd
  - verilog
  - debounce
---

# 04주차 결과 - Text-LCD 문자열 출력과 버튼 회전

이전: [[04주차 예비 - Text-LCD 컨트롤러와 타이밍]]  
다음: [[05주차 예비 - TFT-LCD 구조와 BRAM 영상 표시]]

## 핵심 요약

이번 실습에서는 Text-LCD controller RTL을 분석하고, LCD 2행 16문자 출력이 어떤 timing과 mode sequence로 이루어지는지 확인했다. Quiz에서는 버튼 입력에 따라 문자열을 1/4 단위로 좌우 회전하거나 새로운 메시지로 교체하도록 수정했다.

## `text_lcd.v` 구조

| 구성 | 역할 |
|---|---|
| 문자 macro | 각 문자에 대응하는 8비트 ASCII-like 값 정의 |
| `reg_a~reg_h` | 32비트씩 총 8개, 4문자 단위 문자열 저장 |
| `delay_lcdclk` | LCD enable pulse timing 생성 |
| `count_lcd` | LCD sequence 진행 counter |
| `lcd_mode` | 현재 LCD 동작 mode |
| `set_data[9:0]` | `{RS, R/W, DB[7:0]}` 묶음 |

`reg_a~reg_d`는 1행 16문자, `reg_e~reg_h`는 2행 16문자를 담당한다. 1문자는 8비트이므로 32비트 register 하나가 4문자를 담는다.

## Counter와 Mode Sequence

`delay_lcdclk`는 0~1999를 반복한다. 이 counter가 특정 값일 때 `lcd_en`을 제어한다.

- `delay_lcdclk == 200`: `lcd_en = 1`
- `delay_lcdclk == 1800`: `lcd_en = 0`

`count_lcd`는 `delay_lcdclk == 0`일 때 증가하며, LCD 초기화와 문자 쓰기 순서를 만든다.

| `count_lcd` | mode | 동작 |
|---|---|---|
| 0 | `mode_pwron` | 전원 설정 |
| 1 | `mode_fnset` | function set |
| 2 | `mode_onoff` | display on/off |
| 3~5 | `mode_entr*` | entry/home/clear |
| 6 | `mode_seta1` | 1행 DDRAM address 설정 |
| 7~22 | `mode_wr1st` | 1행 16문자 write |
| 23 | `mode_seta2` | 2행 DDRAM address 설정 |
| 24~39 | `mode_wr2nd` | 2행 16문자 write |
| 40 | `mode_delay` | 안정화 지연 |

초기 설정 이후에는 `count_lcd`가 6으로 돌아가므로, 1행/2행 쓰기 sequence를 반복하며 화면을 갱신한다.

## `top.v`

`top.v`는 Text-LCD 관련 6개 포트만 실제로 사용한다.

- `resetn`
- `lcdclk`
- `lcd_rs`
- `lcd_rw`
- `lcd_en`
- `lcd_data[7:0]`

Processor System 관련 포트가 포함되어 있으나 이번 실습에서는 PL 단독 Text-LCD 구동이므로 실질적으로 사용되지 않는다.

## Quiz: 버튼 기반 문자열 회전

수정 목표:

- Button 0: 각 행의 문자열을 1/4만큼 right rotate
- Button 1: 각 행의 문자열을 1/4만큼 left rotate
- Button 2: 문자열을 `"MESSAGE         ROTATION        "`으로 교체

수정 포인트:

- 버튼 입력을 받기 위해 `PushButton[2:0]` 포트 추가
- `reg_a~reg_h`를 `wire`에서 `reg`로 변경
- 버튼 edge 검출을 위해 `RegPushButton` 추가
- reset 시 `"2025 EMBEDDED    SYSTEM LAB      "` 초기화

1행과 2행을 따로 회전시킨 이유는 `reg_a~reg_d`가 1행, `reg_e~reg_h`가 2행을 담당하기 때문이다. 16문자를 4문자 단위로 이동하므로 1/4 회전이 된다.

## Debouncer 고찰

빠른 버튼 입력이나 불안정한 접점 때문에 LCD 문자가 깨지는 문제가 있었다. 이를 줄이기 위해 debouncer를 고려했다. Debouncer는 버튼 입력을 flip-flop으로 안정화하고, 유효한 edge만 짧게 만들어 노이즈 영향을 줄인다.

다만 적용 과정에서 reset 후 문자열이 의도치 않게 1/4 right rotate되는 문제가 발생했다. 가능한 원인은 다음과 같이 정리된다.

- 실제 보드에서 glitch 또는 clock skew로 timing constraint가 깨졌을 가능성
- decoder output always block의 sensitivity 또는 reset 반영 방식이 불완전했을 가능성
- debouncer 인스턴스 위치와 reset sequence가 LCD 초기화 sequence와 충돌했을 가능성

개선 방향은 debouncer를 Text-LCD 모듈 내부 clock domain에 맞춰 인스턴스화하고, reset edge가 decoder output에도 즉시 반영되도록 always 조건을 보완하는 것이다.

## 정리

4주차 결과의 핵심은 Text-LCD 출력이 단순 문자열 대입이 아니라, **DDRAM address 설정, 문자 단위 write, enable timing, mode sequence** 의 조합이라는 점이다. 버튼 회전 Quiz는 이 출력 데이터 저장부(`reg_a~reg_h`)를 동적으로 바꾸는 응용이다.
