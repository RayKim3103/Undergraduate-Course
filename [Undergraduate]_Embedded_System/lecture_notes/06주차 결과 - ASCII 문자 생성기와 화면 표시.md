---
title: "06주차 결과 - ASCII 문자 생성기와 화면 표시"
course: "Embedded System"
week: 6
type: "result"
tags:
  - embedded-system
  - ascii
  - char-generator
  - video
---

# 06주차 결과 - ASCII 문자 생성기와 화면 표시

이전: [[06주차 예비 - TFT-LCD ASCII 문자 출력]]  
다음: [[07주차 예비 - PS 기반 LED 7-Segment AXI 제어]]

## 핵심 요약

이번 실습에서는 ASCII 텍스트를 TFT-LCD에 출력하는 video pipeline을 분석했다. `SVGA_TIMING_GENERATION`이 pixel/character 좌표를 만들고, `CHAR_DISPLAY`가 표시할 문자를 선택하며, `CHAR_GEN`이 ASCII를 8x8 bitmap pixel로 바꾼다. Quiz에서는 조원 정보와 지정된 시 문구를 정렬 및 색상 조건에 맞게 출력했다.

## 전체 출력 파이프라인

1. Pixel clock 생성
2. 수평/수직 sync 및 blanking 생성
3. 현재 pixel 좌표와 character 좌표 계산
4. `text_line` 배열에서 현재 위치의 ASCII 선택
5. `CHAR_DPRAM`에 ASCII 저장/읽기
6. `CHAR_GEN_ROM`에서 ASCII의 8x8 bitmap row 읽기
7. 현재 sub-pixel 위치에 따라 `pixel_on` 생성
8. `pixel_on`에 따라 foreground/background RGB 선택
9. `VIDEO_OUT`에서 blanking을 반영하여 최종 출력

## `SVGA_TIMING_GENERATION.v`

주요 기능은 12가지로 정리된다.

| 기능 | 내용 |
|---|---|
| Horizontal pixel counter | 한 row 안의 현재 pixel 위치 계산 |
| Horizontal sync | `H_ACTIVE`, front/back porch 기준 sync pulse 생성 |
| Vertical line counter | frame 안의 현재 line 위치 계산 |
| Vertical sync | `V_ACTIVE`, front/back porch 기준 sync pulse 생성 |
| H/V blank | active video가 아닌 구간 표시 |
| Composite blank | H 또는 V blank가 있으면 출력 비활성 |
| Subchar line | 8x8 문자 내부의 세로 위치 |
| Subchar pixel | 8x8 문자 내부의 가로 위치 |
| Character column | pixel column을 8로 나눈 문자 열 |
| Character line | pixel line을 8로 나눈 문자 행 |
| Reset char column | active line 끝에서 column 초기화 |
| Reset char line | frame 끝에서 line 초기화 |

`subchar_pixel`을 5로 초기화하는 이유는 `CHAR_GEN`에서 ROM data를 곧바로 가져오는 timing을 맞추기 위한 안정성 확보로 해석된다.

## `CHAR_GEN_ROM.v`

이 모듈은 ASCII 문자 모양을 저장한 ROM이다. address는 다음처럼 구성된다.

```verilog
chargen_rom_address = {ascii_code[7:0], subchar_line[2:0]};
```

즉, 어떤 문자 ASCII인지와 그 문자의 8개 row 중 몇 번째 row인지로 ROM address가 결정된다.

## `CHAR_GEN.v`

`CHAR_GEN`은 ASCII code를 현재 pixel의 on/off 정보로 바꾼다.

- DPRAM에서 현재 문자 ASCII를 읽는다.
- ASCII와 `subchar_line`으로 ROM address를 만든다.
- ROM에서 8비트 bitmap row를 가져온다.
- 상위 4비트와 하위 4비트를 나누어 timing에 맞게 shift한다.
- 현재 sub-pixel 위치의 bit를 `pixel_on`으로 내보낸다.

정리하면 `CHAR_GEN`은 문자 bitmap을 pixel stream으로 직렬화하는 모듈이다.

## `CHAR_DISPLAY.v`

`CHAR_DISPLAY`는 화면 내용과 색상을 결정한다.

- `text_line` 배열에 출력할 문자열 저장
- `char_line`, `char_column`을 이용해 현재 표시할 문자 선택
- direction에 따라 좌표 반전 가능
- 선택한 문자의 ASCII를 DPRAM에 write
- `pixel_on`에 따라 foreground/background RGB 선택
- 위치별 색상 조건 지정

문자열을 Verilog string으로 작성하면 컴파일러가 각 문자를 ASCII 값으로 저장하므로, 직접 hex code를 모두 쓰지 않아도 된다.

## Quiz 1: 조원 정보 출력

목표는 좌측 상단에 조원 학번과 이름을 표시하는 것이다.

- `text_line[0]`, `text_line[1]`에 조원 정보 입력
- 나머지 line은 공백 또는 NULL로 채움
- 55자 기준으로 오른쪽 공백을 맞춰 왼쪽 정렬 유지
- line별 foreground color 설정

공백 대신 NULL을 넣어도 문자열 종료로 처리되지 않고 pixel이 꺼진 상태처럼 보인다. Verilog에서 C 문자열처럼 NULL이 문자열 끝으로 동작하지 않기 때문이다.

## Quiz 2: 지정 문구 정렬과 색상

목표는 지정된 긴 문구를 중앙 정렬하고, 작가명은 우측 정렬하며, 특정 line과 column 범위에 색상을 다르게 주는 것이다.

구현 포인트:

- 모든 `text_line`을 55자 길이에 맞게 공백 포함 작성
- 제목, 반복 문구, 작가명, 조원명에 서로 다른 RGB foreground/background 조건 적용
- `inter_char_line`, `inter_char_column` 범위를 계산하여 영역별 색상 결정

## 고찰

- 실제 보드에서는 이론상 60x34 문자보다 작은 55x33 범위가 안정적으로 사용되었다.
- `text_line[2]~[32]`를 명시하지 않으면 이전 data나 index 0번 줄이 잘린 형태로 반복될 수 있다.
- 화면 가장자리 문자 잘림을 피하려면 active video 범위와 character decode delay를 함께 고려해야 한다.

## 정리

6주차 결과의 핵심은 ASCII 출력이 단순 문자열 출력이 아니라 **timing, memory, ROM bitmap, foreground/background color 선택** 의 결합이라는 점이다. 이후 PS가 문자열이나 데이터를 공급하는 실습에서도 같은 문자/영상 출력 파이프라인 이해가 기반이 된다.
