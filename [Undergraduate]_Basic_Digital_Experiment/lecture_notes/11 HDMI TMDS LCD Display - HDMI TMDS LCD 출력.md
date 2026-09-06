# HDMI, TMDS, LCD Display

tags: #basic-digital-experiment #hdmi #tmds #lcd #display-timing #verilog

관련 노트: [[10 Audio IP Digital Filters Stream Delay - 오디오 IP 디지털필터]], [[12 Sprite BRAM Graphics - 스프라이트 BRAM 그래픽]]

## 핵심 요약

이 자료는 HDMI 출력 구조와 TMDS encoding, LCD timing, test card image generation을 다룬다. Verilog로 timing generator, TMDS encoder, serializer, image pattern module을 연결해 1280x720 화면에 색상 패턴, gradient, square, circle 등을 출력한다.

## HDMI 개요

HDMI(High Definition Multimedia Interface)는 비압축 digital video/audio를 전송하는 interface이다. 일반 Type A HDMI connector는 19개 pin을 사용한다.

## HDMI Pin 기능

| Pin 범위 | 기능 |
|---|---|
| 1-3 | TMDS Data2, 주로 red data |
| 4-6 | TMDS Data1, 주로 green data |
| 7-9 | TMDS Data0, 주로 blue data |
| 10-12 | TMDS clock |
| 13 | CEC |
| 14 | HEAC |
| 15-16 | I2C/EDID용 SCL/SDA |
| 17 | Ground |
| 18 | VDD |
| 19 | Hot Plug Detect |

TMDS data와 clock은 differential pair로 전송되어 noise에 강하다.

## HDMI 송수신 흐름

```text
Source video
-> HDCP encryptor
-> TMDS encoder
-> Serializer
-> Differential driver
-> HDMI cable
-> Receiver CDR/deserializer
-> TMDS decoder
-> HDCP decryptor
-> Display
```

## HDCP

HDCP는 video content 보호를 위한 encryption/authentication 구조이다. 송신기와 수신기는 key exchange를 수행하고, 인증된 장치 사이에서만 video data를 복호화할 수 있다.

## TMDS Encoding

TMDS(Transition Minimized Differential Signaling)는 8-bit video data를 10-bit code로 바꿔 전송한다.

### 목적

- bit transition 수를 줄여 고속 전송 안정성을 높인다.
- DC balance를 맞춰 장시간 0 또는 1로 치우치지 않게 한다.
- differential signaling과 함께 noise에 강한 link를 만든다.

### Encoding 흐름

1. 8-bit 입력에서 1의 개수를 센다.
2. XOR 또는 XNOR 기반 누적 encoding 중 transition이 적은 방식을 선택한다.
3. 9번째 bit에 선택한 방식을 기록한다.
4. running disparity 또는 bias를 고려해 전체 code를 반전할지 결정한다.
5. 10번째 bit에 반전 여부를 기록한다.

## LCD Timing

LCD는 보이는 pixel뿐 아니라 sync와 porch interval을 포함한 일정한 timing으로 구동된다.

| 용어 | 의미 |
|---|---|
| HSYNC | 한 line 시작을 알리는 horizontal sync |
| VSYNC | 한 frame 시작을 알리는 vertical sync |
| HBP/VBP | sync 뒤의 back porch |
| HFP/VFP | 다음 sync 전의 front porch |
| HSLEN/VSLEN | sync pulse width |
| hactive/vactive | 실제 표시 영역 |

계산식:

```text
clocks per line = HSYNC + HBP + hactive + HFP + HSLEN
lines per frame = VSYNC + VBP + vactive + VFP + VSLEN
refresh rate = pixel clock / (clocks per line * lines per frame)
```

## `display_timings`

`display_timings` module은 현재 pixel 좌표와 sync/de 신호를 만든다.

주요 출력:

- `sx`, `sy`: 현재 pixel coordinate
- `hsync`, `vsync`: sync signal
- `de`: display enable, visible 영역 여부

자료의 1280x720 설정에서는 visible coordinate가 `sx >= 0`, `sy >= 0`, 그리고 해상도 범위 안일 때 active display가 된다.

## Test Card Image

### Simple Color Bar

화면을 여러 vertical region으로 나누고 각 region에 다른 RGB 값을 출력한다. HDMI 출력 경로가 정상인지 빠르게 확인하는 데 적합하다.

### Gradient

좌표 bit를 RGB 값에 연결해 자연스러운 색상 변화나 반복 패턴을 만든다. 예를 들어 `i_x`의 하위 bit 폭이 작으면 화면 가로 방향으로 같은 pattern이 여러 번 반복된다.

| x bit 폭 | 1280 화면에서 반복 느낌 |
|---:|---|
| 6 bit | 20회 반복 |
| 7 bit | 10회 반복 |
| 8 bit | 5회 반복 |

### Squares와 Circles

좌표 범위를 조건문으로 나누면 사각형을 만들 수 있다. 원은 중심 `(a, b)`와 반지름 `r`에 대해 다음 조건을 사용한다.

```text
(x - a)^2 + (y - b)^2 <= r^2
```

좌표 차이가 음수가 될 수 있으므로 signed 연산을 주의해야 한다.

## 버튼으로 이미지 선택

버튼 입력을 top module에서 받아 이미지 선택 신호로 사용한다.

| 버튼 | 이미지 |
|---|---|
| BTN3 | square pattern |
| BTN2 | gradient |
| BTN1 | simple color bar |
| BTN0 | reset |

버튼을 누른 순간만 바뀌고 떼어도 선택이 유지되도록 latch/register 형태의 선택 값을 둔다.

## Clock Generation

HDMI에는 pixel clock과 그보다 빠른 serialization clock이 필요하다. 자료에서는 Xilinx `MMCME2_BASE`와 `BUFG`를 이용해 필요한 clock을 만든다.

주요 parameter:

- `CLKFBOUT_MULT_F`
- `DIVCLK_DIVIDE`
- `CLKOUTx_DIVIDE`
- `CLKIN1_PERIOD`
- `CLKOUTx_PHASE`

`BUFG`는 FPGA 내부 clock network에 안정적으로 clock을 배포하기 위한 buffer이다.

## HDMI_TOP 구조

```text
image generator
-> dvi_generator
-> tmds_encoder
-> serializer
-> OBUFDS differential output
-> HDMI connector
```

`OBUFDS` 출력에는 TMDS differential pair용 I/O standard가 지정된다.

## 시험ㆍ복습 체크포인트

- TMDS가 8-bit를 10-bit로 바꾸는 이유를 설명할 수 있어야 한다.
- HSYNC, VSYNC, front porch, back porch, active 영역을 구분할 수 있어야 한다.
- 좌표 기반 image generator가 RGB 값을 만드는 방식을 이해해야 한다.
- serializer와 differential output이 HDMI 전송에서 필요한 이유를 말할 수 있어야 한다.

