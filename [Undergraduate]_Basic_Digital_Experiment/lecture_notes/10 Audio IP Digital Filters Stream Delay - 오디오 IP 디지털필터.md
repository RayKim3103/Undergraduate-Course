# Audio IP, Digital Filters, Stream Delay

tags: #basic-digital-experiment #audio #ip #axi-stream #digital-filter #i2s #spi

관련 노트: [[09 AMBA AHB APB Memory Transfer - AMBA AHB APB 메모리전송]], [[11 HDMI TMDS LCD Display - HDMI TMDS LCD 출력]]

## 핵심 요약

이 자료는 Zynq 기반 audio system을 구성하고, I2S/SPI/custom prescaler/AXI-Stream FIFO IP를 이용해 오디오를 생성ㆍ필터링ㆍ출력하는 실험이다. Low-pass, high-pass, band-pass filter와 equalizer를 C 코드에서 구현하고, AXI-Stream delay module이 오디오 흐름에 주는 영향을 분석한다.

## Audio System의 주요 IP

| IP | 역할 |
|---|---|
| Zynq7 Processing System | C 코드 실행, AXI master 역할 |
| Processor System Reset | clock domain에 맞춘 reset 생성 |
| AXI Interconnect | master/slave 간 AXI 연결 |
| AXI-Stream FIFO | sample stream buffering |
| `myPrescaler` | clock 분주 |
| `myI2STx_v1_0` | I2S audio transmit |
| `mySPIRxTx_v1_0` | SPI 송수신 |

## I2S Transmitter

I2S는 digital audio sample을 serial data로 전송하는 interface이다. `myI2STx_v1_0`은 AXI-Stream으로 받은 audio data를 buffer에 저장한 뒤, bit clock과 left/right clock에 맞춰 `sdata`로 내보낸다.

주요 신호:

- `mclk`: master clock
- `bclk`: bit clock
- `lrclk`: left/right channel 구분
- `sdata`: serial audio data
- `S00_AXIS`: AXI-Stream slave interface

내부 FSM은 idle, data load, bit shifting 같은 상태를 거치며 sample을 전송한다.

## SPI Transceiver

SPI는 master/slave 기반 serial 통신이다. `mySPIRxTx_v1_0`은 AXI-Stream으로 데이터를 받고 SPI clock, MOSI, MISO, SS를 제어한다.

주요 신호:

- `SCLK`: serial clock
- `MOSI`: master out slave in
- `MISO`: master in slave out
- `SS`: slave select

## Prescaler

Prescaler는 빠른 입력 clock을 느린 clock으로 나눈다. counter가 설정값에 도달할 때마다 출력 clock을 toggle하면 전체 주기는 두 번의 toggle로 완성된다.

```text
toggle every N input cycles -> output period = 2N cycles
```

따라서 128번 edge마다 toggle하면 출력 frequency는 입력 대비 256분주가 된다.

## AXI-Stream FIFO

AXI-Stream은 ready/valid handshake로 data stream을 전달한다. FIFO는 producer와 consumer의 속도가 다를 때 sample을 임시 저장해 흐름을 안정화한다.

주요 신호:

- `TVALID`: 송신 측 data valid
- `TREADY`: 수신 측 ready
- `TDATA`: data
- `TLAST`: packet 끝 표시

## Digital Filter 기본

Digital filter는 현재와 과거 sample을 이용해 출력 sample을 계산한다.

```text
y[n] = b0*x[n] + b1*x[n-1] + ... - a1*y[n-1] - ...
```

실험에서는 sampling frequency `fs = 39062.5 Hz` 조건에서 LPF, HPF, BPF를 사용했다.

## Low-Pass Filter

Cutoff frequency는 400 Hz이다.

```text
y[n] - 0.939678*y[n-1] = 0.0604416*x[n]
H(z) = 0.0604416 / (1 - 0.939678*z^-1)
```

낮은 주파수 성분은 통과시키고 높은 주파수 성분은 감쇠시킨다.

## High-Pass Filter

Cutoff frequency는 2000 Hz이다.

```text
y[n] - 0.7566776*y[n-1]
= 0.7566776*x[n] - 0.7566776*x[n-1]
```

높은 주파수 성분을 상대적으로 강조한다.

## Band-Pass Filter

Band-pass filter는 400 Hz부터 2000 Hz 사이의 중간 대역을 통과시킨다.

```text
y[n] - 1.769*y[n-1] + 0.779*y[n-2]
= 0.114*x[n] + 0.114*x[n-2]
```

자료의 계수 표기에는 일부 자리 차이가 있지만, 핵심은 2차 IIR 형태로 중간 주파수 대역만 남기는 것이다.

## Equalizer

Equalizer는 여러 filter 출력을 gain으로 조합해 전체 음색을 바꾼다.

```text
y[n] = g1*y_lpf[n] + g2*y_bpf[n] + g3*y_hpf[n]
```

이론적으로는 원 신호 `x[n]`를 더하는 구조도 가능하지만, 실험 코드에서는 세 filter output을 입력으로 받아 gain을 곱해 합산하는 방식으로 구현했다.

## Fixed-Point와 Scaling

FPGA/embedded 환경에서는 floating-point 연산이 비싸기 때문에 coefficient를 정수화하고 shift로 scale을 조정한다.

```text
실수 계수 * 2^shift -> 정수 계수
계산 결과 >> shift -> scale 복원
```

필터 출력 amplitude가 서로 다르면 equalizer 합산 시 overflow가 발생할 수 있다. 실험에서는 각 filter 출력의 최대값을 찾아 약 20000 수준으로 normalize해 16-bit unsigned 범위 안에 들어오도록 했다.

## Audio Generation

음표 생성 함수는 주파수와 길이를 받아 sample 배열을 만든다. note length를 128에서 256으로 바꾸면 sample 수가 두 배가 되어 음이 약 두 배 길게 들린다.

실험에서는 200 Hz, 2000 Hz, 5000 Hz 같은 test tone을 넣어 LPF/BPF/HPF가 각각 낮은ㆍ중간ㆍ높은 주파수를 어떻게 통과시키는지 확인했다.

## Stream Delay Module

`stream_delay.v`는 AXI-Stream 입력 sample과 이전 sample을 조합해 더 부드러운 출력을 만든다.

```text
out = current sample + previous sample
```

하지만 ready/valid handshake 때문에 module이 입력을 기다리고 출력을 내보내는 과정에서 여러 cycle이 필요하다. 그 결과 오디오가 부드러워지는 대신 재생 속도가 느려지는 현상이 나타났다.

### 속도 저하 완화 방법

- note length를 줄인다.
- `PERIODSAMPLES`를 256에서 128로 줄인다.
- clock을 높인다.
- stream pipeline을 더 효율적으로 설계한다.

## 시험ㆍ복습 체크포인트

- AXI-Stream의 `TVALID`/`TREADY` handshake를 설명할 수 있어야 한다.
- LPF, HPF, BPF가 어떤 주파수 성분을 통과시키는지 구분할 수 있어야 한다.
- IIR filter difference equation을 C 코드로 옮길 때 과거 입력ㆍ출력 sample 배열이 왜 필요한지 이해해야 한다.
- fixed-point scaling과 normalization이 필요한 이유를 설명할 수 있어야 한다.

