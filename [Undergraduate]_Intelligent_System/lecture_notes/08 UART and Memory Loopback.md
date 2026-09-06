---
title: "08. UART and Memory Loopback"
pages: 24
tags: [intelligent-system, lecture-note, UART, communication, BRAM, assignment]
---

# 08. UART and Memory Loopback

> 이전: [[07 FIFO and Line Buffer]]
> 다음: [[09 PS PL AXI PYNQ and ILA]]

## 학습 목표

Week6-2 자료는 hardware communication과 UART Rx/Tx timing을 설명하고, Assignment 2인 memory loopback 구조를 제시한다.

## Hardware Communication

일반적인 hardware system은 binary scale로 동작하므로 data movement도 binary rule에 따라 수행된다.

용어:

- TX: transmitter, data를 보내는 쪽
- RX: receiver, data를 받는 쪽

## Serial vs Parallel Communication

| 방식 | 설명 | 예 |
|---|---|---|
| Serial | 하나의 signal line으로 bit를 순차 전송 | UART |
| Parallel | 여러 signal line으로 여러 bit를 동시에 전송 | bus |

Serial communication은 배선 수가 적지만 시간에 따라 bit를 해석해야 한다. Parallel bus는 bandwidth가 높을 수 있지만 wiring과 timing이 복잡하다.

## Asynchronous vs Synchronous

### Asynchronous

TX와 RX가 clock을 공유하지 않는다. 대신 다음 rule을 미리 약속한다.

- baud rate
- data length
- start bit
- stop bit
- parity 여부

### Synchronous

TX와 RX가 clock signal을 공유한다. clock 기준으로 data를 sampling한다.

## UART 기본 규칙

UART(Universal Asynchronous Receiver Transmitter)의 기본 frame:

```text
IDLE(1) -> Start bit(0) -> Data bits(LSB to MSB) -> Stop bit(1)
```

자료 기준:

- idle value: 1
- start bit: 0
- stop bit: 1
- data: 8-bit
- bit order: LSB to MSB
- baud rate: 115200 bps, 57600 bps, 38400 bps 등

## Baud Rate

baud rate는 초당 signal event 수이다.

$$
\text{Baud rate}=\frac{\text{number of signals}}{\text{time in seconds}}
$$

중요성:

- communication bandwidth 결정
- timing mismatch에 따른 error rate에 직접 영향

clock frequency에서 baud tick을 만들 때 divisor 계산이 필요하다.

## UART Rx Design

주요 신호:

| 신호 | 역할 |
|---|---|
| `CLK` | 내부 기준 clock |
| `RST` | reset |
| `RxD` | 외부 serial input |
| `RxD_CLK_Rx` | 내부 clock에 동기화된 RxD |
| `Frm_ERR` | stop bit가 감지되지 않았을 때 high |
| `Rx_DATA[7:0]` | 수신된 8-bit data |
| `Rx_DATA_rdy` | 수신 data valid |
| `Index[10:0]` | 수신 data 개수 |

Rx flow:

1. idle high 상태 유지
2. falling edge 또는 low level로 start bit 감지
3. baud timing에 맞춰 8 data bits sampling
4. stop bit 확인
5. data valid 시 `Rx_DATA_rdy` high
6. stop bit가 1이 아니면 frame error

주의:

- asynchronous input은 metastability 방지를 위해 clock domain synchronization이 필요하다.
- `Rx_DATA_rdy`가 여러 clock 동안 high일 수 있으므로 edge detection이 필요할 수 있다.

## UART Tx Design

주요 신호:

| 신호 | 역할 |
|---|---|
| `Tx_Dout[7:0]` | 전송할 8-bit input data |
| `Empty` | 전송할 data가 없을 때 high |
| `Rd_en` | Tx가 다음 data를 요청하는 1-clock pulse |
| `TxD` | serial output |

Tx flow:

1. idle high 유지
2. 전송 data가 있으면 `Rd_en`으로 data 요청
3. start bit 0 전송
4. 8 data bits를 LSB부터 전송
5. stop bit 1 전송
6. 다음 data 요청

`Rd_en`이 1 clock pulse이므로 memory/FIFO controller는 그 순간 다음 data를 준비해야 한다.

## Assignment 2: Memory Loopback

목표:

FPGA가 UART Rx로 image data를 받아 BRAM에 저장한 뒤, UART Tx로 다시 PC/tester에 보내는 구조를 설계한다.

data:

- 128 x 128 image
- 8-bit per pixel

구조:

```text
Tester TX -> DUT UART_RX -> Memory Controller -> BRAM
BRAM -> Memory Controller -> UART_TX -> Tester RX
```

## Memory Loopback 동작

1. `rx_switch`를 켜면 module이 PC에서 들어오는 UART data를 기다린다.
2. PC가 image data를 UART Rx로 전송한다.
3. image size만큼 data가 수신되면 LED가 high가 되고 대기한다.
4. `tx_switch`를 켜면 BRAM에서 data를 읽어 UART Tx로 PC에 다시 보낸다.
5. PC/testbench가 받은 data를 grading 기준과 비교한다.

## Controller 설계 포인트

- Rx side와 Tx side를 별도 FSM으로 나누거나 mode FSM으로 관리한다.
- Rx data ready pulse를 정확히 잡아 BRAM write enable로 변환한다.
- 128x128 = 16384 byte address를 순차 증가시킨다.
- Tx가 `Rd_en`을 줄 때 BRAM read latency에 맞춰 data를 공급한다.
- `rx_switch`, `tx_switch`, LED 상태를 FSM state와 일관되게 연결한다.

## 체크포인트

- UART는 clock 공유가 없으므로 baud timing이 핵심이다.
- Rx는 start bit, data bit sampling, stop bit validation이 필요하다.
- Tx는 idle, start, data, stop sequence를 정확히 만든다.
- ready/valid 또는 empty/rd_en handshake를 놓치면 data loss가 생긴다.
- memory loopback은 UART, BRAM, FSM controller를 통합하는 과제이다.
