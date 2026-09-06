# 순차논리, Shift Register, Counter

tags: #basic-digital-experiment #sequential-logic #flip-flop #shift-register #counter #verilog

관련 노트: [[03 PYNQ GPIO DEMUX Decoder - GPIO DEMUX Decoder]], [[05 FSM Traffic Light - 유한상태머신 신호등]]

## 핵심 요약

이 자료는 조합논리에서 순차논리로 넘어가는 주차이다. RS latch, edge-triggered D flip-flop, shift register, binary counter, BCD counter를 구현하면서 clock, reset, 상태 저장, blocking/nonblocking assignment의 차이를 학습한다.

## 순차논리의 의미

조합논리는 현재 입력만으로 출력이 결정된다. 반면 순차논리는 현재 입력뿐 아니라 과거 상태도 출력에 영향을 준다.

```text
next state = f(current state, input)
output     = g(current state, input)
```

따라서 순차논리에는 상태를 저장하는 latch나 flip-flop이 필요하다.

## RS Latch

NOR 기반 RS latch는 `R`, `S` 입력으로 1-bit 상태를 저장한다.

| S | R | 동작 |
|---:|---:|---|
| 0 | 0 | 유지 |
| 1 | 0 | Set |
| 0 | 1 | Reset |
| 1 | 1 | 금지 상태 |

`S=R=1`은 두 출력이 동시에 0이 되어 보수 관계가 깨지므로 피해야 한다.

## Edge-Triggered D Flip-Flop

D flip-flop은 clock edge에서만 입력 `D`를 출력 `Q`로 복사한다.

```verilog
always @(posedge clk or negedge resetn) begin
    if (!resetn)
        q <= 1'b0;
    else
        q <= d;
end
```

positive edge-triggered DFF는 상승 에지에서 동작하고, negative edge-triggered DFF는 하강 에지에서 동작한다.

## Shift Register

Shift register는 여러 flip-flop을 직렬로 연결해 데이터를 한 bit씩 이동시키는 회로이다.

| 종류 | 의미 |
|---|---|
| SISO | Serial In Serial Out |
| SIPO | Serial In Parallel Out |
| PISO | Parallel In Serial Out |
| PIPO | Parallel In Parallel Out |

### 실험 구현

- 10-bit LED 출력 `o_led[9:0]`를 shift register처럼 사용한다.
- 입력은 `clk`, `resetn`, `i_val`이다.
- `always @(negedge resetn or posedge clk)`에서 reset과 clock 동작을 정의한다.
- PYNQ 보드에서는 BTN0을 reset으로 사용하고, 스위치 입력값을 1초 clock마다 밀어 넣는다.

### 결과 해석

스위치를 켜면 LED가 한 칸씩 채워지고, 스위치를 끄면 LED가 한 칸씩 꺼지는 방향으로 이동한다. 이는 입력 bit가 매 clock마다 register chain을 따라 이동한다는 뜻이다.

## Binary Counter

Binary counter는 clock마다 값을 증가 또는 감소시키는 순차회로이다.

### 실험 동작

| push 입력 | 동작 |
|---:|---|
| 001 | +1 |
| 010 | -1 |
| 100 | -4 |

4-bit counter이므로 값 범위는 `0000`부터 `1111`까지이다. overflow와 underflow가 발생하면 2의 보수 표현처럼 wrap-around된다.

예:

```text
1111 + 1 = 0000
0000 - 1 = 1111
```

## BCD Counter

BCD counter는 0부터 9까지만 유효한 decimal digit counter이다. 4-bit를 사용하지만 `1010`부터 `1111`까지는 정상 BCD 숫자가 아니다.

### 실험 동작

| push 입력 | 동작 |
|---:|---|
| 001 | +1 |
| 010 | -1 |
| 100 | +2 |

값이 9를 넘으면 10을 빼고, 0보다 작아지는 경우에는 9로 돌아가도록 보정한다.

```text
9 + 2 -> 1
0 - 1 -> 9
```

## 1초 Clock Generator

FPGA의 기본 clock은 사람이 관찰하기에 너무 빠르다. 따라서 clock divider를 사용해 1초마다 toggle되는 느린 clock을 만든다.

원리는 다음과 같다.

```text
입력 clock edge를 count
목표 count에 도달하면 출력 clock toggle
counter reset
```

PYNQ 보드의 기준 clock을 이용해 half-period에 해당하는 count 값을 정하면 사람이 LED 변화를 볼 수 있는 1초 주기 신호를 만들 수 있다.

## Blocking과 Nonblocking

순차논리에서는 일반적으로 nonblocking assignment `<=`를 사용한다. 같은 clock edge에서 모든 register가 동시에 갱신되는 하드웨어 동작을 표현하기 좋기 때문이다.

```verilog
q1 <= d;
q2 <= q1;
```

blocking assignment `=`는 절차적으로 즉시 대입되는 것처럼 동작한다. BCD counter처럼 한 블록 안에서 값을 먼저 바꾸고 그 결과를 다시 보정하는 코드에서는 동작 차이를 주의해야 한다. 더 안전한 방법은 중간 변수나 명확한 next-state 계산을 두는 것이다.

## 시험ㆍ복습 체크포인트

- 조합논리와 순차논리의 차이를 설명할 수 있어야 한다.
- RS latch의 금지 상태를 이해해야 한다.
- `posedge`, `negedge`, asynchronous reset의 의미를 구분할 수 있어야 한다.
- binary counter와 BCD counter의 overflow 처리 차이를 설명할 수 있어야 한다.
- sequential logic에서 nonblocking assignment를 선호하는 이유를 말할 수 있어야 한다.

