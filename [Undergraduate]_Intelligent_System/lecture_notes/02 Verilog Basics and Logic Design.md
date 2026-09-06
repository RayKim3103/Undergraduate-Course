---
title: "02. Verilog Basics and Logic Design"
pages: 57
tags: [intelligent-system, lecture-note, Verilog, combinational-logic, sequential-logic]
---

# 02. Verilog Basics and Logic Design

> 이전: [[01 Vivado Installation and Basic Flow]]
> 다음: [[03 FSM and Sequential System Design]]

## 학습 목표

Week2-1 자료는 Vivado 사용 흐름을 다시 확인한 뒤 Verilog HDL의 기본 문법과 조합/순차 회로 설계를 다룬다.

- Vivado project/source/testbench/simulation
- HDL과 software programming의 차이
- Verilog module, port, instance
- wire와 reg
- blocking/non-blocking assignment
- parameter/localparam
- number representation
- combinational logic
- sequential logic
- 4-bit full adder, counter, decoder/demux practice

## HDL은 Programming Language가 아니다

HDL은 hardware description language이다. 코드를 “실행 순서”로만 읽으면 안 되고, 어떤 회로와 연결이 만들어지는지 생각해야 한다.

예:

```verilog
z[3] <= a[3] + b[3];
z[2] <= a[2] + b[2];
z[1] <= a[1] + b[1];
z[0] <= a[0] + b[0];
```

이는 네 bit 연산이 clock edge에서 동시에 일어나는 hardware 동작으로 이해해야 한다.

## System IC Design Flow

설계 흐름:

1. idea
2. algorithm
3. specification
4. paper specification
5. HDL coding
6. RTL simulation
7. synthesis
8. gate-level simulation
9. place and route
10. FPGA programming 또는 fabrication

RTL(Register Transfer Level)은 register 사이에서 data가 어떻게 이동하고 어떤 combinational logic을 거치는지 표현하는 수준이다.

## Verilog Module 구조

기본 구조:

```verilog
module adder (
    output reg [4:0] y,
    input wire [3:0] a,
    input wire [3:0] b
);
    // logic
endmodule
```

비교:

| Python | Verilog |
|---|---|
| function | module |
| function parameter | port |
| variable | wire/register |
| call | instantiation |

## Module Instantiation

module을 다른 module 안에서 사용할 수 있다.

위치 기반 연결:

```verilog
adder m1(y1, a1, b1);
```

이름 기반 연결:

```verilog
adder m2(.a(a2), .b(b2), .y(c2));
```

권장: 이름 기반 연결. port 순서를 헷갈려도 의도가 분명하다.

## Wire와 Reg

| 타입 | 용도 |
|---|---|
| `wire` | continuous assignment, module 간 연결 |
| `reg` | procedural assignment, always/initial 내부에서 값 유지 |

주의:

- input port는 일반적으로 wire로 취급된다.
- output은 wire 또는 reg가 될 수 있다.
- `reg`라고 해서 반드시 flip-flop이 되는 것은 아니다. always block의 sensitivity와 할당 방식에 따라 조합논리도 될 수 있다.

## Blocking Assignment

`=` 사용. 문장이 순차적으로 평가된다.

```verilog
always @(posedge clk) begin
    C = B;
    B = A;
    A = D;
end
```

앞 문장의 결과가 뒤 문장에 영향을 줄 수 있다. sequential logic에는 부적합한 경우가 많다.

## Non-Blocking Assignment

`<=` 사용. clock edge에서 동시에 update되는 register 동작을 표현한다.

```verilog
always @(posedge clk) begin
    C <= B;
    B <= A;
    A <= D;
end
```

sequential logic에는 non-blocking assignment를 쓰는 것이 표준적이다.

## Parameter와 Localparam

`parameter`와 `localparam`은 module 내 상수 정의에 사용된다.

```verilog
parameter IDLE = 0;
localparam SIZE = 16;
reg [SIZE-1:0] ram;
```

- `parameter`: module instance 시 override 가능
- `localparam`: 내부 고정 상수

## Number Representation

형식:

```verilog
<bit width>'<base><value>
```

예:

```verilog
4'b1111
4'd15
4'hF
4'o17
```

logic value:

| 값 | 의미 |
|---|---|
| 0 | false/low |
| 1 | true/high |
| x | unknown/don't care |
| z | high impedance |

## 주요 Operator

| 종류 | 예 |
|---|---|
| arithmetic | `+`, `-`, `*`, `/`, `%` |
| bitwise | `~`, `&`, `|`, `^`, `^~` |
| shift | `>>`, `<<` |
| equality | `==`, `!=` |
| relational | `>`, `<`, `>=`, `<=` |
| logical | `!`, `&&`, `||` |
| reduction | `&a`, `|a`, `^a` |
| concatenation | `{a,b}` |
| replication | `{4{a}}` |
| conditional | `sel ? a : b` |

## Combinational Logic

간단한 조합논리는 `assign`으로 구현한다.

```verilog
assign out = ~in;
```

복잡한 조합논리는 `always @(*)`와 `if/case`를 사용한다.

```verilog
always @(*) begin
    if (sel == 1'b0) out = a;
    else out = b;
end
```

주의:

- 조합논리 always block에서는 모든 출력에 모든 branch에서 값을 할당해야 latch inference를 피할 수 있다.
- 비교에는 `==`를 사용한다. `=`는 대입이다.

## 4-bit Full Adder

1-bit full adder:

```verilog
assign S1 = A ^ B;
assign S  = S1 ^ Cin;
assign C1 = S1 & Cin;
assign C2 = A & B;
assign Cout = C1 | C2;
```

4-bit adder는 1-bit full adder 4개를 cascade로 연결한다.

```verilog
Full_Adder FA0 (A[0], B[0], C0, S[0], C1);
Full_Adder FA1 (A[1], B[1], C1, S[1], C2);
Full_Adder FA2 (A[2], B[2], C2, S[2], C3);
Full_Adder FA3 (A[3], B[3], C3, S[3], C4);
```

이 구조는 ripple-carry adder이다.

## Sequential Logic

clock edge에서 상태가 바뀌는 회로이다.

D flip-flop:

```verilog
always @(posedge clk)
    q <= d;
```

asynchronous active-low reset 포함:

```verilog
always @(posedge clk or negedge rst) begin
    if (!rst) q <= 0;
    else q <= d;
end
```

## 4-bit Counter

```verilog
always @(posedge clk or negedge rst) begin
    if (!rst) cnt <= 0;
    else cnt <= cnt + 1;
end
```

핵심:

- clock period와 reset signal을 testbench에서 확인한다.
- reset 후 counter가 0에서 시작해야 한다.

## 74LS138 Decoder/Demultiplexer Practice

74LS138은 enable signal이 있는 active-low 3-to-8 decoder이다.

입력:

- select: A, B, C
- enable: G1, G2A, G2B

출력:

- Y[0]~Y[7], active-low

동작:

- enable 조건이 맞지 않으면 모든 출력이 1
- enable 조건이 맞으면 select 값에 해당하는 출력 하나만 0

구현 방식:

- gate-level: NOT, AND, NAND 등으로 직접 구성
- behavioral: `if/else` 또는 `case` 사용

## Testbench 체크

- DUT input은 `reg`
- DUT output은 `wire`
- 입력을 시간에 따라 바꾸며 truth table과 비교
- `$display`, `$monitor`로 TCL console debugging
- testbench 마지막에 `$finish`

## 체크포인트

- HDL 코드는 회로 구조와 동시성을 표현한다.
- 조합논리는 `assign` 또는 `always @(*)`.
- 순차논리는 `always @(posedge clk)`와 non-blocking assignment.
- `wire`는 연결, `reg`는 procedural assignment 대상이다.
- decoder처럼 truth table이 명확한 회로는 `case`문이 읽기 좋다.
- testbench는 합성 대상이 아니며 simulation 검증용이다.
