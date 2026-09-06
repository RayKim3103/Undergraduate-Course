# Vivado Verilog 기본과 논리게이트

tags: #basic-digital-experiment #verilog #vivado #logic-gate #testbench

관련 노트: [[02 Adders Two Complement Debugging - 가산기 보수 디버깅]]

## 핵심 요약

이 자료는 Vivado에서 Verilog 프로젝트를 만들고, 논리 게이트와 2-bit adder를 설계ㆍ시뮬레이션ㆍFPGA 보드에 연결하는 기본 흐름을 다룬다. 핵심은 하드웨어 모듈을 `design source`로 작성하고, 입력 변화를 만들어 주는 `simulation source`/testbench로 검증한 뒤, top module과 constraint 파일을 통해 실제 보드의 스위치ㆍ버튼ㆍLED 핀에 연결하는 것이다.

## Vivado 설계 흐름

1. 프로젝트 생성
   - 사용할 보드나 FPGA part를 선택한다.
   - 설계 코드는 `Design Sources`, 검증용 코드는 `Simulation Sources`에 둔다.

2. 모듈 작성
   - Verilog의 기본 단위는 `module`이다.
   - 입력은 `input`, 출력은 `output`으로 선언한다.
   - 조합논리는 `assign` 문으로 직접 표현할 수 있다.

3. 테스트벤치 작성
   - 테스트벤치는 실제 회로로 합성되는 코드가 아니라 시뮬레이션용 코드이다.
   - 입력 신호는 값을 바꿔야 하므로 보통 `reg`로 선언하고, 출력은 DUT가 구동하므로 `wire`로 둔다.
   - `initial begin ... end` 블록에서 시간 순서대로 입력을 바꾼다.
   - `#10`은 `timescale 1ns/1ps` 기준으로 10 ns 지연을 뜻한다.
   - 모든 입력 조합을 순서대로 넣고 waveform이 진리표와 일치하는지 확인한다.

4. Top module과 constraint
   - 하위 논리 모듈을 top module에서 인스턴스화한다.
   - top module의 포트를 보드의 스위치, 버튼, LED와 연결한다.
   - XDC constraint 파일에서 실제 핀 번호와 I/O standard를 지정한다.

## Verilog 기본 문법

### 모듈 인스턴스화

하위 모듈을 사용할 때는 다음 두 방식이 가능하다.

```verilog
// 순서 기반 연결
and_gate u0(a, b, y);

// 이름 기반 연결
and_gate u0(
    .a(a),
    .b(b),
    .y(y)
);
```

이름 기반 연결은 포트 수가 많아질수록 실수를 줄이기 쉽다.

### 논리 연산자

| 게이트 | Verilog 표현 | 의미 |
|---|---:|---|
| AND | `a & b` | 둘 다 1일 때 1 |
| OR | `a | b` | 하나라도 1이면 1 |
| NOT | `~a` | 반전 |
| NAND | `~(a & b)` | AND 후 반전 |
| NOR | `~(a | b)` | OR 후 반전 |
| XOR | `a ^ b` | 서로 다르면 1 |

## 논리 게이트 실험

### 실험 목적

기본 논리 게이트를 Verilog로 구현하고, 각 입력 조합에 대한 출력이 진리표와 일치하는지 확인한다.

### 구현 포인트

- 2입력 게이트는 입력 조합이 `00`, `01`, `10`, `11` 네 가지이다.
- 테스트벤치에서 각 조합을 10 ns 간격으로 넣어 waveform을 비교한다.
- FPGA 보드에서는 스위치를 입력으로, LED를 출력으로 연결해 실제 동작을 확인한다.

### 해석

시뮬레이션 waveform에서 각 게이트의 출력은 이론 진리표와 일치한다. 실제 보드에서도 스위치 조작에 따라 LED가 같은 방식으로 켜지고 꺼지므로, RTL 코드와 핀 연결이 정상임을 확인할 수 있다.

## 2-bit Adder 실험

### 실험 목적

2-bit 입력 두 개를 더해 합을 출력하는 회로를 설계한다. 기본 게이트 실험에서 배운 조합논리 표현이 더 복잡한 산술 회로로 확장되는 과정을 확인한다.

### 구현 포인트

- 2-bit 수의 합은 최대 `3 + 3 = 6`이므로 결과 표현에는 최소 3 bit가 필요하다.
- 입력 조합을 testbench에서 바꿔가며 예상 합과 waveform 출력을 비교한다.
- 보드에서는 입력 스위치와 출력 LED를 사용해 덧셈 결과를 직접 확인한다.

## 자주 틀리는 지점

- 테스트벤치의 입력을 `wire`로 선언하면 절차문 안에서 값을 대입할 수 없다.
- top module 포트 이름과 XDC constraint의 이름이 다르면 보드 입출력이 연결되지 않는다.
- `#10` 같은 delay는 합성용 회로 지연이 아니라 시뮬레이션 시간 제어이다.
- 논리 NOT `~`와 논리 부정 `!`는 용도가 다르다. 비트 단위 반전에는 `~`를 사용한다.

## 시험ㆍ복습 체크포인트

- `Design Source`와 `Simulation Source`의 차이를 설명할 수 있어야 한다.
- `reg`와 `wire`를 테스트벤치 관점에서 구분할 수 있어야 한다.
- AND, OR, NOT, NAND, NOR, XOR의 Verilog 표현과 진리표를 쓸 수 있어야 한다.
- top module, module instantiation, constraint 파일이 각각 어떤 역할을 하는지 설명할 수 있어야 한다.

