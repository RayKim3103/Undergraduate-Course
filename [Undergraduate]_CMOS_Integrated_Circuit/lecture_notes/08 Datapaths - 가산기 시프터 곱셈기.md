# Datapaths: 가산기, 시프터, 곱셈기

tags: #cmos-integrated-circuit #datapath #adder #carry-lookahead #shifter #multiplier #carry-save

관련 노트: [[07 Gates - 고급 CMOS 게이트]], [[09 Sequential Circuit Design - 순차회로와 Timing]]

## 핵심 요약

이 장은 datapath의 핵심 building block인 adder, comparator, shifter, multi-input adder, multiplier를 다룬다. 특히 carry propagate adder의 여러 구조가 area, power, delay tradeoff를 어떻게 만드는지 비교한다.

## Boolean Logical Operations

Datapath는 arithmetic뿐 아니라 bitwise logical operation을 포함한다. AND, OR, XOR, XNOR, inversion은 ALU의 기본 연산이고, equality comparator나 zero detector에도 쓰인다.

## Half Adder와 Full Adder

### Half Adder

```text
S = A xor B
C = A B
```

### Full Adder

```text
S = A xor B xor Cin
Cout = majority(A, B, Cin)
```

Full adder는 3개 1-bit 입력을 받아 sum과 carry를 만든다.

## Generate, Propagate, Kill

Carry 동작을 이해하기 위해 bit별로 generate, propagate, kill을 정의한다.

| 상태 | 조건 | 의미 |
|---|---|---|
| Generate | `A B = 1` | `Cin`과 무관하게 `Cout=1` |
| Propagate | `A xor B = 1` | `Cout=Cin` |
| Kill | `A+B=0` | `Cin`과 무관하게 `Cout=0` |

Group generate/propagate를 사용하면 여러 bit 구간의 carry를 병렬로 계산할 수 있다.

## Full Adder Circuit 구현

자료는 여러 full adder 구현을 비교한다.

| 방식 | 특징 |
|---|---|
| Static CMOS | 안정적, noise margin 우수 |
| Optimized static | carry path를 빠르게 설계 |
| CPL | complementary pass transistor logic, 빠르지만 swing/복잡도 주의 |
| Dual-rail domino | 매우 빠르지만 area와 power가 큼 |

Ripple adder에서는 보통 `Cin`에서 `Cout`으로 이어지는 carry path가 critical path이다.

## Carry Propagate Adder

N-bit adder는 carry가 bit들을 지나 최종 sum이 결정되는 구조이며 CPA라고 부른다.

### Carry-Ripple Adder

가장 단순한 구조이다.

```text
FA0 -> FA1 -> FA2 -> ... -> FAN-1
```

장점은 area가 작고 설계가 단순하다는 것이다. 단점은 carry가 모든 bit를 통과할 수 있어 delay가 `O(N)`으로 증가한다.

## Carry-Skip Adder

Carry-skip adder는 group 전체가 propagate 상태이면 carry가 ripple chain을 건너뛰도록 skip path를 둔다.

```text
P_group = P_i * P_{i+1} * ... * P_j
if P_group = 1, carry skips group
```

적절한 block 크기를 고르면 delay가 대략 `O(sqrt(N))`로 줄어든다.

## Carry-Lookahead Adder

Carry-lookahead adder는 group generate와 propagate를 병렬로 계산해 carry를 빠르게 구한다.

```text
G_i:0 = G_i + P_i G_{i-1:0}
P_i:0 = P_i P_{i-1:0}
```

Carry-ripple보다 빠르지만 generate/propagate logic과 wire가 복잡해진다.

## Carry-Select Adder

Carry-select adder는 각 block에서 `Cin=0`일 때와 `Cin=1`일 때의 sum을 미리 계산하고, 실제 carry가 도착하면 mux로 선택한다.

장점:

- carry 대기 시간을 줄인다.

단점:

- 두 경우를 모두 계산하므로 area와 power가 증가한다.

## Tree Adder

Tree adder는 recursive lookahead로 carry를 `O(log N)` depth에 계산한다. Parallel-prefix adder 계열로 볼 수 있으며, Kogge-Stone, Brent-Kung 같은 구조가 대표적이다.

Tradeoff:

- 빠른 delay
- 많은 wire track
- routing 복잡도
- 큰 capacitance와 power

## Adder Architecture Tradeoff

| 구조 | Delay | Area/Power | 특징 |
|---|---|---|---|
| Ripple | 큼, `O(N)` | 작음 | 단순 |
| Skip | 중간 | 중간 | group propagate 사용 |
| Lookahead | 작음 | 큼 | carry 병렬 계산 |
| Select | 작음 | 큼 | 두 carry 경우 미리 계산 |
| Tree | 매우 작음, `O(log N)` | 큼 | wiring 복잡 |

## Comparator와 Detector

Equality check는 bit별 XNOR 후 AND로 구현한다.

```text
equal = AND_i (A_i XNOR B_i)
```

1's detector는 N-input AND gate이고, zero detector는 모든 bit가 0인지 확인하는 구조이다.

## Shifters

Shifter는 bit 위치를 이동한다.

| 종류 | 설명 |
|---|---|
| Logical shift | 빈 bit를 0으로 채움 |
| Arithmetic shift | signed number의 부호 bit 유지 |
| Rotate | 밀려난 bit를 반대쪽으로 돌림 |

### Funnel Shifter

2N-bit input에서 N-bit field를 선택한다. Shift와 rotate를 일반화한 구조이다.

### Barrel Shifter

Barrel shifter는 여러 단계의 mux로 한 cycle에 임의 shift 또는 rotate를 수행한다. Logarithmic barrel shifter는 shift amount의 각 bit에 해당하는 단계로 구성된다.

## Multi-Input Adders와 Carry-Save Addition

Full adder는 3개 입력을 2개 출력으로 줄인다. N개의 full adder를 병렬로 두면 여러 operand를 carry-save 형태로 줄일 수 있다.

```text
3 inputs -> sum + carry
```

Carry output은 sum output보다 한 bit 높은 weight를 가진다. Carry-save adder는 최종 carry propagation을 마지막까지 미뤄 multi-operand addition을 빠르게 만든다.

## Multiplier

Binary multiplier는 partial product를 만들고 이를 더한다.

```text
X * Y = sum_i (x_i ? Y << i : 0)
```

Array multiplier는 규칙적인 구조가 장점이지만 partial product 수와 adder delay/power가 크다. Booth encoding이나 carry-save reduction tree를 사용하면 성능을 개선할 수 있다.

## 시험ㆍ복습 체크포인트

- Generate, propagate, kill의 의미와 carry 관계를 설명할 수 있어야 한다.
- Ripple, skip, lookahead, select, tree adder의 tradeoff를 비교할 수 있어야 한다.
- Carry-save adder가 multi-input addition에 유리한 이유를 말할 수 있어야 한다.
- Logical shift, arithmetic shift, rotate의 차이를 이해해야 한다.
- Multiplier가 partial product와 adder tree로 구성되는 원리를 설명할 수 있어야 한다.

