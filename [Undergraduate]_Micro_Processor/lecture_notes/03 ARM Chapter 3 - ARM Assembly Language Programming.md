# ARM Chapter 3 - ARM Assembly Language Programming

tags: #micro-processor #arm #assembly #ldr #str #branch #stack #conditional-execution

관련 노트: [[02 ARM Chapter 2 - ARM Architecture]], [[04 Computer Organization Chapter 6 - Storage and IO Topics]]

## 핵심 요약

이 자료는 ARM assembly programming의 기본 instruction을 설명한다. Data processing, data transfer, control flow instruction을 중심으로 register operand, immediate, shifted register operand, load/store addressing, multiple register transfer, stack, branch, conditional execution, subroutine call을 다룬다.

## Assembly Programming 관점

Assembly에서는 개별 machine instruction 수준에서 생각해야 한다. ARM instruction은 기본적으로 32-bit이며, register와 condition code, addressing mode를 직접 다룬다.

## Data Processing Instructions

Data processing instruction은 register 안의 값을 연산하고 결과를 register에 저장한다.

예:

```armasm
ADD r0, r1, r2
SUB r3, r4, r5
MOV r0, r1
AND r0, r1, r2
ORR r0, r1, r2
EOR r0, r1, r2
```

결과는 32-bit register에 저장된다.

## Arithmetic Instructions

| 명령 | 의미 |
|---|---|
| `ADD` | 덧셈 |
| `ADC` | carry 포함 덧셈 |
| `SUB` | 뺄셈 |
| `SBC` | carry/borrow 포함 뺄셈 |
| `RSB` | reverse subtraction |
| `RSC` | reverse subtraction with carry |

`S` suffix를 붙이면 연산 결과에 따라 CPSR condition code가 갱신된다.

```armasm
ADDS r0, r1, r2
```

## Logical and Move Instructions

| 명령 | 의미 |
|---|---|
| `AND` | bitwise AND |
| `ORR` | bitwise OR |
| `EOR` | bitwise XOR |
| `BIC` | bit clear |
| `MOV` | move |
| `MVN` | move NOT |
| `CMP` | 비교, condition code만 설정 |
| `TST` | bit test |

`CMP`는 내부적으로 subtraction을 수행하지만 결과 register를 저장하지 않고 flag만 바꾼다.

## Immediate Operand

Immediate operand는 instruction 안에 직접 들어가는 상수이다.

```armasm
ADD r0, r0, #1
MOV r1, #10
```

ARM immediate는 encoding 제한이 있으므로 모든 32-bit 상수를 한 instruction으로 표현할 수 있는 것은 아니다.

## Shifted Register Operand

ARM data processing instruction은 두 번째 register operand에 shift를 결합할 수 있다.

```armasm
ADD r0, r1, r2, LSL #2
```

이 명령은 `r1 + (r2 << 2)`를 계산한다. 별도 shift instruction 없이 multiply by power of two와 address 계산을 효율적으로 수행할 수 있다.

Shift 종류:

- `LSL`: logical shift left
- `LSR`: logical shift right
- `ASR`: arithmetic shift right
- `ROR`: rotate right

## Multiply

ARM은 multiplication instruction을 제공한다. 일부 multiply instruction에서는 결과 register가 첫 번째 source register와 같으면 안 되는 제약이 있을 수 있다. Assembly 작성 시 instruction별 operand restriction을 확인해야 한다.

## Data Transfer Instructions

ARM은 load-store 구조이므로 memory 접근은 `LDR`, `STR` 계열이 담당한다.

```armasm
LDR r0, [r1]
STR r0, [r2]
```

여기서 `r1`, `r2`는 base register로 memory address를 담는다.

## Addressing Modes

### Register-Indirect

```armasm
LDR r0, [r1]
```

`r1`이 가리키는 주소에서 word를 읽어 `r0`에 저장한다.

### Base Plus Offset

```armasm
LDR r0, [r1, #4]
```

`r1 + 4` 주소에서 읽는다.

### Post-Indexed

```armasm
LDR r0, [r1], #4
```

먼저 `[r1]`에서 읽고, 이후 `r1 = r1 + 4`로 갱신한다.

### Byte Access

```armasm
LDRB r0, [r1]
STRB r0, [r2]
```

Byte 단위 접근에 사용한다.

## Multiple Register Transfer

여러 register를 한 번에 memory로 저장하거나 읽을 수 있다.

```armasm
STMIA r9!, {r0, r1, r5}
LDMIA r9!, {r0, r1, r5}
```

낮은 register 번호가 낮은 memory address에 대응된다. `!`는 base register write-back을 의미한다.

## Stack Addressing

ARM stack은 ascending/descending, full/empty 개념으로 설명된다.

| 개념 | 의미 |
|---|---|
| Full stack | SP가 마지막 유효 data item을 가리킴 |
| Empty stack | SP가 비어 있는 slot을 가리킴 |
| Ascending | push 시 address 증가 |
| Descending | push 시 address 감소 |

일반적으로 full descending stack을 많이 사용하며, `STMFD`, `LDMFD` pseudo instruction으로 표현한다.

```armasm
STMFD r13!, {r0-r2, r14}
LDMFD r13!, {r0-r2, pc}
```

## Branch와 Conditional Execution

### Branch

```armasm
B label
BL subroutine
```

`BL`은 branch and link로, return address를 link register `r14`에 저장한다.

### Conditional Branch

```armasm
CMP r0, #0
BEQ zero_case
BNE nonzero_case
```

Condition code는 CPSR flag를 기반으로 판단된다.

### Conditional Execution

ARM은 opcode 뒤에 condition suffix를 붙여 짧은 if block을 branch 없이 실행할 수 있다.

```armasm
CMP r0, #0
ADDEQ r1, r1, #1
SUBNE r1, r1, #1
```

짧은 조건부 sequence에서는 branch penalty를 줄이고 code size를 줄일 수 있다.

## Jump Table

여러 subroutine 중 하나를 index로 호출하려면 jump table을 사용할 수 있다.

```armasm
LDRLS pc, [r1, r0, LSL #2]
SUBTAB
    DCD SUB0
    DCD SUB1
    DCD SUB2
```

`DCD`는 word constant를 배치하는 assembler directive이다.

## Hello World 예제

자료는 primitive version과 block copy version의 Hello World program을 통해 문자열을 memory에서 복사하고 byte 단위로 출력하는 흐름을 보여준다.

핵심 instruction:

- `ADR`: label 주소를 register에 적재
- `LDR`: word load
- `STR`: word store
- `LDRB`: byte load
- `CMP`: 종료 문자 검사
- conditional branch: loop 제어

## 시험ㆍ복습 체크포인트

- `LDR`, `STR`, `LDRB`, `STRB`의 차이를 설명할 수 있어야 한다.
- Pre/post-indexed addressing과 write-back의 의미를 이해해야 한다.
- `S` suffix와 `CMP`가 CPSR flag에 미치는 영향을 말할 수 있어야 한다.
- `BL`, `r14`, `pc`를 이용한 subroutine call/return을 설명할 수 있어야 한다.
- `STMFD`/`LDMFD`를 stack save/restore와 연결할 수 있어야 한다.

