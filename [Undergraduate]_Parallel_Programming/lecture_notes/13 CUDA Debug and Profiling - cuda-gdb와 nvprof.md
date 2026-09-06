---
title: "13 CUDA Debug and Profiling - cuda-gdb와 nvprof"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - cuda
  - debugging
  - profiling
---

# 13 CUDA Debug and Profiling - cuda-gdb와 nvprof

이전: [[12 CUDA Stream Updated - 업데이트본 요약]]  
다음: [[14 Multi GPU - 단일 노드와 MPI]]

## 핵심 요약

이 강의는 CUDA kernel의 correctness와 performance를 확인하기 위한 debugging/profiling 도구를 다룬다. `cuda-gdb`로 kernel 내부 thread/block/warp/lane에 focus를 맞추고, memcheck/racecheck로 silent error를 찾으며, profiler로 kernel timeline과 memory behavior를 분석한다.

## Compile Option

CUDA debugging에는 host와 device debug 정보가 필요하다.

| 옵션 | 의미 |
|---|---|
| `-g` | host code debug symbol |
| `-G` | device code debug symbol |

`-G`는 device optimization을 비활성화하고 성능을 크게 낮출 수 있으므로 debugging 때만 사용한다.

## cuda-gdb 기본 명령

| 명령 | 역할 |
|---|---|
| `run` 또는 `r` | 프로그램 시작 |
| `continue` 또는 `c` | 실행 재개 |
| `list` | source 표시 |
| `next` | 다음 source line |
| `step` | 함수 안으로 진입 |
| `nexti`, `stepi` | assembly instruction 단위 실행 |
| `print var` | 변수 값 출력 |
| `print $pc`, `print $R0` | register 확인 |

## Breakpoint

```gdb
break bcast
break bcast.cu:12
set cuda break_on_launch application
```

Kernel launch 시점에 breakpoint를 걸거나, 특정 source line에 걸 수 있다.

## CUDA Focus

CUDA kernel은 grid/block/thread/warp/lane이 많으므로 현재 focus를 확인하고 바꿀 수 있어야 한다.

```gdb
info cuda kernels
cuda kernel block thread
cuda kernel 0 block 0 thread 3
cuda lane 5
```

`*` 표시는 현재 focus를 나타낸다. Divergent thread/warp는 별도로 표시될 수 있다.

## Memcheck

Memcheck는 runtime error checker다. 특히 out-of-bounds access처럼 결과만 보고 찾기 어려운 silent error에 유용하다.

사용 방식:

- cuda-gdb 내부: `set cuda memcheck on`
- standalone tool
- 옵션: `--leak-check full`
- race 확인: `--tool racecheck`

## Profiling

성능 분석 도구:

- `nvprof`
- CSV 출력: `nvprof --csv`
- GPU trace: `nvprof --print-gpu-trace`
- 파일 출력: `nvprof -o tp.prof ./app`
- Visual profiler timeline

최신 GPU에서는 `nvprof` 지원이 제한될 수 있으므로 Nsight 계열 도구를 함께 고려해야 한다.

## 정리

CUDA debugging은 일반 C++ debugging보다 focus 차원이 많다. block/thread/warp/lane을 명시적으로 좁혀야 하며, memory error와 race는 memcheck/racecheck로 확인해야 한다. 성능 최적화는 profiler timeline과 metric을 보고 병목을 찾는 과정이다.
