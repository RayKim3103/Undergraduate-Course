---
title: "14 Multi GPU - 단일 노드와 MPI"
course: "Parallel Programming"
type: "lecture"
tags:
  - parallel-programming
  - multi-gpu
  - mpi
  - distributed-memory
---

# 14 Multi GPU - 단일 노드와 MPI

이전: [[13 CUDA Debug and Profiling - cuda-gdb와 nvprof]]  
다음: [[15 More Notes - DL Compiler와 LLM Inference]]

## 핵심 요약

이 강의는 하나의 GPU를 넘어 single-node multi-GPU와 multi-node multi-GPU를 다룬다. 단일 노드에서는 `cudaSetDevice`, peer copy, unified addressing, event를 사용하고, 여러 노드에서는 distributed memory model과 MPI message passing이 중심이 된다.

## 왜 Multi-GPU인가

- 단일 GPU memory 또는 compute가 부족할 수 있다.
- DNN training은 큰 data/model 때문에 여러 GPU가 필요하다.
- Vector addition처럼 쉽게 분할 가능한 작업은 GPU별로 나누기 좋다.
- Server/cluster 환경에서는 여러 GPU와 여러 node를 동시에 활용한다.

## Single-Node Multi-GPU

`cudaSetDevice()`로 현재 host thread가 사용할 GPU를 선택한다.

```cpp
cudaSetDevice(0);
// GPU 0 allocation/copy/kernel

cudaSetDevice(1);
// GPU 1 allocation/copy/kernel
```

비동기 copy와 stream을 함께 사용하면 한 GPU가 copy 중일 때 다른 GPU 작업을 진행할 수 있다.

## Unified Addressing

CUDA 4.0 이후 CPU와 GPU들이 private virtual memory address region을 가진 unified virtual addressing을 지원한다. Pointer 값만으로 어느 memory space인지 구분하기 쉬워지고, peer access와 runtime 관리가 단순해진다.

## GPU-GPU Communication과 Topology

GPU 간 통신 성능은 topology에 영향을 받는다.

- 같은 PCIe switch 아래 있는 GPU
- CPU socket을 가로질러 통신하는 GPU
- NVLink 연결 여부
- topology conflict로 인한 slow path

Multi-GPU 성능은 compute 분할뿐 아니라 data movement 경로가 중요하다.

## CUDA Events

Event는 timing 측정뿐 아니라 stream/device 작업 완료를 기다리는 데도 사용한다.

용도:

- kernel execution time 측정
- 특정 stream 작업 완료 확인
- GPU 간 dependency 표현

## Multi-Node Multi-GPU

여러 서버를 함께 쓰면 shared memory model만으로는 확장하기 어렵다. 각 node는 private memory를 가지며, network를 통해 data를 교환한다. 이것이 distributed memory model이다.

## Message Passing

Message passing에서는 data 교환이 명시적 send/receive로 이루어진다. 두 process가 matching되는 send/recv 호출을 해야 통신이 성립한다.

## MPI 기본

주요 함수:

| 함수 | 역할 |
|---|---|
| `MPI_Init` | MPI 환경 초기화 |
| `MPI_Comm_size` | rank 수 확인 |
| `MPI_Comm_rank` | 현재 process rank 확인 |
| `MPI_Send` | message 전송 |
| `MPI_Recv` | message 수신 |

`MPI_Send`는 buffer, count, datatype, destination, tag, communicator를 받는다. `MPI_Recv`는 source, tag, status를 통해 수신 정보를 얻는다.

## GPU와 MPI

현대 MPI/CUDA 환경에서는 host/device memory copy가 모두 가능하며, GPU buffer를 MPI 통신에 직접 사용할 수 있는 기능도 발전했다. 과거에는 host staging이 필요했지만, 최신 환경에서는 GPU-aware MPI가 성능을 개선할 수 있다.

## 정리

Multi-GPU는 병렬화를 한 단계 확장하지만 data movement가 더 중요해진다. 단일 노드는 device selection, stream, peer communication, event가 핵심이고, multi-node는 distributed memory와 MPI message passing을 이해해야 한다.
