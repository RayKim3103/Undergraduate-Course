---
title: "14주차 예비 - Linux Device Driver와 Device Control"
course: "Embedded System"
week: 14
type: "pre"
tags:
  - embedded-system
  - linux
  - device-driver
  - syscall
---

# 14주차 예비 - Linux Device Driver와 Device Control

이전: [[13주차 결과 - U-Boot Kernel Device Tree와 Driver 개념]]  
다음: [[14주차 결과 - Sevenseg Driver 8-Byte Read Write]]

## 핵심 요약

이번 예비보고서는 Linux booting 후 application이 device driver를 통해 실제 hardware를 제어하는 전체 흐름을 정리한다. 핵심은 user application이 직접 hardware address를 만지는 것이 아니라, device file과 system call을 통해 kernel driver에 요청하고, driver가 mapped virtual address로 PL device를 제어한다는 점이다.

## Linux Booting 요소

| 요소 | 역할 |
|---|---|
| Bootloader | hardware 초기화, device tree와 kernel image load |
| Kernel image | process, memory, network, device resource 관리 |
| Device tree | hardware 구조, address, size, interrupt 정보 전달 |
| File system | command, library, config, application, device file 제공 |

## 전체 제어 흐름

```text
User Application
-> open("/dev/zynq_sevenseg")
-> file descriptor 획득
-> write/read system call
-> device file
-> chrdev[] major number lookup
-> device_fops
-> device driver
-> copy_from_user / copy_to_user
-> ioremap된 virtual address
-> AXI interconnect
-> PL sevenseg IP
-> 7-segment / LED
```

## File Descriptor

각 process는 file descriptor table을 가진다. `open()`을 호출하면 비어 있는 index에 file object가 연결되고, 그 index가 fd로 반환된다. 이후 `read(fd)`, `write(fd)`, `close(fd)`는 이 fd를 통해 같은 file/device를 참조한다.

기본 fd:

- `0`: stdin
- `1`: stdout
- `2`: stderr

## Device File

Linux에서는 hardware도 파일처럼 다룬다. `/dev/zynq_sevenseg`는 application과 device driver 사이의 entry point다. 실제 driver 선택은 device file의 major number를 통해 이루어진다.

## `open()`

`open()`은 device file과 application 사이의 연결을 만든다.

Driver 관점에서는:

- device 사용 준비
- 필요한 초기화 수행 가능
- 성공 시 0 반환

Kernel은 driver의 `open` 함수를 호출하기 전후로 권한 확인, fd 관리, file object 생성 등 기본 처리를 수행한다.

## `write()`

`write()`는 user space data를 device로 보낸다. Kernel은 user memory를 직접 신뢰하거나 임의 접근하지 않으므로 `copy_from_user()`를 통해 user buffer를 kernel buffer로 복사한다. 이후 driver는 mapped virtual address에 값을 써서 hardware register를 제어한다.

## `read()`

`read()`는 반대 방향이다. Driver가 hardware register 또는 device memory에서 값을 읽어 kernel buffer에 저장하고, `copy_to_user()`로 user buffer에 전달한다.

## `close()`

`close()`는 fd를 닫고 device와 application 사이의 연결을 해제한다. Device 종류에 따라 close 시 hardware 상태를 초기화하거나 resource를 반환할 수 있다.

## PS와 PL의 관계

Zynq PS는 Linux kernel과 user application을 실행하는 CPU다. PL에는 sevenseg 같은 사용자 정의 hardware가 있고, PS와 PL은 AXI interconnect를 통해 memory mapped 방식으로 연결된다.

## 정리

14주차 예비의 핵심은 Linux 위에서 hardware를 제어할 때도 결국 핵심은 address mapping과 system call 흐름이라는 점이다. Application은 `/dev` 파일만 알고, driver가 kernel 내부에서 hardware 접근의 세부사항을 책임진다.
