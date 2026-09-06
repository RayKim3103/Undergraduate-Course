---
title: "12주차 결과 - Zynq Ubuntu Root File System 구성"
course: "Embedded System"
week: 12
type: "result"
tags:
  - embedded-system
  - linux
  - rootfs
  - zynq
---

# 12주차 결과 - Zynq Ubuntu Root File System 구성

이전: [[11주차 결과 - PL Interrupt와 PS Handler]]  
다음: [[13주차 결과 - U-Boot Kernel Device Tree와 Driver 개념]]

## 핵심 요약

이번 실습은 Zynq 보드에서 Linux를 부팅하기 위한 Ubuntu 기반 root file system을 구성하는 과정이다. 핵심은 ARM용 rootfs를 x86 Linux server에서 `qemu-user-static`과 `chroot`로 준비하고, SD 카드의 ext4 partition에 복사하여 보드가 사용할 사용자 공간 환경을 만드는 것이다.

## 목표

- Linux OS 부팅에 필요한 root file system을 만든다.
- ARM 32bit 환경을 x86 host에서 emulation하여 package를 설치한다.
- SD 카드 partition을 boot용 FAT32와 rootfs용 ext4로 구성한다.
- UART console login을 위한 설정을 준비한다.

## 실험 과정

1. MobaXterm으로 Linux server 접속
2. `u-boot-xlnx` 작업 폴더에서 `zynq_xenial_rootfs` 생성
3. ARM processor용 Ubuntu rootfs 다운로드
4. `qemu-user-static` 설치 및 rootfs 내부로 복사
5. `/proc` mount 및 DNS 설정용 `resolv.conf` 복사
6. `chroot`로 ARM rootfs 환경 진입
7. root password와 user 계정 설정
8. 필요한 package 설치
9. UART login과 mount 관련 설정 파일 수정

## `qemu-user-static`

x86 host에서 ARM binary를 실행하기 위한 emulation 도구다. ARM rootfs 내부에서 `apt-get`, shell, ARM binary 등을 실행하려면 필요하다.

## `/proc` mount

`/proc`은 kernel이 제공하는 virtual file system이다. process, CPU, memory 상태 같은 runtime 정보를 제공한다. `chroot` 내부는 별도 root처럼 보이므로 `/proc`을 mount하지 않으면 `ps`, `top`, `/proc/cpuinfo` 등 system 정보 접근이 제대로 되지 않는다.

## `resolv.conf`

DNS 설정 파일이다. `apt-get`, `wget` 등이 domain name을 IP 주소로 바꾸려면 nameserver 정보가 필요하다. `chroot` 환경에는 독립적인 네트워크 설정이 부족하므로 host의 `/etc/resolv.conf`를 복사해 외부 인터넷 접근을 가능하게 한다.

## 설치 package와 역할

| Package | 역할 |
|---|---|
| `sudo` | 일반 사용자에게 root 권한 명령 허용 |
| `vim` | 설정 파일 편집 |
| `make` | Makefile 기반 build |
| `gcc` | C compiler |
| `build-essential` | compile에 필요한 기본 도구 묶음 |
| `kmod` | kernel module load/unload |
| `libssl-dev` | SSL/TLS 개발 header |
| `libncurses5-dev` | terminal UI 기반 build 지원 |
| `bc` | kernel build 중 계산 도구 |
| `udev` | device node 자동 생성 |
| `dialog` | terminal UI script 지원 |
| `net-tools` | `ifconfig`, `netstat` 등 |
| `iproute2` | `ip addr`, `ip link` 등 modern network tool |

## UART Console 설정

| 파일 | 목적 |
|---|---|
| `/etc/init/ttyPS0.conf` | boot 후 `ttyPS0`에 login prompt 표시 |
| `/etc/securetty` | `ttyPS0`에서 root login 허용 |
| `/etc/fstab` | boot 시 filesystem 자동 mount |

`ttyPS0`는 Zynq PS의 UART console 장치로, MobaXterm serial port와 연결되어 보드 Linux shell에 접근할 수 있게 한다.

## SD 카드 partition 구성

Zynq에서 Ubuntu를 부팅하려면 SD 카드에 두 영역이 필요하다.

| Partition | Format | 내용 |
|---|---|---|
| 1번 | FAT32 | boot image, device tree, kernel image, uEnv |
| 2번 | ext4 | root file system |

작업 순서:

1. 기존 mount 해제
2. `fdisk`로 기존 partition 삭제
3. 1GB boot partition 생성
4. boot 가능 flag 설정
5. 남은 공간으로 rootfs partition 생성
6. boot partition을 FAT32로 format
7. rootfs partition을 ext4로 format
8. rootfs partition mount
9. `zynq_xenial_rootfs` 내용 복사
10. mount 해제

Windows는 ext4를 기본 인식하지 못하므로 rootfs partition 작업은 Linux 환경에서 수행해야 한다.

## 왜 `/proc`은 복사하지 않는가

`/proc`은 실제 파일 모음이 아니라 kernel이 runtime에 제공하는 virtual file system이다. rootfs를 SD 카드로 복사할 때 `/proc` 내용은 의미가 없고, 오히려 예기치 않은 오류를 만들 수 있으므로 복사하지 않는다.

## 정리

12주차 결과의 핵심은 embedded Linux boot를 위해 boot image뿐 아니라 제대로 구성된 root file system이 필요하다는 점이다. 이번 주차는 SD 카드의 root partition을 준비하고, 다음 주차에서 boot image와 kernel, device tree를 결합해 실제 Linux boot로 이어진다.
