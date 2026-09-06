---
과목: Electric Circuits 2
유형: Lecture Note
주제: CS with degeneration frequency response, coupling/bypass capacitor, CG frequency response
tags:
  - electric-circuits-2
  - frequency-response
  - source-degeneration
  - common-gate
---

# Frequency Response of CS Degeneration and CG - Degeneration CG 응답

## 핵심 요약

이 강의는 source degeneration이 있는 CS 증폭기의 저주파/고주파 응답과 CG 증폭기의 고주파 응답을 다룬다. coupling capacitor는 DC bias를 분리하면서 low-frequency pole/zero를 만들고, bypass capacitor는 저주파에서는 degeneration으로 안정성을 주고 고주파에서는 source를 AC ground에 가깝게 만들어 gain을 회복한다. CG는 input-output을 직접 잇는 `Cgd` Miller 효과가 없어 CS보다 빠르다.

## Coupling Capacitor

입력 coupling capacitor `Ci`는 DC를 차단하고 AC 신호를 통과시킨다.

전달:

```text
Vx/Vin = s(Rsig + Rin)Ci / [1 + s(Rsig + Rin)Ci]
```

input pole:

```text
wp,in = 1 / [(Rsig + Rin) Ci]
```

관심 최소 주파수 `wmin`에서 충분히 short처럼 보이려면:

```text
Ci > 1 / [(Rsig + Rin) wmin]
```

## CS with Bypassed Degeneration

source resistor `RS`에 bypass capacitor `Cb`를 병렬로 둔다.

저주파:

- `Cb` open
- gain은 degeneration 때문에 작음

```text
Av ≈ -gm RD / (1 + gm RS)
```

고주파:

- `Cb` short
- source가 AC ground에 가까워짐

```text
Av ≈ -gm RD
```

## Bypass Capacitor의 Pole/Zero

전달함수는 대략:

```text
Vout/Vx = -gm RD (1 + s RS Cb) /
          (1 + gm RS + s RS Cb)
```

zero:

```text
wz,out = 1 / (RS Cb)
```

pole:

```text
wp,out = (1 + gm RS) / (RS Cb)
```

따라서 frequency가 올라가면 gain이 `-gmRD/(1+gmRS)`에서 `-gmRD`로 증가한다.

## High-Frequency에서의 CS

고주파에서는 `Cgd`에 의한 Miller pole이 다시 중요해진다.

dominant pole 근사:

```text
wp ≈ 1 / [(Rsig || Rin) Cgd (1 + gm RD)]
```

결국 CS 고주파 응답은 [[12 Frequency Response of CS - CS 주파수 응답]]과 같은 Miller effect를 따른다.

## CG Amplifier Frequency Response

CG는 input과 output을 직접 연결하는 capacitor가 없다고 단순화할 수 있어 Miller effect가 작다.

input pole:

```text
wp,in ≈ (gm RS + 1) / [RS (Cgs + Csb)]
```

또는 source에서 보는 저항을 이용해:

```text
wp,in ≈ 1 / [(RS || 1/gm)(Cgs + Csb)]
```

output pole:

```text
wp,out ≈ 1 / [RD (Cgd + Cdb)]
```

## CG가 CS보다 빠른 이유

CS:

- `Cgd`가 input-output을 연결
- Miller effect로 input capacitance가 커짐

CG:

- gate가 AC ground
- `Cgd`가 output-to-ground capacitance처럼 작용
- Miller multiplication이 거의 없음

따라서 일반적으로 CG가 CS보다 bandwidth가 크다.

## 시험 포인트

- coupling capacitor가 high-pass pole을 만든다.
- bypass capacitor는 저주파 gain을 낮추고 고주파 gain을 회복한다.
- `wz = 1/(RS Cb)`, `wp = (1+gmRS)/(RS Cb)`를 기억한다.
- CG는 Miller effect가 작아 CS보다 빠르다.

## 같이 보면 좋은 노트

- [[04 Common-Source Amplifier - CS 증폭기]]
- [[05 Source Follower and Common-Gate - SF CG 증폭기]]
- [[12 Frequency Response of CS - CS 주파수 응답]]

