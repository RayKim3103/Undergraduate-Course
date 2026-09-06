# Sprite, BRAM, FPGA Graphics

tags: #basic-digital-experiment #sprite #bram #graphics #hdmi #verilog

관련 노트: [[11 HDMI TMDS LCD Display - HDMI TMDS LCD 출력]]

## 핵심 요약

이 자료는 HDMI 화면 위에 sprite를 합성하고, 버튼이나 frame timing으로 sprite를 움직이며, BRAM을 이용해 image data를 저장ㆍ출력하는 실험이다. 배경, sprite pixel, palette index, hit signal, BRAM 주소 계산이 핵심이다.

## Image Compression

Image compression은 image data의 저장 공간과 전송량을 줄이는 기술이다.

| 방식 | 설명 | 특징 |
|---|---|---|
| Lossless | 원본을 완전히 복원 가능 | 압축률은 낮지만 품질 손실 없음 |
| Lossy | 일부 정보를 버림 | 압축률이 높지만 품질 손실 발생 |

PNG는 lossless와 alpha channel에 강하고, JPEG는 사진처럼 자연 영상에 적합한 lossy compression을 사용한다.

## Image File Format

| Format | 특징 |
|---|---|
| JPEG | lossy, direct color, 사진에 적합 |
| PNG | lossless, alpha 지원 |
| GIF | indexed color, animation 가능 |
| BMP | 단순한 bitmap 저장 |
| Raw pixel array | header 없이 pixel data만 저장 가능 |

## Direct Color와 Indexed Color

### Direct Color

각 pixel이 RGB 값을 직접 가진다. 24-bit true color에서는 R, G, B가 각각 8-bit이다.

```text
pixel = {R[7:0], G[7:0], B[7:0]}
```

### Indexed Color

각 pixel은 실제 RGB 값이 아니라 palette table의 index를 저장한다.

```text
pixel index -> palette[index] -> RGB
```

색 수가 적은 sprite에서는 indexed color가 memory를 크게 줄여 준다.

## Sprite Graphics

Sprite는 배경과 독립적으로 움직이는 2D image object이다. 게임이나 embedded graphics에서 배경 전체를 매번 다시 그리지 않고 움직이는 object만 합성할 수 있어 효율적이다.

### 장점

- object 이동 구현이 단순하다.
- 배경과 전경을 분리할 수 있다.
- 작은 image data로 animation을 만들 수 있다.
- 투명색을 사용해 sprite 외곽을 배경과 자연스럽게 합성할 수 있다.

## BRAM

BRAM(Block RAM)은 FPGA 내부에 있는 고속 memory block이다.

### 특징

- FPGA fabric 가까이에 있어 접근 지연이 작다.
- image buffer, lookup table, FIFO, coefficient 저장에 활용된다.
- width와 depth 설정에 따라 저장 구조가 달라진다.

### BRAM 종류

| 종류 | 설명 |
|---|---|
| Single-port RAM | 하나의 port로 read/write |
| Simple dual-port RAM | write port와 read port 분리 |
| True dual-port RAM | 두 port 모두 read/write 가능 |
| Single-port ROM | 하나의 read port |
| Dual-port ROM | 두 read port |

## Experiment 1: Background와 Sprite 출력

### 구조

```text
test_card_simple background
+ sprite_compositor
+ gfx top module
-> HDMI output
```

`sprite_compositor`는 현재 pixel 좌표가 sprite 영역 안에 있는지 판단한다. 영역 안이라면 sprite image data에서 palette index를 읽고, palette table에서 RGB 값을 선택한다.

### Hit Signal

```text
sprite_hit = within_sprite_area && palette_index != 0
```

palette index 0을 transparent color로 두면, sprite 외부 또는 투명 pixel에서는 배경이 그대로 보인다.

### Flip

sprite를 좌우 또는 상하 반전하려면 sprite 내부 좌표를 바꿔 읽으면 된다.

```text
x_read = flip_x ? (15 - sprite_x) : sprite_x
y_read = flip_y ? (15 - sprite_y) : sprite_y
```

## Experiment 2: 자동 이동 Sprite

frame 시작을 나타내는 `i_v_sync` edge에서 sprite 위치를 갱신한다. 매 pixel clock마다 위치를 바꾸면 너무 빠르고 불안정하므로, frame 단위로 한 번만 이동시키는 것이 자연스럽다.

### Boundary Bounce

sprite가 화면 경계에 닿으면 방향을 반전한다.

```text
if x <= left edge  -> x direction = right
if x >= right edge -> x direction = left
```

1280x720 화면에서 sprite 크기를 고려해 `1280 - sprite_width`, `720 - sprite_height`가 최대 위치가 된다.

## Experiment 3: 버튼으로 Sprite 이동

버튼 입력을 top module에서 `gfx`, `sprite_compositor`로 전달해 sprite 위치를 조정한다.

버튼이 눌리면 x 또는 y 방향 좌표가 증가하거나 감소하고, 방향에 따라 flip bit를 바꿔 sprite가 이동 방향을 바라보도록 만들 수 있다.

## Experiment 4: 두 Sprite와 Overlap

두 sprite를 동시에 출력할 때는 각각의 hit signal을 계산한 뒤 우선순위를 정해야 한다.

```text
if sprite1_hit and sprite2_hit:
    overlap color or priority sprite
else if sprite1_hit:
    sprite1 color
else if sprite2_hit:
    sprite2 color
else:
    background color
```

겹침 영역을 별도 색으로 표시하면 collision detection이나 object interaction의 기초가 된다.

## Experiment 5: BRAM Image

BRAM을 이용하면 화면 배경이나 image data를 logic 조건식이 아니라 memory 값으로 저장할 수 있다.

### 주소 계산

1280x720 화면은 총 pixel 수가 다음과 같다.

```text
1280 * 720 = 921600
```

각 pixel을 하나의 memory word로 저장한다면 주소는 다음처럼 계산된다.

```text
addr = x + y * 1280
```

실험에서는 `BRAM_image.v`와 `blk_mem_gen_0`을 사용해 BRAM에서 data를 읽고, 읽은 값에 따라 RGB를 정했다.

### Width와 색 표현

BRAM data width가 3-bit이면 0부터 7까지 8가지 값을 표현할 수 있다. width가 2-bit이면 0부터 3까지 4가지 값만 표현되므로, 8가지 색을 구분하려던 패턴은 반복되거나 일부 색이 사라진다.

## Yellow Line 현상

BRAM image 출력에서 화면 왼쪽에 노란 선이 보인 현상은 read timing이나 초기 data 값과 관련된 것으로 해석된다. 특정 data 값의 색을 바꾸면 선의 색도 같이 바뀌므로, 해당 위치에서 의도와 다른 BRAM output이 먼저 읽히는 상황일 가능성이 높다.

## 시험ㆍ복습 체크포인트

- Sprite와 background 합성에서 hit signal과 transparent color의 역할을 설명할 수 있어야 한다.
- indexed color가 sprite memory를 줄이는 이유를 이해해야 한다.
- BRAM width와 depth가 저장 가능한 이미지 정보에 어떤 영향을 주는지 말할 수 있어야 한다.
- `addr = x + y * width` 형태의 2D 좌표 to 1D memory 주소 변환을 계산할 수 있어야 한다.

