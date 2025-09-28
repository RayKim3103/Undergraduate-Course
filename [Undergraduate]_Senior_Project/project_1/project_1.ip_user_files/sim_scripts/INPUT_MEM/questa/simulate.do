onbreak {quit -f}
onerror {quit -f}

vsim -lib xil_defaultlib INPUT_MEM_opt

do {wave.do}

view wave
view structure
view signals

do {INPUT_MEM.udo}

run -all

quit -force
