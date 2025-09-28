onbreak {quit -f}
onerror {quit -f}

vsim -lib xil_defaultlib IA_ROW_MEM_opt

do {wave.do}

view wave
view structure
view signals

do {IA_ROW_MEM.udo}

run -all

quit -force
