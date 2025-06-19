obj-m += sevenseg_driver.o

# Linux for FPGA does not show its kernel version with uname -r
#KERNEL_VERSION=$(shell uname -r)
# 4.9.0-dirty
KERNEL_VERSION=xlnx

all:
	make -C /lib/modules/${KERNEL_VERSION}/build M=$(PWD) modules

clean:
	make -C /lib/modules/${KERNEL_VERSION}/build M=$(PWD) clean
