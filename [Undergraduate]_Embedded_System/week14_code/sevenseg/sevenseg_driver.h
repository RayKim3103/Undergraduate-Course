#ifndef __SEVENSEG_DRIVER_H
#define __SEVENSEG_DRIVER_H

#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
// #include <asm/uaccess.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <asm/io.h>

#define SEVENSEG_DEVICE_MAJOR 222
#define SEVENSEG_DEVICE_NAME "zynq_sevenseg"
#define SEVENSEG_DEVICE_ADDRESS 0x43C00000
#define SEVENSEG_DEVICE_RANGE 0x1000

void *sevenseg_virtual_addr = NULL;

int sevenseg_open(struct inode* inode, struct file* filep);
int sevenseg_close(struct inode* inode, struct file* filep);
ssize_t sevenseg_write(struct file* filep, const char* user_buffer, size_t length, loff_t* f_pos);
ssize_t sevenseg_read(struct file* filep, char* user_buffer, size_t length, loff_t* f_pos);

#endif