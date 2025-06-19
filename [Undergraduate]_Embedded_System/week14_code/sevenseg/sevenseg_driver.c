#include "sevenseg_driver.h"

void *device_buffer = NULL;

static const struct file_operations sevenseg_fops =
{
	.open    = sevenseg_open,
	.release = sevenseg_close,
	.write   = sevenseg_write,
	.read    = sevenseg_read,
};

static int __init sevenseg_init(void){
	printk(KERN_INFO "Initiating sevenseg_driver\n");

	// register device driver code to the OS
	if(register_chrdev(SEVENSEG_DEVICE_MAJOR, SEVENSEG_DEVICE_NAME, &sevenseg_fops) < 0)
		printk(KERN_ALERT"<SEVENSEG_DRIVER> Cannot open %s module**********************\n", SEVENSEG_DEVICE_NAME);

	// maps device's physical address to os virtual address
	sevenseg_virtual_addr = ioremap(SEVENSEG_DEVICE_ADDRESS, SEVENSEG_DEVICE_RANGE);
	printk(KERN_DEBUG"<SEVENSEG_DRIVER> sevenseg_virtual_addr = 0x%lx\n", (unsigned int*)sevenseg_virtual_addr);
	printk(KERN_DEBUG"<SEVENSEG_DRIVER> Controller init success. Major number: %d\n", SEVENSEG_DEVICE_MAJOR);

	// allocate kernel heap memory space to store some data
	device_buffer = (void *)kmalloc(1024, GFP_KERNEL);

	return 0;
}

static void __exit sevenseg_exit(void){
	printk(KERN_INFO "Exiting sevenseg_driver\n");
	// unmap device driver's virtual address and physical address
	iounmap(sevenseg_virtual_addr);
	// disconnect device driver code from the OS
	unregister_chrdev(SEVENSEG_DEVICE_MAJOR, SEVENSEG_DEVICE_NAME);
	kfree(device_buffer);
	return;
}

int sevenseg_open(struct inode* inode, struct file* filep) {
	printk(KERN_INFO"<SEVENSEG_DRIVER> sevenseg_zynq is opened. \n");
	return 0;
}

int sevenseg_close(struct inode* inode, struct file* filep) {
	printk(KERN_INFO"<SEVENSEG_DRIVER> sevenseg_zynq is closed\n");
	return 0;
}

ssize_t sevenseg_write(struct file* filep, const char* user_buffer, size_t length, loff_t* f_pos) {
	int err;

	printk(KERN_INFO"<SEVENSEG_DRIVER> called sevenseg_write()\n");

	if(length != sizeof(unsigned) && length != sizeof(unsigned long long)) {
		printk(KERN_ALERT"<SEVENSEG_DRIVER> write length should be sizeof int or long long\n");
		printk(KERN_DEBUG"<SEVENSEG_DRIVER> current write length %d\n", length);
		return length;
	}

	err = copy_from_user(device_buffer, user_buffer, length);
	if(err < 0) {
		printk(KERN_ALERT"<SEVENSEG_DRIVER> device write error; error code %d\n", err);
		return err;
	}

	if(length == sizeof(unsigned))
		*(volatile unsigned *)sevenseg_virtual_addr = *(volatile unsigned *)device_buffer;
	else
		//TODO3: Write the code for 8-Byte data write
		*(volatile unsigned long long*)sevenseg_virtual_addr = *(volatile unsigned long long*)device_buffer;
		////////////////////////////////////////////
	return length;
}

ssize_t sevenseg_read(struct file* filep, char* user_buffer, size_t length, loff_t* f_pos) {
	int err;

	printk(KERN_INFO"<SEVENSEG_DRIVER> called sevenseg_read()\n");

	if(length != sizeof(unsigned) && length != sizeof(unsigned long long)) {
		printk(KERN_ALERT"<SEVENSEG_DRIVER> read length should be sizeof int or long long\n");
		printk(KERN_DEBUG"<SEVENSEG_DRIVER> current read length %d\n", length);
		return length;
	}

	if(length == sizeof(unsigned))
		*(volatile unsigned *)device_buffer = *(volatile unsigned *)sevenseg_virtual_addr;
	else
		//TODO4: Write the code for 8-Byte data read
		*(volatile unsigned long long*)device_buffer = *(volatile unsigned long long*)sevenseg_virtual_addr;
		////////////////////////////////////////////

	err = copy_to_user(user_buffer, device_buffer, length);
	if(err < 0) {
		printk(KERN_ALERT"<SEVENSEG_DRIVER> device read error; error code %d\n", err);
		return err;
	}

	return length;
}

module_init(sevenseg_init);
module_exit(sevenseg_exit);
