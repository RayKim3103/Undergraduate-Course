#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define LED_VALUE 0x87654321//TODO1: Put the right value to print LED as the video shows
#define SEG_VALUE 0x20872186//TODO2: Print your last 8 digits of student ID through sevenseg

int device_ctrl = -1;

int wr_len = 0;
int rd_len = 0;

unsigned wr_short_value = 0;
unsigned rd_short_value = 0;

unsigned long long wr_long_value = 0;
unsigned long long rd_long_value = 0;

int main(int argc, char **argv) {

	if(argc > 3) {
		printf("Up to 3 arguments are available\n");
		printf("Usage: %s, (arg1), (arg2) ", argv[0]);
	}
	printf("A simple program to test sevenseg driver\n");

	device_ctrl = open("/dev/zynq_sevenseg", O_RDWR);
	if(device_ctrl < 0) {
		printf("device open failed\n");
		return 1;
	}

	printf("device opened with fd number of %d\n", device_ctrl);

	// read and write a 4-byte data
	wr_short_value = SEG_VALUE;
	wr_len = write(device_ctrl, &wr_short_value, sizeof(unsigned));
	rd_len = read(device_ctrl, &rd_short_value, sizeof(unsigned));
	printf("write length : %d, read length : %d\n", wr_len, rd_len);
	printf("read value : 0x%08x\n", rd_short_value);

	// TODO2: read and write a 8-byte data
	// **8-byte data should be written or read by a single pointer
	wr_long_value = (LED_VALUE<<32LLU) | SEG_VALUE;
	wr_len = write(device_ctrl, &wr_long_value, sizeof(unsigned long long));
	rd_len = read(device_ctrl, &rd_long_value, sizeof(unsigned long long));
	printf("write length : %d, read length : %d\n", wr_len, rd_len);
	printf("read value : 0x%016llx\n", rd_long_value);
	//////////////////////////////////////

	close(device_ctrl);

	return 0;
}
