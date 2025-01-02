#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

// A xv6-riscv syscall can take up to six arguments.
#define max_args 6

// Print a help message.
void print_help(int argc, char **argv) {
  fprintf(2, "%s <options: pid or S/R/X/Z>%s\n",
             argv[0], argc > 7 ? ": too many args" : "");
}

int main(int argc, char **argv) {
  // Print a help message.
  if(argc > 7) { print_help(argc, argv); exit(1); }

  // Argument vector
  int args[max_args];
  memset(args, 0, max_args * sizeof(int));

  /* Assignment 2: System Call and Process
     Convert the char inputs of argv[] to integers in args[].
     In this skeleton code, args[] is initialized to zeros,
     so technically no arguments are passed to the pstate() syscall. */
  
  /**********************************************************/

  // for loop to convert argv[] char inputs to integers
  for (int i = 1; i < argc; i++) 
  {
    // min_bound to detect whether wrong arguments have been given
    // e.g. when 1+5 is given, atoi(1+5) = 1 and min_bound = 100
    // e.g. when -5 is given, atoi(-5) = 0 and min_bound = 10
    // Therefore, we can detect wrong arguments
    int min_bound = 1;
    for(int j = 0; j < strlen(argv[i])-1; j++)
    {
      min_bound = min_bound*10;
    }
    if((strlen(argv[i])>1) && (atoi(argv[i]) < min_bound))
    {
      // printf("## atoi: %d\n", atoi(argv[i]));
      // printf("## min_bound: %d\n", min_bound);
      print_help(argc, argv);
      exit(1);
    }
    // When correct argument is given convert argv[] char inputs to integers
    else if (argv[i][0] == 'S') args[i] = -1;            // SLEEPING
    else if (argv[i][0] == 'R') args[i] = -2;            // RUNNABLE
    else if (argv[i][0] == 'X') args[i] = -3;            // RUNNING
    else if (argv[i][0] == 'Z') args[i] = -4;            // ZOMBIE
    else if (atoi(argv[i]) > 0) args[i] = atoi(argv[i]); // Process ID
    else
    {
      print_help(argc, argv);
      exit(1);
    }
  }
  /**********************************************************/

  // Call the pstate() syscall.
  int ret = pstate(args[0], args[1], args[2], args[3], args[4], args[5]);
  if(ret) { fprintf(2, "pstate failed\n"); exit(1); }

  exit(0);
}
