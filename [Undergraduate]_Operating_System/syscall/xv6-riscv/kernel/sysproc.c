#include "types.h"
#include "riscv.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "proc.h"

uint64
sys_exit(void)
{
  int n;
  argint(0, &n);
  exit(n);
  return 0;  // not reached
}

uint64
sys_getpid(void)
{
  return myproc()->pid;
}

uint64
sys_fork(void)
{
  return fork();
}

uint64
sys_wait(void)
{
  uint64 p;
  argaddr(0, &p);
  return wait(p);
}

uint64
sys_sbrk(void)
{
  uint64 addr;
  int n;

  argint(0, &n);
  addr = myproc()->sz;
  if(growproc(n) < 0)
    return -1;
  return addr;
}

uint64
sys_sleep(void)
{
  int n;
  uint ticks0;

  argint(0, &n);
  acquire(&tickslock);
  ticks0 = ticks;
  while(ticks - ticks0 < n){
    if(killed(myproc())){
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}

uint64
sys_kill(void)
{
  int pid;

  argint(0, &pid);
  return kill(pid);
}

// return how many clock tick interrupts have occurred
// since start.
uint64
sys_uptime(void)
{
  uint xticks;

  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}

/**********************************************************/
/* Define function which counts open files */
int count_open_files(struct proc *p);
void print_process_info(struct proc *p, int hasParent, int ticks);
/**********************************************************/

uint64
sys_pstate(void)
{
  // EEE3535 Operating Systems
  // Assignment 2: System Call and Process
  /*****************************************************************
  * 1. To get the start time when the process starts,
  *    we need int variable to store it's ticks
  *    -> Add int start_ticks; to struct proc in proc.h
  * 2. To get the start time when the process starts 
  *    -> Add p->start_ticks = ticks; to allocproc(void) in proc.c
  * 3. To get char inputs of argv[] to integers, 
  *    change S, R, X, Z to integers which sign is negative 
  *    (since process pid is positive)
  *    change pid which is given as char to integers
  *    -> ps.c converts char inputs to integers
  * 4. Iterate through all processes and print process state
  *    -> sys_pstate in sysproc.c print process state
  *****************************************************************/

  struct proc *p; // Declare a struct variable to iterate through all processes and store each process  
  
  int arg_values[6]; // there can be max 6 arguments

  // argint(int n, int *ip) give us input arguments
  for (int i = 0; i < 6; i++)
  {
    argint(i, &arg_values[i]);
  }

  // printf Header
  printf("PID\tPPID\tState\tRuntime\tOFiles\tName\n");

   // iterate through all processes
  for(p = proc; p < &proc[NPROC]; p++) 
  {
    // values to determine whether ps cmd has extra options or not
    // e.g. $ps   -> it has no option (hasOption = optionPid = optionState = 0)
    // e.g. $ps 1 -> it has pid option (hasOption = optionPid = 1, optionState = 0)
    // e.g. $ps R -> it has state option (hasOption = optionState = 1, optionPid = 0)
    int hasOption = 0;
    int optionPid = 0;
    int optionState = 0;

    // iterates arg_values and determine which options it have
    // process id is converted in to positive integers in ps.c
    // S, R, X. Z is converted in to negative integers in ps.c
    for(int j = 1; j < 6; j++)
    {
      if(arg_values[j] != 0)
        {hasOption = 1;}
      
      if(arg_values[j] == p->pid)
        {optionPid = 1;}
      
      if((arg_values[j] == -1 && p->state == SLEEPING)
      || (arg_values[j] == -2 && p->state == RUNNABLE)
      || (arg_values[j] == -3 && p->state == RUNNING)
      || (arg_values[j] == -4 && p->state == ZOMBIE))
        {optionState = 1;}
    }

    int hasParent; // Check whether p->parent is valid or not

    // If p->parent is 0, which is root process, set hasParent is 0
    if (p->parent == 0) {hasParent = 0;}

    // If p->parent is not 0, get parent process's pid.
    else {hasParent = p->parent->pid;}

    // print processes, depending on its options
    if(hasOption == 0)
      print_process_info(p, hasParent, ticks);
    else if(optionPid)
      print_process_info(p, hasParent, ticks);
    else if(optionState)
      print_process_info(p, hasParent, ticks);
  }
  /**********************************************************/
  return 0;
}

/**********************************************************/
int count_open_files(struct proc *p) 
{
  // for loop to count number of files in ofile 
  // which is defined as array in proc.h
  int count = 0;
  for(int i = 0; i < NOFILE; i++) 
  {
      if(p->ofile[i] != 0) 
      {
          count++;
      }
  }
  return count;
}

void print_process_info(struct proc *p, int hasParent, int ticks) 
{
  // calculte runtime
  uint64 runtime_ticks = ticks - p->start_ticks;
  int minutes = runtime_ticks / 600;
  int seconds = (runtime_ticks / 10) % 60;
  int tenths = (runtime_ticks % 10);

  // print process information depending on its state (S, R, X ,Z)
  if (p->state == SLEEPING)
    printf("%d\t%d\tS\t%d:%d.%d\t%d\t%s\n", 
    p->pid, hasParent, 
    minutes, seconds, tenths, 
    count_open_files(p), p->name);

  else if (p->state == RUNNING)
    printf("%d\t%d\tX\t%d:%d.%d\t%d\t%s\n",     
    p->pid, hasParent, 
    minutes, seconds, tenths, 
    count_open_files(p), p->name);

  else if (p->state == RUNNABLE)
    printf("%d\t%d\tR\t%d:%d.%d\t%d\t%s\n",     
    p->pid, hasParent, 
    minutes, seconds, tenths, 
    count_open_files(p), p->name);

  else if (p->state == ZOMBIE)
    printf("%d\t%d\tZ\t%d:%d.%d\t%d\t%s\n",     
    p->pid, hasParent, 
    minutes, seconds, tenths, 
    count_open_files(p), p->name);
}
/**********************************************************/


  // int procout_int = 0;
  // printf("\t# procout_int: %d\n", procout_int);

  // char *pointer_char = (char *)malloc(sizeof(char));
  // *pointer_char = 'A';
  // printf("\t# procout_char: %c\n", pointer_char);

  // char procout_char = 'Y';
  // printf("\t# procout_char: %c\n", procout_char);


  // char proc_state[2]; // = {'X', 0}
  // char* proc_state;

  // if(p->state == RUNNING)
  // {
  //   strncpy(proc_state, "X", 2);
  //   //proc_state = "X";
  //   //safestrcpy(proc_state, "X", 1);
  //   printf("THIS PROC is Running: %s\n", proc_state);
  // }