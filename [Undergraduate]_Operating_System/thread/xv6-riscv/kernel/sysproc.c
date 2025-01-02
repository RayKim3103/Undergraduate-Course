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

/****************************************/
uint64
sys_texit(void) {
  uint64 ret;
  argaddr(0, &ret);
  texit((void *)ret);
  return 0; // texit는 반환하지 않음
}
/****************************************/

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
/****************************************/
uint64
sys_tfork(void) {
    uint64 func_ptr, arg_ptr;

    argaddr(0, &func_ptr);
    argaddr(1, &arg_ptr);
    
    // (*)(void*): 함수 포인터를 나타냄
    // 즉, (*)(void*)는 "인자로 void*를 받는 함수"에 대한 포인터
    // void* (*)(void*)는 "인자로 void*를 받고, void*를 반환하는 함수의 포인터"
    void *(*func)(void *) = (void *(*)(void *))func_ptr;
    void *arg = (void *)arg_ptr;

    return tfork(func, arg);
}
/****************************************/

uint64
sys_wait(void)
{
  uint64 p;
  argaddr(0, &p);
  return wait(p);
}

/****************************************/
uint64
sys_twait(void) {
    int tid;
    uint64 addr;

    argint(0, (int *)&tid);
    argaddr(1, &addr);

    return twait(tid, addr);
}
/****************************************/

uint64
sys_sbrk(void)
{
  uint64 addr;
  int n;

  argint(0, &n);
  // addr = myproc()->sz;
  /****************************************/
  addr = *(myproc()->sz_ptr);
  /****************************************/
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
