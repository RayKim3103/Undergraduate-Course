#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "proc.h"
#include "defs.h"

struct cpu cpus[NCPU];

struct proc proc[NPROC];

struct proc *initproc;

int nextpid = 1;
struct spinlock pid_lock;

extern void forkret(void);
static void freeproc(struct proc *p);

extern char trampoline[]; // trampoline.S

/****************************************/
struct spinlock tid_lock;               // lock for creating TID 

// lock for twait(), but we don't need it
// because, thread and process uses same scheduler
// struct spinlock twait_lock; 

int nexttid = 1;                        // counter for assigning TID
static void freethread(struct proc *p); // free thread()
pagetable_t t_pagetable(struct proc *t, struct proc *p);
/****************************************/

// helps ensure that wakeups of wait()ing
// parents are not lost. helps obey the
// memory model when using p->parent.
// must be acquired before any p->lock.
struct spinlock wait_lock;

// Allocate a page for each process's kernel stack.
// Map it high in memory, followed by an invalid
// guard page.
void
proc_mapstacks(pagetable_t kpgtbl)
{
  struct proc *p;
  
  for(p = proc; p < &proc[NPROC]; p++) {
    char *pa = kalloc();
    if(pa == 0)
      panic("kalloc");
    uint64 va = KSTACK((int) (p - proc));
    kvmmap(kpgtbl, va, (uint64)pa, PGSIZE, PTE_R | PTE_W);
  }
}

// initialize the proc table.
void
procinit(void)
{
  struct proc *p;
  
  initlock(&pid_lock, "nextpid");
  initlock(&wait_lock, "wait_lock");
  for(p = proc; p < &proc[NPROC]; p++) {
    initlock(&p->lock, "proc");
    p->state = UNUSED;
    p->kstack = KSTACK((int) (p - proc));
  }
}

// Must be called with interrupts disabled,
// to prevent race with process being moved
// to a different CPU.
int
cpuid()
{
  int id = r_tp();
  return id;
}

// Return this CPU's cpu struct.
// Interrupts must be disabled.
struct cpu*
mycpu(void)
{
  int id = cpuid();
  struct cpu *c = &cpus[id];
  return c;
}

// Return the current struct proc *, or zero if none.
struct proc*
myproc(void)
{
  push_off();
  struct cpu *c = mycpu();
  struct proc *p = c->proc;
  pop_off();
  return p;
}

int
allocpid()
{
  int pid;
  
  acquire(&pid_lock);
  pid = nextpid;
  nextpid = nextpid + 1;
  release(&pid_lock);

  return pid;
}

/****************************************/
/* same with allocpid                   */
/* tid is used in threads               */
/* tid is independent with its process  */
/****************************************/
/****************************************/
int
alloctid() {
  int tid;
  
  acquire(&tid_lock);
  tid = nexttid;
  nexttid = nexttid + 1;
  release(&tid_lock);

  return tid;
}
/****************************************/

// Look in the process table for an UNUSED proc.
// If found, initialize state required to run in the kernel,
// and return with p->lock held.
// If there are no free procs, or a memory allocation fails, return 0.
static struct proc*
allocproc(void)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state == UNUSED) {
      goto found;
    } else {
      release(&p->lock);
    }
  }
  return 0;

found:
  p->pid = allocpid();
  p->state = USED;
/***************************************************************************/
/* 1. Why?                                                                 */
/* - thread should share virual memory space with its own process          */
/* 2. Which variable in struct proc is shared?                             */
/* - sz, ofile, cwd is shared. other variable is not shared                */
/* 3. How should we change the default sz, ofile, cwd?                     */
/* - use pointers                                                          */
/*                                                                         */
/* # NOTICE                                                                */
/* - non pointer variable (i.e. sz) is used in processes "independently"   */
/* - non pointer variable is created when process is created               */
/* - this is why allocproc is proper place to initialize pointer variable  */
/*                                                                         */
/* - pointer variable is used in threads                                   */
/* - pointer variable (i.e. sz_ptr) should points to non pointer variable  */
/*                                                                         */
/* - when making new thread, (i.e. tfork)                                  */
/* - it just copies the process's variable address to pointer variable     */
/***************************************************************************/
  /****************************************/
  p->sz_ptr = &(p->sz);               // sz(value of the virtual memory) address
  for(int i = 0; i < NOFILE; i++) {
    p->ofile_ptr[i] = &(p->ofile[i]); // ofile address
  }
  p->cwd_ptr = &(p->cwd);             // cwd address
  /****************************************/

  // Allocate a trapframe page.
  if((p->trapframe = (struct trapframe *)kalloc()) == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // An empty user page table.
  p->pagetable = proc_pagetable(p);
  if(p->pagetable == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // Set up new context to start executing at forkret,
  // which returns to user space.
  memset(&p->context, 0, sizeof(p->context));
  p->context.ra = (uint64)forkret;
  p->context.sp = p->kstack + PGSIZE;

  return p;
}

/****************************************************************************/
/* 1. thread uses same struct proc just like processes                      */
/*    thus, for loop is used (same with allocproc)                          */
/* 2. thread needs variable "tid", and have same "pid" with it's process    */
/* 3. t is treates almost same as process so it needs it's own trapframe    */
/*    thus, memory space for trapframe should be allocated                  */
/* 4. t's pagetable is independent with it's process                        */
/*    but, it should be same as it's parent's pagetable                     */
/*    (why? : thread should share virtual memory space with it's parent)    */
/*    allocated virtual memory space for its own trapframe should be mapped */
/* 5. t's context is initialized to 0                                       */
/*    and it should be initialized with parent's context return address     */
/****************************************************************************/
/****************************************/
static struct proc* 
allocthread(struct proc* p) {
  struct proc *t;
  for(t = proc; t < &proc[NPROC]; t++) {
    acquire(&t->lock);
    if(t->state == UNUSED) {
      goto found_t;
    } 
    else {
      release(&t->lock);
    }
  }
  return 0;
  
found_t:
  t->tid = alloctid();
  t->state = USED;
  t->pid = p->pid;  // 부모 PID 공유, not using allocpid()

  // Allocate a trapframe page. trapframe을 위한 메모리 할당
  if((t->trapframe = (struct trapframe *)kalloc()) == 0){
    freethread(t);
    release(&t->lock);
    return 0;
  }


  t->pagetable = t_pagetable(t, p);
  if(t->pagetable == 0){
    freethread(t);
    release(&t->lock);
    return 0;
  }

  // Set up new context to start executing at forkret,
  // which returns to user space.
  memset(&t->context, 0, sizeof(t->context));
  t->context.ra = (uint64)forkret;
  t->context.sp = t->kstack + PGSIZE;

  return t;
}
/****************************************/

// free a proc structure and the data hanging from it,
// including user pages.
// p->lock must be held.
static void
freeproc(struct proc *p)
{
  if(p->trapframe)
    kfree((void*)p->trapframe);
  p->trapframe = 0;
  if(p->pagetable)
  /****************************************/
    proc_freepagetable(p, p->pagetable, *(p->sz_ptr));
    // proc_freepagetable(p, p->pagetable, p->sz);
  /****************************************/
  p->pagetable = 0;
  p->sz = 0;
  /****************************************/
  // for process (not thread), p->sz = 0 is same with *(p->sz_ptr) = 0
  // *(p->sz_ptr) = 0; 
  /****************************************/
  p->pid = 0;
  p->parent = 0;
  p->name[0] = 0;
  p->chan = 0;
  p->killed = 0;
  p->xstate = 0;
  p->state = UNUSED;
}

/*************************************************************************/
/* 1. for threads only trapframe should be initialized                   */
/*    - why? process and other thread will use pagetable                 */
/*    - and use trampoline page and their own trapframes                 */
/* 2. other variables which freed thread was using should be initialized */
/*    - why? When a new thread is reallocated with allocthread,          */
/*    - it can reuse the memory of a thread that was in the UNUSED state.*/
/*************************************************************************/
static void
freethread(struct proc *p)
{
  if(p->trapframe)
    kfree((void*)p->trapframe);
  p->trapframe = 0;

  if(p->pagetable)
    uvmunmap(p->pagetable, TRAPFRAME(p-proc), 1, 0);
  // proc_freepagetable(p, p->pagetable, *(p->sz_ptr));

  p->xret = 0;
  p->pagetable = 0;
  p->sz = 0;
  // *(p->sz_ptr) = 0; // just initialize thread's sz, not process's sz
  p->pid = 0;
  p->parent = 0;
  p->name[0] = 0;
  p->chan = 0;
  p->killed = 0;
  p->xstate = 0;
  p->state = UNUSED;
}
/****************************************/

// Create a user page table for a given process, with no user memory,
// but with trampoline and trapframe pages.
pagetable_t
proc_pagetable(struct proc *p)
{
  pagetable_t pagetable;

  // An empty page table.
  pagetable = uvmcreate();
  if(pagetable == 0)
    return 0;

  // map the trampoline code (for system call return)
  // at the highest user virtual address.
  // only the supervisor uses it, on the way
  // to/from user space, so not PTE_U.
  if(mappages(pagetable, TRAMPOLINE, PGSIZE,
              (uint64)trampoline, PTE_R | PTE_X) < 0){
    uvmfree(pagetable, 0);
    return 0;
  }

  // map the trapframe page just below the trampoline page, for
  // trampoline.S.
  if(mappages(pagetable, TRAPFRAME(p-proc), PGSIZE,
              (uint64)(p->trapframe), PTE_R | PTE_W) < 0){
    uvmunmap(pagetable, TRAMPOLINE, 1, 0);
    uvmfree(pagetable, 0);
    return 0;
  }

  return pagetable;
}

/***********************************************************************/
/* thread's pagetable is independent with its parent pagetable         */
/* but, thread's pagetable should be same with its parent pagetable    */
/* furthermore, it should share trampoline virtual address with process*/
/*                                                                     */
/* trapframe is not shared with process,                               */
/* because it is used for thread (not process)                         */
/*                                                                     */
/* # NOTICE                                                            */
/* - proc is global variable (it is always same)                       */
/* - p or t is different with each other,                              */
/* - because it is allocated by allocproc or allocthread               */
/*                                                                     */
/* - each trapframe is used by corresponding p or t                    */
/***********************************************************************/
pagetable_t
t_pagetable(struct proc *t, struct proc *p)
{
  pagetable_t pagetable;

  t->pagetable = p->pagetable;
  // 부모 pagetable 공유
  pagetable = t->pagetable;

  // map the trapframe page just below the trampoline page, for
  // trampoline.S.
  if(mappages(pagetable, TRAPFRAME(t-proc), PGSIZE,
              (uint64)(t->trapframe), PTE_R | PTE_W) < 0){
    uvmunmap(pagetable, TRAMPOLINE, 1, 0);
    uvmfree(pagetable, 0);
    return 0;
  }
  // printf("\t ### t_pagetable: after mappages\n");
  return pagetable;
}
/****************************************/

// Free a process's page table, and free the
// physical memory it refers to.
void
proc_freepagetable(struct proc *p, pagetable_t pagetable, uint64 sz)
{
  uvmunmap(pagetable, TRAMPOLINE, 1, 0);
  uvmunmap(pagetable, TRAPFRAME(p-proc), 1, 0);
  uvmfree(pagetable, sz);
}

// a user program that calls exec("/init")
// assembled from ../user/initcode.S
// od -t xC ../user/initcode
uchar initcode[] = {
  0x17, 0x05, 0x00, 0x00, 0x13, 0x05, 0x45, 0x02,
  0x97, 0x05, 0x00, 0x00, 0x93, 0x85, 0x35, 0x02,
  0x93, 0x08, 0x70, 0x00, 0x73, 0x00, 0x00, 0x00,
  0x93, 0x08, 0x20, 0x00, 0x73, 0x00, 0x00, 0x00,
  0xef, 0xf0, 0x9f, 0xff, 0x2f, 0x69, 0x6e, 0x69,
  0x74, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00
};

// Set up first user process.
void
userinit(void)
{
  struct proc *p;

  p = allocproc();
  initproc = p;
  
  // allocate one user page and copy initcode's instructions
  // and data into it.
  uvmfirst(p->pagetable, initcode, sizeof(initcode));
  // p->sz = PGSIZE;
  /****************************************/
  *(p->sz_ptr) = PGSIZE;
  /****************************************/

  // prepare for the very first "return" from kernel to user.
  p->trapframe->epc = 0;      // user program counter
  p->trapframe->sp = PGSIZE;  // user stack pointer

  safestrcpy(p->name, "initcode", sizeof(p->name));
  // p->cwd = namei("/");
  /****************************************/
  *(p->cwd_ptr) = namei("/");
  /****************************************/

  p->state = RUNNABLE;

  release(&p->lock);
}

// Grow or shrink user memory by n bytes.
// Return 0 on success, -1 on failure.
int
growproc(int n)
{
  uint64 sz;
  struct proc *p = myproc();

  // sz = p->sz;
  /****************************************/
  sz = *(p->sz_ptr);
  /****************************************/
  if(n > 0){
    if((sz = uvmalloc(p->pagetable, sz, sz + n, PTE_W)) == 0) {
      return -1;
    }
  } else if(n < 0){
    sz = uvmdealloc(p->pagetable, sz, sz + n);
  }
  // p->sz = sz;
  /****************************************/
  *(p->sz_ptr) = sz;
  /****************************************/
  return 0;
}

// Create a new process, copying the parent.
// Sets up child kernel stack to return as if from fork() system call.
int
fork(void)
{
  int i, pid;
  struct proc *np;
  struct proc *p = myproc();

  // Allocate process.
  if((np = allocproc()) == 0){
    return -1;
  }

  // Copy user memory from parent to child.
  if(uvmcopy(p->pagetable, np->pagetable, p->sz) < 0){
    freeproc(np);
    release(&np->lock);
    return -1;
  }
  np->sz = p->sz;

  // copy saved user registers.
  *(np->trapframe) = *(p->trapframe);

  // Cause fork to return 0 in the child.
  np->trapframe->a0 = 0;

  // increment reference counts on open file descriptors.
  for(i = 0; i < NOFILE; i++)
    if(p->ofile[i])
      np->ofile[i] = filedup(p->ofile[i]);
  np->cwd = idup(p->cwd);

  safestrcpy(np->name, p->name, sizeof(p->name));

  pid = np->pid;

  release(&np->lock);

  acquire(&wait_lock);
  np->parent = p;
  release(&wait_lock);

  acquire(&np->lock);
  np->state = RUNNABLE;
  release(&np->lock);

  return pid;
}

/****************************************************************************/
/* # tfork() is similar with fork()                                         */
/* 1. thread should be created using allocthread                            */
/*    - allocthread() allocates new thread in proc table and initialize it  */
/* 2. p->sz of process and thread should be shared                          */
/*    - thus, pointer "sz_ptr" is used to share sz                          */
/* 3. register a0 is used for reading first argument of function to execute */
/*    - thus, a0 is used to pass argument to thread                         */
/* 4. "ofile" and "cwd" should be shared just like p->sz                    */
/*    - thus, pointer "ofile_ptr" and "cwd_ptr" is used                     */
/* 5. parent name is copied                                                 */
/* 6. stack pointer is set to the top of the stack                          */
/* 7. epc (program counter) is set to the function address                  */
/*    - function exec will execute the function                             */
/* 8. after initializing and setting state of tid, tid is returned          */
/****************************************************************************/

/****************************************/
int tfork(void *(*func)(void *), void *arg) {
  int i, tid;
  struct proc *nt;           // new thread
  struct proc *p = myproc(); // 부모 프로세스

  // allocate new thread
  if ((nt = allocthread(p)) == 0) {
    return -1;
  }


  // sz를 공유하도록 새로운 pointer를 소개
  nt->sz_ptr = p->sz_ptr;  // sz(value of the virtual memory) address

  // Cause tfork to return arg in the thread
  nt->trapframe->a0 = (uint64)arg;

  // increment reference counts on open file descriptors.
  for(i = 0; i < NOFILE; i++) {
    nt->ofile_ptr[i] = p->ofile_ptr[i];
    if(*(p->ofile_ptr[i])) {
      // should not write filedup(p->ofile[i]) because, p may be parent "thread"
      filedup(*(p->ofile_ptr[i])); // *(nt->ofile_ptr[i])
    }
  }
  nt->cwd_ptr = p->cwd_ptr; // cwd address
  idup(*(p->cwd_ptr)); // *(nt->cwd_ptr)

  safestrcpy(nt->name, p->name, sizeof(p->name));

  nt->trapframe->sp = PGROUNDUP(p->trapframe->sp) + (nt - p) * PGSIZE; // 스택 위치 설정
  nt->trapframe->epc = (uint64)func; // 함수 시작 주소 설정
  tid = nt->tid;

  release(&nt->lock);

  acquire(&wait_lock);
  nt->parent = p;
  release(&wait_lock);

  acquire(&nt->lock);
  nt->state = RUNNABLE;
  release(&nt->lock);

  return tid;
}
/****************************************/

// Pass p's abandoned children to init.
// Caller must hold wait_lock.
void
reparent(struct proc *p)
{
  struct proc *pp;

  for(pp = proc; pp < &proc[NPROC]; pp++){
    if(pp->parent == p){
      pp->parent = initproc;
      wakeup(initproc);
    }
  }
}

// Exit the current process.  Does not return.
// An exited process remains in the zombie state
// until its parent calls wait().
void
exit(int status)
{
  struct proc *p = myproc();

  if(p == initproc)
    panic("init exiting");

  // Close all open files.
  for(int fd = 0; fd < NOFILE; fd++){
    // if(p->ofile[fd]){
    //   struct file *f = p->ofile[fd];
    //   fileclose(f);
    //   p->ofile[fd] = 0;
    // }
    /****************************************/
    if(*(p->ofile_ptr[fd])){
      struct file *f = *(p->ofile_ptr[fd]);
      fileclose(f);
      *(p->ofile_ptr[fd]) = 0;
    }
    /****************************************/
  }

  begin_op();
  // iput(p->cwd);
  /****************************************/
  iput(*(p->cwd_ptr));
  /****************************************/
  end_op();
  // p->cwd = 0;
  /****************************************/
  *(p->cwd_ptr) = 0;
  /****************************************/

  acquire(&wait_lock);

  // Give any children to init.
  reparent(p);

  // Parent might be sleeping in wait().
  wakeup(p->parent);
  
  acquire(&p->lock);

  p->xstate = status;
  p->state = ZOMBIE;

  release(&wait_lock);

  // Jump into the scheduler, never to return.
  sched();
  panic("zombie exit");
}
/****************************************************************************/
/* # texit is similar with exit                                             */
/* 1. For thread exit there is no need to close files or drop ref. count    */
/*    - because, thread shares it's process's file or directory             */
/*    - (it does not have it's own file or directory)                       */
/* 2. If there is unfinished child thread,                                  */
/*      - it should be adopted by other process, which will be initproc     */
/* 3. As, thread is finished it's parent needs to be woke up                */
/* 4. xret is set to return value                                           */
/* 5. state is set to ZOMBIE,                                               */
/*    - (when thread or process exits, it's state goes to ZOMBIE)           */
/* 6. sched() is called to switch context                                   */
/****************************************************************************/
/****************************************/
void 
texit(void *ret) {
    struct proc *p = myproc();

    acquire(&wait_lock);

    // 자식 스레드 재배치
    reparent(p);

    // Parent might be sleeping in wait().
    wakeup(p->parent);

    // 반환 값 설정
    acquire(&p->lock);

    p->xret = (uint64)ret;  // xret(return 값) 설정  
    p->state = ZOMBIE;      // ZOMBIE 상태로 설정

    release(&wait_lock);

    sched(); // 컨텍스트 전환
    panic("texit: should not reach here");
}
/****************************************/

// Wait for a child process to exit and return its pid.
// Return -1 if this process has no children.
int
wait(uint64 addr)
{
  struct proc *pp;
  int havekids, pid;
  struct proc *p = myproc();

  acquire(&wait_lock);

  for(;;){
    // Scan through table looking for exited children.
    havekids = 0;
    for(pp = proc; pp < &proc[NPROC]; pp++){
      if(pp->parent == p){
        // make sure the child isn't still in exit() or swtch().
        acquire(&pp->lock);

        havekids = 1;
        if(pp->state == ZOMBIE){
          // Found one.
          pid = pp->pid;
          if(addr != 0 && copyout(p->pagetable, addr, (char *)&pp->xstate,
                                  sizeof(pp->xstate)) < 0) {
            release(&pp->lock);
            release(&wait_lock);
            return -1;
          }
          freeproc(pp);
          release(&pp->lock);
          release(&wait_lock);
          return pid;
        }
        release(&pp->lock);
      }
    }

    // No point waiting if we don't have any children.
    if(!havekids || killed(p)){
      release(&wait_lock);
      return -1;
    }
    
    // Wait for a child to exit.
    sleep(p, &wait_lock);  //DOC: wait-sleep
  }
}
/****************************************************************************/
/* # twait is similar with wait                                             */
/* 1. twait is used for waiting child thread to be finished                 */
/* 2. havekids is used to check whether there is child thread               */
/* 3. for loop is used to scan through table looking for exited children    */
/*    - since, there can be multiple thread in one process tid is checked   */
/* 4. if child thread is finished, it's xret(return value) is copied to addr*/
/*     - this operation let parent to read it's child return state value    */
/* 5. freethread is called to free the thread                               */
/*    - freethread only frees and unmappes the trapframe of thread          */
/*    - and it initialize child thread's struct proc variables              */
/*    - (why? it is explained in freethread's comment)                      */
/* 6. if, there is no childs or parent thread is killed it return -1        */
/* 7. if there is child thread, it sleeps and wait for child to exit        */
/****************************************************************************/
 /****************************************/
// Wait for a child thread to exit and return its pid.
// Return -1 if this thread has no children.
int 
twait(int tid, uint64 addr) {
  
  // printf("\t ### starting twait\n");

  struct proc *ct;  // child thread
  int havekids;
  struct proc *p = myproc();
  
  acquire(&wait_lock);

  for (;;) {
    havekids = 0;
    for (ct = proc; ct < &proc[NPROC]; ct++) {
      if (ct->parent == p && ct->tid == tid) {
        acquire(&ct->lock);

        havekids = 1;
        if (ct->state == ZOMBIE) {
          tid = ct->tid;

          // printf("\t ### twait: after np->state == ZOMBIE\n");

          if (addr != 0 && copyout(p->pagetable, addr, (char *)&ct->xret, 
                                    sizeof(ct->xret)) < 0) {
              release(&ct->lock);
              release(&wait_lock);
              return -1;
          }

          freethread(ct); // 자원 해제
          release(&ct->lock);
          release(&wait_lock);
          return 0;
        }
      release(&ct->lock);
      }
    }

    if (!havekids || killed(p)) {
      release(&wait_lock);
      return -1;
    }
      
    sleep(p, &wait_lock); // 대기
  }
}
 /****************************************/



// Per-CPU process scheduler.
// Each CPU calls scheduler() after setting itself up.
// Scheduler never returns.  It loops, doing:
//  - choose a process to run.
//  - swtch to start running that process.
//  - eventually that process transfers control
//    via swtch back to the scheduler.
void
scheduler(void)
{
  struct proc *p;
  struct cpu *c = mycpu();
  
  c->proc = 0;
  for(;;){
    // Avoid deadlock by ensuring that devices can interrupt.
    intr_on();

    for(p = proc; p < &proc[NPROC]; p++) {
      acquire(&p->lock);
      if(p->state == RUNNABLE) {
        // Switch to chosen process.  It is the process's job
        // to release its lock and then reacquire it
        // before jumping back to us.
        p->state = RUNNING;
        c->proc = p;
        swtch(&c->context, &p->context);

        // Process is done running for now.
        // It should have changed its p->state before coming back.
        c->proc = 0;
      }
      release(&p->lock);
    }
  }
}

// Switch to scheduler.  Must hold only p->lock
// and have changed proc->state. Saves and restores
// intena because intena is a property of this
// kernel thread, not this CPU. It should
// be proc->intena and proc->noff, but that would
// break in the few places where a lock is held but
// there's no process.
void
sched(void)
{
  int intena;
  struct proc *p = myproc();

  if(!holding(&p->lock))
    panic("sched p->lock");
  if(mycpu()->noff != 1)
    panic("sched locks");
  if(p->state == RUNNING)
    panic("sched running");
  if(intr_get())
    panic("sched interruptible");

  intena = mycpu()->intena;
  swtch(&p->context, &mycpu()->context);
  mycpu()->intena = intena;
}

// Give up the CPU for one scheduling round.
void
yield(void)
{
  struct proc *p = myproc();
  acquire(&p->lock);
  p->state = RUNNABLE;
  sched();
  release(&p->lock);
}

// A fork child's very first scheduling by scheduler()
// will swtch to forkret.
void
forkret(void)
{
  static int first = 1;

  // Still holding p->lock from scheduler.
  release(&myproc()->lock);

  if (first) {
    // File system initialization must be run in the context of a
    // regular process (e.g., because it calls sleep), and thus cannot
    // be run from main().
    first = 0;
    fsinit(ROOTDEV);
  }

  usertrapret();
}

// Atomically release lock and sleep on chan.
// Reacquires lock when awakened.
void
sleep(void *chan, struct spinlock *lk)
{
  struct proc *p = myproc();
  
  // Must acquire p->lock in order to
  // change p->state and then call sched.
  // Once we hold p->lock, we can be
  // guaranteed that we won't miss any wakeup
  // (wakeup locks p->lock),
  // so it's okay to release lk.

  acquire(&p->lock);  //DOC: sleeplock1
  release(lk);

  // Go to sleep.
  p->chan = chan;
  p->state = SLEEPING;

  sched();

  // Tidy up.
  p->chan = 0;

  // Reacquire original lock.
  release(&p->lock);
  acquire(lk);
}

// Wake up all processes sleeping on chan.
// Must be called without any p->lock.
void
wakeup(void *chan)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    if(p != myproc()){
      acquire(&p->lock);
      if(p->state == SLEEPING && p->chan == chan) {
        p->state = RUNNABLE;
      }
      release(&p->lock);
    }
  }
}

// Kill the process with the given pid.
// The victim won't exit until it tries to return
// to user space (see usertrap() in trap.c).
int
kill(int pid)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++){
    acquire(&p->lock);
    if(p->pid == pid){
      p->killed = 1;
      if(p->state == SLEEPING){
        // Wake process from sleep().
        p->state = RUNNABLE;
      }
      release(&p->lock);
      return 0;
    }
    release(&p->lock);
  }
  return -1;
}

void
setkilled(struct proc *p)
{
  acquire(&p->lock);
  p->killed = 1;
  release(&p->lock);
}

int
killed(struct proc *p)
{
  int k;
  
  acquire(&p->lock);
  k = p->killed;
  release(&p->lock);
  return k;
}

// Copy to either a user address, or kernel address,
// depending on usr_dst.
// Returns 0 on success, -1 on error.
int
either_copyout(int user_dst, uint64 dst, void *src, uint64 len)
{
  struct proc *p = myproc();
  if(user_dst){
    return copyout(p->pagetable, dst, src, len);
  } else {
    memmove((char *)dst, src, len);
    return 0;
  }
}

// Copy from either a user address, or kernel address,
// depending on usr_src.
// Returns 0 on success, -1 on error.
int
either_copyin(void *dst, int user_src, uint64 src, uint64 len)
{
  struct proc *p = myproc();
  if(user_src){
    return copyin(p->pagetable, dst, src, len);
  } else {
    memmove(dst, (char*)src, len);
    return 0;
  }
}

// Print a process listing to console.  For debugging.
// Runs when user types ^P on console.
// No lock to avoid wedging a stuck machine further.
void
procdump(void)
{
  static char *states[] = {
  [UNUSED]    "unused",
  [USED]      "used",
  [SLEEPING]  "sleep ",
  [RUNNABLE]  "runble",
  [RUNNING]   "run   ",
  [ZOMBIE]    "zombie"
  };
  struct proc *p;
  char *state;

  printf("\n");
  for(p = proc; p < &proc[NPROC]; p++){
    if(p->state == UNUSED)
      continue;
    if(p->state >= 0 && p->state < NELEM(states) && states[p->state])
      state = states[p->state];
    else
      state = "???";
    printf("%d %s %s", p->pid, state, p->name);
    printf("\n");
  }
}
