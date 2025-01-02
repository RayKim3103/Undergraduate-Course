#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "user/uthread.h"

int thread_create(thread_t *t, void*(*func)(void*), void *arg) {
  // 새로운 스레드 생성
  *t = tfork(func, arg);
  if (*t < 0)
      return -1;  // 스레드 생성 실패

  // 성공적으로 스레드 생성, 0 반환
  return 0;
  // return -1;
}

int thread_join(thread_t t, void **ret) {
  // printf("\n\t ### starting thread_join\n");
  return twait(t, (uint64)ret);
  // return -1;
}

void thread_exit(void *ret) {
  texit(ret);
  // exit(1);
}

void thread_mutex_init(thread_mutex_t *mtx) {
  mtx->locked = 0;
}

void thread_mutex_destroy(thread_mutex_t *mtx) {
}

void thread_mutex_lock(thread_mutex_t *mtx) {
  while (__sync_lock_test_and_set(&mtx->locked, 1));

  __sync_synchronize();
}

void thread_mutex_unlock(thread_mutex_t *mtx) {
  __sync_synchronize();
  
  __sync_lock_release(&mtx->locked);
}
