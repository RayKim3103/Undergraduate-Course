#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

unsigned urand() {
    static unsigned rnd = 19;
    return rnd = (16807 * rnd) % ((unsigned)-1 >> 1);
}
