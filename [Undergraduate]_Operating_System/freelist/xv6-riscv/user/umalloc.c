#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/param.h"

// Memory allocator by Kernighan and Ritchie,
// The C programming Language, 2nd ed.  Section 8.7.

typedef long Align;

union header {
  struct {
    union header *ptr;
    uint size;
  } s;
  Align x;
};

typedef union header Header;

static Header base;
static Header *freep;

void
freelist(void) { // Show the chain of free blocks.
  int i = 0;
  Header* p = &base;

  printf("Free list:\n");
  if(!freep) { printf("--\n"); return; } // Free list hasn't been created.
  for(p = p->s.ptr; p != &base; p = p->s.ptr) {
    printf("[%d] p = %p, p->s.size = %d bytes, p->s.ptr = %p\n",
           ++i, p, sizeof(Header) * p->s.size, p->s.ptr);
  } printf("\n");
}

void
free(void *ap)
{
  Header *bp, *p;

  bp = (Header*)ap - 1;
  for(p = freep; !(bp > p && bp < p->s.ptr); p = p->s.ptr)
    if(p >= p->s.ptr && (bp > p || bp < p->s.ptr))
      break;
  if(bp + bp->s.size == p->s.ptr){
    bp->s.size += p->s.ptr->s.size;
    bp->s.ptr = p->s.ptr->s.ptr;
  } else
    bp->s.ptr = p->s.ptr;
  if(p + p->s.size == bp){
    p->s.size += bp->s.size;
    p->s.ptr = bp->s.ptr;
  } else
    p->s.ptr = bp;
  freep = p;
}

static Header*
morecore(uint nu)
{
  char *p;
  Header *hp;

  if(nu < 4096)
    nu = 4096;
  p = sbrk(nu * sizeof(Header));
  if(p == (char*)-1)
    return 0;
  hp = (Header*)p;
  hp->s.size = nu;
  free((void*)(hp + 1));
  return freep;
}

void* malloc(uint nbytes) 
{
  // Flag to indicate if a suitable block is found
  int is_found = 0;
  // Pointers to traverse and manage the free list
  Header *p, *prevp, *bestp = 0, *bestprevp = 0;
  // Calculate the number of units needed, including space for the header
  uint nunits = (nbytes + sizeof(Header) - 1) / sizeof(Header) + 1;

  // If the free list is empty, initialize it with the base block
  if ((prevp = freep) == 0) 
  {
      base.s.ptr = freep = prevp = &base;
      base.s.size = 0;
  }

  // Traverse the free list to find the best fit block
  for (p = prevp->s.ptr; ; prevp = p, p = p->s.ptr) 
  {
      // Check if the current block is large enough
      if (p->s.size >= nunits) 
      {
          // Update the best fit block if it's the first found or larger than the current best
          if (bestp == 0 || p->s.size > bestp->s.size) 
          {
              bestp = p;
              bestprevp = prevp;
              is_found = 1;
          }
      }
      // If we have traversed the entire list and found no suitable block, request more memory
      if(p == freep && is_found == 0)
      {
        if ((p = morecore(nunits)) == 0)
          return 0; // Return null if more memory cannot be allocated
      }
      else if(p == freep)
        break; // Exit the loop if we have traversed the entire list
  }

  // If a suitable block is found, allocate memory from it
  if (bestp != 0) 
  {
      p = bestp;
      prevp = bestprevp;
      // If the block size matches exactly, remove it from the free list
      if (p->s.size == nunits) 
      {
          prevp->s.ptr = p->s.ptr;
      } 
      else 
      {
          // Otherwise, split the block and update the free list
          p->s.size -= nunits;
          p += p->s.size;
          p->s.size = nunits;
      }
      freep = prevp;
      return (void*)(p + 1); // Return a pointer to the allocated memory
  }

  // Return a pointer to the allocated memory (this line should not be reached)
  return (void*)(p + 1);
}

// void*
// malloc(uint nbytes)
// {
//   Header *p, *prevp;
//   uint nunits;

//   nunits = (nbytes + sizeof(Header) - 1)/sizeof(Header) + 1;
//   if((prevp = freep) == 0){
//     base.s.ptr = freep = prevp = &base;
//     base.s.size = 0;
//   }
//   for(p = prevp->s.ptr; ; prevp = p, p = p->s.ptr){
//     if(p->s.size >= nunits){
//       if(p->s.size == nunits)
//         prevp->s.ptr = p->s.ptr;
//       else {
//         p->s.size -= nunits;
//         p += p->s.size;
//         p->s.size = nunits;
//       }
//       freep = prevp;
//       return (void*)(p + 1);
//     }
//     if(p == freep)
//       if((p = morecore(nunits)) == 0)
//         return 0;
//   }
// }
