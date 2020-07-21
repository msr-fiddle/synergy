//This code is derived from https://github.com/awreece/memory-bandwidth-demo/blob/master/main.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <limits.h>
#include <cmath>
#include <string.h>
#include <omp.h>
#include <assert.h>
#include <unistd.h>
#define SAMPLES 5
#define TIMES 5
#define BYTES_PER_GB (1024*1024*1024LL)
#define SIZE (1*BYTES_PER_GB)
#define PAGE_SIZE (1<<12)
char array[SIZE + PAGE_SIZE] __attribute__((aligned (32)));
static inline double to_bw(size_t bytes, double secs) {
    double size_bytes = (double) bytes;
    double size_gb = size_bytes / ((double) BYTES_PER_GB);
    return size_gb / secs;
}

#define NANOS_PER_SECF 1000000000.0
#define USECS_PER_SEC 1000000
#if _POSIX_TIMERS > 0 && defined(_POSIX_MONOTONIC_CLOCK)
  // If we have it, use clock_gettime and CLOCK_MONOTONIC.
  #include <time.h>
  double monotonic_time() {
    struct timespec time;
    // Note: Make sure to link with -lrt to define clock_gettime.
    clock_gettime(CLOCK_MONOTONIC, &time);
    return ((double) time.tv_sec) + ((double) time.tv_nsec / (NANOS_PER_SECF));
}
#endif


void read_memory_loop(void* array, size_t size) {
  size_t* carray = (size_t*) array;
  size_t val = 0;
  size_t i;
  for (i = 0; i < size / sizeof(size_t); i++) {
    val += carray[i];
  }
}


// Time a function, printing out time to perform the memory operation and
// the computed memory bandwidth.
#define timefun(f, a) timeit(f, #f, a)
double timeit(void (*function)(void*, size_t), char* name, int num_rounds) {
  double min = INFINITY;
  size_t i;
  printf("Threads used : %d, num_rounds= %d\n", omp_get_max_threads(), num_rounds);
  for (i = 0; i < SAMPLES; i++) {
    double before, after, total;
    assert(SIZE % omp_get_max_threads() == 0);
    size_t chunk_size = SIZE / omp_get_max_threads();
#pragma omp parallel
    {
#pragma omp barrier
#pragma omp master
      before = monotonic_time();
      int j;
      for (j = 0; j < num_rounds; j++) {
	        function(&array[chunk_size * omp_get_thread_num()], chunk_size);
      }
#pragma omp barrier
#pragma omp master
      after = monotonic_time();
    }
    total = after - before;
    if (total < min) {
      min = total;
    }
  }
  double bw = to_bw(SIZE * num_rounds, min);
  printf("%28s_omp: %5.2f GiB/s\n", name, bw);
  return int(bw);
}



int main(int argc, char** argv) {
    int num_threads = 4;
    if (argc < 2)
       printf("Using default thread count = 4\n");
    else {
       num_threads = atoi(argv[1]);
       if (SIZE % num_threads != 0) {
           num_threads = pow(2, floor(log(num_threads)/log(2)));
           printf("Rounding num threads to %d\n", num_threads);
       }
    }
    omp_set_num_threads(num_threads);
    memset(array, 0xFF, SIZE);
    * ((uint64_t *) &array[SIZE]) = 0;

    int num_rounds = 5*num_threads;
    int bw = timefun(read_memory_loop, num_rounds);
    printf("%d\n", bw);
    return bw;
}
