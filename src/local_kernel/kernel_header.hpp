#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include <cstring>
#include <iostream>
#include <cmath>
#include <array>
#include <tuple>
#include <type_traits>

#define MAX_BITS_ROWS 16
#define MAX_BITS_COLS 64
#define MAX_ROWS 16
#define MAX_COLS 32 // hard to change this oddly enough
#define STRIDE 64
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

#define restrict __restrict__
#define noinline __attribute__((noinline))
#define GCC_COMPILER (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))

//NOTE: NOTHING WRONG WITH MASS KERNEL. HOWEVER, THEY NEED FLUSH_SUBNORMAL=false !!!!
#if !defined(KERNEL_HEADER)
    #define KERNEL_HEADER ../ffcx/kernels/poisson_kernel_hexahedron_4_action.hpp
#endif

#define str(s) #s
#define xstr(s) str(s)
#include xstr(KERNEL_HEADER)

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

#if !defined(KERNEL_TYPE_POISSON) && !defined(KERNEL_TYPE_MASS)
    #error "Only poisson and mass kernels are supported"
#endif

#if defined(KERNEL_TYPE_POISSON) && defined(KERNEL_TYPE_ACTION)
    #define POISSON_ACTION
#endif

#if defined(KERNEL_TYPE_POISSON) && !defined(KERNEL_TYPE_ACTION)
    #define POISSON_BILINEAR
#endif

#if defined(KERNEL_TYPE_MASS) && defined(KERNEL_TYPE_ACTION)
    #define MASS_ACTION
#endif

#if defined(KERNEL_TYPE_MASS) && !defined(KERNEL_TYPE_ACTION)
    #define MASS_BILINEAR
#endif

#define NTIMES 7

#define N_BATCH_CELLS (64)
static_assert(N_BATCH_CELLS%16 == 0);

#define BASE_N_TOTAL_CELLS 2000000
# if defined(CELL_TYPE_TETRAHEDRON) || defined(CELL_TYPE_HEXAHEDRON)
    #if defined(CELL_TYPE_TETRAHEDRON)
        #include "./meshes/tetrahedron_mesh.hpp"
    #else
        #include "./meshes/hexahedron_mesh.hpp"
    #endif
    #define N_TOTAL_CELLS ((long int) MIN(N_BATCH_CELLS*((BASE_N_TOTAL_CELLS/ELDIM)/N_BATCH_CELLS + 1), N_BATCH_CELLS*(N_MESH_CELLS/N_BATCH_CELLS)))
#else
    #define N_TOTAL_CELLS ((long int) (N_BATCH_CELLS*((BASE_N_TOTAL_CELLS/ELDIM)/N_BATCH_CELLS + 1)))
#endif

static_assert(N_TOTAL_CELLS%N_BATCH_CELLS == 0);

#if defined(KERNEL_TYPE_ACTION)
    #define ASIZE (ELDIM)
#else
    #define ASIZE (ELDIM*ELDIM)
#endif

#define BROWS (ELDIM/16 + ((ELDIM%16) > 0))
#define PADDEDROWS (BROWS*16)
#define NCOLS (NQUAD)
#define BCOLS (2*(NCOLS/32 + ((NCOLS)%32 > 0)))
#define PADDEDCOLS (BCOLS*16)

//Define tile config data structure 
typedef struct __tile_config
{
  uint8_t palette_id=0;
  uint8_t start_row=0;
  uint8_t reserved_0[14]={};
  uint16_t cols[16]={}; 
  uint8_t rows[16]={}; 
} __tilecfg;

__tilecfg tile_data = {0};

/* Initialize tile config */
void init_tile_config (__tilecfg *tileinfo)
{
  int i;
  tileinfo->palette_id = 1;
  tileinfo->start_row = 0;

  for (i = 0; i < 1; ++i)
  {
    tileinfo->cols[i] = MAX_BITS_ROWS;
    tileinfo->rows[i] =  MAX_BITS_ROWS;
  }

  for (i = 1; i < 4; ++i)
  {
    tileinfo->cols[i] = MAX_BITS_COLS;
    tileinfo->rows[i] =  MAX_BITS_ROWS;
  }

  _tile_loadconfig (tileinfo);
}

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
bool set_tiledata_use(){
   if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)){
      printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
      return false;
   }else{
      //printf("\n TILE DATA USE SET - OK \n\n");
      return true;
   }
   return true;
}

template <typename T>
void print_buffer(T* buf, int rows, int cols){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            printf("%.2e ", (double) buf[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

inline double get_time(const clock_t tic, const clock_t toc) noexcept {
    return (toc-tic)/((double) CLOCKS_PER_SEC);
}

template <typename TA, typename TB>
inline double errnorm(const TA * restrict Aex, const TB * restrict A, const int NN){

    double s = 0.;
    double t = 0.;

    for(int i=0; i<NN; i++){
        s = std::max(s, std::fabs((double) Aex[i] - ((double) A[i])));
        t = std::max(t, std::fabs((double) Aex[i]));
    }

    return s/t;
}

