#include "kernel_header.hpp"
#include "geometry.hpp"
#include "bf16_helpers.hpp"
#include "AVX512.hpp"
#include "AVX_BF16.hpp"
#include "AMX_BF16.hpp"
#include "cell_coordinates.hpp"
#include "geometry_test.hpp"

// If coeff and action then in ffcx the two coefficient arrays
// are concatenated. The first array is the coefficient one, and
// the second one the action one.
#if defined(KERNEL_TYPE_COEFFICIENT) && defined(KERNEL_TYPE_ACTION)
    #define FFCX_COEFF_SIZE 2*ELDIM
#else
    #define FFCX_COEFF_SIZE ELDIM
#endif

#define FFCX -1
#define AVX512 0
#define AVX512_BF16 1
#define AMX_BF16 2

//#undef N_TOTAL_CELLS
//#define N_TOTAL_CELLS 32

const std::array<const char *, 4> kernel_names = {"FFCX", "AVX512", "AVX512_BF16", "AMX_BF16"};

// FIXME: STEP 1 - Put all kernels in their own functions WITHOUT CHANGING ANYTHING and check it runs
//        STEP 2 - Add a template int to get_coordinate_dofs and kernel_test_ffcx  so that you
//                 can directly jump to computing what you need.
//        STEP 3 - Finish separating the cell batches in the code to save RAM.

template <long int CELL_BATCH=1, typename TG>
inline void get_coordinate_dofs(TG * restrict coordinate_dofs, const long int n=-1){
    if (n<0){
        #if !defined(N_MESH_CELLS)
            // Cast cdofs to type TG
            for (long int j = 0; j < N_TOTAL_CELLS; j++){
                for (long int i = 0; i < DOMAIN_ELDIM*GDIM; ++i){
                   coordinate_dofs[j*DOMAIN_ELDIM*GDIM + i] = (TG) cdofs[i];
                }
            }
            
        #else
            for (long int i = 0; i < N_TOTAL_CELLS*DOMAIN_ELDIM*GDIM; i++){
                coordinate_dofs[i] = (TG) mesh_coordinates[i];
            }
        #endif
    }else{
        #if !defined(N_MESH_CELLS)
            // Cast cdofs to type TG
            for (long int j = 0; j < CELL_BATCH; j++){
                for (long int i = 0; i < DOMAIN_ELDIM*GDIM; ++i){
                   coordinate_dofs[j*DOMAIN_ELDIM*GDIM + i] = (TG) cdofs[i];
                }
            }
            
        #else
            for (long int i = 0; i < CELL_BATCH*DOMAIN_ELDIM*GDIM; i++){
                coordinate_dofs[i] = (TG) mesh_coordinates[n*DOMAIN_ELDIM*GDIM + i];
            }
        #endif
    }
}

double kernel_test_ffcx(double * restrict times){

    clock_t tic, toc;

    double e = 0.;

    std::array<double, ASIZE> A{};

    std::array<double, DOMAIN_ELDIM*GDIM> coordinate_dofs{};
    std::array<double, FFCX_COEFF_SIZE> ffcx_coeffs{};

    // FIXME: We use the same coeffs on all cells so fine. In general this has to be put into the for loop
    for(long int j=0; j<ELDIM; j++){
        for(long int k=0; k<(FFCX_COEFF_SIZE/ELDIM); k++){
            ffcx_coeffs[k*ELDIM + j] = (double) (2.123456789*(j+1)/((double) ELDIM));
        }
    }

    for(long int n = 0; n < N_TOTAL_CELLS; n++){
        get_coordinate_dofs(coordinate_dofs.data(), n);
        tic = clock();
        ffcx_kernel(A.data(), coordinate_dofs.data(), ffcx_coeffs.data());
        toc = clock();
        times[NTIMES-1] += get_time(tic, toc);
    }

    return e;

}

template <typename TA, typename TG, typename TP, typename TS>
double kernel_test_avx(double * restrict times, TG * restrict weights, TG * restrict fe_domain_grad, TP * restrict fe0_geom, TP * restrict fe_grad_p){

    double e = 0.;

    #if defined(KERNEL_TYPE_POISSON)
        std::array<TP, NQUAD*GDIM*ELDIM> action_fe_grad{};
        restructure_array_action<TP, GDIM>(fe_grad_p, action_fe_grad.data());

        std::array<TS, NQUAD*GDIM*ELDIM> fe_grad{};
        for(int i=0; i<NQUAD*GDIM*ELDIM; i++){
            fe_grad[i] = (TS) fe_grad_p[i];
        }

    #else
        std::array<TP, NQUAD*ELDIM> action_fe0{};
        restructure_array_action<TP, 1>(fe0_geom, action_fe0.data());

        std::array<TS, NQUAD*ELDIM> fe0{};
        for(int i = 0; i<NQUAD*ELDIM; i++){
            fe0[i] = (TS) fe0_geom[i];
        }

    #endif

    std::array<double, N_BATCH_CELLS*ASIZE> Aex{};
    std::array<TA, N_BATCH_CELLS*ASIZE> A{};

    std::array<double, N_BATCH_CELLS*DOMAIN_ELDIM*GDIM> ffcx_coordinate_dofs{};
    std::array<TG, N_BATCH_CELLS*DOMAIN_ELDIM*GDIM> coordinate_dofs{};

    // NOTE: if only coeff or action, then ffcx_coeffs contains only coeff or action
    //       be careful in case you initialise coeffs and action_coeffs to have different values.
    std::array<TP, N_BATCH_CELLS*ELDIM> coeffs{};
    std::array<TP, N_BATCH_CELLS*ELDIM> action_coeffs{};
    std::array<double, N_BATCH_CELLS*FFCX_COEFF_SIZE> ffcx_coeffs{};

    // FIXME: We use the same coeffs on all cells so fine. In general this has to be put into the for loop
    for(long int i=0; i<N_BATCH_CELLS; i++){
        for(long int j=0; j<ELDIM; j++){
            coeffs[i*ELDIM + j] = (TP) (2.123456789*(j+1)/((double) ELDIM));
            action_coeffs[i*ELDIM + j] = (TP) (2.123456789*(j+1)/((double) ELDIM));
            for(long int k=0; k<(FFCX_COEFF_SIZE/ELDIM); k++){
                ffcx_coeffs[i*FFCX_COEFF_SIZE + k*ELDIM + j] = (double) (2.123456789*(j+1)/((double) ELDIM));
            }
        }
    }

    for(long int n = 0; n < N_TOTAL_CELLS/N_BATCH_CELLS; n++){
        std::fill(Aex.begin(), Aex.end(), 0.);

        get_coordinate_dofs<N_BATCH_CELLS>(ffcx_coordinate_dofs.data(), n*N_BATCH_CELLS);
        get_coordinate_dofs<N_BATCH_CELLS>(coordinate_dofs.data(), n*N_BATCH_CELLS);

        #if defined(POISSON_ACTION)
            local_kernel_poisson(times, A.data(), coordinate_dofs.data(), weights, fe_domain_grad, fe_grad.data(), action_fe_grad.data(), action_coeffs.data(), fe0_geom, coeffs.data());
        #elif defined(MASS_ACTION)
            local_kernel_mass(times, A.data(), coordinate_dofs.data(), weights, fe_domain_grad, action_coeffs.data(), fe0.data(), fe0_geom, action_fe0.data(), coeffs.data());
        #endif
        for(long int m = 0; m < N_BATCH_CELLS; m++){
            #if defined(POISSON_BILINEAR)
                local_kernel_poisson(times, A.data() + m*ASIZE, coordinate_dofs.data() + m*DOMAIN_ELDIM*GDIM, weights, fe_domain_grad, fe_grad.data(), action_fe_grad.data(), action_coeffs.data() + m*ELDIM, fe0_geom, coeffs.data() + m*ELDIM);
            #elif defined(MASS_BILINEAR)
                local_kernel_mass(times, A.data() + m*ASIZE, coordinate_dofs.data() + m*DOMAIN_ELDIM*GDIM, weights, fe_domain_grad, action_coeffs.data() + m*ELDIM, fe0.data(), fe0_geom, action_fe0.data(), coeffs.data() + m*ELDIM);
            #endif
            ffcx_kernel(Aex.data() + m*ASIZE, ffcx_coordinate_dofs.data() + m*DOMAIN_ELDIM*GDIM, ffcx_coeffs.data() + m*FFCX_COEFF_SIZE);
            e = std::max(e, errnorm(Aex.data() + m*ASIZE, A.data() + m*ASIZE, ASIZE));
        }
        
    }

    return e;
}

template <typename TG, typename TP, typename TS>
double kernel_test_avxbf16(double * restrict times, TG * restrict weights, TG * restrict fe_domain_grad, double * restrict fe0_d, double * restrict fe_grad_d){

    double e = 0.;

    std::array<TP, NQUAD*ELDIM> fe0_geom{};
    array_safecast<NQUAD*ELDIM>(fe0_d, fe0_geom.data());

    #if defined(KERNEL_TYPE_POISSON)
        std::array<TP, NQUAD*GDIM*ELDIM> action_fe_grad{};
        {   // Auto-deleting fe_grad_p by making it go out of scope
            std::array<TP, NQUAD*GDIM*ELDIM> fe_grad_p{};
            array_safecast<NQUAD*GDIM*ELDIM>(fe_grad_d, fe_grad_p.data());
            restructure_array_action<TP, GDIM>(fe_grad_p.data(), action_fe_grad.data());
        }
    #elif defined(KERNEL_TYPE_MASS)
        std::array<TP, NQUAD*ELDIM> action_fe0{};
        restructure_array_action<TP, 1>(fe0_geom.data(), action_fe0.data());
    #endif

    #if defined(POISSON_ACTION)
        constexpr int unpadded_cols = GDIM*NQUAD;
        constexpr int cols = unpadded_cols%32 == 0 ? unpadded_cols : 32*(unpadded_cols/32 + 1);

        std::array<TS, 1> fe_grad{};  // Empty array. Does nothing
        std::array<__bf16, ELDIM*cols> bf16_fe_grad{};
        for(int i = 0; i<GDIM; i++){
            for(int ic = 0; ic < ELDIM; ic++){
                for(int iq = 0; iq<NQUAD; iq++){
                    bf16_fe_grad[ic*cols + i*NQUAD + iq] = (__bf16) fe_grad_d[i*ELDIM*NQUAD + iq*ELDIM + ic];
                }
            }
        }
    #elif defined(POISSON_BILINEAR)
        std::array<TS, GDIM*ELDIM*NQUAD> fe_grad{};
        std::array<__bf16, GDIM*ELDIM*PADDEDCOLS> bf16_fe_grad{};
        for(int i = 0; i<GDIM; i++){
            for(int ic = 0; ic < ELDIM; ic++){
                for(int iq = 0; iq<NQUAD; iq++){
                    bf16_fe_grad[i*ELDIM*PADDEDCOLS + ic*PADDEDCOLS + iq] = (__bf16) fe_grad_d[i*ELDIM*NQUAD + iq*ELDIM + ic];
                    fe_grad[i*ELDIM*NQUAD + ic*NQUAD + iq] = (TS) fe_grad_d[i*ELDIM*NQUAD + iq*ELDIM + ic];
                }
            }
        }
    #elif defined(MASS_ACTION)
        std::array<TS, 1> fe0{}; // Empty array. Does nothing.
        std::array<__bf16, ELDIM*PADDEDCOLS> bf16_fe0{};
        for(int ic = 0; ic < ELDIM; ic++){
            for(int iq = 0; iq < NQUAD; iq++){
                bf16_fe0[ic*PADDEDCOLS + iq] = (__bf16) fe0_d[iq*ELDIM + ic];
            }
        }
    #elif defined(MASS_BILINEAR)
        std::array<TS, ELDIM*PADDEDCOLS> fe0{};
        std::array<__bf16, ELDIM*PADDEDCOLS> bf16_fe0{};
        for(long int ic = 0; ic < ELDIM; ic++){
            for(long int iq = 0; iq < NQUAD; iq++){
                bf16_fe0[ic*PADDEDCOLS + iq] = (__bf16) fe0_d[iq*ELDIM + ic];
                fe0[ic*PADDEDCOLS + iq] = (TS) fe0_d[iq*ELDIM + ic];
            }
        }
    #endif

    std::array<double, N_BATCH_CELLS*ASIZE> Aex{};
    std::array<float, N_BATCH_CELLS*ASIZE> A{};

    std::array<double, N_BATCH_CELLS*DOMAIN_ELDIM*GDIM> ffcx_coordinate_dofs{};
    std::array<TG, N_BATCH_CELLS*DOMAIN_ELDIM*GDIM> coordinate_dofs{};

    // NOTE: if only coeff or action, then ffcx_coeffs contains only coeff or action
    //       be careful in case you initialise coeffs and action_coeffs to have different values.
    std::array<TP, N_BATCH_CELLS*ELDIM> coeffs{};
    std::array<TP, N_BATCH_CELLS*ELDIM> action_coeffs{};
    std::array<double, N_BATCH_CELLS*FFCX_COEFF_SIZE> ffcx_coeffs{};

    // FIXME: We use the same coeffs on all cells so fine. In general this has to be put into the for loop
    for(long int i=0; i<N_BATCH_CELLS; i++){
        for(long int j=0; j<ELDIM; j++){
            coeffs[i*ELDIM + j] = (TP) (2.123456789*(j+1)/((double) ELDIM));
            action_coeffs[i*ELDIM + j] = (TP) (2.123456789*(j+1)/((double) ELDIM));
            for(long int k=0; k<(FFCX_COEFF_SIZE/ELDIM); k++){
                ffcx_coeffs[i*FFCX_COEFF_SIZE + k*ELDIM + j] = (double) (2.123456789*(j+1)/((double) ELDIM));
            }
        }
    }

    for(long int n = 0; n < N_TOTAL_CELLS/N_BATCH_CELLS; n++){
        std::fill(Aex.begin(), Aex.end(), 0.);

        get_coordinate_dofs<N_BATCH_CELLS>(ffcx_coordinate_dofs.data(), n*N_BATCH_CELLS);
        get_coordinate_dofs<N_BATCH_CELLS>(coordinate_dofs.data(), n*N_BATCH_CELLS);

        #if defined(POISSON_ACTION)
            local_kernel_avxbf16_poisson(times, A.data(), coordinate_dofs.data(), weights, fe_domain_grad, fe_grad.data(), bf16_fe_grad.data(), action_fe_grad.data(), action_coeffs.data(), fe0_geom.data(), coeffs.data());
        #elif defined(MASS_ACTION)
            local_kernel_avxbf16_mass(times, A.data(), coordinate_dofs.data(), weights, fe_domain_grad, action_coeffs.data(), fe0.data(), bf16_fe0.data(), fe0_geom.data(), action_fe0.data(), coeffs.data());
        #endif
        for(long int m = 0; m < N_BATCH_CELLS; m++){
            #if defined(POISSON_BILINEAR)
                local_kernel_avxbf16_poisson(times, A.data() + m*ASIZE, coordinate_dofs.data() + m*DOMAIN_ELDIM*GDIM, weights, fe_domain_grad, fe_grad.data(), bf16_fe_grad.data(), action_fe_grad.data(), action_coeffs.data() + m*ELDIM, fe0_geom.data(), coeffs.data() + m*ELDIM);
            #elif defined(MASS_BILINEAR)
                    local_kernel_avxbf16_mass(times, A.data() + m*ASIZE, coordinate_dofs.data() + m*DOMAIN_ELDIM*GDIM, weights, fe_domain_grad, action_coeffs.data() + m*ELDIM, fe0.data(), bf16_fe0.data(), fe0_geom.data(), action_fe0.data(), coeffs.data() + m*ELDIM);
            #endif
            ffcx_kernel(Aex.data() + m*ASIZE, ffcx_coordinate_dofs.data() + m*DOMAIN_ELDIM*GDIM, ffcx_coeffs.data() + m*FFCX_COEFF_SIZE);
            e = std::max(e, errnorm(Aex.data() + m*ASIZE, A.data() + m*ASIZE, ASIZE));
        }
    }

    return e;
}

template <typename TG, typename TP, typename TS>
double kernel_test_AMX(double * restrict times, TG * restrict weights, TG * restrict fe_domain_grad, double * restrict fe0_d, double * restrict fe_grad_d){

    constexpr int unpadded_rows = ELDIM;
    constexpr int rows = unpadded_rows%32 == 0 ? unpadded_rows : 32*(unpadded_rows/32 + 1);

    double e = 0.;

    std::array<TP, NQUAD*ELDIM> fe0_geom{};
    array_safecast<NQUAD*ELDIM>(fe0_d, fe0_geom.data());

    #if defined(KERNEL_TYPE_POISSON)
        std::array<TP, NQUAD*GDIM*ELDIM> action_fe_grad{};
        {   // Auto-deleting fe_grad_p by making it go out of scope
            std::array<TP, NQUAD*GDIM*ELDIM> fe_grad_p{};
            array_safecast<NQUAD*GDIM*ELDIM>(fe_grad_d, fe_grad_p.data());
            restructure_array_action<TP, GDIM>(fe_grad_p.data(), action_fe_grad.data());
        }
    #else
        std::array<TP, NQUAD*ELDIM> action_fe0{};
        restructure_array_action<TP, 1>(fe0_geom.data(), action_fe0.data());
    #endif

    std::array<double, N_BATCH_CELLS*ASIZE> Aex{};
    std::array<float, N_BATCH_CELLS*ASIZE> A{};

    std::array<double, N_BATCH_CELLS*DOMAIN_ELDIM*GDIM> ffcx_coordinate_dofs{};
    std::array<TG, N_BATCH_CELLS*DOMAIN_ELDIM*GDIM> coordinate_dofs{};

    // NOTE: if only coeff or action, then ffcx_coeffs contains only coeff or action
    //       be careful in case you initialise coeffs and action_coeffs to have different values.
    //FIXME: Make sure direct casting of double to __bf16 in padded_action_coeffs works
    std::array<__bf16, N_BATCH_CELLS*rows> padded_action_coeffs{};
    std::array<TP, N_BATCH_CELLS*ELDIM> coeffs{};
    std::array<double, N_BATCH_CELLS*FFCX_COEFF_SIZE> ffcx_coeffs{};

    for(long int i=0; i<N_BATCH_CELLS; i++){
        for(long int j=0; j<ELDIM; j++){
            coeffs[i*ELDIM + j] = (TP) (2.123456789*(j+1)/((double) ELDIM));
            padded_action_coeffs[i*rows + j] = (__bf16) (2.123456789*(j+1)/((double) ELDIM));
            for(long int k=0; k<(FFCX_COEFF_SIZE/ELDIM); k++){
                ffcx_coeffs[i*FFCX_COEFF_SIZE + k*ELDIM + j] = (double) (2.123456789*(j+1)/((double) ELDIM));
            }
        }
    }

    // Request permission to linux kernel to run AMX 
    if (!set_tiledata_use())
        exit(-1);

    init_tile_config(&tile_data);
    
    //////////////////// ADDITIONAL AMX PREPARATION BEGIN ///////////////
    
    #if defined(POISSON_BILINEAR)
        std::array<__bf16, GDIM*PADDEDROWS*PADDEDCOLS> AMX_fe_grad{};
        std::array<TS, GDIM*PADDEDROWS*PADDEDCOLS> AMX_fe_grad_transposed{};
        restructure_array_AMX<double, __bf16, GDIM, ELDIM, NQUAD, true, false>(fe_grad_d, AMX_fe_grad.data());
        restructure_array_AMX<double, TS, GDIM, ELDIM, NQUAD, true, true>(fe_grad_d, AMX_fe_grad_transposed.data());

    #elif defined(MASS_BILINEAR)
        std::array<__bf16, PADDEDROWS*PADDEDCOLS> AMX_fe0{};
        std::array<TS, PADDEDROWS*PADDEDCOLS> AMX_fe0_transposed{};
        restructure_array_AMX<double, __bf16, 1, ELDIM, NQUAD, true, false>(fe0_d, AMX_fe0.data());
        restructure_array_AMX<double, TS, 1, ELDIM, NQUAD, true, true>(fe0_d, AMX_fe0_transposed.data());

    #elif defined (POISSON_ACTION)
        std::array<__bf16, GDIM*PADDEDROWS*PADDEDCOLS> AMX_fe_grad_transposed{};
        restructure_array_AMX<double, __bf16, 1, ELDIM, GDIM*NQUAD, true, true>(fe_grad_d, AMX_fe_grad_transposed.data());

        std::array<__bf16, GDIM*PADDEDCOLS*rows> AMX_action_fe_grad{};
        {   // Auto-deleting padded_action_fe_grad by making it go out of scope
            std::array<double, GDIM*PADDEDCOLS*rows> padded_action_fe_grad{};
            for(int i = 0; i < GDIM; i++){
                for (int iq = 0; iq < NQUAD; iq++){
                    for(int ic = 0; ic < ELDIM; ic++){
                        padded_action_fe_grad[i*PADDEDCOLS*rows + iq*rows + ic] = fe_grad_d[i*NQUAD*ELDIM + iq*ELDIM + ic];
                    }
                }
            }

            restructure_array_AMX<double, __bf16, GDIM, PADDEDCOLS, rows, false, true>(padded_action_fe_grad.data(), AMX_action_fe_grad.data());
        }
        
    #elif defined(MASS_ACTION)
        std::array<__bf16, PADDEDROWS*PADDEDCOLS> AMX_fe0_transposed{};
        restructure_array_AMX<double, __bf16, 1, ELDIM, NQUAD, true, true>(fe0_d, AMX_fe0_transposed.data());

        std::array<__bf16, PADDEDCOLS*rows> AMX_action_fe0{};
        {   // Auto-deleting padded_action_fe0 by making it go out of scope
            std::array<double, PADDEDCOLS*rows> padded_action_fe0{};
            for (int iq = 0; iq < NQUAD; iq++){
                for(int ic = 0; ic < ELDIM; ic++){
                    padded_action_fe0[iq*rows + ic] = fe0_d[iq*ELDIM + ic];
                }
            }

            restructure_array_AMX<double, __bf16, 1, PADDEDCOLS, rows, false, true>(padded_action_fe0.data(), AMX_action_fe0.data());
        }

    #endif
    
    //////////////////// ADDITIONAL AMX PREPARATION END ////////////////

    for(long int n = 0; n < N_TOTAL_CELLS/N_BATCH_CELLS; n++){
        std::fill(Aex.begin(), Aex.end(), 0.);

        get_coordinate_dofs<N_BATCH_CELLS>(ffcx_coordinate_dofs.data(), n*N_BATCH_CELLS);
        get_coordinate_dofs<N_BATCH_CELLS>(coordinate_dofs.data(), n*N_BATCH_CELLS);

        #if defined(POISSON_ACTION)
            AMX_kernel_poisson_action<TG, TP, TS>(times, A.data(), coordinate_dofs.data(), weights, fe_domain_grad, padded_action_coeffs.data(), fe0_geom.data(), coeffs.data(), AMX_action_fe_grad.data(), AMX_fe_grad_transposed.data());
        #elif defined(MASS_ACTION)
            AMX_kernel_mass_action<TG, TP, TS>(times, A.data(), coordinate_dofs.data(), weights, fe_domain_grad, padded_action_coeffs.data(), fe0_geom.data(), coeffs.data(), AMX_action_fe0.data(), AMX_fe0_transposed.data());
        #endif
        for(long int m = 0; m < N_BATCH_CELLS; m++){
            #if defined(POISSON_BILINEAR)
                AMX_kernel_poisson(times, A.data() + m*ASIZE, coordinate_dofs.data() + m*DOMAIN_ELDIM*GDIM, weights, fe_domain_grad, fe0_geom.data(), coeffs.data() + m*ELDIM, AMX_fe_grad.data(), AMX_fe_grad_transposed.data());
            #elif defined(MASS_BILINEAR)
                AMX_kernel_mass(times, A.data() + m*ASIZE, coordinate_dofs.data() + m*DOMAIN_ELDIM*GDIM, weights, fe_domain_grad, fe0_geom.data(), coeffs.data() + m*ELDIM, AMX_fe0.data(), AMX_fe0_transposed.data());
            #endif
            ffcx_kernel(Aex.data() + m*ASIZE, ffcx_coordinate_dofs.data() + m*DOMAIN_ELDIM*GDIM, ffcx_coeffs.data() + m*FFCX_COEFF_SIZE);
            e = std::max(e, errnorm(Aex.data() + m*ASIZE, A.data() + m*ASIZE, ASIZE));
        }
    }

    _tile_release();

    return e;
}

template <typename TA, typename TG, typename TP, typename TS, int variant=AVX512>
noinline std::tuple<double, std::array<int, 5>> kernel_test(double * restrict times){

    static_assert((variant >= -1) && (variant <= 2));
    static_assert((variant != AMX_BF16 || std::is_same_v<TA, float> ));
    static_assert((variant != AVX512_BF16 || std::is_same_v<TA, float> ));
    static_assert(variant != FFCX || (std::is_same_v<TA, double> && std::is_same_v<TG, double> && std::is_same_v<TP, double> && std::is_same_v<TS, double>));

    double e;

    for(int i = 0; i < NTIMES; i++)
        times[i] = 0.;

    std::array<TG, DOMAIN_NQUAD*GDIM*DOMAIN_ELDIM> fe_domain_grad{};
    std::array<TG, NQUAD> weights{};

    // Use double precision for BF16 variants to avoid possible errors in double casting double->fp16->bf16 in the offline computations
    using TTP = std::conditional<variant == AMX_BF16 || variant == AVX512_BF16, double, TP>::type;
    std::array<TTP, NQUAD*ELDIM> fe0_geom{};
    std::array<TTP, NQUAD*GDIM*ELDIM> fe_grad_p{};

    preprocess_tabulated_values(weights.data(), fe_domain_grad.data(), fe_grad_p.data(), fe0_geom.data());

    if constexpr(variant == AVX512){
        e = kernel_test_avx<TA, TG, TP, TS>(times, weights.data(), fe_domain_grad.data(), fe0_geom.data(), fe_grad_p.data());
    }else if constexpr(variant == AVX512_BF16){
        e = kernel_test_avxbf16<TG, TP, TS>(times, weights.data(), fe_domain_grad.data(), fe0_geom.data(), fe_grad_p.data());
    }else if constexpr(variant == AMX_BF16){
        e = kernel_test_AMX<TG, TP, TS>(times, weights.data(), fe_domain_grad.data(), fe0_geom.data(), fe_grad_p.data());
    }else if constexpr(variant == FFCX){
        e = kernel_test_ffcx(times);
    }else{
        static_assert(false);
    }

    for(int i = 0; i < NTIMES; i++)
        times[i] /= ((double) N_TOTAL_CELLS);

    const std::array<int, 5> input_info = {variant, 8*sizeof(TA), 8*sizeof(TG), 8*sizeof(TP), 8*sizeof(TS)};
    return std::make_tuple(e, input_info);
}

template <typename T, int variant=AVX512>
std::tuple<double, std::array<int, 5>> kernel_test_s(double * restrict times){
    if constexpr(variant != AMX_BF16 && variant != AVX512_BF16){
        return kernel_test<T, T, T, T, variant>(times);
    }else{
        return kernel_test<float, T, T, _Float16, variant>(times);
    }
}

void print_kernel_test_results(const double e, const double * restrict t, const std::array<int, 5> ii){
    printf("\n\nResults %s (fp%d, fp%d, fp%d, fp%d):\n\n", kernel_names[ii[0]+1], ii[1], ii[2], ii[3], ii[4]);
    printf("ERROR:                 %e\n\n", e);
    printf("Total time:            %e\n\n", t[6]);
    printf("Jacobian construction: %e\n",   t[0]);
    printf("Jacobian adj and det:  %e\n",   t[1]);
    printf("Geometry mults:        %e\n",   t[2]);
    printf("Geometry total:        %e\n",   t[3]);
    printf("Preparation sum2gemm:  %e\n",   t[4]);
    printf("Sum2gemm3:             %e\n",   t[5]);
}

int main(){

    double e = 0.;
    double t[8][NTIMES]={};
    std::array<int, 5> ii = {};

    ///////////////////// TESTS BEGIN //////////////////

    ////NOTE: DO NOT PRINT "ERROR" IN CAPITAL LETTERS:
    ////      It will screw with the run_all_kernels.py script!!!
    //printf("\n########### TESTS ############\n\n");

    //// geometry test
    //for(int i = 0; i<10; i++){
    //    
    //    double eps = std::pow(4.0, -i);
    //    printf("\n\nepsilon: %e", eps);

    //    e = geometry_test<double>(eps);
    //    printf("\nGeometry error fp64: %e", e);

    //    e = geometry_test<float>(eps);
    //    printf("\nGeometry error fp32: %e", e);

    //    e = geometry_test<_Float16>(eps);
    //    printf("\nGeometry error fp16: %e", e);

    //    printf("\n\n");
    //}

    /////////////////////// TESTS END //////////////////

    printf("\n\n\n########### ACTUAL RUN ############\n\n");

    //////////// kernel_test<TA, TG, TP, TS, variant>(...); 
    
    //std::tie(e,ii) = kernel_test_s<double, FFCX>(t[0]);
    print_kernel_test_results(e, t[0], ii);

    std::tie(e,ii) = kernel_test_s<double>(t[1]);
    print_kernel_test_results(e, t[1], ii);

    std::tie(e,ii) = kernel_test_s<float>(t[2]);
    print_kernel_test_results(e, t[2], ii);

    //std::tie(e,ii) = kernel_test_s<_Float16>(t[3]);
    std::tie(e,ii) = kernel_test<_Float16, float, _Float16, _Float16>(t[3]);
    print_kernel_test_results(e, t[3], ii);

    std::tie(e,ii) = kernel_test<_Float16, float, float, float>(t[4]);
    print_kernel_test_results(e, t[4], ii);

    std::tie(e,ii) = kernel_test_s<float, AVX512_BF16>(t[5]);
    print_kernel_test_results(e, t[5], ii);

    std::tie(e,ii) = kernel_test_s<float, AMX_BF16>(t[6]);
    print_kernel_test_results(e, t[6], ii);

    //std::tie(e,ii) = kernel_test_s<_Float16, AMX_BF16>(t[7]);
    print_kernel_test_results(e, t[7], ii);

}
