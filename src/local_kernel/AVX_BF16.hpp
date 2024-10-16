template<int m, int n, int k, int n_vecs_0=1, int n_vecs_1=1>
inline void perform_sum2gemm3_AVXbf16(float * restrict A, const __bf16 * restrict AVX_F0, const __bf16 * restrict AVX_F1){
    //FIXME: Currently assumes AVX_F0 is made of n_vecs_0 arrays and AVX_F1 of n_vecs_0*n_vecs_1 arrays.
    //       For poisson and mass bilinear forms this is fine. Probably not so fine in general.

    constexpr int A_ROWS = m;
    constexpr int INNER_SIZE = n;
    constexpr int A_COLS = k;

    constexpr int INNER_BSIZE = INNER_SIZE/32; // 16 bits for A and 2 16-bit blocks for the bf16 arrays

    static_assert(INNER_SIZE%32 == 0);

    __m512bh a;
    __m512bh b;
    __m512   c;

    // NOTE: Most of the cost is spent here, even if you don't use reduce_add_ps
    std::array<float, 16> temp{};
    for(int i=0; i<A_ROWS; i++){
        for(int j=0; j<A_COLS; j++){
            c = _mm512_loadu_ps(temp.data());
            for(int id = 0; id < n_vecs_0; id++){
                for(int bk = 0; bk < INNER_BSIZE; bk++){
                    a = reinterpret_cast<__m512bh>(_mm512_loadu_ph(AVX_F0 + id*A_ROWS*INNER_SIZE + i*INNER_SIZE + 32*bk));
                    for(int jd = 0; jd < n_vecs_1; jd++){

                        b = reinterpret_cast<__m512bh>(_mm512_loadu_ph(AVX_F1 + (id*n_vecs_1 + jd)*A_COLS*INNER_SIZE + j*INNER_SIZE + 32*bk));
                        c = _mm512_dpbf16_ps(c, a, b);
                    }
                }
            }
            A[i*A_COLS + j] = _mm512_reduce_add_ps(c);
        }
    }

}

template <typename TG, typename TP>
inline void sum2gemm3_avxbf16_poisson_action(double * restrict times, float * restrict A, const __bf16 * restrict bf16_fe_grad, const TP * restrict action_fe_grad, const TP * restrict action_coeffs, TG * restrict G){

    clock_t tic, toc;

    tic = clock();

    constexpr int unpadded_cols = GDIM*NQUAD;
    constexpr int cols = unpadded_cols%32 == 0 ? unpadded_cols : 32*(unpadded_cols/32 + 1);
    constexpr int SIZE = GDIM*N_BATCH_CELLS*NQUAD;
    constexpr int PADDEDSIZE = N_BATCH_CELLS*cols;

    //NOTE Simply using float here since using AVX-bf16 would give the same timings (it's slow).
    std::array<TP, SIZE> temp_coeff_val{};
    for(int ic = 0; ic < ELDIM; ic++){
        for(int i = 0; i < GDIM; i++){
            for(int ii = 0; ii < N_BATCH_CELLS; ii++){
                for (int iq = 0; iq < NQUAD; iq++){
                    temp_coeff_val[i*N_BATCH_CELLS*NQUAD + ii*NQUAD + iq] += action_coeffs[ii*ELDIM + ic]*action_fe_grad[ic*NQUAD*GDIM + i*NQUAD + iq];
                }
            }
        }
    }

    // NOTE: Double casting does not make too much sense, but it's for the paper (makes theory simpler).
    [[maybe_unused]] std::array<TG, SIZE> TGbuf{};
    TG * coeff_val = array_cast<SIZE>(temp_coeff_val.data(), TGbuf.data());

    std::array<TG, PADDEDSIZE> G_coeff_val{};
    for(int i = 0; i < GDIM; i++){
        for(int j = 0; j < GDIM; j++){
            for(int ii = 0; ii < N_BATCH_CELLS; ii++){
                for (int iq = 0; iq < NQUAD; iq++){
                    G_coeff_val[ii*cols + i*NQUAD + iq] += G[ii*GDIM*GDIM*NQUAD + (i*GDIM + j)*NQUAD + iq]*coeff_val[j*N_BATCH_CELLS*NQUAD + ii*NQUAD + iq];
                }
            }
        }
    }

    [[maybe_unused]] std::array<__bf16, PADDEDSIZE> bf16buf{};
    __bf16 * Gcval = array_cast<PADDEDSIZE>(G_coeff_val.data(), bf16buf.data());

    toc = clock();
    times[4] += get_time(tic, toc);
    tic = clock();
   
    constexpr int A_ROWS = N_BATCH_CELLS;
    constexpr int A_COLS = ELDIM;
    constexpr int INNER_SIZE = cols;
    constexpr int n_vecs_0 = 1;
    constexpr int n_vecs_1 = 1;

    // Gcval must be of sizes [N_BATCH_CELLS][cols]
    // bf16_fe_grad must be of sizes [ELDIM][cols]
    perform_sum2gemm3_AVXbf16<A_ROWS, INNER_SIZE, A_COLS, n_vecs_0, n_vecs_1>(A, Gcval, bf16_fe_grad);

    toc = clock();
    times[5] += get_time(tic, toc);

}

template <typename TS>
inline void sum2gemm3_avxbf16_poisson(double * restrict times, float * restrict A, const TS * restrict fe_grad, const __bf16 * restrict bf16_fe_grad, const TS * restrict G){
    
    constexpr int A_ROWS = ELDIM;
    constexpr int A_COLS = ELDIM;
    constexpr int INNER_SIZE = PADDEDCOLS;
    constexpr int n_vecs_0 = GDIM;
    constexpr int n_vecs_1 = GDIM;

    constexpr int SIZE = GDIM*GDIM*ELDIM*PADDEDCOLS;

    clock_t tic, toc;

    tic = clock();

    std::array<TS, SIZE> G_fe_grad{};
    for(int j = 0; j < GDIM; j++){
        for(int k = 0; k < GDIM; k++){
            for(int ic = 0; ic < ELDIM; ic++){
                for(int iq = 0; iq < NQUAD; iq++){
                    G_fe_grad[(j*GDIM + k)*ELDIM*PADDEDCOLS + ic*PADDEDCOLS + iq] = G[j*GDIM*NQUAD + k*NQUAD + iq]*fe_grad[k*ELDIM*NQUAD + ic*NQUAD + iq];
                }
            }
        }
    }

    [[maybe_unused]] std::array<__bf16, SIZE> bf16_buffer{};
    __bf16 * Gcval = array_cast<SIZE>(G_fe_grad.data(), bf16_buffer.data());

    toc = clock();
    times[4] += get_time(tic, toc);
    tic = clock();

    // bf16_fe_grad must be of sizes [GDIM][ELDIM][PADDEDCOLS]
    // G_fe_grad must be of sizes [GDIM][GDIM][ELDIM][PADDEDCOLS]
    perform_sum2gemm3_AVXbf16<A_ROWS, INNER_SIZE, A_COLS, n_vecs_0, n_vecs_1>(A, bf16_fe_grad, Gcval);

    toc = clock();
    times[5] += get_time(tic, toc);

}

template <typename TS>
inline void sum2gemm3_avxbf16_mass(double * restrict times, float * restrict A, const TS * restrict fe0, const __bf16 * restrict bf16_fe0, const TS * restrict G){
   
    constexpr int A_ROWS = ELDIM;
    constexpr int A_COLS = ELDIM;
    constexpr int INNER_SIZE = PADDEDCOLS;
    constexpr int n_vecs_0 = 1;
    constexpr int n_vecs_1 = 1;

    clock_t tic, toc;

    tic = clock();

    std::array<TS, ELDIM*PADDEDCOLS> Gfe0{};
    std::memcpy(Gfe0.data(), fe0, ELDIM*PADDEDCOLS*sizeof(TS)); //NOTE: can only do this since for AVX_BF16 fe0 is padded
    for(int iq = 0; iq < NQUAD; iq++){
        for(int ic = 0; ic < ELDIM; ic++){
            Gfe0[ic*PADDEDCOLS + iq] *= G[iq];
        }
    }

    [[maybe_unused]] std::array<__bf16, ELDIM*PADDEDCOLS> bf16_buffer{};
    __bf16 * Gcval = array_cast<ELDIM*PADDEDCOLS>(Gfe0.data(), bf16_buffer.data());

    toc = clock();
    times[4] += get_time(tic, toc);
    tic = clock();
    
    // bf16_fe0 must be of sizes [ELDIM][PADDEDCOLS]
    // Gcval must be of sizes [ELDIM][PADDEDCOLS]
    perform_sum2gemm3_AVXbf16<A_ROWS, INNER_SIZE, A_COLS, n_vecs_0, n_vecs_1>(A, bf16_fe0, Gcval);

    toc = clock();
    times[5] += get_time(tic, toc);

}

template <typename TG, typename TP>
inline void action_sum_avxbf16_mass(double * restrict times, float * restrict A, const __bf16 * restrict bf16_fe0, const TP * restrict action_fe0, const TP * restrict action_coeffs, const TG * restrict G){

    clock_t tic, toc;

    tic = clock();

    constexpr int SIZE = N_BATCH_CELLS*NQUAD;
    constexpr int PADDEDSIZE = N_BATCH_CELLS*PADDEDCOLS;

    std::array<TP, PADDEDSIZE> temp_coeff_val{};
    for(int ic = 0; ic<ELDIM; ic++){
        for(int ii = 0; ii < N_BATCH_CELLS; ii++){
            for (int iq = 0; iq < NQUAD; iq++){
                temp_coeff_val[ii*PADDEDCOLS + iq] += action_coeffs[ii*ELDIM + ic]*action_fe0[ic*NQUAD + iq];
            }
        }
    }

    // NOTE: Extra casting does not make too much sense, but it's for the paper (makes theory simpler).
    [[maybe_unused]] std::array<TG, PADDEDSIZE> TGbuf{};
    TG * coeff_val = array_cast<PADDEDSIZE>(temp_coeff_val.data(), TGbuf.data());

    for(int ii = 0; ii < N_BATCH_CELLS; ii++){
        for(int iq = 0; iq < NQUAD; iq++){
             coeff_val[ii*PADDEDCOLS + iq] *= G[ii*NQUAD + iq];
        }
    }

    [[maybe_unused]] std::array<__bf16, PADDEDSIZE> bf16_buf{};
    __bf16 * cval = array_cast<PADDEDSIZE>(coeff_val, bf16_buf.data());
    
    toc = clock();
    times[4] += get_time(tic, toc);
    tic = clock();

    constexpr int A_ROWS = N_BATCH_CELLS;
    constexpr int A_COLS = ELDIM;
    constexpr int INNER_SIZE = PADDEDCOLS;
    constexpr int n_vecs_0 = 1;
    constexpr int n_vecs_1 = 1;

    // cval must be of sizes [N_BATCH_CELLS][PADDEDCOLS]
    // bf16_fe0 must be of sizes [ELDIM][PADDEDCOLS]
    perform_sum2gemm3_AVXbf16<A_ROWS, INNER_SIZE, A_COLS, n_vecs_0, n_vecs_1>(A, cval, bf16_fe0);

    toc = clock();
    times[5] += get_time(tic, toc);

}

template <typename TG, typename TP, typename TS>
noinline void local_kernel_avxbf16_mass(double * restrict times, float * restrict A, const TG * restrict coordinate_dofs, const TG * restrict weights, const TG * restrict fe_domain_grad, const TP * restrict action_coeffs, [[maybe_unused]] const TS * restrict fe0, const __bf16 * restrict bf16_fe0, const TP * restrict fe0_geom, [[maybe_unused]] const TP * restrict action_fe0, const TP * restrict coeffs){
    // degree 3 Lagrange element on hexahedrons

    clock_t firsttic, lasttoc;

    firsttic = clock();

    #if defined(KERNEL_TYPE_ACTION)
        std::fill(A, A + N_BATCH_CELLS*ASIZE, (float) 0.);

        std::array<TG, N_BATCH_CELLS*NQUAD> G{}; // Geometry[NQUAD]
        for(int i = 0; i < N_BATCH_CELLS; i++)
            construct_geometry_mass(times, G.data() + i*NQUAD, coordinate_dofs + i*DOMAIN_ELDIM*GDIM, weights, fe_domain_grad, fe0_geom, coeffs + i*ELDIM);

        action_sum_avxbf16_mass(times, A, bf16_fe0, action_fe0, action_coeffs, G.data());
    #else
        std::fill(A, A + ASIZE, (float) 0.);

        std::array<TG, NQUAD> Gtemp{}; // Geometry[NQUAD]
        construct_geometry_mass(times, Gtemp.data(), coordinate_dofs, weights, fe_domain_grad, fe0_geom, coeffs);
        std::array<TS, NQUAD> G{};
        for(int i=0; i < NQUAD; i++)
            G[i] = (TS) Gtemp[i];

        sum2gemm3_avxbf16_mass(times, A, fe0, bf16_fe0, G.data());
    #endif

    lasttoc = clock();
    times[6] += get_time(firsttic, lasttoc);

}

template <typename TG, typename TP, typename TS>
noinline void local_kernel_avxbf16_poisson(double * restrict times, float * restrict A, const TG * restrict coordinate_dofs, const TG * restrict weights, const TG * restrict fe_domain_grad, [[maybe_unused]] const TS * restrict fe_grad, const __bf16 * restrict bf16_fe_grad, [[maybe_unused]] const TP * restrict action_fe_grad, const TP * restrict action_coeffs, const TP * restrict fe0_geom, const TP * restrict coeffs){
    // degree 3 Lagrange element on hexahedrons


    clock_t firsttic, lasttoc;

    firsttic = clock();

    #if defined(KERNEL_TYPE_ACTION)
        std::fill(A, A + N_BATCH_CELLS*ASIZE, (float) 0.);

        std::array<TG, N_BATCH_CELLS*NQUAD*GDIM*GDIM> G{}; // Geometry[NQUAD][GDIM][GDIM]
        for(int i = 0; i < N_BATCH_CELLS; i++)
            construct_geometry_poisson(times, G.data() + i*GDIM*GDIM*NQUAD, coordinate_dofs + i*DOMAIN_ELDIM*GDIM, weights, fe_domain_grad, fe0_geom, coeffs + i*ELDIM);

        sum2gemm3_avxbf16_poisson_action(times, A, bf16_fe_grad, action_fe_grad, action_coeffs, G.data());
    #else
        std::fill(A, A + ASIZE, (float) 0.);

        std::array<TG, NQUAD*GDIM*GDIM> Gtemp{}; // Geometry[NQUAD][GDIM][GDIM]
        construct_geometry_poisson(times, Gtemp.data(), coordinate_dofs, weights, fe_domain_grad, fe0_geom, coeffs);

        std::array<TS, NQUAD*GDIM*GDIM> G{};
        for(int i=0; i < NQUAD*GDIM*GDIM; i++)
            G[i] = (TS) Gtemp[i];

        sum2gemm3_avxbf16_poisson(times, A, fe_grad, bf16_fe_grad, G.data());
    #endif

    lasttoc = clock();
    times[6] += get_time(firsttic, lasttoc);

}

