template<typename T, int n_vecs, int rows, int cols, bool transpose_form>
inline void to_AMX_form(const T * restrict vec, T * restrict AMX_vec){

    static_assert(rows%16 == 0);
    static_assert(cols%32 == 0);

    constexpr int block_rows = rows/16;
    constexpr int block_cols = cols/16;

    for (int id=0; id < n_vecs; id++){
        for (int bi=0; bi < block_rows; bi++){
            for (int bj=0; bj < block_cols/2; bj++){
                for (int ii=0; ii<16; ii++){
                    for (int jj=0; jj<16; jj++){
                        // For interleaving, take 2 16x16 blocks at a time: we have BCOLS/2 16x32 blocks
                        // (2*16*16) = n entries per 16x32 block. bi*(BCOLS/2) + bj = current 16x32 block in C indexing.
                        // 32*ii is used since each row has 32 cols in the 16x32 block. 2*jj and 2*jj+1 with jj<16 takes
                        // subsequent entries in the block. 16*(2*bj) = current 16x16 block in feg, 16*(2*bj+1) = subsequent 16x16 block in feg.
                        // Therefore 16*(2*bj) + jj, and 16*(2*bj+1) + jj are corresponding entries in subsequent blocks on the same row.
                        if constexpr(!transpose_form){
                            AMX_vec[id*rows*cols + (2*16*16)*(bi*(block_cols/2) + bj) + 32*ii + 2*jj]    = vec[id*rows*cols + cols*(16*bi + ii) + 16*(2*bj)       + jj];
                            AMX_vec[id*rows*cols + (2*16*16)*(bi*(block_cols/2) + bj) + 32*ii + 2*jj+1]  = vec[id*rows*cols + cols*(16*bi + ii) + 16*(2*bj+1)     + jj];
                        }else{
                            AMX_vec[id*rows*cols + (2*16*16)*(bi*(block_cols/2) + bj) + 32*ii + 2*jj]    = vec[id*rows*cols + cols*(16*bi + jj) + 16*(2*bj)       + ii];
                            AMX_vec[id*rows*cols + (2*16*16)*(bi*(block_cols/2) + bj) + 32*ii + 2*jj+1]  = vec[id*rows*cols + cols*(16*bi + jj) + 16*(2*bj+1)     + ii];
                        }
                    }
                }
            }
        }
    }

}

template<typename INTYPE, typename OTYPE, int n_vecs, int unpadded_rows, int unpadded_cols, bool pad_and_reshape=false, bool transpose_form=false>
inline void restructure_array_AMX(const INTYPE * restrict old_vec, OTYPE* restrict AMX_vec){
    // NOTE: if pad_and_reshape == true, then the input must have size [n_vecs][unpadded_cols][unpadded_rows], and it will be padded and reshaped
    //       into an array of sizes [n_vecs][rows][cols]

    static_assert(pad_and_reshape || (unpadded_rows%16 == 0 && unpadded_cols%32 == 0));
    static_assert(n_vecs > 0);

    constexpr int rows = unpadded_rows%16 == 0 ? unpadded_rows : 16*(unpadded_rows/16 + 1);
    constexpr int cols = unpadded_cols%32 == 0 ? unpadded_cols : 32*(unpadded_cols/32 + 1);
    constexpr int SIZE = n_vecs*rows*cols;

    constexpr bool nocast = std::is_same_v<OTYPE, INTYPE>;
    constexpr bool fp16tobf16flag = (std::is_same_v<OTYPE, __bf16> && std::is_same_v<INTYPE, _Float16>);

    std::array<OTYPE, SIZE> vec_arr{};
    OTYPE * restrict vec = vec_arr.data();

    if constexpr(!pad_and_reshape){
        if constexpr(nocast){
            vec = old_vec;
        }else if constexpr(fp16tobf16flag){
            //// Copy _Float16 memory to vec without casting, then cast inplace.
            std::memcpy(vec, old_vec, SIZE*sizeof(__bf16));
            fp16tobf16<SIZE>(reinterpret_cast<uint16_t*>(vec));
        }else{
            for(int i=0; i<SIZE; i++)
                vec[i] = (OTYPE) old_vec[i];
        }

    }else{
        pad_and_reshape_bf16<INTYPE, OTYPE, n_vecs, unpadded_rows, unpadded_cols>(old_vec, vec);
    }

    to_AMX_form<OTYPE, n_vecs, rows, cols, transpose_form>(vec, AMX_vec);
}

template<typename TS>
inline void restructure_geometry_AMX(const TS * restrict G, TS * restrict AMX_G){

    //NOTE: G here is padded: it is of size [GDIM][GDIM][PADDEDCOLS]
    //      AMX_G is instead of size [GDIM*GDIM][PADDEDROWS][PADDEDCOLS]
    
    auto G_ = [G](auto k, auto i, auto j) -> TS { return G[k*GDIM*PADDEDCOLS + i*PADDEDCOLS + j]; };
    auto AMX_G_ = [AMX_G](auto i, auto j, auto k) -> TS& { return AMX_G[(i*GDIM + j)*PADDEDCOLS*PADDEDROWS + k]; };

    // Transpose-only version
    for (int j=0; j<GDIM; j++){
        for (int k=0; k<GDIM; k++){
            for (int bj=0; bj<BCOLS/2; bj++){
                for (int ii=0; ii<16; ii++){
                    for (int bi=0; bi<BROWS; bi++){
                        for (int jj=0; jj<16; jj++){
                            AMX_G_(j, k, (2*16*16)*(bi*(BCOLS/2) + bj) + 32*ii + 2*jj)    = G_(j, k, 16*(2*bj)   + ii);
                            AMX_G_(j, k, (2*16*16)*(bi*(BCOLS/2) + bj) + 32*ii + 2*jj+1)  = G_(j, k, 16*(2*bj+1) + ii);
                        }
                    }
                }
            }
        }
    }

}


template<int m, int n, int k, int n_vecs_0=1, int n_vecs_1=1>
inline void perform_sum2gemm3_AMX(float * restrict A_tile, const __bf16 * restrict AMX_F0, const __bf16 * restrict AMX_F1){
    //FIXME: Currently assumes AMX_F0 is made of n_vecs_0 arrays and AMX_F1 of n_vecs_0*n_vecs_1 arrays.
    //       For poisson and mass bilinear forms this is fine. Probably not so fine in general.

    constexpr int A_ROWS = m;
    constexpr int INNER_SIZE = n;
    constexpr int A_COLS = k;

    constexpr int A_BROWS = A_ROWS/16;
    constexpr int A_BCOLS = A_COLS/16;
    constexpr int INNER_BSIZE = INNER_SIZE/16;

    static_assert(A_ROWS%16 == 0);
    static_assert(A_COLS%16 == 0);
    static_assert(INNER_SIZE%32 == 0);

    for(int bi = 0; bi < A_BROWS; bi++){
        for(int bj = 0; bj < A_BCOLS; bj++){
            _tile_loadd (1, A_tile + (16*16)*(bi*A_BCOLS + bj), STRIDE);
            for(int id = 0; id < n_vecs_0; id++){
                for(int bk = 0; bk < INNER_BSIZE/2; bk++){

                    _tile_loadd (2, AMX_F0 + id*A_ROWS*INNER_SIZE + (2*16*16)*(bi*(INNER_BSIZE/2) + bk), STRIDE);

                    for(int jd = 0; jd < n_vecs_1; jd++){
                        // RECALL: AMX does two blocks at a time

                        //NOTE: the commented line below was probably wrong, but went fine since it only causes issues when the sizes are not consistent and when n_vecs>1 so it was never detected.
                        //_tile_loadd (3, AMX_F1 + (id*n_vecs_1 + jd)*A_ROWS*INNER_SIZE + (2*16*16)*(bj*(INNER_BSIZE/2) + bk), STRIDE);
                        _tile_loadd (3, AMX_F1 + (id*n_vecs_1 + jd)*A_COLS*INNER_SIZE + (2*16*16)*(bj*(INNER_BSIZE/2) + bk), STRIDE);

                        _tile_dpbf16ps (1, 2, 3);

                    }
                }
            }
            _tile_stored (1, A_tile + (16*16)*(bi*A_BCOLS + bj), STRIDE);
        }
    }

}

template<int A_ROWS, int A_COLS, int A_REAL_ROWS, int A_REAL_COLS>
inline void unblock_AMX(float * restrict A, const float * restrict A_tile){

    constexpr int A_BROWS = A_ROWS/16;
    constexpr int A_BCOLS = A_COLS/16;

    static_assert(A_ROWS%16 == 0);
    static_assert(A_COLS%16 == 0);

    auto A_ = [A](auto i, auto j) -> float& { return A[i*A_REAL_COLS + j]; };

    //Need to "unblock" A_tile.
    if constexpr(A_COLS == A_REAL_COLS && A_ROWS == A_REAL_ROWS){

        for (int bi=0; bi < A_BROWS; bi++){
            for (int bj=0; bj < A_BCOLS; bj++){
                for (int ii=0; ii<16; ii++){
                    for (int jj=0; jj<16; jj++){
                        A_(16*bi + ii, 16*bj + jj) = A_tile[16*16*(bi*A_BCOLS + bj) + 16*ii + jj];
                    }
                }
            }
        }
    } else {

        // Use a copy to avoid an if statement in the for loop
        std::array<float, A_ROWS*A_COLS> A_tile_reordered{};

        for (int bi=0; bi < A_BROWS; bi++){
            for (int bj=0; bj < A_BCOLS; bj++){
                for (int ii=0; ii<16; ii++){
                    for (int jj=0; jj<16; jj++){
                        A_tile_reordered[(16*bi + ii)*A_COLS + 16*bj + jj] = A_tile[16*16*(bi*A_BCOLS + bj) + 16*ii + jj];
                    }
                }
            }
        }

        for(int i=0; i < A_REAL_ROWS; i++){
            for(int j=0; j < A_REAL_COLS; j++){
                A_(i,j) = A_tile_reordered[i*A_COLS + j];
            }
        }

    }
}



template<typename TS>
inline void prepare_sum2gemm3_AMX_mass(const TS * restrict G, const TS * restrict AMX_fe0_transposed, TS * restrict AMX_G){

    // Transpose-only version
    for (int bj=0; bj<BCOLS/2; bj++){
        for (int bi=0; bi<BROWS; bi++){
            for (int ii=0; ii<16; ii++){
                for (int jj=0; jj<16; jj++){
                    AMX_G[(2*16*16)*(bi*(BCOLS/2) + bj) + 32*ii + 2*jj]    = G[16*(2*bj)   + ii]*AMX_fe0_transposed[(2*16*16)*(bi*(BCOLS/2) + bj) + 32*ii + 2*jj];
                    AMX_G[(2*16*16)*(bi*(BCOLS/2) + bj) + 32*ii + 2*jj+1]  = G[16*(2*bj+1) + ii]*AMX_fe0_transposed[(2*16*16)*(bi*(BCOLS/2) + bj) + 32*ii + 2*jj+1];
                }
            }
        }
    }

    //FIXME NOTE: If geometry done and restructured offline then the code above could perhaps be made more efficient. See similar note in prepare_sum2gemm3_AMX_poisson 

}

template<typename TS>
inline void prepare_sum2gemm3_AMX_poisson(const TS * restrict G, const TS * restrict AMX_fe_grad_transposed, TS * restrict AMX_G){

    auto G_ = [G](auto k, auto i, auto j) -> TS { return G[k*GDIM*PADDEDCOLS + i*PADDEDCOLS + j]; };
    auto AMX_G_ = [AMX_G](auto i, auto j, auto k) -> TS& { return AMX_G[(i*GDIM + j)*PADDEDCOLS*PADDEDROWS + k]; };
    auto afegt = [AMX_fe_grad_transposed](auto i, auto j) -> TS { return AMX_fe_grad_transposed[i*PADDEDCOLS*PADDEDROWS + j]; };

    // Transpose-only version
    for (int j=0; j<GDIM; j++){
        for (int k=0; k<GDIM; k++){
            for (int bj=0; bj<BCOLS/2; bj++){
                for (int bi=0; bi<BROWS; bi++){
                    for (int ii=0; ii<16; ii++){
                        for (int jj=0; jj<16; jj++){
                            AMX_G_(j, k, (2*16*16)*(bi*(BCOLS/2) + bj) + 32*ii + 2*jj)    = G_(j, k, 16*(2*bj)   + ii)*afegt(k, (2*16*16)*(bi*(BCOLS/2) + bj) + 32*ii + 2*jj);
                            AMX_G_(j, k, (2*16*16)*(bi*(BCOLS/2) + bj) + 32*ii + 2*jj+1)  = G_(j, k, 16*(2*bj+1) + ii)*afegt(k, (2*16*16)*(bi*(BCOLS/2) + bj) + 32*ii + 2*jj+1);
                        }
                    }
                }
            }
        }
    }

    //// FIXME: The code below is slower than the code above. However, if we do the geomety offline it may be more advantageous. DO NOT DELETE!
    //// NOTE: overwriting AMX_G to avoid copy.
    
    //restructure_geometry_AMX(G, AMX_G);
    //
    //constexpr int MM = GDIM*PADDEDCOLS*PADDEDROWS;
    //for (int s=0; s<MM; s++){
    //    for (int j=0; j<GDIM; j++){
    //        AMX_G[j*MM + s] *= AMX_fe_grad_transposed[s];
    //    }
    //}

}

template <typename TS>
inline void sum2gemm3_AMX(double * restrict times, float * restrict A, const TS * restrict G, const __bf16 * restrict AMX_fe0, const TS * restrict AMX_fe0_transposed){

    #if defined(KERNEL_TYPE_POISSON)
        constexpr int SIZE = GDIM*GDIM*PADDEDCOLS*PADDEDROWS;
        constexpr int n_vecs_0 = GDIM;
        constexpr int n_vecs_1 = GDIM;
    #else // MASS
        constexpr int SIZE = PADDEDCOLS*PADDEDROWS;
        constexpr int n_vecs_0 = 1;
        constexpr int n_vecs_1 = 1;
    #endif

    constexpr int A_ROWS = PADDEDROWS;
    constexpr int A_COLS = PADDEDROWS;
    constexpr int A_REAL_ROWS = ELDIM;
    constexpr int A_REAL_COLS = ELDIM;
    constexpr int INNER_SIZE = PADDEDCOLS;

    clock_t tic, toc;

    tic = clock();

    std::array<TS, SIZE> AMX_G{};
    #if defined(KERNEL_TYPE_POISSON)
        prepare_sum2gemm3_AMX_poisson(G, AMX_fe0_transposed, AMX_G.data());
    #else // MASS
        prepare_sum2gemm3_AMX_mass(G, AMX_fe0_transposed, AMX_G.data());
    #endif

    std::array<__bf16, SIZE> bf16_buffer{};
    __bf16 * agf0 = array_cast<SIZE>(AMX_G.data(), bf16_buffer.data());

    toc = clock();
    times[4] += get_time(tic, toc);
    tic = clock();

    std::array<float, A_ROWS*A_COLS> A_tile{};
    perform_sum2gemm3_AMX<A_ROWS, INNER_SIZE, A_COLS, n_vecs_0,n_vecs_1>(A_tile.data(), AMX_fe0, agf0);
    unblock_AMX<A_ROWS, A_COLS, A_REAL_ROWS, A_REAL_COLS>(A, A_tile.data());

    toc = clock();
    times[5] += get_time(tic, toc);

}

template<typename TG>
inline void prepare_sum2gemm3_AMX_mass_action(double * times, const TG * restrict G, const __bf16 * restrict AMX_action_fe0, const __bf16 * restrict action_coeffs, __bf16 * restrict AMX_G){

    clock_t tic, toc;

    ////////////////// EVALUATE FE FUNCTION BEGIN /////////////////////

    constexpr int unpadded_rows = ELDIM;
    constexpr int rows = unpadded_rows%32 == 0 ? unpadded_rows : 32*(unpadded_rows/32 + 1);

    // CAST action_coeffs to AMX form
    std::array<__bf16, N_BATCH_CELLS*rows> AMX_action_coeffs{};
    to_AMX_form<__bf16, 1, N_BATCH_CELLS, rows, false>(action_coeffs, AMX_action_coeffs.data());

    // Sizes for AMX product
    constexpr int A_ROWS = N_BATCH_CELLS;
    constexpr int A_COLS = PADDEDCOLS;
    constexpr int INNER_SIZE = rows;
    constexpr int A_REAL_ROWS = N_BATCH_CELLS;
    constexpr int A_REAL_COLS = PADDEDCOLS;

    // Perform AMX product
    std::array<float, N_BATCH_CELLS*PADDEDCOLS> new_coeff_val{};
    std::array<float, A_ROWS*A_COLS> A_tile{};
    perform_sum2gemm3_AMX<A_ROWS, INNER_SIZE, A_COLS, 1, 1>(A_tile.data(), AMX_action_coeffs.data(), AMX_action_fe0);
    unblock_AMX<A_ROWS, A_COLS, A_REAL_ROWS, A_REAL_COLS>(new_coeff_val.data(), A_tile.data());

    tic = clock();

    // If TG \neq float, then cast float output to TG 
    [[maybe_unused]] std::array<TG, N_BATCH_CELLS*PADDEDCOLS> coeff_buffer{};
    TG * cval = array_cast<N_BATCH_CELLS*PADDEDCOLS>(new_coeff_val.data(), coeff_buffer.data());
    
    ////////////////// EVALUATE FE FUNCTION END /////////////////////

    // Multiply function and geometry
    for(int ii = 0; ii < N_BATCH_CELLS; ii++){
        for(int iq = 0; iq < NQUAD; iq++){
             cval[ii*PADDEDCOLS + iq] *= G[ii*NQUAD + iq];
        }
    }

    [[maybe_unused]] std::array<__bf16, N_BATCH_CELLS*PADDEDCOLS> bf16_buffer{};
    __bf16 * bf16_cval = array_cast<N_BATCH_CELLS*PADDEDCOLS>(cval, bf16_buffer.data());

    toc = clock();
    times[4] += get_time(tic, toc);

    // reshape into a [N_BATCH_CELLS][PADDEDCOLS] AMX (non-transposed) array.
    to_AMX_form<__bf16, 1, N_BATCH_CELLS, PADDEDCOLS, false>(bf16_cval, AMX_G);

}

template<typename TG>
inline void prepare_sum2gemm3_AMX_poisson_action(double * times, const TG * restrict G, const __bf16 * restrict AMX_action_fe_grad, const __bf16 * restrict action_coeffs, __bf16 * restrict AMX_G){

    clock_t tic, toc;

    ////////////////// EVALUATE FE FUNCTION BEGIN /////////////////////
    
    constexpr int unpadded_cols = GDIM*NQUAD;
    constexpr int cols = unpadded_cols%32 == 0 ? unpadded_cols : 32*(unpadded_cols/32 + 1);
    constexpr int unpadded_rows = ELDIM;
    constexpr int rows = unpadded_rows%32 == 0 ? unpadded_rows : 32*(unpadded_rows/32 + 1);

    // CAST action_coeffs to AMX form
    std::array<__bf16, N_BATCH_CELLS*rows> AMX_action_coeffs{};
    to_AMX_form<__bf16, 1, N_BATCH_CELLS, rows, false>(action_coeffs, AMX_action_coeffs.data());

    // Sizes for AMX product
    constexpr int A_ROWS = N_BATCH_CELLS;
    constexpr int A_COLS = PADDEDCOLS;
    constexpr int INNER_SIZE = rows;
    constexpr int A_REAL_ROWS = N_BATCH_CELLS;
    constexpr int A_REAL_COLS = NQUAD;

    // Perform AMX product: proceed dimension by dimensions
    std::array<float, GDIM*N_BATCH_CELLS*NQUAD> new_coeff_val{};
    for(int i=0; i<GDIM; i++){
        std::array<float, A_ROWS*A_COLS> A_tile{};
        perform_sum2gemm3_AMX<A_ROWS, INNER_SIZE, A_COLS, 1, 1>(A_tile.data(), AMX_action_coeffs.data(), &AMX_action_fe_grad[i*PADDEDCOLS*rows]);
        unblock_AMX<A_ROWS, A_COLS, A_REAL_ROWS, A_REAL_COLS>(&new_coeff_val[i*N_BATCH_CELLS*NQUAD], A_tile.data());
    }

    tic = clock();

    // If TG \neq float, then cast float output to TG 
    [[maybe_unused]] std::array<TG, N_BATCH_CELLS*GDIM*NQUAD> coeff_buffer{};
    TG * cval = array_cast<N_BATCH_CELLS*GDIM*NQUAD>(new_coeff_val.data(), coeff_buffer.data());
    
    ////////////////// EVALUATE FE FUNCTION END /////////////////////

    // Multiply function and geometry
    std::array<TG, N_BATCH_CELLS*cols> G_coeff_val{};
    for(int i = 0; i < GDIM; i++){
        for(int j = 0; j < GDIM; j++){
            for(int ii = 0; ii < N_BATCH_CELLS; ii++){
                for (int iq = 0; iq < NQUAD; iq++){
                    G_coeff_val[ii*cols + i*NQUAD + iq] += G[ii*GDIM*GDIM*NQUAD + (i*GDIM + j)*NQUAD + iq]*cval[j*N_BATCH_CELLS*NQUAD + ii*NQUAD + iq];
                }
            }
        }
    }

    [[maybe_unused]] std::array<__bf16, N_BATCH_CELLS*cols> bf16_buffer{};
    __bf16 * Gcval = array_cast<N_BATCH_CELLS*cols>(G_coeff_val.data(), bf16_buffer.data());

    toc = clock();
    times[4] += get_time(tic, toc);
    
    // reshape into a [N_BATCH_CELLS][cols] AMX (non-transposed) array.
    to_AMX_form<__bf16, 1, N_BATCH_CELLS, cols, false>(Gcval, AMX_G);

}

template <typename TG>
inline void sum2gemm3_AMX_action(double * restrict times, float * restrict A, const TG * restrict G, const __bf16 * restrict action_fe0, const __bf16 * restrict AMX_fe0_transposed, const __bf16 * restrict action_coeffs){
    
    // NOTE: action_fe0, and AMX_fe0_transposed must be fe_grad variables for poisson

    #if defined(KERNEL_TYPE_POISSON)
        constexpr int unpadded_cols = GDIM*NQUAD;
        constexpr int cols = unpadded_cols%32 == 0 ? unpadded_cols : 32*(unpadded_cols/32 + 1);
        constexpr int SIZE = N_BATCH_CELLS*cols;
        constexpr int INNER_SIZE = cols;
    #else // MASS
        constexpr int SIZE = N_BATCH_CELLS*PADDEDCOLS;
        constexpr int INNER_SIZE = PADDEDCOLS;
    #endif

    constexpr int n_vecs_0 = 1;
    constexpr int n_vecs_1 = 1;
    constexpr int A_ROWS = N_BATCH_CELLS;
    constexpr int A_COLS = PADDEDROWS;
    constexpr int A_REAL_ROWS = N_BATCH_CELLS;
    constexpr int A_REAL_COLS = ELDIM;

    clock_t tic, toc;

    //tic = clock();
    //toc = clock();
    //times[4] += get_time(tic, toc);
    
    tic = clock();

    std::array<__bf16, SIZE> AMX_G{};
    #if defined(KERNEL_TYPE_POISSON)
        prepare_sum2gemm3_AMX_poisson_action(times, G, action_fe0, action_coeffs, AMX_G.data());
    #else // MASS
        prepare_sum2gemm3_AMX_mass_action(times, G, action_fe0, action_coeffs, AMX_G.data());
    #endif

    std::array<float, A_ROWS*A_COLS> A_tile{};
    perform_sum2gemm3_AMX<A_ROWS, INNER_SIZE, A_COLS, n_vecs_0, n_vecs_1>(A_tile.data(), AMX_G.data(), AMX_fe0_transposed);
    unblock_AMX<A_ROWS, A_COLS, A_REAL_ROWS, A_REAL_COLS>(A, A_tile.data());

    toc = clock();
    times[5] += get_time(tic, toc);

}

template<typename TG, typename TP, typename TS>
noinline void AMX_kernel_mass(double * restrict times, float * restrict A, const TG * restrict coordinate_dofs, const TG * restrict weights, const TG * restrict fe_domain_grad, const TP * restrict fe0_geom, const TP * restrict coeffs, const __bf16 * restrict AMX_fe0, const TS * restrict AMX_fe0_transposed){
    
    std::fill(A, A + ASIZE, (float) 0.);

    clock_t firsttic, lasttoc;

    firsttic = clock();

    std::array<TG, PADDEDCOLS> Gtemp{}; // Geometry[PADDEDCOLS]
    construct_geometry_mass<TG, TP, PADDEDCOLS>(times, Gtemp.data(), coordinate_dofs, weights, fe_domain_grad, fe0_geom, coeffs);

    std::array<TS, PADDEDCOLS> G{};
    for(int i=0; i < PADDEDCOLS; i++)
        G[i] = (TS) Gtemp[i];

    sum2gemm3_AMX(times, A, G.data(), AMX_fe0, AMX_fe0_transposed);

    lasttoc = clock();
    times[6] += get_time(firsttic, lasttoc);

}

template<typename TG, typename TP, typename TS>
noinline void AMX_kernel_poisson(double * restrict times, float * restrict A, const TG * restrict coordinate_dofs, const TG * restrict weights, const TG * restrict fe_domain_grad, const TP * restrict fe0_geom, const TP * restrict coeffs, const __bf16 * restrict AMX_fe_grad, const TS * restrict AMX_fe_grad_transposed){
    
    std::fill(A, A + ASIZE, (float) 0.);

    clock_t firsttic, lasttoc;

    firsttic = clock();

    std::array<TG, GDIM*GDIM*PADDEDCOLS> Gtemp{}; // Geometry[GDIM][GDIM][PADDEDCOLS]
    construct_geometry_poisson<TG, TP, PADDEDCOLS>(times, Gtemp.data(), coordinate_dofs, weights, fe_domain_grad, fe0_geom, coeffs);

    std::array<TS, PADDEDCOLS*GDIM*GDIM> G{};
    for(int i=0; i < PADDEDCOLS*GDIM*GDIM; i++)
        G[i] = (TS) Gtemp[i];

    sum2gemm3_AMX(times, A, G.data(), AMX_fe_grad, AMX_fe_grad_transposed);

    lasttoc = clock();
    times[6] += get_time(firsttic, lasttoc);

}

template<typename TG, typename TP, typename TS>
noinline void AMX_kernel_mass_action(double * restrict times, float * restrict A, const TG * restrict coordinate_dofs, const TG * restrict weights, const TG * restrict fe_domain_grad, const __bf16 * restrict action_coeffs, const TP * restrict fe0_geom, const TP * restrict coeffs, const __bf16 * restrict AMX_action_fe0, const __bf16 * restrict AMX_fe0_transposed){
    
    std::fill(A, A + N_BATCH_CELLS*ASIZE, (float) 0.);

    clock_t firsttic, lasttoc;

    firsttic = clock();

    std::array<TG, N_BATCH_CELLS*NQUAD> G{}; // Geometry[PADDEDCOLS]
    for(int i = 0; i < N_BATCH_CELLS; i++)
        construct_geometry_mass(times, G.data() + i*NQUAD, coordinate_dofs + i*DOMAIN_ELDIM*GDIM, weights, fe_domain_grad, fe0_geom, coeffs + i*ELDIM);

    sum2gemm3_AMX_action(times, A, G.data(), AMX_action_fe0, AMX_fe0_transposed, action_coeffs);

    lasttoc = clock();
    times[6] += get_time(firsttic, lasttoc);

}

template<typename TG, typename TP, typename TS>
noinline void AMX_kernel_poisson_action(double * restrict times, float * restrict A, const TG * restrict coordinate_dofs, const TG * restrict weights, const TG * restrict fe_domain_grad, const __bf16 * restrict action_coeffs, const TP * restrict fe0_geom, const TP * restrict coeffs, const __bf16 * restrict AMX_action_fe_grad, const __bf16 * restrict AMX_fe_grad_transposed){
    
    std::fill(A, A + N_BATCH_CELLS*ASIZE, (float) 0.);

    clock_t firsttic, lasttoc;

    firsttic = clock();

    std::array<TG, N_BATCH_CELLS*GDIM*GDIM*NQUAD> G{}; // Geometry[N_BATCH_CELLS][GDIM][GDIM][NQUAD]
    for(int i = 0; i < N_BATCH_CELLS; i++)
        construct_geometry_poisson(times, G.data() + i*GDIM*GDIM*NQUAD, coordinate_dofs + i*DOMAIN_ELDIM*GDIM, weights, fe_domain_grad, fe0_geom, coeffs + i*ELDIM);

    sum2gemm3_AMX_action(times, A, G.data(), AMX_action_fe_grad, AMX_fe_grad_transposed, action_coeffs);

    lasttoc = clock();
    times[6] += get_time(firsttic, lasttoc);

}
