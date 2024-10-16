template <typename TG, typename Tfeg, typename Tfe0>
void preprocess_tabulated_values(TG* restrict weights, TG* restrict fe_domain_grad, Tfeg* restrict fe_grad, Tfe0 * restrict fe0_geom){

    auto fdg = [fe_domain_grad](auto k, auto i, auto j) -> TG& { return fe_domain_grad[k*DOMAIN_ELDIM*DOMAIN_NQUAD + i*DOMAIN_NQUAD + j]; };
    auto feg = [fe_grad](auto k, auto i, auto j) -> Tfeg& { return fe_grad[k*ELDIM*NQUAD + i*ELDIM + j]; };

    for(int iq=0; iq<NQUAD; iq++)
        weights[iq] = (TG) weights_[iq];

    #if defined(KERNEL_TYPE_COEFFICIENT) || defined(KERNEL_TYPE_MASS)
        for(int iq=0; iq<NQUAD; iq++){
            for(int ic=0; ic<ELDIM; ic++){
                fe0_geom[iq*ELDIM + ic] = (Tfe0) FE_C0[iq][ic];
            }
        }

    #endif

    #if GDIM == 3
        for(int iq=0; iq<DOMAIN_NQUAD; iq++){
            for(int j=0; j<DOMAIN_ELDIM; j++){
                fdg(0, j, iq) = (TG) FE_DOMAIN_D100[iq][j];
                fdg(1, j, iq) = (TG) FE_DOMAIN_D010[iq][j];
                fdg(2, j, iq) = (TG) FE_DOMAIN_D001[iq][j];
            }
        }

        #if defined(KERNEL_TYPE_POISSON)
            for(int iq=0; iq<NQUAD; iq++){
                for(int j=0; j<ELDIM; j++){
                    feg(0, iq, j) = (Tfeg) FE_C0_D100[iq][j];
                    feg(1, iq, j) = (Tfeg) FE_C0_D010[iq][j];
                    feg(2, iq, j) = (Tfeg) FE_C0_D001[iq][j];
                }
            }
        #endif
    #else
        for(int iq=0; iq<DOMAIN_NQUAD; iq++){
            for(int j=0; j<DOMAIN_ELDIM; j++){
                fdg(0, j, iq) = (TG) FE_DOMAIN_D10[iq][j];
                fdg(1, j, iq) = (TG) FE_DOMAIN_D01[iq][j];
            }
        }

        #if defined(KERNEL_TYPE_POISSON)
            for(int iq=0; iq<NQUAD; iq++){
                for(int j=0; j<ELDIM; j++){
                    feg(0, iq, j) = (Tfeg) FE_C0_D10[iq][j];
                    feg(1, iq, j) = (Tfeg) FE_C0_D01[iq][j];
                }
            }
        #endif

    #endif
}

template <typename TP, int nvecs>
inline void restructure_array_action(const TP * restrict vec, TP * restrict action_vec){
    for(int i = 0; i < nvecs; i++){
        for(int ic = 0; ic < ELDIM; ic++){
            for (int iq = 0; iq < NQUAD; iq++){
                action_vec[ic*NQUAD*nvecs + i*NQUAD + iq] = vec[i*ELDIM*NQUAD + iq*ELDIM + ic];
            }
        }
    }
}

//////////////////////////////////// MASS FORM /////////////////////////////////////////

template <typename TA, typename TS>
inline void sum2gemm3_mass(double * restrict times, TA * restrict A, const TS * restrict fe0, const TS * restrict G){

    clock_t tic, toc;

    tic = clock();

    std::array<TS, NQUAD*ELDIM> Gfe0{};
    std::memcpy(Gfe0.data(), fe0, NQUAD*ELDIM*sizeof(TS));
    for(int iq = 0; iq < NQUAD; iq++){
        for(int ic = 0; ic < ELDIM; ic++){
            Gfe0[iq*ELDIM + ic] *= G[iq];
        }
    }

    toc = clock();
    times[4] += get_time(tic, toc);
    tic = clock();

    for(int iq = 0; iq < NQUAD; iq++){
        for(int i = 0; i < ELDIM; i++){
            for(int j = 0; j < ELDIM; j++){
                A[i*ELDIM + j] += ((TA) fe0[iq*ELDIM + i])*((TA) Gfe0[iq*ELDIM + j]);
            }
        }
    }

    toc = clock();
    times[5] += get_time(tic, toc);

}

template <typename TA, typename TG, typename TP, typename TS>
inline void action_sum_mass(double * restrict times, TA * restrict A, const TS * restrict fe0, const TP * restrict action_fe0, const TP * restrict action_coeffs, const TG * restrict G){

    clock_t tic, toc;

    tic = clock();

    constexpr int SIZE = N_BATCH_CELLS*NQUAD;

    std::array<TP, SIZE> temp_coeff_val{};
    for(int ic = 0; ic<ELDIM; ic++){
        for(int ii = 0; ii < N_BATCH_CELLS; ii++){
            for (int iq = 0; iq < NQUAD; iq++){
                temp_coeff_val[ii*NQUAD + iq] += action_coeffs[ii*ELDIM + ic]*action_fe0[ic*NQUAD + iq];
            }
        }
    }

    // NOTE: Extra casting does not make too much sense, but it's for the paper (makes theory simpler).
    [[maybe_unused]] std::array<TG, SIZE> TGbuf{};
    TG * coeff_val = array_cast<SIZE>(temp_coeff_val.data(), TGbuf.data());

    for(int ii = 0; ii < SIZE; ii++){
         coeff_val[ii] *= G[ii];
    }

    [[maybe_unused]] std::array<TS, SIZE> TSbuf{};
    TS * cval = array_cast<SIZE>(coeff_val, TSbuf.data());
    
    toc = clock();
    times[4] += get_time(tic, toc);
    tic = clock();

    for(int iq = 0; iq < NQUAD; iq++){
        for(int ii = 0; ii < N_BATCH_CELLS; ii++){
            for(int ic = 0; ic < ELDIM; ic++){
                A[ii*ASIZE + ic] += ((TA) fe0[iq*ELDIM + ic])*((TA) cval[ii*NQUAD + iq]);
            }
        }
    }

    toc = clock();
    times[5] += get_time(tic, toc);

}

template <typename TA, typename TG, typename TP, typename TS>
noinline void local_kernel_mass(double * restrict times, TA * restrict A, const TG * restrict coordinate_dofs, const TG * restrict weights, const TG * restrict fe_domain_grad, const TP * restrict action_coeffs, const TS * restrict fe0, const TP * restrict fe0_geom, [[maybe_unused]] const TP * restrict action_fe0, const TP * restrict coeffs){
    // degree 3 Lagrange element on hexahedrons

    clock_t firsttic, lasttoc;

    firsttic = clock();

    #if defined(KERNEL_TYPE_ACTION)
        std::fill(A, A + N_BATCH_CELLS*ASIZE, (TA) 0.);

        std::array<TG, N_BATCH_CELLS*NQUAD> G{}; // Geometry[NQUAD]
        for(int i = 0; i < N_BATCH_CELLS; i++)
            construct_geometry_mass(times, G.data() + i*NQUAD, coordinate_dofs + i*DOMAIN_ELDIM*GDIM, weights, fe_domain_grad, fe0_geom, coeffs + i*ELDIM);

        action_sum_mass(times, A, fe0, action_fe0, action_coeffs, G.data());
    #else
        std::fill(A, A + ASIZE, (TA) 0.);

        std::array<TG, NQUAD> Gtemp{}; // Geometry[NQUAD]
        construct_geometry_mass(times, Gtemp.data(), coordinate_dofs, weights, fe_domain_grad, fe0_geom, coeffs);
        std::array<TS, NQUAD> G{};
        for(int i=0; i < NQUAD; i++)
            G[i] = (TS) Gtemp[i];

        sum2gemm3_mass(times, A, fe0, G.data());
    #endif

    lasttoc = clock();
    times[6] += get_time(firsttic, lasttoc);

}

////////////////////////////////////// POISSON FORM /////////////////////////////////////////

template <typename TS>
inline void prepare_sum2gemm3_poisson(TS * restrict G_fe_grad, const TS * restrict fe_grad, const TS * restrict G){

    auto G_ = [G](auto k, auto i, auto j) -> TS { return G[k*GDIM*NQUAD + i*NQUAD + j]; };
    auto feg = [fe_grad](auto k, auto i, auto j) -> TS { return fe_grad[k*ELDIM*NQUAD + i*ELDIM + j]; }; // fe_grad[GDIM][NQUAD][ELDIM]
    auto Gfeg = [G_fe_grad](auto k, auto i, auto j) -> TS& { return G_fe_grad[k*ELDIM*NQUAD + i*ELDIM + j]; }; // fe_grad[GDIM*GDIM][NQUAD][ELDIM]

    for(int j = 0; j < GDIM; j++){
        for(int k = 0; k < GDIM; k++){
            for(int iq = 0; iq < NQUAD; iq++){
                for(int l = 0; l < ELDIM; l++){
                    Gfeg(j*GDIM + k, iq, l) = G_(j, k, iq)*feg(k, iq, l);
                }
            }
        }
    }
}

template <typename TA, typename TS>
inline void perform_sum2gemm3_poisson(TA * restrict A, const TS * restrict fe_grad, const TS * restrict G_fe_grad){

    auto A_ = [A](auto i, auto j) -> TA& { return A[i*ELDIM + j]; };
    auto feg = [fe_grad](auto k, auto i, auto j) -> TS { return fe_grad[k*ELDIM*NQUAD + i*ELDIM + j]; }; // fe_grad[GDIM][NQUAD][ELDIM]

    for(int iq = 0; iq < NQUAD; iq++){
        for(int i = 0; i < ELDIM; i++){
            for(int l = 0; l < ELDIM; l++){
                for(int j = 0; j < GDIM; j++){
                    for(int k = 0; k < GDIM; k++){
                        A_(i,l) += ((TA) feg(j, iq, i))*((TA) G_fe_grad[(j*GDIM + k)*ELDIM*NQUAD + iq*ELDIM + l]);
                    }
                }
            }
        }
    }

}

template <typename TA, typename TS>
inline void sum2gemm3_poisson(double * restrict times, TA * restrict A, const TS * restrict fe_grad, const TS * restrict G){

    clock_t tic, toc;

    tic = clock();

    std::array<TS, GDIM*GDIM*NQUAD*ELDIM> G_fe_grad{};
    prepare_sum2gemm3_poisson(G_fe_grad.data(), fe_grad, G);

    toc = clock();
    times[4] += get_time(tic, toc);
    tic = clock();

    perform_sum2gemm3_poisson(A, fe_grad, G_fe_grad.data());

    toc = clock();
    times[5] += get_time(tic, toc);

}

template <typename TA, typename TG, typename TP, typename TS>
inline void sum2gemm3_poisson_action(double * restrict times, TA * restrict A, const TS * restrict fe_grad, const TP * restrict action_fe_grad, const TP * restrict action_coeffs, TG * restrict G){

    clock_t tic, toc;

    tic = clock();

    constexpr int cols = N_BATCH_CELLS*NQUAD;
    constexpr int SIZE = GDIM*cols;

    std::array<TP, SIZE> temp_coeff_val{};
    for(int ic = 0; ic < ELDIM; ic++){
        for(int i = 0; i < GDIM; i++){
            for(int ii = 0; ii < N_BATCH_CELLS; ii++){
                for (int iq = 0; iq < NQUAD; iq++){
                    temp_coeff_val[i*cols + ii*NQUAD + iq] += action_coeffs[ii*ELDIM + ic]*action_fe_grad[ic*NQUAD*GDIM + i*NQUAD + iq];
                }
            }
        }
    }

    // NOTE: Double casting does not make too much sense, but it's for the paper (makes theory simpler).
    [[maybe_unused]] std::array<TG, SIZE> TGbuf{};
    TG * coeff_val = array_cast<SIZE>(temp_coeff_val.data(), TGbuf.data());

    std::array<TG, SIZE> G_coeff_val{};
    for(int i = 0; i < GDIM; i++){
        for(int j = 0; j < GDIM; j++){
            for(int ii = 0; ii < N_BATCH_CELLS; ii++){
                for (int iq = 0; iq < NQUAD; iq++){
                    G_coeff_val[i*cols + ii*NQUAD + iq] += G[ii*GDIM*GDIM*NQUAD + (i*GDIM + j)*NQUAD + iq]*coeff_val[j*cols + ii*NQUAD + iq];
                }
            }
        }
    }

    [[maybe_unused]] std::array<TS, SIZE> TSbuf{};
    TS * Gcval = array_cast<SIZE>(G_coeff_val.data(), TSbuf.data());

    toc = clock();
    times[4] += get_time(tic, toc);
    tic = clock();

    for(int j = 0; j < GDIM; j++){
        for(int iq = 0; iq < NQUAD; iq++){
            for(int ii = 0; ii < N_BATCH_CELLS; ii++){
                for(int ic = 0; ic < ELDIM; ic++){
                    A[ii*ASIZE + ic] += ((TA) fe_grad[j*ELDIM*NQUAD + iq*ELDIM + ic])*((TA) Gcval[j*cols + ii*NQUAD + iq]);
                }
            }
        }
    }

    toc = clock();
    times[5] += get_time(tic, toc);

}

template <typename TA, typename TG, typename TP, typename TS>
noinline void local_kernel_poisson(double * restrict times, TA * restrict A, const TG * restrict coordinate_dofs, const TG * restrict weights, const TG * restrict fe_domain_grad, const TS * restrict fe_grad, [[maybe_unused]] const TP * restrict action_fe_grad, const TP * restrict action_coeffs, const TP * restrict fe0_geom, const TP * restrict coeffs){
    // degree 3 Lagrange element on hexahedrons


    clock_t firsttic, lasttoc;

    firsttic = clock();

    #if defined(KERNEL_TYPE_ACTION)
        std::fill(A, A + N_BATCH_CELLS*ASIZE, (TA) 0.);

        std::array<TG, N_BATCH_CELLS*NQUAD*GDIM*GDIM> G{}; // Geometry[NQUAD][GDIM][GDIM]
        for(int i = 0; i < N_BATCH_CELLS; i++)
            construct_geometry_poisson(times, G.data() + i*GDIM*GDIM*NQUAD, coordinate_dofs + i*DOMAIN_ELDIM*GDIM, weights, fe_domain_grad, fe0_geom, coeffs + i*ELDIM);

        sum2gemm3_poisson_action(times, A, fe_grad, action_fe_grad, action_coeffs, G.data());
    #else
        std::fill(A, A + ASIZE, (TA) 0.);

        std::array<TG, NQUAD*GDIM*GDIM> Gtemp{}; // Geometry[NQUAD][GDIM][GDIM]
        construct_geometry_poisson(times, Gtemp.data(), coordinate_dofs, weights, fe_domain_grad, fe0_geom, coeffs);

        std::array<TS, NQUAD*GDIM*GDIM> G{};
        for(int i=0; i < NQUAD*GDIM*GDIM; i++)
            G[i] = (TS) Gtemp[i];

        sum2gemm3_poisson(times, A, fe_grad, G.data());
    #endif

    lasttoc = clock();
    times[6] += get_time(firsttic, lasttoc);

}
