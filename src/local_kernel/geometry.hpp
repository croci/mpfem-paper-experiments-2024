template <typename TG>
inline TG det2(const TG a, const TG d, const TG b, const TG c){
    // Computes the determinant of a 2-by-2 matrix a*d - b*c using Kahan's FMA trick
    // which is accurate even at low precisions.
    // NOTE: up until C++23 std::fma is not implemented in the half precision formats (even if the hardware supports it).
    #if defined(FP_FAST_FMA)
        const TG w = b*c;
        return std::fma(a,d,-w) + std::fma(-b,c,w);
    #else
        return a*d - b*c;
    #endif
}

template <typename TG, bool DETONLY=false>
inline void adj_det3_inplace(TG * J, TG * detJ){
    // Overwrites J with its adjugate and returns abs(detJ)

    auto J_ = [J](auto i, auto j, auto k) -> TG& { return J[i*GDIM*DOMAIN_NQUAD + j*DOMAIN_NQUAD + k]; };

    #if !GCC_COMPILER
        uint16_t * u_detJ = NULL;
        if constexpr(std::is_same_v<TG, __bf16> || std::is_same_v<TG, _Float16>){
            u_detJ = reinterpret_cast<uint16_t*>(&detJ);
        }
    #endif

    // NOTE that if DOMAIN_NQUAD == 1 there is no actual loop here
    for(int iq=0; iq<DOMAIN_NQUAD; iq++){

        #if GDIM == 3

            TG a = J_(0, 0, iq);
            TG b = J_(0, 1, iq);
            TG c = J_(0, 2, iq);
            TG d = J_(1, 0, iq);
            TG e = J_(1, 1, iq);
            TG f = J_(1, 2, iq);
            TG g = J_(2, 0, iq);
            TG h = J_(2, 1, iq);
            TG i = J_(2, 2, iq);

            TG A = det2(e, i, f, h);
            TG B = det2(f, g, d, i);
            TG C = det2(d, h, e, g);
            if constexpr(!DETONLY){
                TG D = det2(c, h, b, i);
                TG E = det2(a, i, c, g);
                TG F = det2(b, g, a, h);
                TG G = det2(b, f, c, e);
                TG H = det2(c, d, a, f);
                TG I = det2(a, e, b, d);

                J_(0, 0, iq) = A;
                J_(1, 0, iq) = B;
                J_(2, 0, iq) = C;
                J_(0, 1, iq) = D;
                J_(1, 1, iq) = E;
                J_(2, 1, iq) = F;
                J_(0, 2, iq) = G;
                J_(1, 2, iq) = H;
                J_(2, 2, iq) = I;
            }

            //NOTE: fine for Poisson to take the abs but in general be careful about the sign
            // fabs not defined for half precision in clang 18.1.1 yet
            detJ[iq] = (a*A + b*B + c*C);
            #if GCC_COMPILER
                detJ[iq] = std::fabs(detJ[iq]);
            #else
                if constexpr(std::is_same_v<TG, __bf16> || std::is_same_v<TG, _Float16>){
                    u_detJ[iq] &= (uint16_t) 0b0111111111111111; // remove the sign
                }else{
                    detJ[iq] = std::fabs(detJ[iq]);
                }
            #endif

        #else
            TG a = J_(0, 0, iq);
            TG b = J_(0, 1, iq);
            TG c = J_(1, 0, iq);
            TG d = J_(1, 1, iq);

            if constexpr(!DETONLY){
                J_(0, 0, iq) = d;
                J_(0, 1, iq) = -b;
                J_(1, 0, iq) = -c;
                J_(1, 1, iq) = a;
            }

            //NOTE: fine for Poisson to take the abs but in general be careful about the sign
            // fabs not defined for half precision in clang 18.1.1 yet
            detJ[iq] = det2(a,d,b,c);
            #if GCC_COMPILER
                detJ[iq] = std::fabs(detJ[iq]);
            #else
                if constexpr(std::is_same_v<TG, __bf16> || std::is_same_v<TG, _Float16>){
                    u_detJ[iq] &= (uint16_t) 0b0111111111111111; // remove the sign
                }else{
                    detJ[iq] = std::fabs(detJ[iq]);
                }
            #endif

        #endif

    }

}

//NOTE: MQUAD is only used in the definition of G_ and is used for the AMX kernel
template <typename TG, typename TP, int MQUAD=NQUAD>
inline void construct_geometry_mass(double * restrict times, TG * restrict G, const TG * restrict coordinate_dofs, const TG * restrict weights, const TG * restrict fe_domain_grad, const TP * restrict fe0, const TP * restrict coeffs){

    static_assert(DOMAIN_NQUAD == NQUAD || DOMAIN_NQUAD == 1);

    clock_t firsttic, tic, toc;

    firsttic = clock();
    tic = clock();

    std::array<TG, DOMAIN_NQUAD*GDIM*GDIM> J{}; // Jacobian[DOMAIN_NQUAD][GDIM][GDIM]
    std::array<TG, DOMAIN_NQUAD> detJ{};     // Jacobian determinants

    // Construct Jacobians
    for(int i = 0; i < GDIM; i++){
        for(int j = 0; j < GDIM; j++){
            for(int ic = 0; ic < DOMAIN_ELDIM; ic++){
                for (int iq = 0; iq < DOMAIN_NQUAD; iq++){
                    J[i*DOMAIN_NQUAD*GDIM + j*DOMAIN_NQUAD + iq] += coordinate_dofs[ic*GDIM + i]*fe_domain_grad[j*DOMAIN_ELDIM*DOMAIN_NQUAD + ic*DOMAIN_NQUAD + iq];
                }
            }
        }
    }

    toc = clock();
    times[0] += get_time(tic, toc);
    tic = clock();
        
    // Computes absolute value of determinant of J.
    adj_det3_inplace<TG, true>(J.data(), detJ.data()); // J is still J here since passing DETONLY=true

    toc = clock();
    times[1] += get_time(tic, toc);
    tic = clock();

    std::array<TG, NQUAD> coeff_val{};
    #if defined(KERNEL_TYPE_COEFFICIENT)
        std::array<TP, NQUAD> temp_coeff_val{};
        for(int ic = 0; ic<ELDIM; ic++){
            for (int iq = 0; iq < NQUAD; iq++){
                temp_coeff_val[iq] += coeffs[ic]*fe0[iq*ELDIM + ic];
            }
        }

        for (int iq = 0; iq < NQUAD; iq++){
            coeff_val[iq] = (TG) temp_coeff_val[iq];
        }

    #else
        std::fill(coeff_val.begin(), coeff_val.end(), (TG) 1.);
    #endif

    if constexpr(DOMAIN_NQUAD == NQUAD){
        for (int iq = 0; iq < NQUAD; iq++){
            G[iq] = (weights[iq]*detJ[iq])*coeff_val[iq];
        }
    }else{
        for (int iq = 0; iq < NQUAD; iq++){
            G[iq] = (weights[iq]*detJ[0])*coeff_val[iq];
        }

    }

    toc = clock();
    times[2] += get_time(tic, toc);
    times[3] += get_time(firsttic, toc);

}

//NOTE: MQUAD is only used in the definition of G_ and is used for the AMX kernel
template <typename TG, typename TP, int MQUAD=NQUAD>
inline void construct_geometry_poisson(double * restrict times, TG * restrict G, const TG * restrict coordinate_dofs, const TG * restrict weights, const TG * restrict fe_domain_grad, const TP * restrict fe0, const TP * restrict coeffs){

    static_assert(DOMAIN_NQUAD == NQUAD || DOMAIN_NQUAD == 1);

    clock_t firsttic, tic, toc;

    firsttic = clock();
    tic = clock();

    std::array<TG, DOMAIN_NQUAD*GDIM*GDIM> J{}; // Jacobian[DOMAIN_NQUAD][GDIM][GDIM]
    std::array<TG, DOMAIN_NQUAD> detJ{};     // Jacobian determinants

    // Construct Jacobians
    for (int iq = 0; iq < DOMAIN_NQUAD; iq++){
        for(int ic = 0; ic < DOMAIN_ELDIM; ic++){
            for(int j = 0; j < GDIM; j++){
                for(int i = 0; i < GDIM; i++){
                    J[i*DOMAIN_NQUAD*GDIM + j*DOMAIN_NQUAD + iq] += coordinate_dofs[ic*GDIM + i]*fe_domain_grad[j*DOMAIN_ELDIM*DOMAIN_NQUAD + ic*DOMAIN_NQUAD + iq];
                }
            }
        }
    }

    toc = clock();
    times[0] += get_time(tic, toc);
    tic = clock();
        
    // Computes adjugate and absolute value of determinant of J.
    adj_det3_inplace(J.data(), detJ.data()); // J now contains Jadj

    toc = clock();
    times[1] += get_time(tic, toc);
    tic = clock();

    // Construct geometry as G = (J^A)*(J^A)^T*(weight/detJ)
    // here splitting the loop in 2 to reduce
    // the number of computations (hence rounding errors). Perhaps not essential.
    // NOTE that if DOMAIN_NQUAD == 1 there is no actual loop over iq
    // FIXME: this is now in a weird shape in memory. Double check what is the best way
    // of doing this
    for (int iq = 0; iq < DOMAIN_NQUAD; iq++){
        for(int k = 0; k < GDIM; k++){
            for(int i = 0; i < GDIM; i++){
                for(int j = 0; j < GDIM; j++){
                    G[i*GDIM*MQUAD + j*MQUAD + iq] += (J[i*DOMAIN_NQUAD*GDIM + k*DOMAIN_NQUAD + iq]*J[j*DOMAIN_NQUAD*GDIM + k*DOMAIN_NQUAD + iq]);
                }
            }
        }
    }

    std::array<TG, NQUAD> coeff_val{};
    #if defined(KERNEL_TYPE_COEFFICIENT)
        std::array<TP, NQUAD> temp_coeff_val{};
        for(int ic = 0; ic<ELDIM; ic++){
            for (int iq = 0; iq < NQUAD; iq++){
                temp_coeff_val[iq] += coeffs[ic]*fe0[iq*ELDIM + ic];
            }
        }

        for (int iq = 0; iq < NQUAD; iq++){
            coeff_val[iq] = (TG) temp_coeff_val[iq];
        }

    #else
        std::fill(coeff_val.begin(), coeff_val.end(), (TG) 1.);
    #endif

    if constexpr(DOMAIN_NQUAD == NQUAD){
        for(int i = 0; i < GDIM; i++){
            for(int j = 0; j < GDIM; j++){
                for (int iq = 0; iq < NQUAD; iq++){
                    G[i*GDIM*MQUAD + j*MQUAD + iq] *= (weights[iq]/detJ[iq])*coeff_val[iq];
                }
            }
        }
    }else{
        //FIXME: Possibly when DOMAIN_NQUAD == 1 we can do AMX_G_fe_grad offline apart from
        // the weights[iq] which would be easy to incorporate later.
        // This would effectively reduce the AMX preparation by a lot.

        //NOTE: Here we have DOMAIN_NQUAD == 1 and G_(0, i, j) contains the constant geometry
        //      We do a reverse loop so that G_(0, i, j) is overwritten last
        for (int iq = NQUAD; iq --> 0;){ 
            for(int i = 0; i < GDIM; i++){
                for(int j = 0; j < GDIM; j++){
                    G[i*GDIM*MQUAD + j*MQUAD + iq] = G[i*GDIM*MQUAD + j*MQUAD + 0]*(weights[iq]/detJ[0])*coeff_val[iq];
                }
            }
        }

    }

    toc = clock();
    times[2] += get_time(tic, toc);
    times[3] += get_time(firsttic, toc);

}
