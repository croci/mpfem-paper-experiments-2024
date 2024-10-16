template <typename TG, typename TP=double>
noinline void compute_geometry(TG * restrict G, const double eps){

    std::array<TG, DOMAIN_NQUAD*GDIM*DOMAIN_ELDIM> fe_domain_grad{};
    std::array<TG, NQUAD> weights{};

    std::array<TG, NQUAD*GDIM*ELDIM> fe_grad{};
    std::array<TP, NQUAD*ELDIM> fe0_geom{};
    preprocess_tabulated_values(weights.data(), fe_domain_grad.data(), fe_grad.data(), fe0_geom.data());

    std::array<TG, DOMAIN_ELDIM*GDIM> coordinate_dofs{};

    #if defined(CELL_TYPE_TETRAHEDRON) && defined(KERNEL_TYPE_POISSON)

        // Jacobian condition number in the infinity norm is (2+eps)*(3+eps)/eps \sim 6/eps
        // Jacobian determinant (which is the cell volume) is eps*scale (I think)

        // first entry is the scale, the other ones are the origin coordinates
        double scale = 1.0/std::sqrt(3.0);
        std::array<double, GDIM> origin_coords = {1.0/3.0, 2.0/3.0, 4.0/3.0};
        std::array<double, DOMAIN_ELDIM*GDIM> base_cell = { 1., 1.,  0.,
                                                            1., 0.,  1.,
                                                           eps, 1., -1.,
                                                            0., 0.,  0.};

        for(int j=0; j<DOMAIN_ELDIM; j++){
            for(int i=0; i<GDIM; i++){
                base_cell[j*GDIM + i] = base_cell[j*GDIM + i]*scale + origin_coords[i];
            }
        }

        for (int i = 0; i < DOMAIN_ELDIM*GDIM; i++)
            coordinate_dofs[i] = (TG) base_cell[i];

    #else
        static_assert(false); // TEST ONLY CREATED FOR TETRAHEDRA AND POISSON
        
        //for (int i = 0; i < DOMAIN_ELDIM*GDIM; i++)
        //    coordinate_dofs[i] = (TG) cdofs[i];

    #endif

    // NOTGE: if only coeff or action, then ffcx_coeffs contains only coeff or action
    //       be careful in case you initialise coeffs and action_coeffs to have different values.
    std::array<TP, ELDIM> coeffs{};
    for(int j=0; j<ELDIM; j++)
        coeffs[j] = (TP) (2.123456789*(j+1)/((double) ELDIM));

    double times[NTIMES] = {};
    construct_geometry_poisson(times, G, coordinate_dofs.data(), weights.data(), fe_domain_grad.data(), fe0_geom.data(), coeffs.data());

}

template <typename TG, typename TP=double>
double geometry_test(const double eps){

    #if defined(KERNEL_TGYPE_MASS)
        return 0.;
    #endif

    std::array<double, NQUAD*GDIM*GDIM> Gex{};
    std::array<TG, NQUAD*GDIM*GDIM> G{};

    compute_geometry<double, double>(Gex.data(), eps);
    compute_geometry<TG, TP>(G.data(), eps);

    double e = errnorm(Gex.data(), G.data(), NQUAD*GDIM*GDIM);

    return e;
}
