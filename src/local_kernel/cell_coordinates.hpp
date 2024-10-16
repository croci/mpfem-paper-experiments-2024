// Cell taken from the get_mesh_info.py script
#ifdef CELL_TYPE_HEXAHEDRON
const std::array<double, DOMAIN_ELDIM*GDIM> cdofs = { 0.13157895, -4.86842105, 4.73684211,
                                                      0.        , -5.        , 5.        ,
                                                      0.12927054, -4.61218837, 4.74145891,
                                                      0.        , -4.73684211, 5.        ,
                                                      0.        , -4.74145891, 4.48291782,
                                                     -0.13157895, -4.86842105, 4.73684211,
                                                      0.        , -4.49190844, 4.49190844,
                                                     -0.12927054, -4.61218837, 4.74145891};
const std::array<double, DOMAIN_ELDIM*GDIM> cdofs_ffcx = cdofs;
#endif


// Cell taken from the get_mesh_info.py script
#ifdef CELL_TYPE_TETRAHEDRON
const std::array<double, DOMAIN_ELDIM*GDIM> cdofs = {0.41867970, 0.82805639, 0.61219593,
                                                     0.45008666, 1.        , 0.60627011,
                                                     0.43362131, 0.88745495, 0.73114432,
                                                     0.52191511, 0.88440620, 0.58254484};

const std::array<double, DOMAIN_ELDIM*GDIM> cdofs_ffcx = cdofs;
#endif

#ifdef CELL_TYPE_TRIANGLE
#define BASETRI 1.41867970
#define TRI_EPS 1.0e0
#define TRI_SCALE (1.0e0)

const double X0 = TRI_SCALE*(BASETRI);
const double Y0 = TRI_SCALE*(BASETRI);
const double X1 = TRI_SCALE*(BASETRI + 2.);
const double Y1 = TRI_SCALE*(BASETRI + 2.);
const double X2 = TRI_SCALE*(BASETRI + 1. - TRI_EPS);
const double Y2 = TRI_SCALE*(BASETRI + 1. + TRI_EPS);

// This triangle leads to an ill-conditioned Jacobian (depending on how small eps is).
const std::array<double, DOMAIN_ELDIM*GDIM> cdofs = {X0, Y0,
                                                     X1, Y1,
                                                     X2, Y2};

const std::array<double, DOMAIN_ELDIM*3> cdofs_ffcx = {X0, Y0, 0.,
                                                       X1, Y1, 0.,
                                                       X2, Y2, 0.};
#undef BASETRI
#undef TRI_EPS
#undef TRI_SCALE
#endif

#ifdef CELL_TYPE_QUADRILATERAL
#define BASEQUAD 1.//.41867970
#define QUAD_EPS 5.0e-1

const double X0 = (BASEQUAD);
const double Y0 = (BASEQUAD);
const double X1 = (BASEQUAD + 1. - QUAD_EPS);
const double Y1 = (BASEQUAD + 1. + QUAD_EPS);
const double X2 = (BASEQUAD + 1. + QUAD_EPS);
const double Y2 = (BASEQUAD + 1. - QUAD_EPS);
const double X3 = (BASEQUAD + 2.);
const double Y3 = (BASEQUAD + 2.);

// This quadrilateral leads to an ill-conditioned Jacobian (depending on how small eps is).
const std::array<double, DOMAIN_ELDIM*GDIM> cdofs = {X0, Y0,
                                                     X1, Y1,
                                                     X2, Y2,
                                                     X3, Y3};

const std::array<double, DOMAIN_ELDIM*3> cdofs_ffcx = {X0, Y0, 0.,
                                                       X1, Y1, 0.,
                                                       X2, Y2, 0.,
                                                       X3, Y3, 0.,};

//// Keeping this here to remember the coordinate ordering
//const std::array<double, DOMAIN_ELDIM*GDIM> cdofs = {0.,0.,
//                                                     0.,1.,
//                                                     1.,0.,
//                                                     1.,1.};
//
//const std::array<double, DOMAIN_ELDIM*3> cdofs_ffcx = {0.,0.,0.,
//                                                       0.,1.,0.,
//                                                       1.,0.,0.,
//                                                       1.,1.,0.};

#undef BASEQUAD
#undef QUAD_EPS
#endif
