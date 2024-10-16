#if defined(__clang__)
    #define CLANGCHECK true
#else
    #define CLANGCHECK false
#endif

#if !defined(SAFECAST_DEFAULT)
    #define SAFECAST_DEFAULT false
#endif

// WARNING: WITH FLUSH_SUBNORMALS=true THE MASS KERNELS LOSE ALL ACCURACY!!!!
#if !defined(FLUSH_SUBNORMALS)
    #define FLUSH_SUBNORMALS false
#endif

//#define DEBUG_CAST
#ifdef DEBUG_CAST
    #include <format>
#endif

#if ~FLUSH_SUBNORMALS
    #include <bit>
#endif

#define FP16MANTISSA 0b0000001111111111
#define FP16EXPONENT 0b0111110000000000

template<int SIZE, bool SAFECAST=SAFECAST_DEFAULT>
inline void fp16tobf16(uint16_t * u) noexcept {
    //NOTE: _Float16 inf output does not map to inf. That would require an if statement which is slow.
    
    static_assert(sizeof(_Float16) == sizeof(uint16_t) && sizeof(__bf16) == sizeof(uint16_t));

    #if defined(DEBUG_CAST)
        std::array<_Float16, SIZE> input{};
        uint16_t * iu = reinterpret_cast<uint16_t*>(input.data());
        std::memcpy(input.begin(), u, SIZE*sizeof(uint16_t));
        __bf16 * out = reinterpret_cast<__bf16*>(u);
    #endif
    
    if constexpr(!SAFECAST){
        for(int i=0; i<SIZE; i++){

            const uint16_t sign = u[i] & 0b1000000000000000; // get the sign
            u[i] = u[i] & 0b0111111111111111; // remove the sign
            const bool iszero = u[i] == 0b0;
            const bool is_subnormal = (u[i] & FP16EXPONENT) == 0b0; // if subnormal, then the fp16 exponent is 0

            #if FLUSH_SUBNORMALS
                if (is_subnormal){
                    u[i] = 0b0; // Not much time lost by adding this branch.
                }else{
                    u[i] = u[i] + 0b0000000000000100; // round mantissa to nearest before bitshift.
                    u[i] = u[i] >> ((uint16_t) 3); // after this bitshift the last three bits of the mantissa are truncated and the exponent bits are aligned
                    u[i] = u[i] + 0b0011100000000000; // however, we must fix the exponent since for fp16 2**(exponent - 15) while for bf16 2**(exponent - 127)
                }

            #else
                if (iszero) {
                    u[i] = 0b0; // Not much time lost by adding this branch.
                }else if(is_subnormal){
                    //NOTE: this only works for subnormals. Assuming that fp16 subnormals are cast to bf16 normal numbers we can do the rounding as follows (no RtN here):
                    
                    // NOTE: count the number of left zeros in a uint16, then do -6 + 1 (the -6 counts the zeros in the mantissa)
                    // The +1 is a trick. Without it you would get the first mantissa bit being 1, but we are casting to a bf16 normal number which hardcodes a "1." so we add +1 to get rid of it.
                    const uint16_t mantissa = u[i] & FP16MANTISSA;
                    const uint16_t left_zeros = std::countl_zero(mantissa) - 5;
                    const uint16_t subn_exponent = (0b0000000001110001 - left_zeros) << 7;
                    u[i] = ((mantissa << left_zeros) & FP16MANTISSA);
                    u[i] = u[i] >> ((uint16_t) 3); // after this bitshift the last three bits of the mantissa are truncated and the exponent bits are aligned
                    u[i] = u[i] | subn_exponent;
                } else {
                    u[i] = u[i] + 0b0000000000000100; // round mantissa to nearest before bitshift.
                    u[i] = u[i] >> ((uint16_t) 3); // after this bitshift the last three bits of the mantissa are truncated and the exponent bits are aligned
                    u[i] = u[i] + 0b0011100000000000; // however, we must fix the exponent since for fp16 2**(exponent - 15) while for bf16 2**(exponent - 127)
                }

            #endif

            u[i] = u[i] | sign; // put the sign back

        }
    }else{
        _Float16 * f = reinterpret_cast<_Float16*>(u);
        __bf16 * b = reinterpret_cast<__bf16*>(u);
        for(int i=0; i<SIZE; i++){
            volatile float tempfloatcast = (float) f[i]; // volatile forces the compiler to do the casting
            b[i] = (__bf16) tempfloatcast;
        }
    }

    #if defined(DEBUG_CAST)
        if constexpr(!SAFECAST){
            for(int i=0; i<SIZE; i++){
                double a = input[i];
                double b = out[i];
                double err = std::fabs(a-b);
                if (err > 4.0e-3*std::fabs(a)){
                    std::cout << std::format("\n fp16: {} ({:16b}), bf16: {} ({:16b}), err: {}, \n", a, iu[i], b, u[i], err);
                }

            }
        }
    #endif

}

template<typename INTYPE, typename OTYPE, int n_vecs, int unpadded_rows, int unpadded_cols>
inline void pad_and_reshape_bf16(const INTYPE * old_vec, OTYPE * new_vec){
    // NOTE: Takes an input array of size [n_vecs][unpadded_cols][unpadded_rows] and
    //       returns a padded and reshaped array of sizes [n_vecs][rows][cols].
    //       The output array is cast to OTYPE which can only be of type INTYPE or __bf16.

    static_assert(n_vecs > 0);

    constexpr int rows = unpadded_rows%16 == 0 ? unpadded_rows : 16*(unpadded_rows/16 + 1);
    constexpr int cols = unpadded_cols%32 == 0 ? unpadded_cols : 32*(unpadded_cols/32 + 1);

    constexpr bool fp16tobf16flag = (std::is_same_v<OTYPE, __bf16> && std::is_same_v<INTYPE, _Float16>);

    if constexpr(!fp16tobf16flag){
        //If not fp16tobf16 then casting to OTYPE is either safe or not_needed
        for(int i=0; i < n_vecs; i++){
            for(int iq=0; iq < unpadded_cols; iq++){
                for(int j=0; j < unpadded_rows; j++){
                    new_vec[i*rows*cols + j*cols + iq] = (OTYPE) old_vec[i*unpadded_rows*unpadded_cols + iq*unpadded_rows + j];
                }
            }
        }

    }else{ // Here OTYPE is __bf16 and INTYPE is _Float16
        //// Copy and reshape _Float16 memory to new_vec without casting, then cast inplace.
        const __bf16 * tempbf16 = reinterpret_cast<const __bf16*>(old_vec);
        for(int i=0; i < n_vecs; i++){
            for(int iq=0; iq < unpadded_cols; iq++){
                for(int j=0; j < unpadded_rows; j++){
                    new_vec[i*rows*cols + j*cols + iq] = tempbf16[i*unpadded_rows*unpadded_cols + iq*unpadded_rows + j];
                }
            }
        }

        fp16tobf16<n_vecs*rows*cols>(reinterpret_cast<uint16_t*>(new_vec));

    }
}

template<typename TS, int SIZE>
inline void safecastbf16(TS * restrict a, __bf16 * restrict b){
    
    constexpr bool fp16flag = std::is_same_v<TS, _Float16>;

    // simply casts an array a to a __bf16 array b.
    // The problem is that direct conversion from _Float16 to __bf16
    // is not supported in hardware and compilers currently do not support direct
    // conversion from _Float16 either
    
    if constexpr(fp16flag && !CLANGCHECK){

        std::array<float, SIZE> temp{};

        for(int i=0; i<SIZE; i++)
            temp[i] = (float) a[i];

        for(int i=0; i<SIZE; i++)
            b[i] = (__bf16) temp[i];

    }else if constexpr(fp16flag && CLANGCHECK){

        for(int i=0; i<SIZE; i++)
            b[i] = (__bf16) ((float) a[i]);
    }else{
        for(int i=0; i<SIZE; i++)
            b[i] = (__bf16) a[i];
    }

}

template<int SIZE>
inline void fp32tobf16(float * x, __bf16 * y) {
    // NOTE: Using Intel intrinsics since GCC conversion is painfully slow
    //       and bit operations in GCC for float to bf16 conversion are bugged.

    // Casting two blocks of 16 floats at a time
    constexpr int nblocks = SIZE/32; // 16 floats per 512 bits and we take two 512 bits blocks at a time
    constexpr int rem512 = SIZE - 32*nblocks;
    for(int n = 0; n<nblocks; n++){
        __m512 a = _mm512_loadu_ps(x + 32*n);
        __m512 b = _mm512_loadu_ps(x + 32*n + 16);
        __m512bh c = _mm512_cvtne2ps_pbh(b,a);
        _mm512_storeu_ph((void *)(y + 32*n), reinterpret_cast<__m512h>(c));
    }

    // Casting two blocks of 8 floats at a time
    constexpr int flag256 = rem512/16;
    constexpr int rem256 = rem512 - 16*flag256;
    if constexpr(flag256 == 1){
        __m256 a = _mm256_loadu_ps(x + 32*nblocks);
        __m256 b = _mm256_loadu_ps(x + 32*nblocks + 8);
        __m256bh c = _mm256_cvtne2ps_pbh(b,a);
        _mm256_storeu_ph((void *)(y + 32*nblocks), reinterpret_cast<__m256h>(c));
    }

    // Scalar casting using a remainder loop
    constexpr int start = SIZE-rem256;
    for (int i = 0; i<rem256; i++)
        y[start + i] = _mm_cvtness_sbh(x[start + i]);

}


template<int SIZE, typename OUTTYPE, typename INTYPE>
inline OUTTYPE * array_cast(INTYPE * in, [[maybe_unused]] OUTTYPE * outbuffer){

    OUTTYPE * out = NULL;

    constexpr bool fp16tobf16flag = (std::is_same_v<OUTTYPE, __bf16> && std::is_same_v<INTYPE, _Float16>);
    constexpr bool fp32tobf16flag = (std::is_same_v<OUTTYPE, __bf16> && std::is_same_v<INTYPE, float>);
    constexpr bool nocastflag = std::is_same_v<INTYPE, OUTTYPE>;

    if constexpr(nocastflag){
        return in;

    }else if constexpr(fp16tobf16flag){
        out = reinterpret_cast<__bf16*>(in);
        fp16tobf16<SIZE>(reinterpret_cast<uint16_t*>(out));

    }else if constexpr(fp32tobf16flag){
        out = outbuffer;
        fp32tobf16<SIZE>(in, out);

    }else{
        out = outbuffer;
        for(int i=0; i<SIZE; i++)
            out[i] = (OUTTYPE) in[i];

    }

    return out;

}

template<int SIZE, typename OUTTYPE, typename INTYPE>
void array_safecast(INTYPE * in, OUTTYPE * out){

    static_assert(!std::is_same_v<OUTTYPE, __bf16>); // Use array_cast for casting to bf16

    for(int i=0; i<SIZE; i++)
        out[i] = (OUTTYPE) in[i];

}

//inline float sdpabf16(const __bf16 * a, const __bf16 * b){
//
//    float c[16];
//    std::fill(c, c+16, 0.0);
//
//    __m512 c_ =_mm512_load_ps(c);
//    for(int i=0; i<BCOLS/2; i++){
//        __m512bh a_ = reinterpret_cast<__m512bh>(_mm512_loadu_epi16(a + 32*i));
//        __m512bh b_ = reinterpret_cast<__m512bh>(_mm512_loadu_epi16(b + 32*i));
//        c_ = _mm512_dpbf16_ps(c_, a_, b_);
//    }
//
//    _mm512_store_ps((void *) c, c_);
//
//    float s = 0;
//    for(int i=0; i<16; i++)
//        s += c[i];
//
//    return s;
//}

////NOTE: the following was working with dpabf16(fe_grad + i*PADDEDCOLS, G_fe_grad + j*PADDEDCOLS, Atemp + 16*(i*ELDIM + j));
////      however, it was slow since the final copy of Atem was not aligned in a useful way
//inline void dpabf16(const __bf16 * restrict a, const __bf16 * restrict b, float * restrict c){
//
//    __m512 c_ =_mm512_loadu_ps(c);
//    for(int i=0; i<BCOLS/2; i++){
//        __m512bh a_ = reinterpret_cast<__m512bh>(_mm512_loadu_epi16(a + 32*i));
//        __m512bh b_ = reinterpret_cast<__m512bh>(_mm512_loadu_epi16(b + 32*i));
//        c_ = _mm512_dpbf16_ps(c_, a_, b_);
//    }
//
//    _mm512_storeu_ps((void *) c, c_);
//
//}
