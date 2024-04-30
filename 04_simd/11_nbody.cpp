#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
    const int N = 16;
    float x[N], y[N], m[N], fx[N], fy[N];
    int mask2[N];
    for(int i=0; i<N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
        fx[i] = fy[i] = 0;
    }
    __m512i vindex = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

    printf("[Non-vectorization]\n");

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            if(i != j) {
                float rx = x[i] - x[j];
                float ry = y[i] - y[j];
                float r = std::sqrt(rx * rx + ry * ry);
                float r3 = 1 / (r * r * r);
                fx[i] -= rx * m[j] / (r * r * r);
                fy[i] -= ry * m[j] / (r * r * r);
                float fx_temp = rx * m[j] / (r * r * r);
                float fy_temp = ry * m[j] / (r * r * r);
            }
        }
        printf("%d %g %g\n",i,fx[i],fy[i]);
    }
    printf("--------------------------\n");
    printf("[Vectorization]\n");

    for(int i=0; i<N; i++) {
        __m512 x_vec = _mm512_load_ps(x);
        __m512 y_vec = _mm512_load_ps(y);
        __m512 m_vec = _mm512_load_ps(m);
        __m512 fx_vec = _mm512_setzero_ps();
        __m512 fy_vec = _mm512_setzero_ps();

        __m512 x_i = _mm512_set1_ps(x[i]);
        __m512 y_i = _mm512_set1_ps(y[i]);
        __m512i vi = _mm512_set1_epi32(i);
        __mmask16 mask = _mm512_cmpeq_epi32_mask(vindex, vi); // i番目のビットだけ1，それ以外は0のマスク

        __m512 rx = _mm512_sub_ps(x_i, x_vec);
        __m512 ry = _mm512_sub_ps(y_i, y_vec);
        __m512 r = _mm512_rsqrt14_ps(_mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry))); // 1/sqrt(rx^2 + ry^2)
        __m512 r3 = _mm512_mul_ps(r, _mm512_mul_ps(r, r)); // 1/r^3
        __m512 rx_m_r3 = _mm512_mul_ps(_mm512_mul_ps(rx, m_vec), r3);
        __m512 ry_m_r3 = _mm512_mul_ps(_mm512_mul_ps(ry, m_vec), r3);
        __m512 rx_m_r3_masked = _mm512_mask_blend_ps(mask, rx_m_r3, _mm512_setzero_ps());
        __m512 ry_m_r3_masked = _mm512_mask_blend_ps(mask, ry_m_r3, _mm512_setzero_ps());

        // 配列の水平加算
        __m512 rx_m_r3_sum = _mm512_set1_ps(_mm512_reduce_add_ps(rx_m_r3_masked));
        __m512 ry_m_r3_sum = _mm512_set1_ps(_mm512_reduce_add_ps(ry_m_r3_masked));
      
        // rx_m_r3_sumのi番目の要素以外を0として，fx_vecから減算
        fx_vec = _mm512_sub_ps(fx_vec, _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), rx_m_r3_sum));
        fy_vec = _mm512_sub_ps(fy_vec, _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), ry_m_r3_sum));

        _mm512_store_ps(fx, fx_vec);
        _mm512_store_ps(fy, fy_vec);
        printf("%d %g %g\n",i,fx[i],fy[i]);
    }
}
