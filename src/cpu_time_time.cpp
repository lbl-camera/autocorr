#include <cmath>

void cpu_time_time_brown(float *& intensity, int ntimes, int npixels, float *& tcf) {
    /* DOI:https://doi.org/10.1103/PhysRevE.56.6601 */

    #pragma omp parallel for
    for (int i = 0; i < ntimes * ntimes; i++)
        tcf[i] = 0;

    #pragma omp parallel for
    for (int i = 0; i < ntimes; i++) {
        float I1 = 0;
        float I1square = 0;
        for (int pix = 0; pix < npixels; pix++) {
            I1 += intensity[i*npixels+pix];
            I1square += std::pow(intensity[i*npixels+pix], 2);
        }

        for (int j = 0; j < ntimes; j++) {
            float I2 = 0;
            float I2square = 0;
            float I1I2 = 0;
            for (int pix = 0; pix < npixels; pix++) {
                I2 += intensity[j*npixels+pix];
                I2square += std::pow(intensity[j*npixels+pix], 2);
                I1I2 = intensity[i*npixels+pix] * intensity[j*npixels+pix];
            }

            float norm = std::sqrt((I1square-I1*I1)*(I2square-I2*I2));
            tcf[i * npixels + j] = (I1I2-I1*I2)/norm;
        } 
    }
}

void cpu_time_time_sutton(float *& intensity, int ntimes, int npixels, float *& tcf) {
    /* DOI: https://doi.org/10.1364/OE.11.002268 */

    #pragma omp parallel for
    for (int i =  0; i < ntimes * ntimes; i++)
        tcf[i] = 0;

    #pragma omp parallel for
    for (int i = 0; i < ntimes; i++) {
        float I1 = 0;

        for (int pix = 0; pix < npixels; pix++)
            I1 += intensity[i*npixels+pix];

        for (int j = 0; j < ntimes; j++) {
            float I2 = 0;
            float I1I2 = 0;
            for (int pix = 0; pix < npixels; pix++) {
                I2 += intensity[j*npixels+pix];
                I1I2 += intensity[i*npixels+pix] * intensity[j*npixels+pix];
            }
            tcf[i*ntimes+j] = I1I2 / (I1 * I2);
        }
    }
}
