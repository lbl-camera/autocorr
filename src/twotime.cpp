#include "autocorr.h"

void twotime_corr_func(float *& signal, float *& tcf, int nrow, int ntimes) {

    // allocate memory for normalization values
    float * n1 = new float [ntimes * ntimes];
    float * n2 = new float [ntimes * ntimes]; 

#pragma omp parallel for
    for (int i = 0; i < ntimes*ntimes; i++) {
        tcf[i] = n1[i] = n2[i] = 0.f;
    }

    for (int i = 0; i < nrow; i++)
#pragma omp parallel for
        for (int j = 0; j < ntimes; j++)
            for (int k = 0; k < ntimes; k++) {
                tcf[j*ntimes+k] += signal[i*ntimes+j] * signal[i*ntimes+k];
                n1[j*ntimes+k]  += signal[i*ntimes+j];
                n2[j*ntimes+k]  += signal[i*ntimes+k];
            }

    // normalize
#pragma omp parallel for
    for (int i = 0; i < ntimes*ntimes; i++)
        if ((n1[i] > 0) && (n2[i] > 0))
            tcf[i] /= n1[i] * n2[i];
        else
            tcf[i] = 0.f;

    delete [] n1;
    delete [] n2;
}
