#include <math.h>
#include <stdlib.h>

float EPS = 1e-12;

float mean_squared_error(float *y, float *t, int D);
float cross_entropy(float *y, float *t, int K);
float *d_mean_squared_error(float *y, float *t, int D);
float *d_cross_entropy(float *y, float *t, int K);

float mean_squared_error(float *y, float *t, int D) {
    int d;
    float delta;
    float E = 0.0f;
    
    for(d = 1; d <= D; d++) {
        delta = t[d] - y[d];
        E += delta * delta;
    }
    
    return E / (float) D;
}

float cross_entropy(float *y, float *t, int K) {
    int k;
    float E = 0.0f;
    
    for(k = 1; k <= K; k++) {
        E -= t[k] * logf(y[k] + EPS);
    }
    
    return E;
}

/*===== derivative function =====*/
float *d_mean_squared_error(float *y, float *t, int D) {
    int d;
    float *dE_dy = NULL;
    
    if((dE_dy = (float *)malloc((D + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    dE_dy[0] = 0.0f;
    
    for(d = 1; d <= D; d++) {
        dE_dy[d] = 2.0f * (t[d] - y[d]) / (float) D;
    }
    
    return dE_dy;
}

float *d_cross_entropy(float *y, float *t, int K) {
    int k;
    float *dE_dy = NULL;
    
    if((dE_dy = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    dE_dy[0] = 0.0f;
    
    for(k = 1; k <= K; k++) {
        dE_dy[k] = - (t[k] / (y[k] + EPS));
    }
    
    return dE_dy;
}
