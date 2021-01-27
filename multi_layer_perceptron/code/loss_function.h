#include <math.h>

float EPS = 1e-12;

float mean_squared_error(float *y, float *t, int D, int derivative, float *dE_dy);
float cross_entropy(float *y, float *t, int K, int derivative, float *dE_dy);

float mean_squared_error(float *y, float *t, int D, int derivative, float *dE_dy) {
    int d;
    float delta;
    float E = 0.0f;

    if(derivative) {
        for(d = 1; d <= D; d++) {
            dE_dy[d] = - 2.0f * (t[d] - y[d]) / (float) D;
        }
        return 0.0f;
    }

    for(d = 1; d <= D; d++) {
        delta = t[d] - y[d];
        E += delta * delta;
    }
    
    return E / (float) D;
}

float cross_entropy(float *y, float *t, int K, int derivative, float *dE_dy) {
    int k;
    float E = 0.0f;
    
    if(derivative) {
        for(k = 1; k <= K; k++) {
            dE_dy[k] = - t[k] / (y[k] + EPS);
        }
        return 0.0f;
    }
    
    for(k = 1; k <= K; k++) {
        E -= t[k] * logf(y[k] + EPS);
    }
    
    return E;
}
