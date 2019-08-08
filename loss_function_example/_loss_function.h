#include <math.h>

float mean_squared_error(float *y, float *t, int D);
float cross_entropy(float *y, float *t, int K);

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
        E -= t[k] * logf(y[k]);
    }
    
    return E;
}
