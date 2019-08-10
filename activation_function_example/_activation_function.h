#include <stdlib.h>
#include <math.h>

float *step(float *a, int D);
float *ReLU(float *a, int D);
float *sigmoid(float *a, int D);
float *softmax(float *a, int K);

float *step(float *a, int D) {
    int d;
    float *z = NULL;
    
    if((z = (float *)malloc((D + 1) * sizeof(float))) == NULL) {
        return NULL;
    }
    
    z[0] = 1.0f;
    
    for(d = 1; d <= D; d++) {
        if(a[d] < 0.0f) {
            z[d] = 0.0f;
        } else {
            z[d] = 1.0f;
        }
    }
    
    return z;
}

float *ReLU(float *a, int D) {
    int d;
    float *z = NULL;
    
    if((z = (float *)malloc((D + 1) * sizeof(float))) == NULL) {
        return NULL;
    }
    
    z[0] = 1.0f;
    
    for(d = 1; d <= D; d++) {
        if(a[d] < 0.0f) {
            z[d] = 0.0f;
        } else {
            z[d] = a[d];
        }
    }
    
    return z;
}

float *sigmoid(float *a, int D) {
    int d;
    float *z = NULL;
    
    if((z = (float *)malloc((D + 1) * sizeof(float))) == NULL) {
        return NULL;
    }
    
    z[0] = 1.0f;
    
    for(d = 1; d <= D; d++) {
        z[d] = 1.0f / (1.0f + expf(- a[d]));
    }
    
    return z;
}

float *softmax(float *a, int K) {
    int k;
    float *y = NULL;
    float sum_exp = 0.0f;
    
    if((y = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
        return NULL;
    }

    y[0] = 0.0f;
    
    for(k = 1; k <= K; k++) {
        y[k] = expf(a[k]);
        sum_exp += y[k];
    }
    
    for(k = 1; k <= K; k++) {
        y[k] /= sum_exp;
    }
    
    return y;
}
