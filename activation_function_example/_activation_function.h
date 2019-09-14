#include <stdlib.h>
#include <math.h>

float *step(float *a, int D);
float *ReLU(float *a, int D);
float *sigmoid(float *a, int D);
float *softmax(float *a, int K);
float *d_step(float *a, int D);
float *d_ReLU(float *a, int D);
float *d_sigmoid(float *a, int D);
float **d_softmax(float *a, int K);

float *step(float *a, int D) {
    int b = 0;
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


float *d_step(float *a, int D) {
    // a: never used
    int d;
    float *dz_da = NULL;
    
    if((dz_da = (float *)malloc((D + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    dz_da[0] = 0.0f;
    
    for(d = 1; d <= D; d++) {
        dz_da[d] = 0.0f;
    }
    
    return dz_da;
}

float *d_ReLU(float *a, int D) {
    int d;
    float *dz_da = NULL;
    
    if((dz_da = (float *)malloc((D + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    dz_da[0] = 0.0f;
    
    for(d = 1; d <= D; d++) {
        if(a[d] < 0.0f) {
            dz_da[d] = 0.0f;
        } else {
            dz_da[d] = 1.0f;
        }
    }
    
    return dz_da;
}

float *d_sigmoid(float *a, int D) {
    int d;
    float *z = NULL;
    float *dz_da = NULL;
    
    if((z = (float *)malloc((D + 1) * sizeof(float))) == NULL) {
        return NULL;
    }
    
    if((dz_da = (float *)malloc((D + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    z = sigmoid(a, D);
    
    dz_da[0] = 0.0f;
    
    for(d = 1; d <= D; d++) {
        dz_da[d] = z[d] * (1.0f - z[d]);
    }
    
    return dz_da;
}

float **d_softmax(float *a, int K) {
    int k, k_dash;
    float *y = NULL;
    float **dz_da = NULL;
    float sum_exp = 0.0f;
    float max_a = a[1];
    
    if((y = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
        return NULL;
    }
    
    if((dz_da = (float **)malloc((K + 1) * sizeof(float *))) == NULL) {
        return NULL;
    }
    
    dz_da[0] = NULL;
    
    for(k = 1; k <= K; k++) {
        if((dz_da[k] = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
            return NULL;
        }
    }
    
    for(k = 2; k <= K; k++) {
        if(a[k] > max_a) {
            max_a = a[k];
        }
    }
    
    for(k = 1; k <= K; k++) {
        y[k] = expf(a[k] - max_a);
        sum_exp += y[k];
    }
    
    for(k = 1; k <= K; k++) {
        y[k] /= sum_exp;
    }
    
    
    for(k = 1; k <= K; k++) {
        dz_da[k][0] = 0.0f;
        
        for(k_dash = 1; k_dash < k; k_dash++) {
            dz_da[k][k_dash] = -y[k] * y[k_dash];
            dz_da[k_dash][k] = -y[k] * y[k_dash];
        }
        dz_da[k][k] = y[k] * (1.0f - y[k]);
    }
    
    return dz_da;
}
