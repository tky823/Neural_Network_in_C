#include <stdlib.h>
#include <math.h>

float *step(float *a, int D, int derivative, float **derivative_matrix);
float *ReLU(float *a, int D, int derivative, float **derivative_matrix);
float *sigmoid(float *a, int D, int derivative, float **derivative_matrix);
float *softmax(float *a, int K, int derivative, float **derivative_matrix);

float *step(float *a, int D, int derivative, float **derivative_matrix) {
    int d, d_dash;
    float *z = NULL;
    
    if(derivative) {
        for(d = 1; d <= D; d++) {
            for(d_dash = 1; d_dash <= D; d_dash++) {
                derivative_matrix[d][d_dash] = 0.0f;
            }
        }
        return NULL;
    }
    
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

float *ReLU(float *a, int D, int derivative, float **derivative_matrix) {
    int d, d_dash;
    float *z = NULL;

    if(derivative) {
        for(d = 1; d <= D; d++) {
            for(d_dash = 1; d_dash <= D; d_dash++) {
                derivative_matrix[d][d_dash] = 0.0f;
            }
        }
        
        for(d = 1; d <= D; d++) {
            if(a[d] > 0.0f) {
                derivative_matrix[d][d] = 1.0f;
            }
        }
        
        return NULL;
    }
    
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

float *sigmoid(float *a, int D, int derivative, float **derivative_matrix) {
    int d, d_dash;
    float *z = NULL;
    
    if((z = (float *)malloc((D + 1) * sizeof(float))) == NULL) {
        return NULL;
    }
    
    z[0] = 1.0f;
    
    for(d = 1; d <= D; d++) {
        z[d] = 1.0f / (1.0f + expf(- a[d]));
    }
    
    if(derivative) {
        for(d = 1; d <= D; d++) {
            for(d_dash = 1; d_dash <= D; d_dash++) {
                derivative_matrix[d][d_dash] = 0.0f;
            }
        }
        
        for(d = 1; d <= D; d++) {
            derivative_matrix[d][d] = z[d] * (1.0f - z[d]);
        }
        
        return NULL;
    }
    
    return z;
}

float *softmax(float *a, int K, int derivative, float **derivative_matrix) {
    int k, k_dash;
    float *y = NULL;
    float sum_exp = 0.0f;
    float max_a = a[1];
    
    if((y = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
        return NULL;
    }

    for(k = 2; k <= K; k++) {
        if(a[k] > max_a) {
            max_a = a[k];
        }
    }
    
    y[0] = 1.0f;
    
    for(k = 1; k <= K; k++) {
        y[k] = expf(a[k] - max_a);
        sum_exp += y[k];
    }
    
    for(k = 1; k <= K; k++) {
        y[k] /= sum_exp;
        
    }
    
    if(derivative) {
        for(k = 1; k <= K; k++) {
            for(k_dash = 1; k_dash < k; k_dash++) {
                derivative_matrix[k][k_dash] = -y[k] * y[k_dash];
                derivative_matrix[k_dash][k] = -y[k] * y[k_dash];
            }
            derivative_matrix[k][k] = y[k] * (1.0f - y[k]);
        }
        
        return NULL;
    }
    
    return y;
}
