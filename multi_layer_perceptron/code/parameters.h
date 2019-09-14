#include <stdlib.h>
#include "utils.h"
#ifndef _STRUCT_H
    #include "struct.h"
#endif

float *x = NULL;
int *D = NULL;
float ***w = NULL;
float **z = NULL;
float **a = NULL;
float *y = NULL;
float *t = NULL;

float ***dE_dw = NULL;
float ***dE_dw_total = NULL;
float **dE_da = NULL;

void setup_parameters(MODEL_PARAMETER model_parameter);

void setup_parameters(MODEL_PARAMETER model_parameter) {
    int l, d, i, j, k;
    int L = model_parameter.L;
    int K = model_parameter.K;
    int dim_in = model_parameter.dim_in;
    
    initialize_seed();
    
    // Setup x
    if((x = (float *)malloc((model_parameter.dim_in + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }

    // Setup D
    model_parameter.N_unit[0] = dim_in;
    D = model_parameter.N_unit;
    
    // Setup w
    if((w = (float ***)malloc((L + 1) * sizeof(float **))) == NULL) {
        exit(-1);
    }
    
    for(l = 0; l <= L; l++) {
        int d_max = D[l];
        int d_next_layer = D[l + 1];
        
        float sigma = 2.0f / sqrtf((float)d_max);
        
        if((w[l] = (float **)malloc((d_max + 1) * sizeof(float *))) == NULL) {
            exit(-1);
        }
        
        for(i = 0; i <= d_max; i++) {
            if((w[l][i] = (float *)malloc((d_next_layer + 1) * sizeof(float))) == NULL) {
                exit(-1);
            }
            
            for(j = 0; j <= d_next_layer; j++) {
                // Xavier initialization
                w[l][i][j] = rand_normal(0.0f, sigma);
            }
        }
    }
    
    // Setup z
    if((z = (float **)malloc((L + 1) * sizeof(float *))) == NULL) {
        exit(-1);
    }
    
    for(l = 0; l <= L; l++) {
        int d_max = D[l];
        
        if((z[l] = (float *)malloc((d_max + 1) * sizeof(float))) == NULL) {
            exit(-1);
        }
        
        for(d = 0; d <= d_max; d++) {
            z[l][d] = 0.0f;
        }
    }
    
    // Setup a
    if((a = (float **)malloc((L + 2) * sizeof(float *))) == NULL) {
        exit(-1);
    }
    
    for(l = 0; l <= L; l++) {
        int d_max = D[l];
        
        if((a[l] = (float *)malloc((d_max + 1) * sizeof(float))) == NULL) {
            exit(-1);
        }
        
        for(d = 0; d <= d_max; d++) {
            a[l][d] = 0.0f;
        }
    }
    
    if((a[L + 1] = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    for(k = 0; k <= K; k++) {
        a[L + 1][k] = 0.0f;
    }
    
    // Setup y
    if((y = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    for(k = 0; k <= K; k++) {
        y[k] = 0.0f;
    }
    
    // Setup t
    if((t = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    for(k = 0; k <= K; k++) {
        t[k] = 0.0f;
    }
    
    // Setup dE_dw & dE_dw_average
    if((dE_dw = (float ***)malloc((L + 1) * sizeof(float **))) == NULL || (dE_dw_total = (float ***)malloc((L + 1) * sizeof(float **))) == NULL) {
        exit(-1);
    }
    
    for(l = 0; l <= L; l++) {
        int d_max = D[l];
        int d_next_layer = D[l + 1];
        
        if((dE_dw[l] = (float **)malloc((d_max + 1) * sizeof(float *))) == NULL || (dE_dw_total[l] = (float **)malloc((d_max + 1) * sizeof(float *))) == NULL) {
            exit(-1);
        }
        
        for(i = 0; i <= d_max; i++) {
            if((dE_dw[l][i] = (float *)malloc((d_next_layer + 1) * sizeof(float))) == NULL || (dE_dw_total[l][i] = (float *)malloc((d_next_layer + 1) * sizeof(float))) == NULL) {
                exit(-1);
            }
            
            for(j = 0; j <= d_next_layer; j++) {
                dE_dw[l][i][j] = 0.0f;
                dE_dw_total[l][i][j] = 0.0f;
            }
        }
    }
    
    // Setup dE_da
    if((dE_da = (float **)malloc((L + 2) * sizeof(float *))) == NULL) {
        exit(-1);
    }
    
    for(l = 0; l <= L; l++) {
        int d_max = D[l];
        
        if((dE_da[l] = (float *)malloc((d_max + 1) * sizeof(float))) == NULL) {
            exit(-1);
        }
        
        for(d = 0; d <= d_max; d++) {
            dE_da[l][d] = 0.0f;
        }
    }
    
    if((dE_da[L + 1] = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    for(k = 0; k <= K; k++) {
        dE_da[L + 1][k] = 0.0f;
    }
}
