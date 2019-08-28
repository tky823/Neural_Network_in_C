#include <stdlib.h>

#define _STRUCT_H

typedef struct {
    int L; //number of hidden layers
    int K; //number of classes
    int dim_in; //input dimension
    int *N_unit; //number of units
    float *(**activation)(float *a, int D, int derivative, float **derivative_matrix); // activation function
    float (*loss)(float *y, float *t, int K, int derivative, float *dE_dy); //loss function
} MODEL_PARAMETER;

typedef struct {
    float *x; // input
    int t; // label
} DATA;

typedef struct {
    float E_total;
    float E_average;
    int true_count;
    float accuracy;
} METRICS;

MODEL_PARAMETER setup_model_parameter(MODEL_PARAMETER model_parameter) {
    // L, K has been decided
    
    if((model_parameter.N_unit = (int *)malloc((model_parameter.L + 1) * sizeof(int))) == NULL) {
        exit(-1);
    }
    
    model_parameter.N_unit[0] = model_parameter.dim_in;
    model_parameter.N_unit[model_parameter.L + 1] = model_parameter.K;
    
    if((model_parameter.activation = (float * (**)(float *, int,  int,  float **))malloc((model_parameter.L + 2) * sizeof(float (**)(float *, int,  int,  float **)))) == NULL) {
        exit(-1);
    }
    model_parameter.activation[0] = NULL;
    
    model_parameter.loss = NULL;
    
    return model_parameter;
}
