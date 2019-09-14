#include <stdio.h>
#ifndef _STRUCT_H
    #include "struct.h"
#endif
#include "parameters.h"
#include "activation_function.h"
#include "loss_function.h"

#define DATA_LINE 5

MODEL_PARAMETER build_model(void);
void forward(MODEL_PARAMETER model_parameter);
DATA *loda_data(char *file_path, int *N_data);

int main(int argc, char **argv) {
    int id, N;
    int k;
    char *file_path = NULL;
    float E;
    
    DATA *train_data = NULL;
    
    if(argc < 2) {
        printf("Specify the data path!\n");
        return -1;
    }
    
    initialize_seed();
    
    file_path = argv[1];
    train_data = loda_data(file_path, &N);
    
    MODEL_PARAMETER model_parameter = build_model();
    
    id = 1; //1st data
    x = train_data[id].x;
    
    for(k = 1; k <= model_parameter.K; k++) {
        t[k] = 0.0f;
    }
    
    t[train_data[id].t] = 1.0f;
    
    //reset_dE_dw_total(model_parameter);
    forward(model_parameter);
    E = model_parameter.loss(y, t, model_parameter.K, 0, NULL);
    
    printf("E: %f\n", E);
    
    return 0;
}

DATA *loda_data(char *file_path, int *N_data) {
    int d, n, N;
    FILE *fp = NULL;
    DATA *data = NULL;
    
    if((fp = fopen(file_path, "r")) == NULL) {
        exit(-1);
    }
    
    fscanf(fp, "%d", &N);
    
    if((data = (DATA *)malloc((N + 1) * sizeof(DATA))) == NULL) {
        exit(-1);
    }
    
    for(n = 1; n <= N; n++) {
        if((data[n].x = (float *)malloc((DATA_LINE * DATA_LINE) * sizeof(float))) == NULL) {
            exit(-1);
        }
        
        fscanf(fp, "%d", &(data[n].t));
        
        for(d = 1; d <= DATA_LINE * DATA_LINE; d++) {
            fscanf(fp, "%f", &(data[n].x[d]));
        }
        
    }
    
    fclose(fp);
    
    *N_data = N;
    
    return data;
}

MODEL_PARAMETER build_model(void){
    int l;
    MODEL_PARAMETER model_parameter;
    
    model_parameter.L = 2;
    model_parameter.K = 3;
    model_parameter.dim_in = DATA_LINE * DATA_LINE;
    
    model_parameter = setup_model_parameter(model_parameter);
    
    for(l = 1; l <= model_parameter.L; l++) {
        model_parameter.N_unit[l] = 16;
    }
    
    for(l = 1; l <= model_parameter.L; l++) {
        //ReLU, sigmoid, ...
        model_parameter.activation[l] = ReLU;
    }
    
    model_parameter.activation[model_parameter.L + 1] = softmax;
    model_parameter.loss = cross_entropy;
    
    setup_parameters(model_parameter);
    
    return model_parameter;
}

void forward(MODEL_PARAMETER model_parameter) {
    int l, i, j, k;
    int L = model_parameter.L;
    int K = model_parameter.K;
    float tmp_a;
    
    // layer: input layer
    z[0] = x;
    
    // layer: hidden layer
    for(l = 1; l <= L; l++) {
        int l_prev = l - 1;
        int d_prev_layer = D[l_prev];
        int d_max = D[l];
        
        for(j = 1; j <= d_max; j++) {
            tmp_a = w[l_prev][0][j]; //bias
            for(i = 1; i <= d_prev_layer; i++) {
                tmp_a += w[l_prev][i][j] * z[l_prev][i];
            }
            a[l][j] = tmp_a;
        }
        
        free(z[l]);
        
        z[l] = model_parameter.activation[l](a[l], D[l], 0, NULL);
        // z[l][0] = 1.0f;
    }
    
    // layer: output layer
    int l_prev = L;
    int d_prev_layer = D[l_prev];
    
    for(k = 1; k <= K; k++) {
        tmp_a = w[L][0][k];
        
        for(i = 0; i <= d_prev_layer; i++) {
            tmp_a += w[L][i][k] * z[L][i];
        }
        
        a[L + 1][k] = tmp_a;
    }
    
    free(y);
    y = model_parameter.activation[L + 1](a[L + 1], K, 0, NULL);
}
