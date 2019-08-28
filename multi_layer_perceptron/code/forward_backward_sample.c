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
void backward(MODEL_PARAMETER model_parameter);
void update_parameters(MODEL_PARAMETER model_parameter, float learning_rate);
DATA *loda_data(char *file_path, int *N_data);

int main(int argc, char **argv) {
    int id;
    int k, n, N_train;
    char *file_path = NULL;
    int print_frequency = 100;
    float LEARNING_RATE = 0.01f;
    float E;
    
    DATA *train_data = NULL;
    
    if(argc < 2) {
        printf("Specify the data path!\n");
        return -1;
    }
    
    file_path = argv[1];
    train_data = loda_data(file_path, &N_train);
    
    MODEL_PARAMETER model_parameter = build_model();
    
    for(n = 1; n <= N_train; n++) {
        id = rand() % N_train + 1;
        x = train_data[id].x;
        
        for(k = 1; k <= model_parameter.K; k++) {
            t[k] = 0.0f;
        }
        
        t[train_data[id].t] = 1.0f;
        
        forward(model_parameter);
        E = model_parameter.loss(y, t, model_parameter.K, 0, NULL);
        
        if(n % print_frequency == 0) {
            printf("%d / %d E: %f\n", n, N_train, E);
        }
        
        backward(model_parameter);
        update_parameters(model_parameter, LEARNING_RATE);
    }
    
    return 0;
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

void backward(MODEL_PARAMETER model_parameter) {
    int l, i, j, k, k_dash;
    int L = model_parameter.L;
    int K = model_parameter.K;
    
    float *dE_dy, **dy_da, **dz_da;
    
    // Setup dE_dy
    if((dE_dy = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    for(k = 0; k <= K; k++) {
        dE_dy[k] = 0.0f;
    }
    
    // Calculate dE/dy
    model_parameter.loss(y, t, K, 1, dE_dy);
    
    // Setup dy_da
    if((dy_da = (float **)malloc((K + 1) * sizeof(float *))) == NULL) {
        exit(-1);
    }
    
    for(k = 1; k <= K; k++) {
        if((dy_da[k] = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
            exit(-1);
        }
    }
    
    /* ========== output layer ========== */
    // Calculate dy/da
    model_parameter.activation[L + 1](a[L + 1], K, 1, dy_da);
    
    for(k = 1; k <= K; k++) {
        float tmp_dE_da = 0.0f;
        
        for(k_dash = 1; k_dash <= K; k_dash++) {
            tmp_dE_da += dE_dy[k_dash] * dy_da[k_dash][k];
        }
        
        dE_da[L + 1][k] = tmp_dE_da;
    }
    
    // Free dE_dy & dy_da
    free(dE_dy);
    
    for(k = 1; k <= K; k++) {
        free(dy_da[k]);
    }
    
    free(dy_da);
    
    /* ========== hidden layer ========== */
    for(l = L; l >= 1; l--) {
        int d_max = D[l];
        int d_next_layer = D[l + 1];
        float z_l_i = 0.0f; // z[l][i]
        
        // Calculate dE/dw
        for(i = 0; i <= d_max; i++) {
            z_l_i = z[l][i];
            for(j = 0; j <= d_next_layer; j++) {
                dE_dw[l][i][j] = z_l_i * dE_da[l + 1][j];
            }
        }
        
        // Setup dz_da
        if((dz_da = (float **)malloc((d_max + 1) * sizeof(float *))) == NULL) {
            exit(-1);
        }
        
        for(i = 0; i <= d_max; i++) {
            if((dz_da[i] = (float *)malloc((d_max + 1) * sizeof(float))) == NULL) {
                exit(-1);
            }
            
            for(j = 0; j <= d_max + 1; j++) {
                dz_da[i][j] = 0.0f;
            }
        }
        
        // Calculate dz/da
        model_parameter.activation[l](a[l], d_max, 1, dz_da);
        
        // Calculate dE/da
        for(i = 1; i <= d_max; i++) {
            float tmp_dE_da = 0.0f;
            
            for(j = 1; j <= d_next_layer; j++) {
                tmp_dE_da += w[l][i][j] * dE_da[l + 1][j];
            }
            
            dE_da[l][i] = dz_da[i][i] * tmp_dE_da;
        }
        
        // Free dz_da
        for(i = 0; i <= d_max; i++) {
            free(dz_da[i]);
        }
        
        free(dz_da);
    }
    
    /* ========== input layer ========== */
    // Calculate dE/dw
    int d_max = model_parameter.dim_in;
    int d_next_layer = D[1];
    float z_0_i = 0.0f; // z[0][i]
    
    for(i = 0; i <= d_max; i++) {
        z_0_i = z[0][i];
        for(j = 0; j <= d_next_layer; j++) {
            dE_dw[0][i][j] = z_0_i * dE_da[1][j];
        }
    }
}

void update_parameters(MODEL_PARAMETER model_parameter, float learning_rate) {
    int l, i, j;
    int L = model_parameter.L;
    
    for(l = 0; l <= L; l++) {
        int d_max = D[l];
        int d_next_layer = D[l + 1];
        
        for(i = 1; i <= d_max; i++) {
            for(j = 0; j <= d_next_layer; j++) {
                w[l][i][j] -= learning_rate * dE_dw[l][i][j];
            }
        }
    }
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
