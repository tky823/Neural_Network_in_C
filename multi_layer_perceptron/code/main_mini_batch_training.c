#include <stdio.h>
#ifndef _STRUCT_H
    #include "struct.h"
#endif
#include "parameters.h"
#include "activation_function.h"
#include "loss_function.h"

#define DATA_LINE 5

MODEL_PARAMETER build_model(void);
METRICS train_per_batch(MODEL_PARAMETER model_parameter, float learning_rate, int n_max, DATA *data, int N_train);
METRICS validate_per_batch(MODEL_PARAMETER model_parameter, DATA *validation_data, int N_validation);
METRICS test_per_batch(MODEL_PARAMETER model_parameter, DATA *test_data, int N_test);
void forward(MODEL_PARAMETER model_parameter);
void backward(MODEL_PARAMETER model_parameter);
void update_parameters(MODEL_PARAMETER model_parameter, float learning_rate, int batch_size);
DATA *loda_data(char *file_path, int *N_data);
void reset_dE_dw_total(MODEL_PARAMETER model_parameter);
void show_probability(MODEL_PARAMETER model_parameter);

int main(int argc, char **argv) {
    int i;
    int iteration, epoch;
    int N_train, N_validation, N_test;
    int EPOCH = 10;
    int BATCH_SIZE = 32;
    float LEARNING_RATE = 0.01f;
    char *file_path = NULL;
    
    METRICS metrics;
    
    DATA *train_data = NULL;
    DATA *validation_data = NULL;
    DATA *test_data = NULL;
    
    if(argc < 4) {
        printf("Specify the data path!\n");
        return -1;
    }
    
    file_path = argv[1];
    train_data = loda_data(file_path, &N_train);
    
    file_path = argv[2];
    validation_data = loda_data(file_path, &N_validation);
    
    file_path = argv[3];
    test_data = loda_data(file_path, &N_test);
    
    MODEL_PARAMETER model_parameter = build_model();
    
    iteration = (N_train - 1) / BATCH_SIZE;
    
    for(epoch = 1; epoch <= EPOCH; epoch++) {
        printf("Epoch: %d / %d\n", epoch, EPOCH);
        
        int n_max = BATCH_SIZE;
        float E_total = 0.0f;
        int true_count = 0;
        
        for(i = 1; i <= iteration; i++) {
            metrics = train_per_batch(model_parameter, LEARNING_RATE, n_max, train_data, N_train);
            E_total += metrics.E_total;
            true_count += metrics.true_count;
        }
        
        n_max = N_train - iteration * BATCH_SIZE;
        metrics = train_per_batch(model_parameter, LEARNING_RATE, n_max, train_data, N_train);
        E_total += metrics.E_total;
        true_count += metrics.true_count;
        
        printf("\tloss: %f, accuracy: %f", E_total / (float) N_train, true_count / (float) N_train);

        metrics = validate_per_batch(model_parameter, validation_data, N_validation);
        
        printf("\tvalidation loss: %f, validation accuracy: %f\n", metrics.E_total / (float) N_validation, metrics.true_count / (float) N_validation);
    }
    
    metrics = test_per_batch(model_parameter, test_data, N_test);
    
    printf("Test\n");
    printf("\taccuracy: %f\n", metrics.true_count / (float) N_test);
    
    return 0;
    
}

MODEL_PARAMETER build_model(void) {
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

METRICS train_per_batch(MODEL_PARAMETER model_parameter, float learning_rate, int batch_size, DATA *train_data, int N_train) {
    int k, n;
    int id;
    METRICS metrics;
    
    metrics.E_total = 0.0f;
    metrics.E_average = 0.0f;
    metrics.true_count = 0;
    metrics.accuracy = 0.0f;
    
    if(batch_size <= 0) {
        return metrics;
    }
    
    reset_dE_dw_total(model_parameter);
    
    for(n = 1; n <= batch_size; n++) {
        id = rand() % N_train + 1;
        x = train_data[id].x;
        
        for(k = 1; k <= model_parameter.K; k++) {
            t[k] = 0.0f;
        }
        t[train_data[id].t] = 1.0f;
        forward(model_parameter);
        metrics.E_total += model_parameter.loss(y, t, model_parameter.K, 0, NULL);
        
        int k_estimation = 1;
        float p_max = y[k_estimation];
        
        for(k = 2; k <= model_parameter.K; k++) {
            if(y[k] > p_max) {
                k_estimation = k;
                p_max = y[k_estimation];
            }
        }
        
        if(t[k_estimation] == 1.0f) {
            metrics.true_count += 1;
        }
        
        backward(model_parameter);
    }
    
    update_parameters(model_parameter, learning_rate, batch_size);
    
    metrics.E_average = metrics.E_total / (float) batch_size;
    metrics.accuracy = metrics.true_count / (float) batch_size;
    
    return metrics;
}

METRICS validate_per_batch(MODEL_PARAMETER model_parameter, DATA *validation_data, int N_validation) {
    int n, k;
    METRICS metrics;
    
    metrics.E_total = 0.0f;
    metrics.E_average = 0.0f;
    metrics.true_count = 0;
    metrics.accuracy = 0.0f;
    
    for(n = 1; n <= N_validation; n++) {
        x = validation_data[n].x;
        
        for(k = 1; k <= model_parameter.K; k++) {
            t[k] = 0.0f;
        }
        
        t[validation_data[n].t] = 1.0f;
        
        forward(model_parameter);
        metrics.E_total += model_parameter.loss(y, t, model_parameter.K, 0, NULL);
        
        int k_estimation = 1;
        float p_max = y[k_estimation];
        
        for(k = 2; k <= model_parameter.K; k++) {
            if(y[k] > p_max) {
                k_estimation = k;
                p_max = y[k_estimation];
            }
        }
        
        if(t[k_estimation] == 1.0f) {
            metrics.true_count += 1;
        }
    }
    
    metrics.E_average = metrics.E_total / (float) N_validation;
    metrics.accuracy = metrics.true_count / (float) N_validation;
    
    return metrics;
}

METRICS test_per_batch(MODEL_PARAMETER model_parameter, DATA *test_data, int N_test) {
    int n, k;
    METRICS metrics;
    
    metrics.E_total = 0.0f;
    metrics.E_average = 0.0f;
    metrics.true_count = 0;
    metrics.accuracy = 0.0f;
    
    for(n = 1; n <= N_test; n++) {
        x = test_data[n].x;
        
        for(k = 1; k <= model_parameter.K; k++) {
            t[k] = 0.0f;
        }
        
        t[test_data[n].t] = 1.0f;
        
        forward(model_parameter);
        metrics.E_total += model_parameter.loss(y, t, model_parameter.K, 0, NULL);
        
        int k_estimation = 1;
        float p_max = y[k_estimation];
        
        for(k = 2; k <= model_parameter.K; k++) {
            if(y[k] > p_max) {
                k_estimation = k;
                p_max = y[k_estimation];
            }
        }
        
        if(t[k_estimation] == 1.0f) {
            metrics.true_count += 1;
        }
    }
    
    metrics.E_average = metrics.E_total / (float) N_test;
    metrics.accuracy = metrics.true_count / (float) N_test;
    
    return metrics;
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
                dE_dw_total[l][i][j] += dE_dw[l][i][j];
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
    int z_0_i = 0.0f; // z[0][i]
    
    for(i = 0; i <= d_max; i++) {
        z_0_i = z[0][i];
        for(j = 0; j <= d_next_layer; j++) {
            dE_dw[0][i][j] = z_0_i * dE_da[1][j];
            dE_dw_total[0][i][j] += dE_dw[0][i][j];
        }
    }
}

void update_parameters(MODEL_PARAMETER model_parameter, float learning_rate, int batch_size) {
    int l, i, j;
    int L = model_parameter.L;
    
    for(l = 0; l <= L; l++) {
        int d_max = D[l];
        int d_next_layer = D[l + 1];
        
        for(i = 1; i <= d_max; i++) {
            for(j = 0; j <= d_next_layer; j++) {
                w[l][i][j] -= learning_rate * (dE_dw_total[l][i][j] / (float) batch_size);
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

void reset_dE_dw_total(MODEL_PARAMETER model_parameter) {
    int l, i, j;
    int L = model_parameter.L;
    
    for(l = 0; l <= L; l++) {
        int d_max = D[l];
        int d_next_layer = D[l + 1];
        
        for(i = 0; i <= d_max; i++) {
            for(j = 0; j <= d_next_layer; j++) {
                dE_dw_total[l][i][j] = 0.0f;
            }
        }
    }
}

void show_probability(MODEL_PARAMETER model_parameter) {
    int k;
    
    for(k = 1; k <= model_parameter.K; k++) {
        printf("class %d: %f ", k, y[k]);
    }
    printf("\n");
}
