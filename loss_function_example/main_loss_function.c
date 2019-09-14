#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "_loss_function.h"

int main(void) {
    int D = 9, K = 3;
    float *y = NULL;
    float *t = NULL;
    float E;
    
    // mean squared error
    printf("=====mean squared error=====\n");
    
    if((y = (float *)malloc((D + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    if((t = (float *)malloc((D + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    y[0] = 0.0f; t[0] = 0.0f;
    y[1] = 0.0f; t[1] = 0.0f;
    y[2] = 1.0f; t[2] = 1.0f;
    y[3] = 4.0f; t[3] = 3.0f;
    y[4] = -2.0f; t[4] = -2.0f;
    y[5] = 1.0f; t[5] = 0.0f;
    y[6] = 1.0f; t[6] = 1.0f;
    y[7] = 2.0f; t[7] = 4.0f;
    y[8] = -1.0f; t[8] = -1.0f;
    y[9] = -2.0f; t[9] = -1.0f;
    
    // y: {(0.0,) 0.0, 1.0, 4.0, -2.0, 1.0, 1.0, 2.0, -1.0, -2.0}
    // t: {(0.0,) 0.0, 1.0, 3.0, -2.0, 0.0, 1.0, 4.0, -1.0, -1.0}
    
    E = mean_squared_error(y, t, D);
    printf("E: %f\n\n", E);
    
    free(y);
    free(t);
    
    // cross entropy
    printf("=====cross entropy=====\n");
    
    if((y = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    if((t = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    y[0] = 0.0f; t[0] = 0.0f;
    y[1] = 0.5f; t[1] = 1.0f;
    y[2] = 0.4f; t[2] = 0.0f;
    y[3] = 0.1f; t[3] = 0.0f;
    
    // y: {(0.0,) 0.0, 1.0, 4.0, -2.0, 1.0, 1.0, 2.0, -1.0, -2.0}
    // t: {(0.0,) 0.0, 1.0, 3.0, -2.0, 0.0, 1.0, 4.0, -1.0, -1.0}
    
    E = cross_entropy(y, t, K);
    printf("E: %f\n\n", E);
    
    return 0;
}
