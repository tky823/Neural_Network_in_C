#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "_activation_function.h"

void print_vector(float *z, int D);

int main(void) {
    int d;
    int D = 5;
    float *a = NULL;
    float *z = NULL;
    
    if((a = (float *)malloc((D + 1) * sizeof(float))) == NULL) {
        exit(-1);
    }
    
    a[0] = 0.0f;
    
    for(d = 1; d <= D; d++) {
        a[d] = d - 3.0f;
    }
    // a: {(0.0,) -2.0, -1.0, 0.0, 1.0, 2.0}
    
    // step function
    z = step(a, D);
    
    if(z == NULL) {
        exit(-1);
    }
    
    printf("=====step function=====\n");
    print_vector(z, D);
    printf("\n");
    free(z);
    
    // sigmoid function
    z = sigmoid(a, D);
    
    if(z == NULL) {
        exit(-1);
    }
    
    printf("=====sigmoid function=====\n");
    print_vector(z, D);
    printf("\n");
    free(z);
    
    // ReLU function
    z = ReLU(a, D);
    
    if(z == NULL) {
        exit(-1);
    }
    
    printf("=====ReLU function=====\n");
    print_vector(z, D);
    printf("\n");
    free(z);
    
    // softmax function
    z = softmax(a, D);
    
    if(z == NULL) {
        exit(-1);
    }
    
    printf("=====softmax function=====\n");
    print_vector(z, D);
    printf("\n");
    free(z);
    
    return 0;
}

void print_vector(float *z, int D) {
    int d;
    
    printf("[\n");
    
    for(d = 1; d <= D; d++) {
        printf("\t%f\n", z[d]);
    }
    
    printf("]\n");
}

