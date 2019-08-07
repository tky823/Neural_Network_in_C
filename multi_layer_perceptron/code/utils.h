#include <stdlib.h>
#include <math.h>
#define _UTILS_H

void initialize_seed(void);
float rand_uniform(float a, float b);
float rand_normal(float mu, float sigma);

void initialize_seed(void) {
    srand(111);
}

float rand_uniform(float a, float b) {
    float x = ((float)rand() + 1.0)/((float)RAND_MAX + 2.0);
    return (b - a) * x + a;
}

float rand_normal(float mu, float sigma) {
    float z = sqrtf(- 2.0 * logf(rand_uniform(0.0f, 1.0f))) * sinf(2.0 * M_PI * rand_uniform(0.0f, 1.0f));
    return mu + sigma * z;
}
