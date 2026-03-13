#ifndef BN_SAMPLER_H
#define BN_SAMPLER_H

#include <stdint.h>

typedef struct {
    int      vocab_size;
    float    temperature;
    float    topp;
    uint64_t rng_state;
} BnSampler;

void bn_sampler_init(BnSampler *s, int vocab_size, float temp, float topp, uint64_t seed);
int  bn_sampler_sample(BnSampler *s, float *logits);

#endif // BN_SAMPLER_H
