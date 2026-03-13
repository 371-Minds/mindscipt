#ifndef BN_TRANSFORMER_H
#define BN_TRANSFORMER_H

#include "model.h"

// Run one token through the transformer, returns pointer to logits
float *bn_transformer_forward(BnModel *m, int token, int pos);

#endif // BN_TRANSFORMER_H
