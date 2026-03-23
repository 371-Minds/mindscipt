#ifndef BN_GENERATE_H
#define BN_GENERATE_H

#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "bn_alloc.h"

// Callback for streaming token output. Return non-zero to stop generation.
typedef int (*bn_token_callback)(const char *piece, int token_id, void *user_data);

// Chat template format
typedef enum {
    BN_CHAT_AUTO,    // auto-detect from tokenizer (ChatML if im_start/im_end present, else LLaMA)
    BN_CHAT_CHATML,  // <|im_start|>role\n{content}<|im_end|>\n
    BN_CHAT_LLAMA,   // Role: {content}<|eot_id|>
    BN_CHAT_RAW,     // no wrapping — encode content directly (caller handles template)
} BnChatFormat;

// Chat message roles
typedef enum {
    BN_ROLE_SYSTEM,
    BN_ROLE_USER,
    BN_ROLE_ASSISTANT,
} BnChatRole;

// A single chat message (role + content).
typedef struct {
    BnChatRole role;
    const char *content;
} BnChatMessage;

// Stop string configuration for generation.
typedef struct {
    const char **strings;  // array of stop strings (NULL-terminated content, not token IDs)
    int n;                 // number of stop strings
} BnStopStrings;

// Generate tokens autoregressively from pre-computed logits.
// The model must have logits ready (from bn_prefill or bn_transformer_forward).
// pos is updated to reflect the new position after generation.
// stop: optional stop strings (NULL to disable). Generation halts when any
//       stop string appears in the output. The stop string is NOT included
//       in the callback output.
// alloc is used for internal scratch buffers (NULL = stdlib default).
// Returns: number of tokens generated, -1 on loop detected, -2 on error,
//          -3 on stop string match.
int bn_generate(BnModel *model, BnTokenizer *tok, BnSampler *sampler,
                int max_tokens, int *pos,
                bn_token_callback cb, void *user_data,
                const BnStopStrings *stop,
                BnAllocator *alloc);

// Speculative decoding: draft K tokens with small model, verify with target.
// Both models must have logits ready from prefill. Greedy only (temperature=0).
// alloc is used for verify_logits buffer (NULL = stdlib default).
// pos is updated. Returns: n_generated, -1 on loop, -2 on error.
int bn_generate_speculative(BnModel *target, BnModel *draft, int draft_k,
                            BnTokenizer *tok, BnSampler *sampler,
                            int max_tokens, int *pos,
                            bn_token_callback cb, void *user_data,
                            BnAllocator *alloc);

// Prefill prompt tokens through the model. Returns logits for the last token,
// or NULL on error. pos is set to pos0 + n_tokens after return.
// If no_prefill is set, runs tokens one at a time (for debugging).
float *bn_prefill(BnModel *model, const int *tokens, int n_tokens,
                  int pos0, int no_prefill);

// Encode text into tokens. Returns number of tokens written.
// alloc is used for scratch buffer (NULL = stdlib default).
int bn_count_tokens(const BnTokenizer *tok, const char *text,
                    BnAllocator *alloc);

// Format a single user message into a chat turn (legacy convenience wrapper).
// fmt=BN_CHAT_AUTO uses tokenizer's detected format.
// alloc is used for message formatting buffer (NULL = stdlib default).
// Writes encoded tokens into out_tokens[0..max_tokens-1].
// Returns number of tokens written.
int bn_chat_format_turn(const BnTokenizer *tok, BnChatFormat fmt,
                        const char *user_msg,
                        int *out_tokens, int max_tokens,
                        BnAllocator *alloc);

// Format a multi-turn conversation into tokens.
// Encodes all messages in order, appends assistant prompt at the end.
// fmt=BN_CHAT_AUTO uses tokenizer's detected format.
// alloc is used for formatting buffers (NULL = stdlib default).
// Writes encoded tokens into out_tokens[0..max_tokens-1].
// Returns number of tokens written.
int bn_chat_format_messages(const BnTokenizer *tok, BnChatFormat fmt,
                            const BnChatMessage *messages, int n_messages,
                            int *out_tokens, int max_tokens,
                            BnAllocator *alloc);

// Return the end-of-turn token ID for the given format.
// Used to feed into KV cache after assistant response completes.
// Returns -1 if no end-of-turn token exists for the format.
int bn_chat_turn_end_id(const BnTokenizer *tok, BnChatFormat fmt);

#endif // BN_GENERATE_H
