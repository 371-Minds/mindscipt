#ifndef BN_PROMPT_CACHE_H
#define BN_PROMPT_CACHE_H

#include "model.h"
#include "bn_alloc.h"

// Forward declaration — full definition in session.h
typedef struct BnSession BnSession;

#define BN_PROMPT_CACHE_MAX_ENTRIES 16

// A cached KV prefix snapshot.
typedef struct {
    int *tokens;            // [n_tokens] token IDs (owned)
    int n_tokens;           // number of cached prefix tokens
    uint64_t hash;          // FNV-1a hash for fast reject
    void *key_cache;        // [n_attn_layers * n_tokens * kv_dim] compact KV data
    void *value_cache;      // same layout
    size_t kv_bytes;        // bytes per cache (key or value)
    int kv_f16;             // FP16 or FP32
    int n_attn_layers;      // for config validation
    int kv_dim;             // for config validation
} BnPromptCacheEntry;

// Prompt cache: stores KV prefix snapshots keyed by token sequences.
// Supports longest-prefix matching for efficient KV reuse.
typedef struct {
    BnPromptCacheEntry entries[BN_PROMPT_CACHE_MAX_ENTRIES];
    int n_entries;
    size_t max_bytes;       // eviction threshold (0 = no limit)
    size_t used_bytes;      // sum of all entries' kv_bytes * 2 + token bytes
    BnAllocator alloc;      // allocator for entry buffers
#if !defined(__EMSCRIPTEN__)
    void *mtx;              // pthread_mutex_t* (opaque, NULL on single-threaded)
#endif
} BnPromptCache;

// Create a prompt cache with a memory budget.
// max_bytes: eviction threshold for KV data (0 = no limit).
// alloc: allocator for all internal buffers (NULL = stdlib default).
BnPromptCache *bn_prompt_cache_create(size_t max_bytes, BnAllocator *alloc);

// Free the prompt cache and all entries.
void bn_prompt_cache_free(BnPromptCache *cache);

// Store the current session's KV prefix into the cache.
// tokens/n_tokens: the token sequence that produced this KV state.
// Returns 0 on success, -1 if hybrid model (SSM layers), -2 on alloc failure.
int bn_prompt_cache_store(BnPromptCache *cache, const BnModel *model,
                          const BnSession *session,
                          const int *tokens, int n_tokens);

// Restore the longest matching prefix into a session.
// tokens/n_tokens: the full prompt token sequence to match against.
// On hit: copies KV data into session's cache, sets session->pos.
// Returns number of tokens restored (0 on miss, minimum match is 1 token).
int bn_prompt_cache_restore(BnPromptCache *cache, const BnModel *model,
                            BnSession *session,
                            const int *tokens, int n_tokens);

// Evict all entries.
void bn_prompt_cache_clear(BnPromptCache *cache);

// Stats.
int bn_prompt_cache_count(const BnPromptCache *cache);
size_t bn_prompt_cache_bytes(const BnPromptCache *cache);

#endif // BN_PROMPT_CACHE_H
