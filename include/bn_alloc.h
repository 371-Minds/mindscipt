#ifndef BN_ALLOC_H
#define BN_ALLOC_H

#include <stddef.h>

// Bring-your-own allocator vtable.
// Compatible with KlAllocator from keel — same signature, different prefix.
typedef struct {
    void *(*malloc)(void *ctx, size_t size);
    void *(*realloc)(void *ctx, void *ptr, size_t old_size, size_t new_size);
    void  (*free)(void *ctx, void *ptr, size_t size);
    void *ctx;
} BnAllocator;

// Return the default stdlib-backed allocator.
BnAllocator bn_allocator_default(void);

// Convenience wrappers.
void *bn_malloc(BnAllocator *a, size_t size);
void *bn_realloc(BnAllocator *a, void *ptr, size_t old_size, size_t new_size);
void  bn_free(BnAllocator *a, void *ptr, size_t size);

#endif // BN_ALLOC_H
