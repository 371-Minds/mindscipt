#include "bn_alloc.h"
#include <stdlib.h>

static void *stdlib_malloc(void *ctx, size_t size) {
    (void)ctx;
    return malloc(size);
}

static void *stdlib_realloc(void *ctx, void *ptr, size_t old_size, size_t new_size) {
    (void)ctx;
    (void)old_size;
    return realloc(ptr, new_size);
}

static void stdlib_free(void *ctx, void *ptr, size_t size) {
    (void)ctx;
    (void)size;
    free(ptr);
}

BnAllocator bn_allocator_default(void) {
    return (BnAllocator){
        .malloc  = stdlib_malloc,
        .realloc = stdlib_realloc,
        .free    = stdlib_free,
        .ctx     = NULL,
    };
}

void *bn_malloc(BnAllocator *a, size_t size) {
    return a->malloc(a->ctx, size);
}

void *bn_realloc(BnAllocator *a, void *ptr, size_t old_size, size_t new_size) {
    return a->realloc(a->ctx, ptr, old_size, new_size);
}

void bn_free(BnAllocator *a, void *ptr, size_t size) {
    a->free(a->ctx, ptr, size);
}
