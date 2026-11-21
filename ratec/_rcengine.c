/* _rcengine.c — RateC tokenizer engine (the library core).
 *
 * Semantics (bit-exact with the validated reference pipeline):
 *   judge   = fixed-weight 6-order KT-mixture byte surprisal  (cold start)
 *   patcher = greedy under-shoot at C bits (searchsorted side='right')
 * Bit-exactness discipline: every double expression keeps the reference
 * shape; build with -ffp-contract=off -fno-fast-math; glibc log2 (same as
 * CPython's math.log2).
 *
 * Library features over the bench kernel (ratec_fast.c):
 *   - reusable stateful engine: judge tables + patcher state persist across
 *     calls, so feed(a); feed(b); flush() == one-shot(a+b) exactly
 *   - patch book: dynamic content->id assignment (first-seen order) with
 *     decode support; optional (off for bulk LM data prep)
 *   - provably exact patcher restore: on close, the new patch budget limit
 *     is lim = cs + C with the running double cs itself (the bench kernel
 *     recomputed cs = cnext - sf, a float roundtrip that is empirically but
 *     not provably exact)
 *   - selectable C (1..64, fractional), selectable prefetch depth (0..4)
 *   - selectable patcher: reference under-shoot or credit carry-over (v0.4)
 *   - batch encoding across threads (GIL released, one engine per doc)
 *   - snapshot save/load: full state (judge + patcher + book) <-> bytes
 *   - bounded-memory mode: map cap bits default 2^22 slots; beyond that the
 *     maps stay in place (load factor rises, throughput degrades gracefully)
 *
 * A hot byte costs up to ~96 B of physical table memory in the orders-3..5
 * maps (3 new contexts/byte worst case x 32 B slots); for multi-MB streams
 * prefer chunked engines or accept the cap and graceful slowdown.
 *
 * Output spans are uint32 byte offsets: one engine covers at most 4 GiB.
 */
#ifndef RC_STANDALONE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#endif
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>

/* v0.6.0: the SWAR paths and the snapshot format are little-endian by
 * construction; big-endian targets are unsupported (README) — fail the
 * build instead of producing silently wrong snapshots. */
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "ratec supports little-endian targets only"
#endif

#define NO 6
static const double W[NO] = {0.02, 0.04, 0.08, 0.16, 0.30, 0.40};
static const uint64_t MASK[NO] = {0ull, 255ull, 65535ull, 16777215ull,
                                  4294967295ull, 1099511627775ull};
#define ALPHA 0.2
#define GOLDEN 0x9E3779B97F4A7C15ull
#define WIN 16                  /* inline occurrence window (u16 counts) */
#define IMAX 65535              /* max inline total (u16 window counts) */
#define PROMOTED 0xFF           /* distinct marker: count map is author */
/* Version is single-sourced from pyproject.toml (setup.py injects
 * -DRC_VERSION); the local define is the fallback for standalone builds. */
#ifndef RC_VERSION
#define RC_VERSION "0.8.0"
#endif
#define PF_DEFAULT 2
#define MAP_CAPBITS_DEFAULT 22
#define ID_NONE 0xFFFFFFFFu

/* ------------------------------------------------------------------ map */
/* Both slot types share the {u64 key; u32 tot;} prefix: tot==0 marks an
 * empty slot, so one generic probe serves both. The tmap slot keeps an
 * inline occurrence window with u16 counts (contexts stay single-probe
 * until 65535 total visits); the cmap slot is a bare 16 B counter
 * (contexts promoted out of their window). */
typedef struct {
    uint64_t key;                /* tmap: (order<<48)|ctx ; cmap: ctx<<8|b */
    uint32_t tot;
    uint8_t  distinct;           /* tmap: inline entries used / PROMOTED   */
    uint8_t  b[WIN];
    uint16_t c[WIN];             /* window counts (<= tot <= 65535)        */
    uint8_t  pad[2];   /* 8+4+1+16+32+2 = 64 */
} Ent;                           /* exactly 64 bytes = one cache line      */
_Static_assert(sizeof(Ent) == 64, "Ent must be one cache line");
typedef struct {
    uint64_t key;                /* ctx<<8|b                               */
    uint32_t tot;
    uint32_t pad;
} CEnt;                          /* 16 bytes, {key,tot} prefix             */
typedef struct {
    void *e; uint64_t capbits, mask, n; uint32_t esize;
    uint64_t *dirty;             /* page-granular touched bitmap (reset)  */
    int page_shift;              /* log2(slots per 4 KiB page)            */
} Map;

static int map_alloc(Map *m, uint64_t capbits, uint32_t esize) {
    if (capbits < 4) capbits = 4;
    if (capbits > 26) capbits = 26;
    m->capbits = capbits; m->mask = (1ull << capbits) - 1; m->n = 0;
    m->esize = esize;
    int ps = 0;                   /* log2(slots per 4 KiB page) */
    while ((1u << ps) < 4096u / esize) ps++;
    m->page_shift = ps;
    m->dirty = (uint64_t *)calloc(((1ull << capbits) >> ps) + 64, 8);
    m->e = calloc(1ull << capbits, esize);
    return (m->e && m->dirty) ? 0 : -1;
}
/* find-or-create; never grows (growth is checked once per byte, after all
 * slot pointers for that byte are done) */
static inline void *fos(Map *m, uint64_t key) {
    uint64_t i = (key * GOLDEN) >> (64 - m->capbits);
    for (;;) {
        uint8_t *base = (uint8_t *)m->e + i * m->esize;
        /* hit path first: the key sits on the same cache line the caller
         * touches anyway, and the insert branch stays off the hot
         * fetch/predict stream in steady state */
        if (*(const uint64_t *)base == key &&
            *(const uint32_t *)(base + 8) != 0) return base;
        if (*(const uint32_t *)(base + 8) == 0) {
            *(uint64_t *)base = key; m->n++;
            {   /* mark the slot's page dirty: bitmap bit = page index */
                const uint64_t pgidx = i >> m->page_shift;
                m->dirty[pgidx >> 6] |= 1ull << (pgidx & 63);
            }
            return base;
        }
        i = (i + 1) & m->mask;
    }
}
/* rehash all occupied slots into a table of capbits (contents preserved);
 * generic over the slot size, streams source lines through prefetch */
static int map_rehash_to(Map *m, uint64_t capbits) {
    if (capbits <= m->capbits) return 0;
    if (capbits > 26) capbits = 26;
    Map t;
    if (map_alloc(&t, capbits, m->esize) < 0) return -1;
    uint64_t cap = 1ull << m->capbits;
    const uint8_t *sb0 = (const uint8_t *)m->e;
    const uint32_t es = m->esize;
    for (uint64_t i = 0; i < cap; i++) {
        const uint8_t *sb = sb0 + i * es;
        if (*(const uint32_t *)(sb + 8)) {
            const uint64_t key = *(const uint64_t *)sb;
            uint64_t j = (key * GOLDEN) >> (64 - t.capbits);
            for (;;) {
                uint8_t *db = (uint8_t *)t.e + j * t.esize;
                if (*(uint32_t *)(db + 8) == 0) {
                    memcpy(db, sb, es);
                    t.n++;
                    {   const uint64_t pgidx = j >> t.page_shift;
                        t.dirty[pgidx >> 6] |= 1ull << (pgidx & 63); }
                    break;
                }
                if (*(const uint64_t *)db == key) break;
                j = (j + 1) & t.mask;
            }
        }
        if ((i & 63) == 0) __builtin_prefetch(sb + 64 * es, 0, 0);
    }
    free(m->e); free(m->dirty); *m = t; return 0;
}
static int map_grow(Map *m) { return map_rehash_to(m, m->capbits + 1); }
static int map_grow_to(Map *m, uint64_t capbits) { return map_rehash_to(m, capbits); }

/* find first window index q < distinct with e->b[q] == b (SWAR over the
 * two window halves, identical result to the scalar first-match loop) */
static inline int win_scan(const Ent *e, int distinct, uint8_t b) {
    uint64_t w0, w1;
    memcpy(&w0, e->b, 8);
    memcpy(&w1, e->b + 8, 8);
    const uint64_t x = 0x0101010101010101ull * (uint64_t)b;
    uint64_t m0 = (w0 ^ x);
    m0 = (m0 - 0x0101010101010101ull) & ~m0 & 0x8080808080808080ull;
    const int lo = distinct > 8 ? 8 : distinct;
    uint64_t vlo = (lo == 8) ? ~0ull : ((1ull << (8 * lo)) - 1ull);
    m0 &= vlo;
    if (m0) return (int)(__builtin_ctzll(m0) >> 3);
    if (distinct <= 8) return -1;
    uint64_t m1 = (w1 ^ x);
    m1 = (m1 - 0x0101010101010101ull) & ~m1 & 0x8080808080808080ull;
    const int hi = distinct - 8;                 /* 1..8 */
    uint64_t vhi = (hi == 8) ? ~0ull : ((1ull << (8 * hi)) - 1ull);
    m1 &= vhi;
    return m1 ? 8 + (int)(__builtin_ctzll(m1) >> 3) : -1;
}

/* --------------------------------------------------------------- OutVec */
typedef struct { uint32_t *a; size_t n, cap; } OutVec;
static int ov_init(OutVec *v) { v->a = NULL; v->n = v->cap = 0; return 0; }
static void ov_free(OutVec *v) { free(v->a); ov_init(v); }
static inline __attribute__((always_inline)) int ov_push(OutVec *v, uint32_t x) {
    if (v->n == v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 1024;
        uint32_t *na = (uint32_t *)realloc(v->a, nc * 4);
        if (!na) return -1;
        v->a = na; v->cap = nc;
    }
    v->a[v->n++] = x; return 0;
}

/* ----------------------------------------------------------------- book */
/* one hash slot: hash + id+1 (0 = empty). 16B = one slot per probe in a
 * single cache line, where the split hk/hv arrays cost two lines per
 * probe; the probe arithmetic (slot index, linear step, values) is
 * unchanged, so ids and outputs are bit-exact by construction. */
typedef struct { uint64_t h; uint32_t ref; } BSlot;
typedef struct {
    uint8_t  *arena; size_t alen, acap;
    uint32_t *off, *ln;          /* by id */
    uint32_t *cnt;               /* by id: occurrences since last prune
                                    (v0.7 prune mode; maintained always,
                                    consumed only when prune_after > 0) */
    size_t n, cap;               /* number of entries */
    BSlot   *slot;               /* open-addressed, linear probing */
    uint64_t capbits, mask;
    int err;
} Book;

static uint64_t fnv1a64(const uint8_t *p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}
static void book_init(Book *b) {
    memset(b, 0, sizeof *b);
    b->capbits = 10; b->mask = (1ull << b->capbits) - 1;
    b->slot = (BSlot *)calloc(1ull << b->capbits, sizeof(BSlot));
    if (!b->slot) { b->err = 1; }
}static int book_grow_hash(Book *b) {
    uint64_t ncap = b->capbits + 1;
    BSlot *sl = (BSlot *)calloc(1ull << ncap, sizeof(BSlot));
    if (!sl) return -1;
    uint64_t nmask = (1ull << ncap) - 1;
    for (size_t id = 0; id < b->n; id++) {
        uint64_t h = fnv1a64(b->arena + b->off[id], b->ln[id]);
        uint64_t j = h & nmask;
        while (sl[j].ref) j = (j + 1) & nmask;
        sl[j].ref = (uint32_t)(id + 1);
        sl[j].h = h;
    }
    free(b->slot);
    b->slot = sl; b->capbits = ncap; b->mask = nmask;
    return 0;
}
/* returns id, or ID_NONE on allocation failure (b->err set) */
static uint32_t book_intern_h(Book *b, const uint8_t *p, size_t len,
                              uint64_t h) {
    if (b->err) return ID_NONE;
    if ((b->n + 1) * 2 >= (1ull << b->capbits)) {
        if (book_grow_hash(b) < 0) { b->err = 1; return ID_NONE; }
    }
    uint64_t i = h & b->mask;
    while (b->slot[i].ref) {
        uint32_t id = b->slot[i].ref - 1;
        if (b->ln[id] == len && memcmp(b->arena + b->off[id], p, len) == 0) {
            if (b->cnt && b->cnt[id] < 0xFFFFFFFFu) b->cnt[id]++;
            return id;
        }
        i = (i + 1) & b->mask;
    }
    if (b->n == b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 256;
        uint32_t *no = (uint32_t *)realloc(b->off, nc * 4);
        if (!no) { b->err = 1; return ID_NONE; }
        b->off = no;
        uint32_t *nl = (uint32_t *)realloc(b->ln, nc * 4);
        if (!nl) { b->err = 1; return ID_NONE; }
        b->ln = nl;
        uint32_t *ncn = (uint32_t *)realloc(b->cnt, nc * 4);
        if (!ncn) { b->err = 1; return ID_NONE; }
        b->cnt = ncn;
        b->cap = nc;
    }
    if (b->alen + len > b->acap) {
        size_t nc = b->acap ? b->acap * 2 : 4096;
        while (nc < b->alen + len) nc *= 2;
        uint8_t *na = (uint8_t *)realloc(b->arena, nc);
        if (!na) { b->err = 1; return ID_NONE; }
        b->arena = na; b->acap = nc;
    }
    memcpy(b->arena + b->alen, p, len);
    b->off[b->n] = (uint32_t)b->alen;
    b->ln[b->n] = (uint32_t)len;
    b->alen += len;
    b->slot[i].ref = (uint32_t)(b->n + 1);
    b->slot[i].h = h;
    if (b->cnt) b->cnt[b->n] = 1;
    b->n++;
    return (uint32_t)(b->n - 1);
}

/* (no general intern entry: every caller either has a rolling hash
 * (encode path) or re-derives fnv1a64 inline (snapshot/prune rebuilds)) */

/* book_prune: drop entries whose since-last-prune count is below minc
 * (the v0.7 frequency-aware prune — singleton starvation remedy), then
 * reset every survivor's window count. Survivors are compacted IN PLACE
 * (write index <= read index) preserving first-seen order and renumbered
 * dense 0..k-1; the hash is rebuilt. No allocation, so this cannot OOM.
 * Returns entries removed, or -1 if the book is in an error state.
 *
 * Id semantics (documented, opt-in): ids emitted BEFORE a prune refer to
 * the pre-prune numbering — decode_bytes rejects them via its bounds
 * check once the book shrinks. Spans are unaffected (the patcher is
 * book-independent); only id assignment churns at prune boundaries. */
static int64_t book_prune(Book *b, uint32_t minc) {
    if (b->err) return -1;
    if (!b->cnt) return 0;                    /* no counts: nothing to do */
    size_t w = 0;                             /* survivor write index */
    size_t wpos = 0;                          /* survivor write position */
    for (size_t id = 0; id < b->n; id++) {
        if (b->cnt[id] >= minc) {
            if (wpos != b->off[id])           /* shift down over the gap */
                memmove(b->arena + wpos, b->arena + b->off[id], b->ln[id]);
            b->off[w] = (uint32_t)wpos;
            b->ln[w] = b->ln[id];
            w++;
            wpos += b->ln[id];
        }
    }
    int64_t removed = (int64_t)b->n - (int64_t)w;
    b->n = w;
    b->alen = wpos;
    memset(b->cnt, 0, b->n * 4);              /* window counts restart */
    /* rebuild the hash over the survivors */
    memset(b->slot, 0, ((size_t)b->mask + 1) * sizeof(BSlot));
    for (size_t id = 0; id < b->n; id++) {
        uint64_t h_ = fnv1a64(b->arena + b->off[id], b->ln[id]);
        uint64_t j = h_ & b->mask;
        while (b->slot[j].ref) j = (j + 1) & b->mask;
        b->slot[j].ref = (uint32_t)(id + 1);
        b->slot[j].h = h_;
    }
    return removed;
}

/* -------------------------------------------------------------- GrowBuf */
typedef struct { uint8_t *p; size_t n, cap; int err; } GrowBuf;
static int gb_put(GrowBuf *g, const void *src, size_t len) {
    if (g->err) return -1;
    if (g->n + len > g->cap) {
        size_t nc = g->cap ? g->cap * 2 : 65536;
        while (nc < g->n + len) nc *= 2;
        uint8_t *np = (uint8_t *)realloc(g->p, nc);
        if (!np) { g->err = 1; return -1; }
        g->p = np; g->cap = nc;
    }
    memcpy(g->p + g->n, src, len); g->n += len; return 0;
}
#define gb_u8(g, v)   gb_put((g), (v), 1)
#define gb_u32(g, v)  gb_put((g), (v), 4)
#define gb_u64(g, v)  gb_put((g), (v), 8)
#define gb_f64(g, v)  gb_put((g), (v), 8)

typedef struct { const uint8_t *p; size_t n, at; int err; } RdBuf;
static int rd_get(RdBuf *r, void *dst, size_t len) {
    if (r->err || r->at + len > r->n) { r->err = 1; return -1; }
    memcpy(dst, r->p + r->at, len); r->at += len; return 0;
}
#define rd_u8(r, v)   rd_get((r), (v), 1)
#define rd_u16(r, v)  rd_get((r), (v), 2)
#define rd_u32(r, v)  rd_get((r), (v), 4)
#define rd_u64(r, v)  rd_get((r), (v), 8)
#define rd_f64(r, v)  rd_get((r), (v), 8)

/* ----------------------------------------------------------------- core */
typedef struct {
    double C;                    /* patch budget, bits (1..64, fractional)*/
    int store_book;
    int pf;                      /* prefetch depth 0..4                   */
    int fast;                    /* non-exact fast lane (poly log2, tree) */
    int pmode;                   /* 0 under-shoot (reference), 1 carry    */
    double carry;                /* carry register (pmode==1), in [0, C]  */
    int poisoned;                /* any failed feed: state invalid        */

    /* judge state */
    uint32_t cnt0[256];
    uint32_t *cnt1;              /* 256 ctx x 256 bytes                   */
    uint32_t *tot1;              /* 256                                   */
    uint32_t *tot2;              /* 65536                                 */
    uint32_t *cnt2;              /* 65536 ctx x 256 bytes (64 MB virtual) */
    uint64_t h[NO];
    Map tmap, cmap;
    uint64_t map_capbits;        /* growth cap for tmap/cmap              */
    uint64_t *dirty2;            /* 65536-bit map of touched order-2 rows */

    /* patcher / stream state */
    int      patch_open;
    uint64_t open_start;         /* absolute stream offset                */
    uint64_t open_len;
    double   cs;                 /* running prefix sum == cs[nseen]       */
    double   lim;                /* cs_at_open + C                        */
    uint8_t *obuf; size_t ocap;  /* open-patch byte buffer (book mode)    */
    uint64_t oh;                 /* rolling fnv1a64 of the open patch —
                                    derived state, not serialized: the
                                    same value fnv1a64 would return over
                                    obuf[0..open_len), maintained O(1)
                                    per extension byte so book_intern
                                    stops re-hashing the whole patch at
                                    every close (the ids path's hot spot) */

    uint64_t nseen;              /* absolute bytes processed              */
    uint64_t n_forced, n_patches;
    uint64_t n_pruned;           /* cumulative book entries removed (v0.7) */

    /* v0.7 frequency-aware book pruning (opt-in; 0 = off, default). At
     * every close whose end offset crosses a multiple of prune_after,
     * book entries whose since-last-prune count < prune_min are dropped
     * and ids renumbered dense. prune_after == 0 keeps the byte-exact
     * default path (no book state changes at all). */
    uint64_t prune_after;        /* bytes between prunes                  */
    uint32_t prune_min;          /* since-last-prune count threshold      */
    uint64_t next_prune_at;      /* next absolute offset to prune at      */

    /* reusable emit vectors: fed by feed/flush/surprise, reset (n=0) per
     * call, capacity retained so steady-state calls never malloc/free */
    OutVec ros, roe, roi;

    Book book;
} RcCore;

static RcCore *core_new(double C, int store_book, int pf, int prebits,
                        int map_capbits, int fast, int pmode,
                        uint64_t prune_after, uint32_t prune_min);
static void core_free(RcCore *e);

static RcCore *core_new(double C, int store_book, int pf, int prebits,
                        int map_capbits, int fast, int pmode,
                        uint64_t prune_after, uint32_t prune_min) {
    RcCore *e = (RcCore *)calloc(1, sizeof(RcCore));
    if (!e) return NULL;
    e->C = C; e->store_book = store_book ? 1 : 0;
    e->pf = pf; e->map_capbits = (uint64_t)map_capbits;
    e->fast = fast ? 1 : 0;
    e->pmode = pmode ? 1 : 0; e->carry = 0.0;
    e->prune_after = prune_after;
    e->prune_min = prune_min ? prune_min : 1;
    e->next_prune_at = prune_after;   /* first boundary; 0 when disabled */
    e->cnt1 = (uint32_t *)calloc(256 * 256, 4);
    e->tot1 = (uint32_t *)calloc(256, 4);
    e->tot2 = (uint32_t *)calloc(65536, 4);
    e->cnt2 = (uint32_t *)calloc(65536ull * 256, 4);   /* zero pages */
    e->dirty2 = (uint64_t *)calloc(1024, 8);
    if (!e->cnt1 || !e->tot1 || !e->tot2 || !e->cnt2 || !e->dirty2) { core_free(e); return NULL; }
    if (map_alloc(&e->tmap, (uint64_t)prebits, sizeof(Ent)) < 0 ||
        map_alloc(&e->cmap, 10, sizeof(CEnt)) < 0) { core_free(e); return NULL; }
    book_init(&e->book);
    if (e->book.err) { core_free(e); return NULL; }
    ov_init(&e->ros); ov_init(&e->roe); ov_init(&e->roi);
    return e;
}

static void core_free(RcCore *e) {
    if (!e) return;
    free(e->cnt1); free(e->tot1); free(e->tot2); free(e->cnt2);
    free(e->dirty2);
    free(e->tmap.e); free(e->tmap.dirty);
    free(e->cmap.e); free(e->cmap.dirty);
    free(e->obuf);
    free(e->book.arena); free(e->book.off); free(e->book.ln);
    free(e->book.cnt);
    free(e->book.slot);
    ov_free(&e->ros); ov_free(&e->roe); ov_free(&e->roi);
    free(e);
}

/* in-place reset: clears state while KEEPING all pages mapped, so pooled
 * engines never pay first-touch faults again. The 64 MB order-2 table is
 * cleared per touched row via the dirty bitmap (~0.3 ms for text). */
static void core_reset_keep(RcCore *e) {
    memset(e->cnt0, 0, sizeof e->cnt0);
    memset(e->cnt1, 0, 256 * 256 * 4);
    memset(e->tot1, 0, 256 * 4);
    for (uint64_t w = 0; w < 1024; w++) {
        uint64_t word = e->dirty2[w];
        if (!word) continue;
        while (word) {
            int bit = __builtin_ctzll(word);
            word &= word - 1;
            uint64_t c = (w << 6) | (uint64_t)bit;
            memset(e->cnt2 + (c << 8), 0, 256 * 4);
            e->tot2[c] = 0;
        }
        e->dirty2[w] = 0;
    }
    memset(e->h, 0, sizeof e->h);
    /* maps: clear only the pages a previous run actually touched (keeps all
     * capacity and pages mapped; cost is proportional to touched pages) */
    for (int which = 0; which < 2; which++) {
        Map *m = which ? &e->cmap : &e->tmap;
        const uint64_t npg = (1ull << m->capbits) >> m->page_shift;
        const size_t pg = 4096;   /* by construction: esize << page_shift */
        for (uint64_t w = 0; w < (npg >> 6) + 1; w++) {
            uint64_t word = m->dirty[w];
            if (!word) continue;
            while (word) {
                const int bit = __builtin_ctzll(word);
                word &= word - 1;
                const uint64_t p = (w << 6) | (uint64_t)bit;
                memset((uint8_t *)m->e + p * pg, 0, pg);
            }
            m->dirty[w] = 0;
        }
    }
    e->tmap.n = 0; e->cmap.n = 0;
    e->patch_open = 0; e->open_start = 0; e->open_len = 0;
    e->oh = 0xcbf29ce484222325ull;
    e->cs = 0.0; e->lim = 0.0; e->carry = 0.0;
    e->nseen = 0; e->n_forced = 0; e->n_patches = 0;
    e->n_pruned = 0;
    e->next_prune_at = e->prune_after;   /* boundary schedule restarts */
    e->poisoned = 0;
    e->book.n = 0; e->book.alen = 0;
    if (!e->book.slot) {
        /* book tables were lost to an earlier OOM: re-init so the
         * advertised "reset() recovers" is actually true */
        free(e->book.slot);
        e->book.slot = NULL;
        book_init(&e->book);
    } else {
        e->book.err = 0;              /* reset clears a book-OOM flag too */
        uint64_t bmask = e->book.mask;
        memset(e->book.slot, 0, (bmask + 1) * sizeof(BSlot));
    }
}

/* pre-size the context map for an upcoming n-byte run (bench-kernel parity:
 * one rehash to 6n slots instead of log-many growth steps mid-encode) */
static int core_reserve(RcCore *e, size_t n) {
    uint64_t want = 6 * ((uint64_t)n + 1);
    uint64_t bits = 8;
    while ((1ull << bits) < want && bits < e->map_capbits) bits++;
    if (bits > e->tmap.capbits && map_grow_to(&e->tmap, bits) < 0) return -1;
    if (bits > e->cmap.capbits && map_grow_to(&e->cmap, bits) < 0) return -1;
    return 0;
}

/* v0.8.8: full-state sync of one bookless core onto another with an
 * IDENTICAL layout (same tmap/cmap capbits, same table sizes). Used by
 * the stream-parallel worker pool: a pristine post-head engine is cloned
 * per worker, and a per-chunk restore replaces load_state's wire parse +
 * full map rebuild. Union-clear: pages the src touched are copied, pages
 * only dst touched (previous chunk) are zeroed, dst's dirty bitmaps
 * become src's — cost is proportional to the UNION of touched pages, and
 * the judge state lands bit-identical to the src's. */
static int core_restore(RcCore *dst, const RcCore *src) {
    if (src->store_book || dst->store_book) return -1;
    if (src->tmap.capbits != dst->tmap.capbits ||
        src->cmap.capbits != dst->cmap.capbits ||
        src->tmap.esize != dst->tmap.esize ||
        src->cmap.esize != dst->cmap.esize) return -1;
    /* scalar state: raw struct copy, then re-seat dst's heap buffers */
    const uint64_t n_tm = src->tmap.n, n_cm = src->cmap.n;
    const Map tm = dst->tmap, cm = dst->cmap;
    const Book bk = dst->book;
    uint32_t *c1 = dst->cnt1, *t1 = dst->tot1, *t2 = dst->tot2;
    uint32_t *c2 = dst->cnt2;
    uint64_t *d2 = dst->dirty2;
    uint8_t *ob = dst->obuf;
    const size_t ocap = dst->ocap;
    OutVec r1 = dst->ros, r2 = dst->roe, r3 = dst->roi;
    *dst = *src;
    dst->cnt1 = c1; dst->tot1 = t1; dst->tot2 = t2;
    dst->cnt2 = c2; dst->dirty2 = d2;
    dst->tmap = tm; dst->cmap = cm; dst->book = bk;
    dst->obuf = ob; dst->ocap = ocap;
    dst->ros = r1; dst->roe = r2; dst->roi = r3;
    dst->ros.n = dst->roe.n = dst->roi.n = 0;
    dst->tmap.n = n_tm; dst->cmap.n = n_cm;
    /* small tables: full copy (sub-MB, always consistent) */
    memcpy(dst->cnt1, src->cnt1, 256 * 256 * 4);
    memcpy(dst->tot1, src->tot1, 256 * 4);
    memcpy(dst->tot2, src->tot2, 65536 * 4);
    /* order-2 rows via the dirty bitmap (union-clear) */
    for (uint64_t w = 0; w < 1024; w++) {
        const uint64_t s = src->dirty2[w];
        uint64_t work = s | (dst->dirty2[w] & ~s);
        while (work) {
            const int bit = __builtin_ctzll(work);
            work &= work - 1;
            const uint64_t c = ((uint64_t)w << 6) | (uint64_t)bit;
            uint32_t *row = dst->cnt2 + ((size_t)c << 8);
            if (s & (1ull << bit))
                memcpy(row, src->cnt2 + ((size_t)c << 8), 256 * 4);
            else
                memset(row, 0, 256 * 4);
        }
        dst->dirty2[w] = s;
    }
    /* maps: page-granular union-clear (identical layouts) */
    for (int which = 0; which < 2; which++) {
        const Map *ms = which ? &src->cmap : &src->tmap;
        Map *md = which ? &dst->cmap : &dst->tmap;
        const uint64_t npg = (1ull << ms->capbits) >> ms->page_shift;
        const size_t pgbytes = (size_t)md->esize << md->page_shift;
        for (uint64_t w = 0; w < (npg >> 6) + 1; w++) {
            const uint64_t s = ms->dirty[w];
            uint64_t work = s | (md->dirty[w] & ~s);
            while (work) {
                const int bit = __builtin_ctzll(work);
                work &= work - 1;
                const uint64_t pg = ((uint64_t)w << 6) | (uint64_t)bit;
                uint8_t *dd = (uint8_t *)md->e + pg * pgbytes;
                if (s & (1ull << bit))
                    memcpy(dd, (const uint8_t *)ms->e + pg * pgbytes, pgbytes);
                else
                    memset(dd, 0, pgbytes);
            }
            md->dirty[w] = s;
        }
    }
    /* open-patch buffer (stream-parallel restores happen between chunks,
     * right after a flush, so open_len is normally 0 — the copy is the
     * general case and costs nothing when empty) */
    if (src->open_len) {
        if (dst->ocap < src->open_len) {
            size_t nc = dst->ocap ? dst->ocap : 256;
            while (nc < src->open_len) nc *= 2;
            uint8_t *nb = (uint8_t *)realloc(dst->obuf, nc);
            if (!nb) return -1;
            dst->obuf = nb; dst->ocap = nc;
        }
        memcpy(dst->obuf, src->obuf, (size_t)src->open_len);
    }
    return 0;
}

/* allocate a layout-matching copy of a bookless core and restore into it
 * (maps are grown to the src's CURRENT capbits so clone == src layout) */
static RcCore *core_clone_from(const RcCore *src) {
    if (src->store_book) return NULL;
    RcCore *dst = core_new(src->C, 0, src->pf, (int)src->tmap.capbits,
                           (int)src->map_capbits, src->fast, src->pmode,
                           src->prune_after, src->prune_min);
    if (!dst) return NULL;
    if (src->cmap.capbits > dst->cmap.capbits &&
        map_grow_to(&dst->cmap, src->cmap.capbits) < 0) {
        core_free(dst); return NULL;
    }
    if (core_restore(dst, src) < 0) { core_free(dst); return NULL; }
    return dst;
}

/* one order-3..5 context: prediction + state update, interleaved.
 * cold -> 1 probe; window hit/new -> 1 probe; promoted -> 2 probes. */
#define DO_ORDER_PM(OI, LN) \
    if (FST) pm##LN = term; else pm += term;
#define DO_ORDER(OI) do { \
    const uint64_t ok_ = ((uint64_t)(OI) << 48) | h[(OI)]; \
    Ent *te = (Ent *)fos(&tl, ok_); \
    const uint32_t t_ = te->tot; \
    if (t_ == 0) { \
        term = W[(OI)] * 0.00390625; DO_ORDER_PM(OI, OI) \
        te->tot = 1; te->distinct = 1; te->b[0] = b; te->c[0] = 1; \
    } else if (te->distinct != PROMOTED) { \
        const int j = win_scan(te, te->distinct, b); \
        const uint32_t e_ = j >= 0 ? te->c[j] : 0; \
        term = W[(OI)] * ((double)e_ + ALPHA); term = term * inv; \
        term = term / (1.0 + (double)t_ * inv); DO_ORDER_PM(OI, OI) \
        if (t_ == IMAX || (j < 0 && te->distinct == WIN)) { \
            for (int q = 0; q < te->distinct; q++) \
                ((CEnt *)fos(&tc, (ok_ << 8) | te->b[q]))->tot = te->c[q]; \
            ((CEnt *)fos(&tc, (ok_ << 8) | b))->tot += 1; \
            te->tot = t_ + 1; te->distinct = PROMOTED; \
        } else if (j >= 0) { te->c[j]++; te->tot = t_ + 1; } \
        else { te->b[te->distinct] = b; te->c[te->distinct] = 1; \
               te->distinct++; te->tot = t_ + 1; } \
    } else { \
        CEnt *ce = (CEnt *)fos(&tc, (ok_ << 8) | b); \
        term = W[(OI)] * ((double)ce->tot + ALPHA); term = term * inv; \
        term = term / (1.0 + (double)t_ * inv); DO_ORDER_PM(OI, OI) \
        ce->tot++; \
        te->tot = t_ + 1; \
    } \
    h[(OI)] = ((h[(OI)] << 8) | b) & MASK[(OI)]; \
} while (0)

/* ------------------------------------------------ fast (non-exact) lane */
/* Degree-11 minimax-squares fit of log2 on m in [1,2): max |err| 5.7e-10
 * (float32 has ~6e-8). Two parallel chains keep the dependency depth ~8.
 * Used only when the engine is constructed with exact=0; surprisal then
 * deviates from the reference in the last ulps and occasional patch cuts
 * shift by one byte (documented; decode/round-trip unaffected). */
static const double RC_L2C[12] = {
    5.73139908829234266e-10, 1.44269494840484547e+00, -7.21343815080402129e-01,
    4.80833472318683619e-01, -3.60052287674538118e-01, 2.84872026466189754e-01,
    -2.26131929175595547e-01, 1.67279251685971603e-01, -1.04398212983452532e-01,
    4.88131318920478827e-02, -1.46256425177586705e-02, 2.05905650043528227e-03
};
static inline float rc_log2_fast(double x) {
    union { double d; uint64_t i; } u;
    u.d = x;
    int e = (int)(u.i >> 52) - 1023;
    u.i = (u.i & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    const double y = u.d - 1.0;
    const double z = y * y;
    /* explicit FMA chains (fast lane only; exact lane never calls this) */
    double E = fma(z, RC_L2C[10], RC_L2C[8]);
    E = fma(z, E, RC_L2C[6]);
    E = fma(z, E, RC_L2C[4]);
    E = fma(z, E, RC_L2C[2]);
    E = fma(z, E, RC_L2C[0]);
    double O = fma(z, RC_L2C[11], RC_L2C[9]);
    O = fma(z, O, RC_L2C[7]);
    O = fma(z, O, RC_L2C[5]);
    O = fma(z, O, RC_L2C[3]);
    O = fma(z, O, RC_L2C[1]);
    return (float)-((double)e + fma(y, O, E));   /* sur = -log2(x) */
}

/* emit one closed patch (start,end,id). ove/ovi may be NULL. Returns 0 or
 * a negative error (-1 OOM, -2 span overflow); *npatch counts successes. */
static int emit_patch(OutVec *ovs, OutVec *ove, OutVec *ovi, int store_book,
                      Book *bk, uint64_t st, uint64_t en, const uint8_t *ct,
                      size_t cl, uint64_t h, size_t *npatch) {
    if (st > 0xFFFFFFFFull || en > 0xFFFFFFFFull) return -2;
    uint32_t id_ = ID_NONE;
    if (ovi && store_book) {
        id_ = book_intern_h(bk, ct, cl, h);
        if (id_ == ID_NONE) return -1;
    }
    /* three independent pushes: the engine's persistent OutVecs diverge
     * in capacity across encode modes (spans-only never grows the ids
     * vector), so a fused lockstep check is NOT safe here — each vector
     * guards its own growth */
    if (ovs && ov_push(ovs, (uint32_t)st) < 0) return -1;
    if (ove && ov_push(ove, (uint32_t)en) < 0) return -1;
    if (ovi && ov_push(ovi, id_) < 0) return -1;
    (*npatch)++;
    return 0;
}

/* prune check after a patch close: fire when the patch end crosses the
 * next prune boundary. Boundaries are absolute stream offsets, so the
 * prune sequence is identical for chunked and one-shot ingestion (the
 * same closes happen at the same offsets). Multiple boundaries crossed
 * by one long patch fire a single prune and advance the schedule. */
static int maybe_prune(RcCore *e, uint64_t abs_end) {
    if (!e->store_book || !e->prune_after || abs_end < e->next_prune_at)
        return 0;
    while (e->next_prune_at <= abs_end) e->next_prune_at += e->prune_after;
    int64_t rem = book_prune(&e->book, e->prune_min);
    if (rem < 0) return -1;
    e->n_pruned += (uint64_t)rem;
    return 0;
}

static inline __attribute__((always_inline)) int
run_body(RcCore *e, const uint8_t *buf, size_t n, int want_patches,
         float *sur_out, OutVec *ovs, OutVec *ove, OutVec *ovi, const int PF)
{
    Map tl = e->tmap, tc = e->cmap;
    uint64_t h[NO]; memcpy(h, e->h, sizeof h);
    uint32_t *cnt0 = e->cnt0, *cnt1 = e->cnt1, *tot1 = e->tot1;
    uint32_t *tot2 = e->tot2, *cnt2 = e->cnt2;
    uint64_t *dirty2 = e->dirty2;
    double cs = e->cs, lim = e->lim;
    int patch_open = e->patch_open;
    uint64_t open_start = e->open_start, open_len = e->open_len;
    uint64_t oh = e->oh;                 /* rolling open-patch fnv hash */
    uint8_t *obuf = e->obuf; size_t ocap = e->ocap;
    const int sbook = e->store_book;
    uint64_t pos0 = e->nseen;
    const double Cloc = e->C;              /* hoisted: no per-byte reloads */
    const int FST = e->fast;
    const int CAR = e->pmode;              /* carry-over patcher mode      */
    double carry = e->carry;
    const uint64_t mapcap = e->map_capbits;
    long frc = 0; size_t npatch = 0;
    int rc = 0;
    const double inv = 1.0 / (256.0 * ALPHA);
    double term;
    size_t i = 0;

    while (i < n) {
        const uint8_t b = buf[i];
        double pm = 0.0, pm0 = 0.0, pm1 = 0.0, pm2 = 0.0, pm3 = 0.0, pm4 = 0.0, pm5 = 0.0;

        /* order 0: global context, total = bytes seen so far */
        {
            const uint64_t tot = pos0 + i;
            const uint32_t c0 = cnt0[b];
            if (tot) {
                term = W[0] * ((double)c0 + ALPHA); term = term * inv;
                term = term / (1.0 + (double)tot * inv);
            } else term = W[0] * 0.00390625;
            if (FST) pm0 = term; else pm += term;
            cnt0[b] = c0 + 1;
        }
        /* order 1: direct arrays */
        {
            const uint32_t t1v = tot1[h[1]];
            const uint32_t c1v = t1v ? cnt1[(h[1] << 8) | b] : 0;
            if (t1v) {
                term = W[1] * ((double)c1v + ALPHA); term = term * inv;
                term = term / (1.0 + (double)t1v * inv);
            } else term = W[1] * 0.00390625;
            if (FST) pm1 = term; else pm += term;
            tot1[h[1]] = t1v + 1;
            cnt1[(h[1] << 8) | b] = c1v + 1;
            h[1] = ((h[1] << 8) | b) & MASK[1];
        }
        /* order 2: direct arrays */
        {
            const uint32_t t2v = tot2[h[2]];
            const uint32_t c2v = t2v ? cnt2[((uint64_t)h[2] << 8) | b] : 0;
            if (t2v) {
                term = W[2] * ((double)c2v + ALPHA); term = term * inv;
                term = term / (1.0 + (double)t2v * inv);
            } else term = W[2] * 0.00390625;
            if (FST) pm2 = term; else pm += term;
            tot2[h[2]] = t2v + 1;
            cnt2[((uint64_t)h[2] << 8) | b] = c2v + 1;
            dirty2[h[2] >> 6] |= 1ull << (h[2] & 63);
            h[2] = ((h[2] << 8) | b) & MASK[2];
        }

        if (PF > 0) {
            /* rolling h[oi] forward over buf[i..i+p-1] yields the context
             * that byte i+p will probe; prefetch its slot line. */
            for (int p = 1; p <= PF && i + p < n; p++) {
                for (int oi = 3; oi < NO; oi++) {
                    uint64_t hc = h[oi];
                    for (int q = 0; q < p; q++)
                        hc = ((hc << 8) | buf[i + q]) & MASK[oi];
                    const uint64_t nok = ((uint64_t)oi << 48) | hc;
                    __builtin_prefetch((uint8_t *)tl.e + (((nok * GOLDEN) >> (64 - tl.capbits))) * tl.esize, 1, 3);
                    /* BET C (v0.8.6 candidate: cmap prefetch) measured by
                     * interleaved A/B (4 rounds, median): +2.3..+8.7% time
                     * on every lane — tc slots are covered by locality
                     * after promotion and the extra hash+prefetch chains
                     * cost more than they save. REJECTED, not shipped. */
                }
                /* BET B (v0.8.5): exact-line prefetch for the order-2
                 * direct row — the future byte buf[i+p] is known, so its
                 * cnt line within the (flat, aligned) row is exact.
                 * Order-1 prefetch measured and REJECTED: the 256 KB
                 * table is L2-resident; extra prefetches cost 2-7%. */
                {
                    uint64_t hc2 = h[2];
                    for (int q = 0; q < p; q++)
                        hc2 = ((hc2 << 8) | buf[i + q]) & MASK[2];
                    __builtin_prefetch(&cnt2[(hc2 << 8) | buf[i + p]], 1, 3);
                    /* BET E (tot2 line prefetch) measured NOISE-level
                     * (-0.9..+2.0%) and reverted — tot2 stays uncovered. */
                }
            }
        }

        /* orders 3-5 */
        DO_ORDER(3); DO_ORDER(4); DO_ORDER(5);

        float sf;
        if (FST) {
            /* relaxed lane: terms live in pm0..pm5; two 3-term partial
             * sums (rounding order is free here) shorten the serial tail */
            const double t03 = pm0 + pm1 + pm2, t45 = pm3 + pm4 + pm5;
            const double pmt = t03 + t45;
            sf = pmt > 0.0 ? rc_log2_fast(pmt) : 64.0f;
        } else {
            sf = pm > 0.0 ? (float)(-log2(pm)) : 64.0f;
        }
        if (sur_out) sur_out[i] = sf;

        if (!want_patches) {
            i++;                                 /* judge-only pass */
        } else {
            const double cnext = cs + (double)sf;
            if (!patch_open) {
                open_start = pos0 + i;
                lim = CAR ? cs + Cloc + carry : cs + (double)Cloc;
                patch_open = 1; open_len = 0;
                oh = 0xcbf29ce484222325ull;      /* fnv offset basis */
            }
            if (cnext <= lim) {
                cs = cnext;
                if (sbook) {
                    if (open_len == ocap) {
                        size_t nc = ocap ? ocap * 2 : 64;
                        uint8_t *nb = (uint8_t *)realloc(obuf, nc);
                        if (!nb) { rc = -1; goto done; }
                        obuf = nb; ocap = nc;
                    }
                    obuf[open_len] = b;
                }
                oh = (oh ^ b) * 0x100000001b3ull; /* fnv step (content joins
                                                     the patch even when the
                                                     buffer is not stored) */
                open_len++; i++;
            } else if (open_len == 0) {          /* forced single byte */
                frc++;
                {   const int erc = emit_patch(ovs, ove, ovi, sbook, &e->book,
                                   open_start, pos0 + i + 1, &b, 1,
                                   (0xcbf29ce484222325ull ^ b)
                                     * 0x100000001b3ull, &npatch);
                    if (erc) { rc = erc; goto done; }
                }
                if (maybe_prune(e, pos0 + i + 1)) { rc = -1; goto done; }
                cs = cnext; patch_open = 0; i++;
                if (CAR) carry = 0.0;            /* forced escape burns carry */
            } else {                             /* close, then byte i */
                int erc_ = emit_patch(ovs, ove, ovi, sbook, &e->book,
                                      open_start, pos0 + i, obuf, open_len,
                                      oh, &npatch);
                if (erc_) { rc = erc_; goto done; }
                if (maybe_prune(e, pos0 + i)) { rc = -1; goto done; }
                patch_open = 0; open_len = 0;
                if (CAR) {                       /* deposit unused budget */
                    const double waste = lim - cs;
                    carry = waste < 0.0 ? 0.0
                          : (waste > Cloc ? Cloc : waste);
                }
                lim = CAR ? cs + Cloc + carry : cs + (double)Cloc;
                open_start = pos0 + i;
                patch_open = 1;
                if (cnext <= lim) {
                    cs = cnext;
                    if (sbook) obuf[0] = b;
                    oh = (0xcbf29ce484222325ull ^ b)   /* new patch, first
                                                          byte joins it */
                       * 0x100000001b3ull;
                    open_len = 1; i++;
                } else {
                    frc++;
                    {   const int erc = emit_patch(ovs, ove, ovi, sbook,
                                                   &e->book, open_start,
                                                   pos0 + i + 1, &b, 1,
                                                   (0xcbf29ce484222325ull ^ b)
                                                     * 0x100000001b3ull,
                                                   &npatch);
                        if (erc) { rc = erc; goto done; }
                    }
                    if (maybe_prune(e, pos0 + i + 1)) { rc = -1; goto done; }
                    cs = cnext; patch_open = 0; i++;
                    if (CAR) carry = 0.0;        /* forced escape burns carry */
                }
            }
        }

        /* map growth between bytes (no slot pointers are held here) */
        if ((tl.n + 1) * 4 >= (1ull << tl.capbits) * 3) {
            if (tl.capbits < mapcap) {
                if (map_grow(&tl)) { rc = -1; goto done; }
            } else if ((tl.n + 1) * 16 >= (1ull << tl.capbits) * 15) {
                rc = -3; goto done;              /* context map nearly full */
            }
        }
        if ((tc.n + 1) * 4 >= (1ull << tc.capbits) * 3) {
            if (tc.capbits < mapcap) {
                if (map_grow(&tc)) { rc = -1; goto done; }
            } else if ((tc.n + 1) * 16 >= (1ull << tc.capbits) * 15) {
                rc = -3; goto done;
            }
        }
    }
done:
    e->tmap = tl; e->cmap = tc;
    memcpy(e->h, h, sizeof h);
    e->cs = cs; e->lim = lim; e->patch_open = patch_open;
    e->carry = carry;
    e->open_start = open_start; e->open_len = open_len;
    e->oh = oh;
    e->obuf = obuf; e->ocap = ocap;
    e->nseen = pos0 + i;
    e->n_forced += (uint64_t)frc;
    e->n_patches += npatch;
    return rc;
}


static int core_run(RcCore *e, const uint8_t *buf, size_t n, int want_patches,
                    float *sur_out, OutVec *ovs, OutVec *ove, OutVec *ovi) {
    switch (e->pf) {
    case 0:  return run_body(e, buf, n, want_patches, sur_out, ovs, ove, ovi, 0);
    case 1:  return run_body(e, buf, n, want_patches, sur_out, ovs, ove, ovi, 1);
    case 2:  return run_body(e, buf, n, want_patches, sur_out, ovs, ove, ovi, 2);
    case 3:  return run_body(e, buf, n, want_patches, sur_out, ovs, ove, ovi, 3);
    case 4:  return run_body(e, buf, n, want_patches, sur_out, ovs, ove, ovi, 4);
    default: return run_body(e, buf, n, want_patches, sur_out, ovs, ove, ovi, PF_DEFAULT);
    }
}

/* close the open patch (if any); end offset = current position */
static int core_flush(RcCore *e, OutVec *ovs, OutVec *ove, OutVec *ovi) {
    if (!e->patch_open || e->open_len == 0) return 0;
    if (e->open_start > 0xFFFFFFFFull || e->nseen > 0xFFFFFFFFull) return -2;
    uint32_t id_ = ID_NONE;
    if (ovi && e->store_book) {
        id_ = book_intern_h(&e->book, e->obuf, (size_t)e->open_len, e->oh);
        if (id_ == ID_NONE) return -1;
    }
    if (ovs && ov_push(ovs, (uint32_t)e->open_start) < 0) return -1;
    if (ove && ov_push(ove, (uint32_t)e->nseen) < 0) return -1;
    if (ovi && ov_push(ovi, id_) < 0) return -1;
    e->n_patches++;
    if (maybe_prune(e, e->nseen)) return -1;
    if (e->pmode) {                        /* deposit unused budget */
        const double waste = e->lim - e->cs;
        e->carry = waste < 0.0 ? 0.0
                 : (waste > e->C ? e->C : waste);
    }
    e->patch_open = 0; e->open_len = 0;
    return 0;
}

/* ------------------------------------------- reference (v1) patcher ---- */
/* greedy under-shoot on a stored float32 surprisal array, any C */
static int patch_from_sur_core(const float *sur, size_t n, double C, OutVec *ov) {
    double *cs = (double *)malloc((n + 1) * 8);
    if (!cs) return -1;
    cs[0] = 0.0;
    for (size_t i = 0; i < n; i++) cs[i + 1] = cs[i] + (double)sur[i];
    size_t s = 0;
    while (s < n) {
        size_t e = s;
        double lim = cs[s] + C;
        for (size_t j = s + 1; j <= n; j++) {
            if (cs[j] <= lim) e = j; else break;
        }
        if (e <= s) e = s + 1;                    /* forced 1-byte escape */
        if (ov_push(ov, (uint32_t)s) < 0) { free(cs); return -1; }
        s = e;
    }
    free(cs);
    return 0;
}

/* ------------------------- credit carry-over patcher (v0.4, pmode 1) --- */
/* Golden spec (mirrored in tests/test_v04.py):
 *   carry starts at 0; each patch opens with lim = cs_open + C + carry
 *   (carry in [0, C], so the spend cap is +C extra); on close,
 *   carry = clamp(lim - cs_close, 0, C) — the unused budget — and a
 *   forced single-byte escape (one sur > budget) burns the carry to 0.
 *   Telescoping: mean bits per patch -> C as the carry is bounded. */
static int patch_from_sur_carry_core(const float *sur, size_t n, double C,
                                     OutVec *ov) {
    double *cs = (double *)malloc((n + 1) * 8);
    if (!cs) return -1;
    cs[0] = 0.0;
    for (size_t i = 0; i < n; i++) cs[i + 1] = cs[i] + (double)sur[i];
    size_t s = 0;
    double carry = 0.0;
    while (s < n) {
        size_t e = s;
        double lim = cs[s] + C + carry;
        for (size_t j = s + 1; j <= n; j++) {
            if (cs[j] <= lim) e = j; else break;
        }
        if (e <= s) { e = s + 1; carry = 0.0; }   /* forced escape */
        else {
            const double waste = lim - cs[e];
            carry = waste < 0.0 ? 0.0 : (waste > C ? C : waste);
        }
        if (ov_push(ov, (uint32_t)s) < 0) { free(cs); return -1; }
        s = e;
    }
    free(cs);
    return 0;
}

/* ------------------------------------------------------------ snapshots */
#define RCST_MAGIC 0x54534352u                  /* "RCST" little-endian */
#define RCST_VERSION 1u

/* Snapshot format history:
 *   v2..v4  integer-C engines (read path kept for backward compatibility)
 *   v5      fractional C + carry mode (v0.4.0)
 *   v6      v0.6.0: u16 per-row nonzero counter (the u8 counter silently
 *           dropped rows with all 256 counts nonzero) + map_capbits is
 *           serialized (a tuned engine used to restore as the default and
 *           could hit map exhaustion later). Byte layout is otherwise the
 *           v5 layout. load reads v2..v6.
 *   v7      v0.7.0: prune-mode engines only — v6 layout + trailing prune
 *           section {prune_after u64, prune_min u32, next_prune_at u64,
 *           n_pruned u64, book cnt[] u32 each}. Written when and only
 *           when prune_after > 0, so non-prune snapshots stay byte-
 *           identical v6. v2..v6 readers reject v7 cleanly (version
 *           check); the strict trailing-garbage check is kept, which is
 *           why the section rides on a version bump rather than trailing
 *           on v6. load reads v2..v7. */
static int save_core(RcCore *e, GrowBuf *g) {
    uint32_t magic = RCST_MAGIC;
    uint32_t ver = e->prune_after ? 7 : 6;
    uint32_t sb = (uint32_t)e->store_book, pf = (uint32_t)e->pf;
    uint32_t fz = (uint32_t)e->fast;
    uint32_t poisoned = 0;
    uint8_t pm8 = (uint8_t)e->pmode;
    uint8_t mcb = (uint8_t)e->map_capbits;
    if (gb_u32(g, &magic) || gb_u32(g, &ver) || gb_f64(g, &e->C) ||
        gb_u32(g, &sb) || gb_u32(g, &pf) || gb_u32(g, &fz) ||
        gb_u32(g, &poisoned) ||
        gb_u8(g, &pm8) || gb_u8(g, &mcb) || gb_f64(g, &e->carry)) return -1;
    if (gb_u64(g, &e->nseen) || gb_u64(g, &e->n_forced) ||
        gb_u64(g, &e->n_patches)) return -1;
    /* patcher state */
    uint8_t po = (uint8_t)e->patch_open;
    if (gb_u8(g, &po) || gb_u64(g, &e->open_start) || gb_u64(g, &e->open_len) ||
        gb_f64(g, &e->cs) || gb_f64(g, &e->lim)) return -1;
    if (e->store_book && e->patch_open && e->open_len &&
        gb_put(g, e->obuf, (size_t)e->open_len)) return -1;   /* obuf only in book mode */
    /* judge direct tables */
    if (gb_put(g, e->cnt0, sizeof e->cnt0)) return -1;
    if (gb_put(g, e->cnt1, 256 * 256 * 4) || gb_put(g, e->tot1, 256 * 4) ||
        gb_put(g, e->tot2, 65536 * 4)) return -1;
    if (gb_put(g, e->h, sizeof e->h)) return -1;
    /* cnt2 RLE: contexts with any nonzero count */
    uint64_t nctx = 0;
    for (uint64_t c = 0; c < 65536; c++) {
        const uint32_t *row = e->cnt2 + (c << 8);
        for (int k = 0; k < 256; k++) if (row[k]) { nctx++; break; }
    }
    if (gb_u64(g, &nctx)) return -1;
    for (uint64_t c = 0; c < 65536; c++) {
        const uint32_t *row = e->cnt2 + (c << 8);
        int any = 0;
        for (int k = 0; k < 256; k++) if (row[k]) { any = 1; break; }
        if (!any) continue;
        uint16_t c16 = (uint16_t)c;
        if (gb_put(g, &c16, 2)) return -1;
        /* v6: u16 nonzero counter — the u8 counter wrapped at a full row
         * (256 nonzero counts) and silently dropped it (P1-4) */
        uint8_t nz[256]; uint32_t vals[256]; int m = 0;
        for (int k = 0; k < 256; k++)
            if (row[k]) { nz[m] = (uint8_t)k; vals[m] = row[k]; m++; }
        uint16_t m16 = (uint16_t)m;
        if (gb_put(g, &m16, 2) || gb_put(g, nz, (size_t)m) ||
            gb_put(g, vals, (size_t)m * 4))
            return -1;
    }
    /* maps */
    if (gb_u64(g, &e->tmap.capbits) || gb_u64(g, &e->tmap.n)) return -1;
    for (uint64_t s = 0; s < (1ull << e->tmap.capbits); s++) {
        const Ent *en = (const Ent *)((const uint8_t *)e->tmap.e + s * e->tmap.esize);
        if (!en->tot) continue;
        if (gb_put(g, &en->key, 8) || gb_put(g, &en->tot, 4) ||
            gb_put(g, &en->distinct, 1) ||
            gb_put(g, en->b, WIN) || gb_put(g, en->c, WIN * 2)) return -1;
    }
    if (gb_u64(g, &e->cmap.capbits) || gb_u64(g, &e->cmap.n)) return -1;
    for (uint64_t s = 0; s < (1ull << e->cmap.capbits); s++) {
        const CEnt *en = (const CEnt *)((const uint8_t *)e->cmap.e + s * e->cmap.esize);
        if (!en->tot) continue;
        if (gb_put(g, &en->key, 8) || gb_put(g, &en->tot, 4)) return -1;
    }
    /* book */
    if (e->store_book) {
        if (gb_u64(g, &e->book.n) || gb_u64(g, &e->book.alen)) return -1;
        if (e->book.alen && gb_put(g, e->book.arena, e->book.alen)) return -1;
        for (size_t id = 0; id < e->book.n; id++)
            if (gb_put(g, &e->book.off[id], 4) || gb_put(g, &e->book.ln[id], 4))
                return -1;
    }
    /* v7 prune section (prune-mode engines only) */
    if (e->prune_after) {
        if (gb_u64(g, &e->prune_after) || gb_u32(g, &e->prune_min) ||
            gb_u64(g, &e->next_prune_at) || gb_u64(g, &e->n_pruned))
            return -1;
        for (size_t id = 0; e->store_book && id < e->book.n; id++)
            if (gb_put(g, &e->book.cnt[id], 4)) return -1;
    }
    return g->err ? -1 : 0;
}

static RcCore *load_core(const uint8_t *p, size_t n) {
    RdBuf r = { p, n, 0, 0 };
    uint32_t magic = 0, ver = 0, C32 = 0, sb = 0, pf = 0, fz = 0, poisoned = 0;
    double C = 20.0, carry = 0.0;
    uint8_t pmode = 0, mcb = (uint8_t)MAP_CAPBITS_DEFAULT;
    if (rd_u32(&r, &magic) || rd_u32(&r, &ver)) return NULL;
    if (ver == 6 || ver == 7) {      /* v6: v5 + serialized map_capbits,
                                      * u16 per-row nonzero counter;
                                      * v7: v6 + trailing prune section */
        if (rd_f64(&r, &C) || rd_u32(&r, &sb) || rd_u32(&r, &pf) ||
            rd_u32(&r, &fz) || rd_u32(&r, &poisoned) ||
            rd_u8(&r, &pmode) || rd_u8(&r, &mcb) || rd_f64(&r, &carry))
            return NULL;
    } else if (ver == 5) {           /* v5: fractional C + carry mode */
        if (rd_f64(&r, &C) || rd_u32(&r, &sb) || rd_u32(&r, &pf) ||
            rd_u32(&r, &fz) || rd_u32(&r, &poisoned) ||
            rd_u8(&r, &pmode) || rd_f64(&r, &carry)) return NULL;
    } else {                         /* v2..v4 legacy: integer C */
        if (rd_u32(&r, &C32) || rd_u32(&r, &sb) || rd_u32(&r, &pf)) return NULL;
        if (ver >= 4) {
            if (rd_u32(&r, &fz)) return NULL;
        }
        if (ver >= 2 && rd_u32(&r, &poisoned)) return NULL;
        C = (double)C32;
    }
    if (magic != RCST_MAGIC || ver < 2 || ver > 7) return NULL;
    /* header sanity (P1-6): reject absurd/NaN tuning and flag fields before
     * they can reach the engine (NaN already fails the range compares) */
    if (sb > 1 || pf > 4 || fz > 1 || poisoned > 1 || pmode > 1) return NULL;
    if (mcb < 8 || mcb > 26) return NULL;
    if (!(C >= 1.0 && C <= 64.0)) return NULL;
    if (!(carry >= 0.0 && carry <= C)) return NULL;   /* finite + in range */
    RcCore *e = core_new(C, (int)sb, (int)pf, 16, (int)mcb,
                         (int)fz, (int)pmode, 0, 1);
    if (!e) return NULL;
    e->carry = carry;
    if (rd_u64(&r, &e->nseen) || rd_u64(&r, &e->n_forced) ||
        rd_u64(&r, &e->n_patches)) goto fail;
    uint8_t po = 0;
    if (rd_u8(&r, &po) || rd_u64(&r, &e->open_start) ||
        rd_u64(&r, &e->open_len) || rd_f64(&r, &e->cs) || rd_f64(&r, &e->lim))
        goto fail;
    if (po > 1) goto fail;
    e->patch_open = (int)po;
    /* patcher scalars must be finite and sane (P1-6): cs is a prefix sum of
     * per-byte surprisals, lim = cs_at_open + C, carry sits in [0, C] */
    if (!isfinite(e->cs) || e->cs < 0.0) goto fail;
    if (!isfinite(e->lim) || e->lim < 0.0) goto fail;
    if (e->store_book && e->patch_open && e->open_len) {
        if (e->open_len > (1ull << 32)) goto fail;
        uint8_t *ob = (uint8_t *)malloc((size_t)e->open_len);
        if (!ob) goto fail;
        if (rd_get(&r, ob, (size_t)e->open_len)) { free(ob); goto fail; }
        free(e->obuf); e->obuf = ob; e->ocap = (size_t)e->open_len;
    }
    /* rolling open-patch hash is derived, not serialized: recompute it
     * from the restored content (bit-identical to a from-scratch fnv1a64,
     * so snapshot bytes stay format v2..v7 unchanged). Book-off engines
     * never intern, so their placeholder value is never consumed. */
    e->oh = (e->store_book && e->patch_open && e->open_len)
          ? fnv1a64(e->obuf, (size_t)e->open_len)
          : 0xcbf29ce484222325ull;
    if (rd_get(&r, e->cnt0, sizeof e->cnt0)) goto fail;
    if (rd_get(&r, e->cnt1, 256 * 256 * 4) || rd_get(&r, e->tot1, 256 * 4) ||
        rd_get(&r, e->tot2, 65536 * 4)) goto fail;
    if (rd_get(&r, e->h, sizeof e->h)) goto fail;
    uint64_t nctx = 0;
    if (rd_u64(&r, &nctx)) goto fail;
    if (nctx > 65536) goto fail;                 /* at most one row per ctx */
    for (uint64_t k = 0; k < nctx; k++) {
        uint16_t c16; uint16_t m16 = 0;
        if (rd_get(&r, &c16, 2)) goto fail;
        if (ver >= 6) { if (rd_u16(&r, &m16)) goto fail; }
        else { uint8_t m8; if (rd_u8(&r, &m8)) goto fail; m16 = m8; }
        if (m16 > 256) goto fail;                /* stack buffer bound */
        uint8_t nz[256];
        if (rd_get(&r, nz, m16)) goto fail;
        uint32_t *row = e->cnt2 + ((uint64_t)c16 << 8);
        for (int q = 0; q < (int)m16; q++) {
            uint32_t v;
            if (rd_u32(&r, &v)) goto fail;
            row[nz[q]] = v;
        }
    }
    /* rebuild the order-2 dirty bitmap from the restored cnt2 rows (the
     * snapshot does not carry it; without this a reset() after load()
     * would leave stale cnt2 rows and corrupt the next engine reuse) */
    memset(e->dirty2, 0, 1024 * 8);
    for (uint64_t c = 0; c < 65536; c++) {
        const uint32_t *row = e->cnt2 + (c << 8);
        for (int k = 0; k < 256; k++)
            if (row[k]) { e->dirty2[c >> 6] |= 1ull << (c & 63); break; }
    }
    uint64_t capb = 0, cnt = 0;
    if (rd_u64(&r, &capb) || rd_u64(&r, &cnt)) goto fail;
    if (capb > 26) goto fail;
    /* P1-2: bound the rebuild before probing. cnt <= 75% of capacity
     * guarantees an empty slot exists, so the insert loop below always
     * terminates — previously a crafted 1 KB snapshot could spin forever
     * with the GIL held. */
    if (cnt > (((1ull << capb) / 4) * 3)) goto fail;
    free(e->tmap.e); free(e->tmap.dirty);          /* P1-6: no bitmap leak */
    e->tmap.e = NULL; e->tmap.dirty = NULL;
    if (map_alloc(&e->tmap, capb, sizeof(Ent)) < 0) goto fail;
    for (uint64_t k = 0; k < cnt; k++) {
        Ent en; memset(&en, 0, sizeof en);
        if (rd_get(&r, &en.key, 8) || rd_get(&r, &en.tot, 4) ||
            rd_get(&r, &en.distinct, 1) ||
            rd_get(&r, en.b, WIN) || rd_get(&r, en.c, WIN * 2)) goto fail;
        /* P1-1: distinct must be a real window occupancy or the PROMOTED
         * marker; 17..254 used to reach win_scan / the cold-byte branch
         * and write past the 64-byte slot. tot must cover every inline
         * count (each carries >= 1 visit). */
        if (en.tot == 0) goto fail;                /* occupancy marker */
        if (en.distinct != PROMOTED) {
            if (en.distinct == 0 || en.distinct > WIN) goto fail;
            if (en.tot < en.distinct) goto fail;
            for (int q = 0; q < en.distinct; q++)
                if (en.c[q] == 0 || en.c[q] > en.tot) goto fail;
        }
        uint64_t j = (en.key * GOLDEN) >> (64 - e->tmap.capbits);
        while (((Ent *)((uint8_t *)e->tmap.e + j * e->tmap.esize))->tot) {
            if (((Ent *)((uint8_t *)e->tmap.e + j * e->tmap.esize))->key
                == en.key) goto fail;              /* duplicate key in file */
            j = (j + 1) & e->tmap.mask;
        }
        *(Ent *)((uint8_t *)e->tmap.e + j * e->tmap.esize) = en;
        {   const uint64_t pgidx = j >> e->tmap.page_shift;
            e->tmap.dirty[pgidx >> 6] |= 1ull << (pgidx & 63); }
        e->tmap.n++;
    }
    if (rd_u64(&r, &capb) || rd_u64(&r, &cnt)) goto fail;
    if (capb > 26) goto fail;
    if (cnt > (((1ull << capb) / 4) * 3)) goto fail;
    free(e->cmap.e); free(e->cmap.dirty);
    e->cmap.e = NULL; e->cmap.dirty = NULL;
    if (map_alloc(&e->cmap, capb, sizeof(CEnt)) < 0) goto fail;
    for (uint64_t k = 0; k < cnt; k++) {
        CEnt en; memset(&en, 0, sizeof en);
        if (rd_get(&r, &en.key, 8) || rd_get(&r, &en.tot, 4)) goto fail;
        if (en.tot == 0) goto fail;
        uint64_t j = (en.key * GOLDEN) >> (64 - e->cmap.capbits);
        while (((CEnt *)((uint8_t *)e->cmap.e + j * e->cmap.esize))->tot) {
            if (((CEnt *)((uint8_t *)e->cmap.e + j * e->cmap.esize))->key
                == en.key) goto fail;
            j = (j + 1) & e->cmap.mask;
        }
        *(CEnt *)((uint8_t *)e->cmap.e + j * e->cmap.esize) = en;
        {   const uint64_t pgidx = j >> e->cmap.page_shift;
            e->cmap.dirty[pgidx >> 6] |= 1ull << (pgidx & 63); }
        e->cmap.n++;
    }
    if (e->store_book) {
        uint64_t bn = 0, alen = 0;
        if (rd_u64(&r, &bn) || rd_u64(&r, &alen)) goto fail;
        /* P1-3/P1-5: entry ids are u32 (hv stores id+1), arena offsets are
         * u32 (book_intern truncates at 4 GiB — the documented limit); a
         * nonempty book needs a nonempty arena, and an empty book has no
         * arena bytes. */
        if (bn > 0xFFFFFFFEull) goto fail;
        if (alen > 0xFFFFFFFFull) goto fail;
        if ((bn == 0) != (alen == 0)) goto fail;
        /* size the book hash for bn entries before rebuilding it; cap
         * raised 26 -> 31 so legitimate large books load (book_grow_hash
         * itself grows unbounded), with the load factor bounded below 0.5
         * so the rebuild probe below always terminates */
        uint64_t want = e->book.capbits;
        while ((bn + 1) * 2 >= (1ull << want) && want < 31) want++;
        if ((bn + 1) * 2 >= (1ull << want)) goto fail;
        if (want != e->book.capbits) {
            free(e->book.slot);
            e->book.capbits = want; e->book.mask = (1ull << want) - 1;
            e->book.slot = (BSlot *)calloc(1ull << want, sizeof(BSlot));
            if (!e->book.slot) goto fail;
        }
        if (alen) {
            uint8_t *ar = (uint8_t *)malloc((size_t)alen);
            if (!ar) goto fail;
            if (rd_get(&r, ar, (size_t)alen)) { free(ar); goto fail; }
            free(e->book.arena); e->book.arena = ar;
            e->book.alen = e->book.acap = alen;
        }
        if (!e->book.off || e->book.cap < bn) {
            /* grow off/ln/cnt arrays to fit bn entries */
            size_t nc = 256;
            while (nc < bn) nc *= 2;
            uint32_t *no = (uint32_t *)realloc(e->book.off, nc * 4);
            if (!no) goto fail;
            e->book.off = no;
            uint32_t *nl = (uint32_t *)realloc(e->book.ln, nc * 4);
            if (!nl) goto fail;
            e->book.ln = nl;
            uint32_t *ncn = (uint32_t *)realloc(e->book.cnt, nc * 4);
            if (!ncn) goto fail;
            e->book.cnt = ncn;
            e->book.cap = nc;
        }
        for (uint64_t id = 0; id < bn; id++) {
            uint32_t off, ln;
            if (rd_u32(&r, &off) || rd_u32(&r, &ln)) goto fail;
            /* P1-3: every (off, ln) must land inside the arena — unchecked
             * values used to reach fnv1a64 and decode_bytes' memcpy */
            if ((uint64_t)off > alen || (uint64_t)ln > alen - (uint64_t)off)
                goto fail;
            e->book.off[id] = off; e->book.ln[id] = ln;
        }
        if (e->book.cnt && ver < 7)
            memset(e->book.cnt, 0, (size_t)bn * 4);   /* no count history */
        e->book.n = bn;
        /* rebuild hash table */
        for (uint64_t id = 0; id < bn; id++) {
            uint64_t h_ = fnv1a64(e->book.arena + e->book.off[id],
                                  e->book.ln[id]);
            uint64_t j = h_ & e->book.mask;
            while (e->book.slot[j].ref) j = (j + 1) & e->book.mask;
            e->book.slot[j].ref = (uint32_t)(id + 1);
            e->book.slot[j].h = h_;
        }
    }
    /* v7 prune section (prune-mode engines only) */
    if (ver == 7) {
        uint64_t pa = 0, npa = 0, npr = 0;
        uint32_t pmin = 0;
        if (rd_u64(&r, &pa) || rd_u32(&r, &pmin) ||
            rd_u64(&r, &npa) || rd_u64(&r, &npr)) goto fail;
        /* header sanity for the prune schedule: the next boundary lies
         * within one interval past the last close (nseen is only ahead of
         * it when an open patch spans the boundary) */
        if (pa < 1 || pa > (1ull << 48) || pmin < 1) goto fail;
        if (npa < 1 || npa > e->nseen + pa) goto fail;
        if (npr > (1ull << 48)) goto fail;
        e->prune_after = pa; e->prune_min = pmin;
        e->next_prune_at = npa; e->n_pruned = npr;
        if (e->store_book) {
            for (uint64_t id = 0; id < e->book.n; id++)
                if (rd_u32(&r, &e->book.cnt[id])) goto fail;
        }
    }
    if (r.at != r.n) goto fail;                    /* trailing garbage */
    return e;
fail:
    core_free(e);
    return NULL;
}

#ifndef RC_STANDALONE
/* ------------------------------------------------------- Python object */
static PyTypeObject Engine_Type;   /* forward: clone() allocates instances */
typedef struct {
    PyObject_HEAD
    RcCore *core;
    int busy;
} EngineObj;

static PyObject *RatecError;

static int Eng_busy(EngineObj *self) {
    if (self->busy) {
        PyErr_SetString(RatecError,
                        "engine already running in another thread; "
                        "RateC engines are not thread-safe (use ratec.batch)");
        return 0;
    }
    if (self->core->poisoned) {
        PyErr_SetString(RatecError,
                        "engine state is invalid after a MemoryError; "
                        "create a new engine");
        return 0;
    }
    return 1;
}

static void Engine_dealloc(EngineObj *self) {
    core_free(self->core);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Engine_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    double C = 20.0;
    int store_book = 1, pf = -1, prebits = 16;
    int map_capbits = MAP_CAPBITS_DEFAULT;
    int fast = 0, carry = 0;
    unsigned long long prune_after = 0;
    int prune_min = 2;
    static char *kwlist[] = {"c", "store_book", "pf", "prebits",
                             "map_capbits", "exact", "carry",
                             "prune_after", "prune_min", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|dpiiiiiKi", kwlist,
                                     &C, &store_book, &pf, &prebits,
                                     &map_capbits, &fast, &carry,
                                     &prune_after, &prune_min))
        return NULL;
    fast = fast ? 0 : 1;   /* kwarg is `exact` (default true); fast = !exact */
    if (!(C >= 1.0 && C <= 64.0)) {
        PyErr_SetString(PyExc_ValueError,
                        "c must be in 1..64 (fractional ok)"); return NULL;
    }
    if (pf < -1 || pf > 4) {
        PyErr_SetString(PyExc_ValueError, "pf must be -1..4"); return NULL;
    }
    if (prune_after > (1ull << 48)) {
        PyErr_SetString(PyExc_ValueError,
                        "prune_after must be 0 (off) .. 2**48"); return NULL;
    }
    if (prune_min < 1 || prune_min > (1 << 30)) {
        PyErr_SetString(PyExc_ValueError,
                        "prune_min must be 1 .. 2**30"); return NULL;
    }
    if (prebits < 4) prebits = 4;
    if (prebits > 24) prebits = 24;
    if (map_capbits < 8) map_capbits = 8;
    if (map_capbits > 26) map_capbits = 26;
    EngineObj *self = (EngineObj *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->busy = 0;
    self->core = core_new(C, store_book, pf < 0 ? PF_DEFAULT : pf,
                          prebits, map_capbits, fast, carry ? 1 : 0,
                          (uint64_t)prune_after, (uint32_t)prune_min);
    if (!self->core) { Py_DECREF(self); return PyErr_NoMemory(); }
    return (PyObject *)self;
}

/* build [(start, end, id|None), ...] from aligned vecs; e may be NULL */
static PyObject *build_patch_list(OutVec *s, OutVec *e, OutVec *id) {
    PyObject *list = PyList_New((Py_ssize_t)s->n);
    if (!list) return NULL;
    for (size_t k = 0; k < s->n; k++) {
        PyObject *pid;
        if (id) {
            if (id->a[k] == ID_NONE) { Py_INCREF(Py_None); pid = Py_None; }
            else {
                pid = PyLong_FromUnsignedLong(id->a[k]);
                if (!pid) { Py_DECREF(list); return NULL; }
            }
        } else { Py_INCREF(Py_None); pid = Py_None; }
        PyObject *tup = Py_BuildValue("(IIO)", s->a[k],
                                      e ? e->a[k] : 0, pid);
        Py_DECREF(pid);
        if (!tup) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, (Py_ssize_t)k, tup);
    }
    return list;
}

static PyObject *Engine_feedflush(EngineObj *self, PyObject *arg, int do_flush) {
    Py_buffer view;
    if (!Eng_busy(self)) return NULL;
    self->busy = 1;
    if (PyObject_GetBuffer(arg, &view, PyBUF_SIMPLE) < 0) {
        self->busy = 0; return NULL;
    }
    /* reusable emit vectors: capacity survives across calls */
    OutVec *s = &self->core->ros, *e2 = &self->core->roe,
           *ids = &self->core->roi;
    s->n = 0; e2->n = 0; ids->n = 0;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = core_reserve(self->core, (size_t)view.len);
    if (rc == 0)
        rc = core_run(self->core, (const uint8_t *)view.buf, (size_t)view.len,
                      1, NULL, s, e2, ids);
    if (rc == 0 && do_flush)
        rc = core_flush(self->core, s, e2, ids);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    self->busy = 0;
    PyObject *out = NULL;
    if (rc == 0) out = build_patch_list(s, e2, ids);
    else if (rc == -1) PyErr_NoMemory();
    else if (rc == -3)
        PyErr_SetString(RatecError, "context map exhausted; raise "
                        "map_capbits or encode in chunks");
    else PyErr_SetString(RatecError, "stream exceeds 4 GiB span limit");
    /* A failed feed leaves the engine unusable: the pipelined judge may
     * have advanced past the failing byte. reset() recovers. The -3 map
     * exhaustion is the exception: it aborts BETWEEN bytes (the growth
     * check runs after a byte completes, before the next), so the state
     * is a valid streaming prefix and the engine stays usable. */
    if (rc == -1 || rc == -2) self->core->poisoned = 1;
    return out;
}

static PyObject *Engine_feed(EngineObj *self, PyObject *arg) {
    return Engine_feedflush(self, arg, 0);
}
static PyObject *Engine_flush(EngineObj *self, PyObject *Py_UNUSED(ignored)) {
    if (!Eng_busy(self)) return NULL;
    self->busy = 1;
    OutVec *s = &self->core->ros, *e2 = &self->core->roe,
           *ids = &self->core->roi;
    s->n = 0; e2->n = 0; ids->n = 0;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = core_flush(self->core, s, e2, ids);
    Py_END_ALLOW_THREADS
    self->busy = 0;
    PyObject *out = NULL;
    if (rc == 0) out = build_patch_list(s, e2, ids);
    else if (rc == -1) PyErr_NoMemory();
    else PyErr_SetString(RatecError, "stream exceeds 4 GiB span limit");
    if (rc < 0) self->core->poisoned = 1;
    return out;
}
static PyObject *Engine_surprise(EngineObj *self, PyObject *arg) {
    Py_buffer view;
    if (!Eng_busy(self)) return NULL;
    self->busy = 1;
    if (PyObject_GetBuffer(arg, &view, PyBUF_SIMPLE) < 0) {
        self->busy = 0; return NULL;
    }
    size_t n = (size_t)view.len;
    float *sur = NULL;
    if (n) {
        sur = (float *)malloc(n * 4);
        if (!sur) { PyBuffer_Release(&view); self->busy = 0;
                    return PyErr_NoMemory(); }
    }
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = core_reserve(self->core, n);
    if (rc == 0)
        rc = core_run(self->core, (const uint8_t *)view.buf, n, 0, sur,
                      NULL, NULL, NULL);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    self->busy = 0;
    PyObject *out = NULL;
    if (rc == 0) {
        out = PyBytes_FromStringAndSize((const char *)sur, (Py_ssize_t)(n * 4));
    } else if (rc == -3) {
        /* recoverable: map exhaustion leaves valid state (matches
         * encode_bytes) — a MemoryError + poison here was wrong */
        PyErr_SetString(RatecError, "context map exhausted; raise "
                        "map_capbits or encode in chunks");
    } else { self->core->poisoned = 1; PyErr_NoMemory(); }
    free(sur);
    return out;
}
static PyObject *Engine_encode_bytes(EngineObj *self, PyObject *arg) {
    /* low-latency path: runs the engine with the id vector only (no span
     * tuples are built, no (start,end,id) list is materialized) */
    if (!self->core->store_book) {
        PyErr_SetString(RatecError,
                        "encode_bytes requires store_book=True");
        return NULL;
    }
    Py_buffer view;
    if (!Eng_busy(self)) return NULL;
    self->busy = 1;
    if (PyObject_GetBuffer(arg, &view, PyBUF_SIMPLE) < 0) {
        self->busy = 0; return NULL;
    }
    OutVec *ids = &self->core->roi;
    ids->n = 0;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = core_reserve(self->core, (size_t)view.len);
    if (rc == 0)
        rc = core_run(self->core, (const uint8_t *)view.buf, (size_t)view.len,
                      1, NULL, NULL, NULL, ids);
    if (rc == 0)
        rc = core_flush(self->core, NULL, NULL, ids);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    self->busy = 0;
    if (rc != 0) {
        /* -1/-2 poison (mid-byte abort possible); -3 aborts between bytes
         * and leaves a valid streaming prefix — see Engine_feedflush */
        if (rc == -3) {
            PyErr_SetString(RatecError, "context map exhausted; raise "
                            "map_capbits or encode in chunks");
            return NULL;
        }
        self->core->poisoned = 1;
        if (rc == -1) return PyErr_NoMemory();
        PyErr_SetString(RatecError, "stream exceeds 4 GiB span limit");
        return NULL;
    }
    PyObject *out = PyList_New((Py_ssize_t)ids->n);
    if (!out) return PyErr_NoMemory();
    for (size_t k = 0; k < ids->n; k++) {
        PyObject *iv = PyLong_FromUnsignedLong(ids->a[k]);
        if (!iv) { Py_DECREF(out); return NULL; }
        PyList_SET_ITEM(out, (Py_ssize_t)k, iv);
    }
    return out;
}
static PyObject *Engine_encode_spans(EngineObj *self, PyObject *arg) {
    return Engine_feedflush(self, arg, 1);
}
static PyObject *Engine_reset(EngineObj *self, PyObject *Py_UNUSED(ig)) {
    if (!Eng_busy(self)) return NULL;
    /* in-place memset-based reset: pages stay mapped, so a pooled engine
     * never pays first-touch faults twice */
    core_reset_keep(self->core);
    Py_RETURN_NONE;
}
static PyObject *Engine_save_state(EngineObj *self, PyObject *Py_UNUSED(ig)) {
    if (!Eng_busy(self)) return NULL;
    if (self->core->poisoned) {
        PyErr_SetString(RatecError, "cannot snapshot a poisoned engine");
        return NULL;
    }
    self->busy = 1;
    GrowBuf g; memset(&g, 0, sizeof g);
    int rc = save_core(self->core, &g);
    self->busy = 0;
    PyObject *out = NULL;
    if (rc == 0)
        out = PyBytes_FromStringAndSize((const char *)g.p, (Py_ssize_t)g.n);
    else PyErr_NoMemory();
    free(g.p);
    return out;
}
/* v0.8.8: worker-side engine cloning for encode_stream_parallel. A clone
 * carries the parent's judge/patcher state bit-identically (no wire
 * roundtrip, no map rebuild); per-chunk restore_from(parent) then costs
 * only a union-of-touched-pages memcpy. Bookless cores only (the stream
 * path never interns). */
static PyObject *Engine_clone(EngineObj *self, PyObject *Py_UNUSED(ig)) {
    if (!Eng_busy(self)) return NULL;
    if (self->core->store_book) {
        PyErr_SetString(RatecError,
                        "clone requires store_book=False (stream/worker use)");
        return NULL;
    }
    RcCore *c = core_clone_from(self->core);
    if (!c) return PyErr_NoMemory();
    EngineObj *out = (EngineObj *)Engine_Type.tp_alloc(&Engine_Type, 0);
    if (!out) { core_free(c); return NULL; }
    out->busy = 0;
    out->core = c;
    return (PyObject *)out;
}
static PyObject *Engine_restore_from(EngineObj *self, PyObject *other) {
    if (!Eng_busy(self)) return NULL;
    if (!PyObject_TypeCheck(other, &Engine_Type)) {
        PyErr_SetString(RatecError, "restore_from expects an Engine");
        return NULL;
    }
    EngineObj *src = (EngineObj *)other;
    if (src->busy) {
        PyErr_SetString(RatecError, "source engine is busy");
        return NULL;
    }
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = core_restore(self->core, src->core);
    Py_END_ALLOW_THREADS
    if (rc < 0) {
        PyErr_SetString(RatecError,
                        "restore_from: incompatible engines (book state or "
                        "map capacity differs)");
        return NULL;
    }
    Py_RETURN_NONE;
}
static PyObject *Engine_reserve(EngineObj *self, PyObject *arg) {
    if (!Eng_busy(self)) return NULL;
    long long n = PyLong_AsLongLong(arg);
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "reserve expects n >= 0");
        return NULL;
    }
    if (core_reserve(self->core, (size_t)n) < 0) return PyErr_NoMemory();
    Py_RETURN_NONE;
}
static PyObject *Engine_decode_bytes(EngineObj *self, PyObject *arg) {
    if (!self->core->store_book) {
        PyErr_SetString(RatecError, "decode_bytes requires store_book=True");
        return NULL;
    }
    /* fast path: a uint32 buffer (numpy array / array('I') / bytes) is
     * decoded straight from memory — no per-id PyLong boxing. Exactness:
     * same ids, same bounds check, same book reads. */
    Py_buffer view;
    if (PyObject_CheckBuffer(arg) &&
        PyObject_GetBuffer(arg, &view, PyBUF_SIMPLE) == 0) {        if (view.len % 4 != 0) {
            PyBuffer_Release(&view);
            PyErr_SetString(PyExc_ValueError,
                            "buffer length must be a multiple of 4 (uint32 ids)");
            return NULL;
        }
        const size_t n = (size_t)view.len / 4;
        const uint32_t *ids = (const uint32_t *)view.buf;
        const uint32_t bn = self->core->book.n;
        size_t total = 0;
        for (size_t k = 0; k < n; k++) {
            const uint32_t v = ids[k];
            if (v >= bn) {
                PyBuffer_Release(&view);
                PyErr_Format(RatecError, "unknown patch id %u "
                             "(book has %u entries; decode only works on the "
                             "tokenizer instance (or same-state copy) that "
                             "produced the ids)", v, bn);
                return NULL;
            }
            total += self->core->book.ln[v];
        }
        PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total);
        if (!out) { PyBuffer_Release(&view); return NULL; }
        uint8_t *wp = (uint8_t *)PyBytes_AS_STRING(out);
        /* gather kernel: software-pipeline the arena reads — load the
         * next patch's (off, ln) and prefetch its first line while the
         * current one memcpys. Patches are short, so the two dependent
         * loads per id (ids[k] -> off[v] -> arena) dominate; the
         * pipeline hides that latency behind the copy. */
        if (n > 0) {
            const uint8_t *cur = self->core->book.arena
                               + self->core->book.off[ids[0]];
            const uint8_t *nxt = cur;
            uint32_t lc = self->core->book.ln[ids[0]];
            uint32_t ln_ = lc;
            __builtin_prefetch(cur, 0, 1);
            for (size_t k = 0; k < n; k++) {
                if (k + 1 < n) {
                    const uint32_t v1 = ids[k + 1];
                    nxt = self->core->book.arena + self->core->book.off[v1];
                    ln_ = self->core->book.ln[v1];
                    __builtin_prefetch(nxt, 0, 1);
                }
                memcpy(wp, cur, lc);
                wp += lc;
                cur = nxt; lc = ln_;
            }
        }
        PyBuffer_Release(&view);
        return out;
    } else if (PyObject_CheckBuffer(arg)) {
        PyErr_Clear();   /* non-contiguous etc. — fall through to sequence */
    }
    PyObject *seq = PySequence_Fast(arg, "ids must be a sequence of ints");
    if (!seq) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    size_t total = 0;
    for (Py_ssize_t k = 0; k < n; k++) {
        PyObject *it = PySequence_Fast_GET_ITEM(seq, k);
        long v = PyLong_AsLong(it);
        if (v == -1 && PyErr_Occurred()) { Py_DECREF(seq); return NULL; }
        if (v < 0 || (uint64_t)v >= self->core->book.n) {
            Py_DECREF(seq);
            PyErr_Format(RatecError, "unknown patch id %ld "
                         "(book has %llu entries; decode only works on the "
                         "tokenizer instance (or same-state copy) that "
                         "produced the ids)",
                         v, (unsigned long long)self->core->book.n);
            return NULL;
        }
        total += self->core->book.ln[v];
    }
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total);
    if (!out) { Py_DECREF(seq); return NULL; }
    uint8_t *wp = (uint8_t *)PyBytes_AS_STRING(out);
    for (Py_ssize_t k = 0; k < n; k++) {
        long v = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, k));
        const uint8_t *src = self->core->book.arena + self->core->book.off[v];
        memcpy(wp, src, self->core->book.ln[v]);
        wp += self->core->book.ln[v];
    }
    Py_DECREF(seq);
    return out;
}

/* getset */
static PyObject *gs_c(EngineObj *self, void *c) {
    return PyFloat_FromDouble(self->core->C);
}
static PyObject *gs_carry(EngineObj *self, void *c) {
    return PyFloat_FromDouble(self->core->carry);
}
static PyObject *gs_carry_mode(EngineObj *self, void *c) {
    return PyBool_FromLong(self->core->pmode);
}
static PyObject *gs_store_book(EngineObj *self, void *c) {
    return PyBool_FromLong(self->core->store_book);
}
static PyObject *gs_pf(EngineObj *self, void *c) {
    return PyLong_FromLong(self->core->pf);
}
static PyObject *gs_book_size(EngineObj *self, void *c) {
    return PyLong_FromSize_t(self->core->store_book ? self->core->book.n : 0);
}
static PyObject *gs_bytes_seen(EngineObj *self, void *c) {
    return PyLong_FromUnsignedLongLong(self->core->nseen);
}
static PyObject *gs_n_patches(EngineObj *self, void *c) {
    return PyLong_FromUnsignedLongLong(self->core->n_patches);
}
static PyObject *gs_n_forced(EngineObj *self, void *c) {
    return PyLong_FromUnsignedLongLong(self->core->n_forced);
}
static PyObject *gs_n_pruned(EngineObj *self, void *c) {
    return PyLong_FromUnsignedLongLong(self->core->n_pruned);
}
static PyObject *gs_prune_after(EngineObj *self, void *c) {
    return PyLong_FromUnsignedLongLong(self->core->prune_after);
}
static PyObject *gs_prune_min(EngineObj *self, void *c) {
    return PyLong_FromUnsignedLong(self->core->prune_min);
}
static PyObject *gs_repr(EngineObj *self) {
    char cbuf[32];
    snprintf(cbuf, sizeof cbuf, "%g", self->core->C);
    return PyUnicode_FromFormat(
        "Engine(c=%s, store_book=%d, pf=%d, bytes_seen=%llu, book=%llu)",
        cbuf, self->core->store_book, self->core->pf,
        (unsigned long long)self->core->nseen,
        (unsigned long long)(self->core->store_book ? self->core->book.n : 0));
}

/* book_dump: the vocabulary as a list of bytes, index == id. The
 * introspection/export surface (tiktoken-ranks parity): enables
 * to_ranks(), vocab diffs, and offline analysis without touching the
 * snapshot wire format. */
static PyObject *Engine_book_dump(EngineObj *self, PyObject *unused) {
    if (!self->core->store_book) {
        PyErr_SetString(RatecError, "book_dump requires store_book=True");
        return NULL;
    }
    Book *b = &self->core->book;
    PyObject *lst = PyList_New((Py_ssize_t)b->n);
    if (!lst) return NULL;
    for (size_t id = 0; id < b->n; id++) {
        PyObject *piece = PyBytes_FromStringAndSize(
            (const char *)(b->arena + b->off[id]), (Py_ssize_t)b->ln[id]);
        if (!piece) { Py_DECREF(lst); return NULL; }
        PyList_SET_ITEM(lst, (Py_ssize_t)id, piece);
    }
    return lst;
}

/* book_counts: per-id occurrence counts (window counts since the last
 * prune, index == id) as raw uint32 bytes — prune-tuning introspection
 * (which patches survive a prune_min threshold) without serializing. */
static PyObject *Engine_book_counts(EngineObj *self, PyObject *unused) {
    if (!self->core->store_book) {
        PyErr_SetString(RatecError, "book_counts requires store_book=True");
        return NULL;
    }
    Book *b = &self->core->book;
    if (!b->cnt) Py_RETURN_NONE;
    return PyBytes_FromStringAndSize((const char *)b->cnt,
                                     (Py_ssize_t)(b->n * 4));
}

static PyMethodDef Engine_methods[] = {
    {"book_counts",  (PyCFunction)Engine_book_counts,  METH_NOARGS,
     "book_counts() -> bytes of uint32 per-id occurrence counts\n"
     "(window counts since last prune; index == id)."},
    {"feed",         (PyCFunction)Engine_feed,         METH_O,
     "feed(data) -> [(start, end, id|None)]; patches closed by this chunk.\n"
     "Patches may span chunk boundaries; call flush() at end of stream."},
    {"flush",        (PyCFunction)Engine_flush,        METH_NOARGS,
     "flush() -> [(start, end, id|None)]; closes the open tail patch."},
    {"surprise",     (PyCFunction)Engine_surprise,     METH_O,
     "surprise(data) -> bytes of float32 little-endian per-byte surprisal\n"
     "(judge state advances; patcher state untouched)."},
    {"encode_bytes", (PyCFunction)Engine_encode_bytes, METH_O,
     "encode_bytes(data) -> [id, ...] (feed + flush)"},
    {"encode_spans", (PyCFunction)Engine_encode_spans, METH_O,
     "encode_spans(data) -> [(start, end, id|None), ...] (feed + flush)"},
    {"decode_bytes", (PyCFunction)Engine_decode_bytes, METH_O,
     "decode_bytes(ids) -> bytes (requires the same book)"},
    {"reset",        (PyCFunction)Engine_reset,        METH_NOARGS,
     "reset() -> cold engine (fresh tables)."},
    {"save_state",   (PyCFunction)Engine_save_state,   METH_NOARGS,
     "save_state() -> bytes snapshot of the full engine state."},
    {"clone",        (PyCFunction)Engine_clone,        METH_NOARGS,
     "clone() -> Engine — bit-identical bookless copy (no wire roundtrip)"},
    {"restore_from", (PyCFunction)(void (*)(void))Engine_restore_from, METH_O,
     "restore_from(engine) — sync this engine's state from another "
     "(bit-identical; union-of-touched-pages cost; bookless only)"},
    {"reserve", (PyCFunction)(void (*)(void))Engine_reserve, METH_O,
     "reserve(n) — pre-size the context maps for an n-byte run"},
    {"book_dump",    (PyCFunction)Engine_book_dump,    METH_NOARGS,
     "book_dump() -> [bytes, ...] the vocabulary, index == id."},
    {NULL, NULL, 0, NULL}
};
static PyGetSetDef Engine_getset[] = {
    {"c",          (getter)gs_c,          NULL, "patch budget (bits, fractional ok)", NULL},
    {"carry",      (getter)gs_carry,      NULL, "carry register value (carry mode)", NULL},
    {"carry_mode", (getter)gs_carry_mode, NULL, "credit carry-over patcher enabled", NULL},
    {"store_book", (getter)gs_store_book, NULL, "book enabled", NULL},
    {"pf",         (getter)gs_pf,         NULL, "prefetch depth", NULL},
    {"book_size",  (getter)gs_book_size,  NULL, "distinct patches in book", NULL},
    {"bytes_seen", (getter)gs_bytes_seen, NULL, "bytes processed", NULL},
    {"n_patches",  (getter)gs_n_patches,  NULL, "patches emitted", NULL},
    {"n_forced",   (getter)gs_n_forced,   NULL, "forced escapes", NULL},
    {"n_pruned",   (getter)gs_n_pruned,   NULL, "book entries removed by pruning", NULL},
    {"prune_after",(getter)gs_prune_after,NULL, "prune interval in bytes (0 = off)", NULL},
    {"prune_min",  (getter)gs_prune_min,  NULL, "since-last-prune count threshold", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};
static PyTypeObject Engine_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ratec._rcengine.Engine",
    .tp_basicsize = sizeof(EngineObj),
    .tp_dealloc = (destructor)Engine_dealloc,
    .tp_repr = (reprfunc)gs_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "RateC tokenizer engine (adaptive judge + rate patcher + book)",
    .tp_methods = Engine_methods,
    .tp_getset = Engine_getset,
    .tp_new = Engine_new,
};

/* ------------------------------------------------------------ batching */
typedef struct {
    const uint8_t *data; size_t n;
    uint32_t *out; size_t nout;          /* starts (always) */
    uint32_t *out_e, *out_i; size_t ne, ni;   /* ids mode only */
    double c; int prebits, err, fast, carry, want_ids;
} BJob;
typedef struct { BJob *jobs; int njobs; atomic_int next; } BPool;

/* free a batch job's output arrays (all three in ids mode) */
static void batch_job_free(BJob *j) {
    free(j->out); free(j->out_e); free(j->out_i);
    j->out = NULL; j->out_e = NULL; j->out_i = NULL;
}

static void *bworker(void *argp) {    BPool *p = (BPool *)argp;
    /* v0.8.7: one POOLED engine per worker. reset_keep between docs keeps
     * every page mapped (cnt2's 64 MB, the maps, the book), so docs after
     * the first on a thread pay zero first-touch faults and zero map
     * re-growth — the same trick the Engine.reset() path uses. Jobs of one
     * rc_batch call share uniform params (c/prebits/fast/carry/ids), so a
     * single core serves the whole thread; any failed job destroys the
     * pool and the next job rebuilds fresh. */
    RcCore *pool = NULL;
    for (;;) {
        int idx = atomic_fetch_add(&p->next, 1);
        if (idx >= p->njobs) break;
        BJob *j = &p->jobs[idx];
        if (!pool) {
            pool = core_new(j->c, j->want_ids, PF_DEFAULT, j->prebits,
                            MAP_CAPBITS_DEFAULT, j->fast, j->carry, 0, 1);
            if (!pool) { j->err = 1; continue; }
        } else {
            core_reset_keep(pool);
        }
        OutVec ov, ove, ovi;
        ov_init(&ov); ov_init(&ove); ov_init(&ovi);
        int rc = core_run(pool, j->data, j->n, 1, NULL, &ov,
                          j->want_ids ? &ove : NULL,
                          j->want_ids ? &ovi : NULL);
        if (rc == 0)
            rc = core_flush(pool, &ov, j->want_ids ? &ove : NULL,
                            j->want_ids ? &ovi : NULL);
        if (rc == 0) {
            j->out = ov.a; j->nout = ov.n;
            if (j->want_ids) {
                j->out_e = ove.a; j->ne = ove.n;
                j->out_i = ovi.a; j->ni = ovi.n;
            }
        } else {
            j->err = 1; ov_free(&ov); ov_free(&ove); ov_free(&ovi);
            core_free(pool); pool = NULL;   /* state suspect: rebuild next */
        }
    }
    if (pool) core_free(pool);
    return NULL;
}

static PyObject *rc_batch(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *docs;
    double c = 20.0;
    int threads = 0, fz = 0, carry = 0, ids = 0;
    static char *kwlist[] = {"docs", "c", "threads", "fast", "carry",
                             "ids", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|diiii", kwlist,
                                     &docs, &c, &threads, &fz, &carry,
                                     &ids))
        return NULL;
    PyObject *fast = PySequence_Fast(docs, "docs must be a sequence");
    if (!fast) return NULL;
    Py_ssize_t nd = PySequence_Fast_GET_SIZE(fast);
    if (nd == 0) { Py_DECREF(fast); return PyList_New(0); }
    if (!(c >= 1.0 && c <= 64.0)) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError,
                        "c must be in 1..64 (fractional ok)"); return NULL;
    }
    Py_buffer *views = (Py_buffer *)calloc((size_t)nd, sizeof(Py_buffer));
    BJob *jobs = (BJob *)calloc((size_t)nd, sizeof(BJob));
    if (!views || !jobs) {
        free(views); free(jobs); Py_DECREF(fast); return PyErr_NoMemory();
    }
    for (Py_ssize_t k = 0; k < nd; k++) {
        if (PyObject_GetBuffer(PySequence_Fast_GET_ITEM(fast, k),
                               &views[k], PyBUF_SIMPLE) < 0) {
            for (Py_ssize_t q = 0; q < k; q++) PyBuffer_Release(&views[q]);
            free(views); free(jobs); Py_DECREF(fast); return NULL;
        }
        jobs[k].data = (const uint8_t *)views[k].buf;
        jobs[k].n = (size_t)views[k].len;
        jobs[k].c = c;
        jobs[k].fast = fz;
        jobs[k].carry = carry ? 1 : 0;
        jobs[k].want_ids = ids ? 1 : 0;
        size_t n = jobs[k].n;
        int bits = 12;
        while ((1ull << bits) < 6 * n && bits < 23) bits++;
        jobs[k].prebits = bits;
    }
    int nthreads = threads;
    if (nthreads <= 0) {
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        nthreads = (int)(ncpu > 0 ? ncpu : 1);
    }
    if (nthreads > nd) nthreads = (int)nd;
    Py_BEGIN_ALLOW_THREADS
    {
        BPool p; p.jobs = jobs; p.njobs = (int)nd; atomic_init(&p.next, 0);
        if (nthreads <= 1) {
            bworker(&p);
        } else {
            pthread_t th[63];
            if (nthreads > 64) nthreads = 64;
            for (int t = 1; t < nthreads; t++)
                pthread_create(&th[t - 1], NULL, bworker, &p);
            bworker(&p);
            for (int t = 1; t < nthreads; t++)
                pthread_join(th[t - 1], NULL);
        }
    }
    Py_END_ALLOW_THREADS
    PyObject *out = PyList_New(nd);
    if (!out) {
        for (Py_ssize_t k = 0; k < nd; k++) PyBuffer_Release(&views[k]);
        for (Py_ssize_t k = 0; k < nd; k++) batch_job_free(&jobs[k]);
        free(views); free(jobs); Py_DECREF(fast);
        return NULL;
    }
    for (Py_ssize_t k = 0; k < nd; k++) {
        if (jobs[k].err) {
            Py_DECREF(out);
            for (Py_ssize_t q = 0; q < nd; q++) PyBuffer_Release(&views[q]);
            for (Py_ssize_t q = 0; q < nd; q++) batch_job_free(&jobs[q]);
            free(views); free(jobs); Py_DECREF(fast);
            PyErr_SetString(RatecError, "batch job failed (OOM?)");
            return NULL;
        }
        PyObject *b = NULL;
        if (jobs[k].want_ids) {
            PyObject *bs = PyBytes_FromStringAndSize(
                (const char *)jobs[k].out, (Py_ssize_t)(jobs[k].nout * 4));
            PyObject *be = bs ? PyBytes_FromStringAndSize(
                (const char *)jobs[k].out_e, (Py_ssize_t)(jobs[k].ne * 4))
                : NULL;
            PyObject *bi = be ? PyBytes_FromStringAndSize(
                (const char *)jobs[k].out_i, (Py_ssize_t)(jobs[k].ni * 4))
                : NULL;
            if (bi) b = PyTuple_Pack(3, bs, be, bi);   /* increfs items */
            Py_XDECREF(bs); Py_XDECREF(be); Py_XDECREF(bi);
        } else {
            b = PyBytes_FromStringAndSize(
                (const char *)jobs[k].out, (Py_ssize_t)(jobs[k].nout * 4));
        }
        if (!b) {
            Py_DECREF(out);
            for (Py_ssize_t q = 0; q < nd; q++) PyBuffer_Release(&views[q]);
            for (Py_ssize_t q = 0; q < nd; q++) batch_job_free(&jobs[q]);
            free(views); free(jobs); Py_DECREF(fast);
            return NULL;
        }
        PyList_SET_ITEM(out, k, b);
    }
    for (Py_ssize_t k = 0; k < nd; k++) PyBuffer_Release(&views[k]);
    for (Py_ssize_t k = 0; k < nd; k++) batch_job_free(&jobs[k]);
    free(views); free(jobs); Py_DECREF(fast);
    return out;
}

static PyObject *rc_patch_from_sur(PyObject *self, PyObject *args,
                                   PyObject *kwds) {
    Py_buffer view;
    double c = 20.0;
    int carry = 0;
    static char *kwlist[] = {"sur", "c", "carry", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "y*|dp", kwlist,
                                     &view, &c, &carry)) return NULL;
    if (!(c >= 1.0 && c <= 64.0)) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError,
                        "c must be in 1..64 (fractional ok)"); return NULL;
    }
    size_t n = (size_t)view.len / 4;
    OutVec ov; ov_init(&ov);
    int rc = carry
        ? patch_from_sur_carry_core((const float *)view.buf, n, c, &ov)
        : patch_from_sur_core((const float *)view.buf, n, c, &ov);
    PyBuffer_Release(&view);
    PyObject *out = NULL;
    if (rc == 0) {
        out = PyBytes_FromStringAndSize((const char *)ov.a,
                                        (Py_ssize_t)(ov.n * 4));
        if (!out) PyErr_NoMemory();
    } else PyErr_NoMemory();
    ov_free(&ov);
    return out;
}

static PyObject *rc_load_state(PyObject *self, PyObject *arg) {
    Py_buffer view;
    if (PyObject_GetBuffer(arg, &view, PyBUF_SIMPLE) < 0)
        return NULL;
    RcCore *core = load_core((const uint8_t *)view.buf, (size_t)view.len);
    PyBuffer_Release(&view);
    if (!core) {
        PyErr_SetString(RatecError, "invalid or truncated ratec state");
        return NULL;
    }
    EngineObj *selfobj = (EngineObj *)Engine_Type.tp_alloc(&Engine_Type, 0);
    if (!selfobj) { core_free(core); return NULL; }
    selfobj->busy = 0;
    selfobj->core = core;
    return (PyObject *)selfobj;
}

static PyObject *rc_version(PyObject *self, PyObject *Py_UNUSED(ig)) {
    return PyUnicode_FromString(RC_VERSION);
}

static PyMethodDef rc_methods[] = {
    {"batch", (PyCFunction)(void (*)(void))rc_batch,
     METH_VARARGS | METH_KEYWORDS,
     "batch(docs, c=20, threads=0, fast=False, carry=False, ids=False)\n"
     "Encode many independent documents in parallel (GIL released, one cold\n"
     "engine per doc). ids=False: each entry is a little-endian uint32 array\n"
     "of patch starts (patch k = [start_k, start_{k+1})). ids=True: each\n"
     "entry is (starts, ends, ids) as a tuple of little-endian uint32 arrays\n"
     "— thread-parallel LM-prep tokens; lens = ends - starts; ids need the\n"
     "book (store_book semantics per doc, spans unaffected)."},
    {"patch_from_sur", (PyCFunction)(void (*)(void))rc_patch_from_sur,
     METH_VARARGS | METH_KEYWORDS,
     "patch_from_sur(sur_f32le, c, carry=False) -> starts_bytes (uint32 LE)\n"
     "Reference patcher on a stored surprisal array: greedy under-shoot\n"
     "(default) or credit carry-over (carry=True), any fractional c."},
    {"load_state", rc_load_state, METH_O,
     "load_state(bytes) -> Engine (restores judge+patcher+book)."},
    {"version", rc_version, METH_NOARGS, "library version."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef rcmodule = {
    PyModuleDef_HEAD_INIT, "ratec._rcengine",
    "RateC tokenizer engine (C core).", -1, rc_methods,
};
PyMODINIT_FUNC PyInit__rcengine(void) {
    if (PyType_Ready(&Engine_Type) < 0) return NULL;
    PyObject *m = PyModule_Create(&rcmodule);
    if (!m) return NULL;
    RatecError = PyErr_NewException("ratec.RatecError", NULL, NULL);
    if (!RatecError) { Py_DECREF(m); return NULL; }
    if (PyModule_AddObject(m, "RatecError", RatecError) < 0) {
        Py_DECREF(RatecError); Py_DECREF(m); return NULL;
    }
    Py_INCREF(&Engine_Type);
    if (PyModule_AddObject(m, "Engine", (PyObject *)&Engine_Type) < 0) {
        Py_DECREF(&Engine_Type); Py_DECREF(m); return NULL;
    }
    PyModule_AddStringConstant(m, "__version__", RC_VERSION);
    return m;
}

#endif /* !RC_STANDALONE */

/* ------------------------------------------------- standalone harness -- */
/* Compile:  gcc -O3 -march=native -ffp-contract=off -fno-fast-math
 *             -DRC_STANDALONE -o /tmp/rcsa ratec/_rcengine.c -lm -lpthread
 * Usage:    /tmp/rcsa <file> <C> <pf> <prebits> <reps> [fast]          */
#ifdef RC_STANDALONE
#include <stdio.h>
#include <time.h>
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s file C pf prebits reps\n",
                            argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) return 2;
    fclose(f);
    double C = atof(argv[2]);
    int pf = atoi(argv[3]), prebits = atoi(argv[4]);
    int reps = atoi(argv[5]);
    int fastarg = (argc > 6) ? atoi(argv[6]) : 0;
    double inv_dummy = 0; (void)inv_dummy;
    for (int r = -1; r < reps; r++) {
        RcCore *core = core_new(C, 0, pf, prebits, MAP_CAPBITS_DEFAULT, fastarg, 0, 0, 1);
        core_reserve(core, (size_t)n);
        OutVec ov; ov_init(&ov);
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        int rc = core_run(core, buf, (size_t)n, 1, NULL, &ov, NULL, NULL);
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (rc != 0) { fprintf(stderr, "run rc=%d\n", rc); return 1; }
        double ns = (ts1.tv_sec - ts0.tv_sec) * 1e9 + (ts1.tv_nsec - ts0.tv_nsec);
        if (r >= 0) printf("REP %d %.0f\n", r, ns);
        if (r == reps - 1)
            printf("SUM %ld %zu %llu\n", n, ov.n,
                   (unsigned long long)core->n_forced);
        ov_free(&ov);
        core_free(core);
    }
    return 0;
}
#endif
