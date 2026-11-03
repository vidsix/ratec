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
 *   - selectable C (1..64), selectable prefetch depth (0..4)
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

#define NO 6
static const double W[NO] = {0.02, 0.04, 0.08, 0.16, 0.30, 0.40};
static const uint64_t MASK[NO] = {0ull, 255ull, 65535ull, 16777215ull,
                                  4294967295ull, 1099511627775ull};
#define ALPHA 0.2
#define GOLDEN 0x9E3779B97F4A7C15ull
#define WIN 8                   /* inline occurrence window             */
#define IMAX 255                /* max inline total (u8 counts)         */
#define PROMOTED 0xFF           /* distinct marker: count map is author */
#define RC_VERSION "0.1.0"
#define PF_DEFAULT 2
#define MAP_CAPBITS_DEFAULT 22
#define ID_NONE 0xFFFFFFFFu

/* ------------------------------------------------------------------ map */
typedef struct {
    uint64_t key;                /* tmap: (order<<48)|ctx ; cmap: ctx<<8|b */
    uint32_t tot;
    uint8_t  distinct;           /* tmap: inline entries used / PROMOTED   */
    uint8_t  b[WIN];
    uint8_t  c[WIN];
    uint8_t  pad[3];
} Ent;                           /* 32 bytes */
typedef struct { Ent *e; uint64_t capbits, mask, n; } Map;

static int map_alloc(Map *m, uint64_t capbits) {
    if (capbits < 4) capbits = 4;
    if (capbits > 26) capbits = 26;
    m->capbits = capbits; m->mask = (1ull << capbits) - 1; m->n = 0;
    m->e = (Ent *)calloc(1ull << capbits, sizeof(Ent));
    return m->e ? 0 : -1;
}
static int map_grow(Map *m) {
    Map t; uint64_t cap = 1ull << m->capbits;
    if (map_alloc(&t, m->capbits + 1) < 0) return -1;
    for (uint64_t i = 0; i < cap; i++) {
        if (m->e[i].tot) {
            uint64_t j = (m->e[i].key * GOLDEN) >> (64 - t.capbits);
            while (t.e[j].tot) j = (j + 1) & t.mask;
            t.e[j] = m->e[i]; t.n++;
        }
    }
    free(m->e); *m = t; return 0;
}
/* grow (rehashing, contents preserved) straight to a target size */
static int map_grow_to(Map *m, uint64_t capbits) {
    if (capbits <= m->capbits) return 0;
    if (capbits > 26) capbits = 26;
    Map t;
    if (map_alloc(&t, capbits) < 0) return -1;
    uint64_t cap = 1ull << m->capbits;
    for (uint64_t i = 0; i < cap; i++) {
        if (m->e[i].tot) {
            uint64_t j = (m->e[i].key * GOLDEN) >> (64 - t.capbits);
            while (t.e[j].tot) j = (j + 1) & t.mask;
            t.e[j] = m->e[i]; t.n++;
        }
    }
    free(m->e); *m = t; return 0;
}
/* core_reserve lives with the core below (needs the full RcCore type) */

/* find-or-create; never grows (growth is checked once per byte, after all
 * slot pointers for that byte are done) */
static inline Ent *fos(Map *m, uint64_t key) {
    uint64_t i = (key * GOLDEN) >> (64 - m->capbits);
    for (;;) {
        Ent *e = &m->e[i];
        if (e->tot == 0) { e->key = key; m->n++; return e; }
        if (e->key == key) return e;
        i = (i + 1) & m->mask;
    }
}
static inline int win_scan(const Ent *e, int distinct, uint8_t b) {
    for (int q = 0; q < distinct; q++) if (e->b[q] == b) return q;
    return -1;
}

/* --------------------------------------------------------------- OutVec */
typedef struct { uint32_t *a; size_t n, cap; } OutVec;
static int ov_init(OutVec *v) { v->a = NULL; v->n = v->cap = 0; }
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
typedef struct {
    uint8_t  *arena; size_t alen, acap;
    uint32_t *off, *ln;          /* by id */
    size_t n, cap;               /* number of entries */
    uint64_t *hk;                /* slot hash */
    uint32_t *hv;                /* slot value = id+1, 0 = empty */
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
    b->hk = (uint64_t *)calloc(1ull << b->capbits, 8);
    b->hv = (uint32_t *)calloc(1ull << b->capbits, 4);
    if (!b->hk || !b->hv) { b->err = 1; }
}
static int book_grow_hash(Book *b) {
    uint64_t ncap = b->capbits + 1;
    uint64_t *hk = (uint64_t *)calloc(1ull << ncap, 8);
    uint32_t *hv = (uint32_t *)calloc(1ull << ncap, 4);
    if (!hk || !hv) { free(hk); free(hv); return -1; }
    uint64_t nmask = (1ull << ncap) - 1;
    for (size_t id = 0; id < b->n; id++) {
        uint64_t h = fnv1a64(b->arena + b->off[id], b->ln[id]);
        uint64_t j = h & nmask;
        while (hv[j]) j = (j + 1) & nmask;
        hv[j] = (uint32_t)(id + 1);
        hk[j] = h;
    }
    free(b->hk); free(b->hv);
    b->hk = hk; b->hv = hv; b->capbits = ncap; b->mask = nmask;
    return 0;
}
/* returns id, or ID_NONE on allocation failure (b->err set) */
static uint32_t book_intern(Book *b, const uint8_t *p, size_t len) {
    if (b->err) return ID_NONE;
    if ((b->n + 1) * 2 >= (1ull << b->capbits)) {
        if (book_grow_hash(b) < 0) { b->err = 1; return ID_NONE; }
    }
    uint64_t h = fnv1a64(p, len);
    uint64_t i = h & b->mask;
    while (b->hv[i]) {
        uint32_t id = b->hv[i] - 1;
        if (b->ln[id] == len && memcmp(b->arena + b->off[id], p, len) == 0)
            return id;
        i = (i + 1) & b->mask;
    }
    if (b->n == b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 256;
        uint32_t *no = (uint32_t *)realloc(b->off, nc * 4);
        if (!no) { b->err = 1; return ID_NONE; }
        b->off = no;
        uint32_t *nl = (uint32_t *)realloc(b->ln, nc * 4);
        if (!nl) { b->err = 1; return ID_NONE; }
        b->ln = nl; b->cap = nc;
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
    b->hv[i] = (uint32_t)(b->n + 1);
    b->hk[i] = h;
    b->n++;
    return (uint32_t)(b->n - 1);
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
#define rd_u32(r, v)  rd_get((r), (v), 4)
#define rd_u64(r, v)  rd_get((r), (v), 8)
#define rd_f64(r, v)  rd_get((r), (v), 8)

/* ----------------------------------------------------------------- core */
typedef struct {
    int C;                       /* patch budget, bits (1..64)            */
    int store_book;
    int pf;                      /* prefetch depth 0..4                   */
    int poisoned;                /* OOM mid-stream: state invalid         */

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

    uint64_t nseen;              /* absolute bytes processed              */
    uint64_t n_forced, n_patches;

    Book book;
} RcCore;

static RcCore *core_new(int C, int store_book, int pf, int prebits,
                        int map_capbits);
static void core_free(RcCore *e);

static RcCore *core_new(int C, int store_book, int pf, int prebits,
                        int map_capbits) {
    RcCore *e = (RcCore *)calloc(1, sizeof(RcCore));
    if (!e) return NULL;
    e->C = C; e->store_book = store_book ? 1 : 0;
    e->pf = pf; e->map_capbits = (uint64_t)map_capbits;
    e->cnt1 = (uint32_t *)calloc(256 * 256, 4);
    e->tot1 = (uint32_t *)calloc(256, 4);
    e->tot2 = (uint32_t *)calloc(65536, 4);
    e->cnt2 = (uint32_t *)calloc(65536ull * 256, 4);   /* zero pages */
    e->dirty2 = (uint64_t *)calloc(1024, 8);
    if (!e->cnt1 || !e->tot1 || !e->tot2 || !e->cnt2 || !e->dirty2) { core_free(e); return NULL; }
    if (map_alloc(&e->tmap, (uint64_t)prebits) < 0 ||
        map_alloc(&e->cmap, 10) < 0) { core_free(e); return NULL; }
    book_init(&e->book);
    if (e->book.err) { core_free(e); return NULL; }
    return e;
}

static void core_free(RcCore *e) {
    if (!e) return;
    free(e->cnt1); free(e->tot1); free(e->tot2); free(e->cnt2);
    free(e->dirty2);
    free(e->tmap.e); free(e->cmap.e); free(e->obuf);
    free(e->book.arena); free(e->book.off); free(e->book.ln);
    free(e->book.hk); free(e->book.hv);
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
    /* maps: clear tot of occupied slots (scan; keeps capacity) */
    for (uint64_t s = 0; s < (1ull << e->tmap.capbits); s++) e->tmap.e[s].tot = 0;
    for (uint64_t s = 0; s < (1ull << e->cmap.capbits); s++) e->cmap.e[s].tot = 0;
    e->tmap.n = 0; e->cmap.n = 0;
    e->patch_open = 0; e->open_start = 0; e->open_len = 0;
    e->cs = 0.0; e->lim = 0.0;
    e->nseen = 0; e->n_forced = 0; e->n_patches = 0;
    e->poisoned = 0;
    e->book.n = 0; e->book.alen = 0;
    uint64_t bmask = e->book.mask;
    memset(e->book.hv, 0, (bmask + 1) * 4);
}

/* pre-size the context map for an upcoming n-byte run (bench-kernel parity:
 * one rehash to 6n slots instead of log-many growth steps mid-encode) */
static int core_reserve(RcCore *e, size_t n) {
    uint64_t want = 6 * ((uint64_t)n + 1);
    uint64_t bits = 8;
    while ((1ull << bits) < want && bits < e->map_capbits) bits++;
    if (bits > e->tmap.capbits && map_grow_to(&e->tmap, bits) < 0) return -1;
    return 0;
}

/* one order-3..5 context: prediction + state update, interleaved.
 * cold -> 1 probe; window hit/new -> 1 probe; promoted -> 2 probes. */
#define DO_ORDER(OI) do { \
    const uint64_t ok_ = ((uint64_t)(OI) << 48) | h[(OI)]; \
    Ent *te = fos(&tl, ok_); \
    const uint32_t t_ = te->tot; \
    if (t_ == 0) { \
        pm += W[(OI)] * 0.00390625; \
        te->tot = 1; te->distinct = 1; te->b[0] = b; te->c[0] = 1; \
    } else if (te->distinct != PROMOTED) { \
        const int j = win_scan(te, te->distinct, b); \
        const uint32_t e_ = j >= 0 ? te->c[j] : 0; \
        term = W[(OI)] * ((double)e_ + ALPHA); term = term * inv; \
        pm += term / (1.0 + (double)t_ * inv); \
        if (t_ == IMAX || (j < 0 && te->distinct == WIN)) { \
            for (int q = 0; q < te->distinct; q++) \
                fos(&tc, (ok_ << 8) | te->b[q])->tot = te->c[q]; \
            fos(&tc, (ok_ << 8) | b)->tot += 1; \
            te->tot = t_ + 1; te->distinct = PROMOTED; \
        } else if (j >= 0) { te->c[j]++; te->tot = t_ + 1; } \
        else { te->b[te->distinct] = b; te->c[te->distinct] = 1; \
               te->distinct++; te->tot = t_ + 1; } \
    } else { \
        Ent *ce = fos(&tc, (ok_ << 8) | b); \
        term = W[(OI)] * ((double)ce->tot + ALPHA); term = term * inv; \
        pm += term / (1.0 + (double)t_ * inv); \
        ce->tot++; \
        te->tot = t_ + 1; \
    } \
    h[(OI)] = ((h[(OI)] << 8) | b) & MASK[(OI)]; \
} while (0)

/* emit one closed patch (start,end,id). ove/ovi may be NULL. */
#define EMIT(ST, EN, CT, CL) do { \
    if ((ST) > 0xFFFFFFFFull || (EN) > 0xFFFFFFFFull) { rc = -2; goto done; } \
    uint32_t id_ = ID_NONE; \
    if (ovi && e->store_book) { \
        id_ = book_intern(&e->book, (const uint8_t *)(CT), (size_t)(CL)); \
        if (id_ == ID_NONE) { rc = -1; goto done; } \
    } \
    if (ovs && ov_push(ovs, (uint32_t)(ST)) < 0) { rc = -1; goto done; } \
    if (ove && ov_push(ove, (uint32_t)(EN)) < 0) { rc = -1; goto done; } \
    if (ovi && ov_push(ovi, id_) < 0) { rc = -1; goto done; } \
    npatch++; \
} while (0)

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
    uint8_t *obuf = e->obuf; size_t ocap = e->ocap;
    const int sbook = e->store_book;
    uint64_t pos0 = e->nseen;
    const int Cloc = e->C;                 /* hoisted: no per-byte reloads */
    const uint64_t mapcap = e->map_capbits;
    long frc = 0; size_t npatch = 0;
    int rc = 0;
    const double inv = 1.0 / (256.0 * ALPHA);
    double term;
    size_t i = 0;

    while (i < n) {
        const uint8_t b = buf[i];
        double pm = 0.0;

        /* order 0: global context, total = bytes seen so far */
        {
            const uint64_t tot = pos0 + i;
            const uint32_t c0 = cnt0[b];
            if (tot) {
                term = W[0] * ((double)c0 + ALPHA); term = term * inv;
                pm += term / (1.0 + (double)tot * inv);
            } else pm += W[0] * 0.00390625;
            cnt0[b] = c0 + 1;
        }
        /* order 1: direct arrays */
        {
            const uint32_t t1v = tot1[h[1]];
            const uint32_t c1v = t1v ? cnt1[(h[1] << 8) | b] : 0;
            if (t1v) {
                term = W[1] * ((double)c1v + ALPHA); term = term * inv;
                pm += term / (1.0 + (double)t1v * inv);
            } else pm += W[1] * 0.00390625;
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
                pm += term / (1.0 + (double)t2v * inv);
            } else pm += W[2] * 0.00390625;
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
                    __builtin_prefetch(&tl.e[(nok * GOLDEN) >> (64 - tl.capbits)], 1, 3);
                }
            }
        }

        /* orders 3-5 */
        DO_ORDER(3); DO_ORDER(4); DO_ORDER(5);

        const float sf = (float)(pm > 0.0 ? -log2(pm) : 64.0);
        if (sur_out) sur_out[i] = sf;

        if (!want_patches) {
            i++;                                 /* judge-only pass */
        } else {
            const double cnext = cs + (double)sf;
            if (!patch_open) {
                open_start = pos0 + i;
                lim = cs + (double)Cloc;
                patch_open = 1; open_len = 0;
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
                open_len++; i++;
            } else if (open_len == 0) {          /* forced single byte */
                frc++;
                EMIT(open_start, pos0 + i + 1, &b, 1);
                cs = cnext; patch_open = 0; i++;
            } else {                             /* close, then byte i */
                EMIT(open_start, pos0 + i, obuf, open_len);
                patch_open = 0; open_len = 0;
                lim = cs + (double)Cloc;         /* cs == cs[pos] exactly */
                open_start = pos0 + i;
                patch_open = 1;
                if (cnext <= lim) {
                    cs = cnext;
                    if (sbook) obuf[0] = b;
                    open_len = 1; i++;
                } else {
                    frc++;
                    EMIT(open_start, pos0 + i + 1, &b, 1);
                    cs = cnext; patch_open = 0; i++;
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
    e->open_start = open_start; e->open_len = open_len;
    e->obuf = obuf; e->ocap = ocap;
    e->nseen = pos0 + i;
    e->n_forced += (uint64_t)frc;
    e->n_patches += npatch;
    return rc;
}
#undef EMIT

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
        id_ = book_intern(&e->book, e->obuf, (size_t)e->open_len);
        if (id_ == ID_NONE) return -1;
    }
    if (ovs && ov_push(ovs, (uint32_t)e->open_start) < 0) return -1;
    if (ove && ov_push(ove, (uint32_t)e->nseen) < 0) return -1;
    if (ovi && ov_push(ovi, id_) < 0) return -1;
    e->n_patches++;
    e->patch_open = 0; e->open_len = 0;
    return 0;
}

/* ------------------------------------------- reference (v1) patcher ---- */
/* greedy under-shoot on a stored float32 surprisal array, any C */
static int patch_from_sur_core(const float *sur, size_t n, int C, OutVec *ov) {
    double *cs = (double *)malloc((n + 1) * 8);
    if (!cs) return -1;
    cs[0] = 0.0;
    for (size_t i = 0; i < n; i++) cs[i + 1] = cs[i] + (double)sur[i];
    size_t s = 0;
    while (s < n) {
        size_t e = s;
        double lim = cs[s] + (double)C;
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

/* ------------------------------------------------------------ snapshots */
#define RCST_MAGIC 0x54534352u                  /* "RCST" little-endian */
#define RCST_VERSION 1u

static int save_core(RcCore *e, GrowBuf *g) {
    uint32_t magic = RCST_MAGIC, ver = RCST_VERSION;
    uint32_t C32 = (uint32_t)e->C;
    uint32_t sb = (uint32_t)e->store_book, pf = (uint32_t)e->pf;
    uint32_t poisoned = 0;
    if (gb_u32(g, &magic) || gb_u32(g, &ver) || gb_u32(g, &C32) ||
        gb_u32(g, &sb) || gb_u32(g, &pf) || gb_u32(g, &poisoned)) return -1;
    if (gb_u64(g, &e->nseen) || gb_u64(g, &e->n_forced) ||
        gb_u64(g, &e->n_patches)) return -1;
    /* patcher state */
    uint8_t po = (uint8_t)e->patch_open;
    if (gb_u8(g, &po) || gb_u64(g, &e->open_start) || gb_u64(g, &e->open_len) ||
        gb_f64(g, &e->cs) || gb_f64(g, &e->lim)) return -1;
    if (e->patch_open && e->open_len &&
        gb_put(g, e->obuf, (size_t)e->open_len)) return -1;
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
        uint8_t nz[256]; uint32_t vals[256]; uint8_t m = 0;
        for (int k = 0; k < 256; k++)
            if (row[k]) { nz[m] = (uint8_t)k; vals[m] = row[k]; m++; }
        if (gb_u8(g, &m) || gb_put(g, nz, m) || gb_put(g, vals, (size_t)m * 4))
            return -1;
    }
    /* maps */
    if (gb_u64(g, &e->tmap.capbits) || gb_u64(g, &e->tmap.n)) return -1;
    for (uint64_t s = 0; s < (1ull << e->tmap.capbits); s++) {
        const Ent *en = &e->tmap.e[s];
        if (!en->tot) continue;
        if (gb_put(g, &en->key, 8) || gb_put(g, &en->tot, 4) ||
            gb_put(g, &en->distinct, 1) ||
            gb_put(g, en->b, WIN) || gb_put(g, en->c, WIN)) return -1;
    }
    if (gb_u64(g, &e->cmap.capbits) || gb_u64(g, &e->cmap.n)) return -1;
    for (uint64_t s = 0; s < (1ull << e->cmap.capbits); s++) {
        const Ent *en = &e->cmap.e[s];
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
    return g->err ? -1 : 0;
}

static RcCore *load_core(const uint8_t *p, size_t n) {
    RdBuf r = { p, n, 0, 0 };
    uint32_t magic = 0, ver = 0, C = 0, sb = 0, pf = 0, poisoned = 0;
    if (rd_u32(&r, &magic) || rd_u32(&r, &ver) ||
        rd_u32(&r, &C) || rd_u32(&r, &sb) || rd_u32(&r, &pf) ||
        rd_u32(&r, &poisoned)) return NULL;
    if (magic != RCST_MAGIC || ver != RCST_VERSION) return NULL;
    RcCore *e = core_new((int)C, (int)sb, (int)pf, 16, MAP_CAPBITS_DEFAULT);
    if (!e) return NULL;
    if (rd_u64(&r, &e->nseen) || rd_u64(&r, &e->n_forced) ||
        rd_u64(&r, &e->n_patches)) goto fail;
    uint8_t po = 0;
    if (rd_u8(&r, &po) || rd_u64(&r, &e->open_start) ||
        rd_u64(&r, &e->open_len) || rd_f64(&r, &e->cs) || rd_f64(&r, &e->lim))
        goto fail;
    e->patch_open = po;
    if (e->patch_open && e->open_len) {
        if (e->open_len > (1ull << 32)) goto fail;
        uint8_t *ob = (uint8_t *)malloc((size_t)e->open_len);
        if (!ob) goto fail;
        if (rd_get(&r, ob, (size_t)e->open_len)) { free(ob); goto fail; }
        free(e->obuf); e->obuf = ob; e->ocap = (size_t)e->open_len;
    }
    if (rd_get(&r, e->cnt0, sizeof e->cnt0)) goto fail;
    if (rd_get(&r, e->cnt1, 256 * 256 * 4) || rd_get(&r, e->tot1, 256 * 4) ||
        rd_get(&r, e->tot2, 65536 * 4)) goto fail;
    if (rd_get(&r, e->h, sizeof e->h)) goto fail;
    uint64_t nctx = 0;
    if (rd_u64(&r, &nctx)) goto fail;
    for (uint64_t k = 0; k < nctx; k++) {
        uint16_t c16; uint8_t m;
        if (rd_get(&r, &c16, 2) || rd_u8(&r, &m)) goto fail;
        uint8_t nz[256];
        if (rd_get(&r, nz, m)) goto fail;
        uint32_t *row = e->cnt2 + ((uint64_t)c16 << 8);
        for (int q = 0; q < m; q++) {
            uint32_t v;
            if (rd_u32(&r, &v)) goto fail;
            row[nz[q]] = v;
        }
    }
    uint64_t capb = 0, cnt = 0;
    if (rd_u64(&r, &capb) || rd_u64(&r, &cnt)) goto fail;
    if (capb > 26) goto fail;
    free(e->tmap.e);
    if (map_alloc(&e->tmap, capb) < 0) goto fail;
    for (uint64_t k = 0; k < cnt; k++) {
        Ent en; memset(&en, 0, sizeof en);
        if (rd_get(&r, &en.key, 8) || rd_get(&r, &en.tot, 4) ||
            rd_get(&r, &en.distinct, 1) ||
            rd_get(&r, en.b, WIN) || rd_get(&r, en.c, WIN)) goto fail;
        uint64_t j = (en.key * GOLDEN) >> (64 - e->tmap.capbits);
        while (e->tmap.e[j].tot) j = (j + 1) & e->tmap.mask;
        e->tmap.e[j] = en; e->tmap.n++;
    }
    if (rd_u64(&r, &capb) || rd_u64(&r, &cnt)) goto fail;
    if (capb > 26) goto fail;
    free(e->cmap.e);
    if (map_alloc(&e->cmap, capb) < 0) goto fail;
    for (uint64_t k = 0; k < cnt; k++) {
        Ent en; memset(&en, 0, sizeof en);
        if (rd_get(&r, &en.key, 8) || rd_get(&r, &en.tot, 4)) goto fail;
        uint64_t j = (en.key * GOLDEN) >> (64 - e->cmap.capbits);
        while (e->cmap.e[j].tot) j = (j + 1) & e->cmap.mask;
        e->cmap.e[j] = en; e->cmap.n++;
    }
    if (e->store_book) {
        uint64_t bn = 0, alen = 0;
        if (rd_u64(&r, &bn) || rd_u64(&r, &alen)) goto fail;
        /* size the book hash for bn entries before rebuilding it */
        uint64_t want = e->book.capbits;
        while ((bn + 1) * 2 >= (1ull << want) && want < 26) want++;
        if (want != e->book.capbits) {
            free(e->book.hk); free(e->book.hv);
            e->book.capbits = want; e->book.mask = (1ull << want) - 1;
            e->book.hk = (uint64_t *)calloc(1ull << want, 8);
            e->book.hv = (uint32_t *)calloc(1ull << want, 4);
            if (!e->book.hk || !e->book.hv) goto fail;
        }
        if (alen) {
            uint8_t *ar = (uint8_t *)malloc((size_t)alen);
            if (!ar) goto fail;
            if (rd_get(&r, ar, (size_t)alen)) { free(ar); goto fail; }
            free(e->book.arena); e->book.arena = ar;
            e->book.alen = e->book.acap = alen;
        }
        if (!e->book.off || e->book.cap < bn) {
            /* grow off/ln arrays to fit bn entries */
            size_t nc = 256;
            while (nc < bn) nc *= 2;
            uint32_t *no = (uint32_t *)realloc(e->book.off, nc * 4);
            if (!no) goto fail;
            e->book.off = no;
            uint32_t *nl = (uint32_t *)realloc(e->book.ln, nc * 4);
            if (!nl) goto fail;
            e->book.ln = nl; e->book.cap = nc;
        }
        for (uint64_t id = 0; id < bn; id++) {
            uint32_t off, ln;
            if (rd_u32(&r, &off) || rd_u32(&r, &ln)) goto fail;
            e->book.off[id] = off; e->book.ln[id] = ln;
        }
        e->book.n = bn;
        /* rebuild hash table */
        for (uint64_t id = 0; id < bn; id++) {
            uint64_t j = fnv1a64(e->book.arena + e->book.off[id],
                                 e->book.ln[id]) & e->book.mask;
            while (e->book.hv[j]) j = (j + 1) & e->book.mask;
            e->book.hv[j] = (uint32_t)(id + 1);
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
    int C = 20, store_book = 1, pf = -1, prebits = 16;
    int map_capbits = MAP_CAPBITS_DEFAULT;
    static char *kwlist[] = {"c", "store_book", "pf", "prebits",
                             "map_capbits", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ipiii", kwlist,
                                     &C, &store_book, &pf, &prebits,
                                     &map_capbits))
        return NULL;
    if (C < 1 || C > 64) {
        PyErr_SetString(PyExc_ValueError, "c must be in 1..64"); return NULL;
    }
    if (pf < -1 || pf > 4) {
        PyErr_SetString(PyExc_ValueError, "pf must be -1..4"); return NULL;
    }
    if (prebits < 4) prebits = 4;
    if (prebits > 24) prebits = 24;
    if (map_capbits < 8) map_capbits = 8;
    if (map_capbits > 26) map_capbits = 26;
    EngineObj *self = (EngineObj *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->busy = 0;
    self->core = core_new(C, store_book, pf < 0 ? PF_DEFAULT : pf,
                          prebits, map_capbits);
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
    OutVec s, e2, ids; ov_init(&s); ov_init(&e2); ov_init(&ids);
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = core_reserve(self->core, (size_t)view.len);
    if (rc == 0)
        rc = core_run(self->core, (const uint8_t *)view.buf, (size_t)view.len,
                      1, NULL, &s, &e2, &ids);
    if (rc == 0 && do_flush)
        rc = core_flush(self->core, &s, &e2, &ids);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    self->busy = 0;
    PyObject *out = NULL;
    if (rc == 0) out = build_patch_list(&s, &e2, &ids);
    else if (rc == -1) PyErr_NoMemory();
    else if (rc == -3)
        PyErr_SetString(RatecError, "context map exhausted; raise "
                        "map_capbits or encode in chunks");
    else PyErr_SetString(RatecError, "stream exceeds 4 GiB span limit");
    if (rc < 0) self->core->poisoned = (rc == -1);
    ov_free(&s); ov_free(&e2); ov_free(&ids);
    return out;
}

static PyObject *Engine_feed(EngineObj *self, PyObject *arg) {
    return Engine_feedflush(self, arg, 0);
}
static PyObject *Engine_flush(EngineObj *self, PyObject *Py_UNUSED(ignored)) {
    if (!Eng_busy(self)) return NULL;
    self->busy = 1;
    OutVec s, e2, ids; ov_init(&s); ov_init(&e2); ov_init(&ids);
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = core_flush(self->core, &s, &e2, &ids);
    Py_END_ALLOW_THREADS
    self->busy = 0;
    PyObject *out = NULL;
    if (rc == 0) out = build_patch_list(&s, &e2, &ids);
    else if (rc == -1) PyErr_NoMemory();
    else PyErr_SetString(RatecError, "stream exceeds 4 GiB span limit");
    if (rc < 0) self->core->poisoned = (rc == -1);
    ov_free(&s); ov_free(&e2); ov_free(&ids);
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
    } else { self->core->poisoned = 1; PyErr_NoMemory(); }
    free(sur);
    return out;
}
static PyObject *Engine_encode_bytes(EngineObj *self, PyObject *arg) {
    if (!self->core->store_book) {
        PyErr_SetString(RatecError,
                        "encode_bytes requires store_book=True");
        return NULL;
    }
    PyObject *spans = Engine_feedflush(self, arg, 1);
    if (!spans) return NULL;
    PyObject *out = PyList_New(PyList_GET_SIZE(spans));
    if (!out) { Py_DECREF(spans); return NULL; }
    for (Py_ssize_t k = 0; k < PyList_GET_SIZE(spans); k++) {
        PyObject *tup = PyList_GET_ITEM(spans, k);   /* borrowed */
        PyObject *pid = PyTuple_GET_ITEM(tup, 2);
        if (pid == Py_None) { Py_DECREF(out); Py_DECREF(spans);
            PyErr_SetString(RatecError, "internal: missing id"); return NULL; }
        PyObject *iv = PyLong_FromLong(PyLong_AsLong(pid));
        if (!iv) { Py_DECREF(out); Py_DECREF(spans); return NULL; }
        PyList_SET_ITEM(out, k, iv);
    }
    Py_DECREF(spans);
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
static PyObject *Engine_decode_bytes(EngineObj *self, PyObject *arg) {
    if (!self->core->store_book) {
        PyErr_SetString(RatecError, "decode_bytes requires store_book=True");
        return NULL;
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
    return PyLong_FromLong(self->core->C);
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
static PyObject *gs_repr(EngineObj *self) {
    return PyUnicode_FromFormat(
        "Engine(c=%d, store_book=%d, pf=%d, bytes_seen=%llu, book=%llu)",
        self->core->C, self->core->store_book, self->core->pf,
        (unsigned long long)self->core->nseen,
        (unsigned long long)(self->core->store_book ? self->core->book.n : 0));
}

static PyMethodDef Engine_methods[] = {
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
    {NULL, NULL, 0, NULL}
};
static PyGetSetDef Engine_getset[] = {
    {"c",          (getter)gs_c,          NULL, "patch budget (bits)", NULL},
    {"store_book", (getter)gs_store_book, NULL, "book enabled", NULL},
    {"pf",         (getter)gs_pf,         NULL, "prefetch depth", NULL},
    {"book_size",  (getter)gs_book_size,  NULL, "distinct patches in book", NULL},
    {"bytes_seen", (getter)gs_bytes_seen, NULL, "bytes processed", NULL},
    {"n_patches",  (getter)gs_n_patches,  NULL, "patches emitted", NULL},
    {"n_forced",   (getter)gs_n_forced,   NULL, "forced escapes", NULL},
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
    uint32_t *out; size_t nout;
    int c, prebits, err;
} BJob;
typedef struct { BJob *jobs; int njobs; atomic_int next; } BPool;

static void *bworker(void *argp) {
    BPool *p = (BPool *)argp;
    for (;;) {
        int idx = atomic_fetch_add(&p->next, 1);
        if (idx >= p->njobs) break;
        BJob *j = &p->jobs[idx];
        RcCore *core = core_new(j->c, 0, PF_DEFAULT, j->prebits,
                                MAP_CAPBITS_DEFAULT);
        if (!core) { j->err = 1; continue; }
        OutVec ov; ov_init(&ov);
        int rc = core_run(core, j->data, j->n, 1, NULL, &ov, NULL, NULL);
        if (rc == 0) rc = core_flush(core, &ov, NULL, NULL);
        if (rc == 0) { j->out = ov.a; j->nout = ov.n; }
        else { j->err = 1; ov_free(&ov); }
        core_free(core);
    }
    return NULL;
}

static PyObject *rc_batch(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *docs;
    int c = 20, threads = 0;
    static char *kwlist[] = {"docs", "c", "threads", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ii", kwlist,
                                     &docs, &c, &threads))
        return NULL;
    PyObject *fast = PySequence_Fast(docs, "docs must be a sequence");
    if (!fast) return NULL;
    Py_ssize_t nd = PySequence_Fast_GET_SIZE(fast);
    if (nd == 0) { Py_DECREF(fast); return PyList_New(0); }
    if (c < 1 || c > 64) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "c must be in 1..64"); return NULL;
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
        for (Py_ssize_t k = 0; k < nd; k++) free(jobs[k].out);
        free(views); free(jobs); Py_DECREF(fast);
        return NULL;
    }
    for (Py_ssize_t k = 0; k < nd; k++) {
        if (jobs[k].err) {
            Py_DECREF(out);
            for (Py_ssize_t q = k; q < nd; q++) PyBuffer_Release(&views[q]);
            for (Py_ssize_t q = k; q < nd; q++) free(jobs[q].out);
            free(views); free(jobs); Py_DECREF(fast);
            PyErr_SetString(RatecError, "batch job failed (OOM?)");
            return NULL;
        }
        PyObject *b = PyBytes_FromStringAndSize(
            (const char *)jobs[k].out, (Py_ssize_t)(jobs[k].nout * 4));
        if (!b) {
            Py_DECREF(out);
            for (Py_ssize_t q = k; q < nd; q++) PyBuffer_Release(&views[q]);
            for (Py_ssize_t q = k; q < nd; q++) free(jobs[q].out);
            free(views); free(jobs); Py_DECREF(fast);
            return NULL;
        }
        PyList_SET_ITEM(out, k, b);
    }
    for (Py_ssize_t k = 0; k < nd; k++) PyBuffer_Release(&views[k]);
    for (Py_ssize_t k = 0; k < nd; k++) free(jobs[k].out);
    free(views); free(jobs); Py_DECREF(fast);
    return out;
}

static PyObject *rc_patch_from_sur(PyObject *self, PyObject *args) {
    Py_buffer view;
    int c;
    if (!PyArg_ParseTuple(args, "y*i", &view, &c)) return NULL;
    if (c < 1 || c > 64) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "c must be in 1..64"); return NULL;
    }
    size_t n = (size_t)view.len / 4;
    OutVec ov; ov_init(&ov);
    int rc = patch_from_sur_core((const float *)view.buf, n, c, &ov);
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
        PyErr_SetString(PyExc_ValueError, "invalid or truncated ratec state");
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
     "batch(docs, c=20, threads=0) -> [starts_bytes, ...]\n"
     "Encode many independent documents in parallel (GIL released, one cold\n"
     "engine per doc). Each entry is a little-endian uint32 array of patch\n"
     "starts (patch k = [start_k, start_{k+1}))."},
    {"patch_from_sur", rc_patch_from_sur, METH_VARARGS,
     "patch_from_sur(sur_f32le, c) -> starts_bytes (uint32 LE)\n"
     "Reference greedy under-shoot patcher on a stored surprisal array."},
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
 * Usage:    /tmp/rcsa <file> <C> <pf> <prebits> <reps>          */
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
    int C = atoi(argv[2]), pf = atoi(argv[3]), prebits = atoi(argv[4]);
    int reps = atoi(argv[5]);
    double inv_dummy = 0; (void)inv_dummy;
    for (int r = -1; r < reps; r++) {
        RcCore *core = core_new(C, 0, pf, prebits, MAP_CAPBITS_DEFAULT);
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
