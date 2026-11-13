// A small oil-paint engine.
//
// The canvas has two layers per pixel:
//   dry  - the reflectance of everything that has already dried
//   wet  - a film of fresh paint: absorption K, scattering S (both as amounts,
//          i.e. coefficient x thickness, per RGB channel) and its volume V
//
// A brush is a reservoir of paint that exchanges paint with the wet film in
// both directions: it lifts some of what is already there and lays down its
// own load until the film reaches a target thickness. That one rule gives
// opaque strokes, wet-in-wet smearing, dry-brush, blending and glazing,
// depending only on how the brush is loaded and pressed.
//
// What the eye sees is the Kubelka-Munk reflectance of the wet film over the
// dry layer, so thin paint is translucent and thick paint covers.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }

inline float smooth(float a, float b, float x) {
    if (b <= a) return x >= b ? 1.f : 0.f;
    float t = clampf((x - a) / (b - a), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed * 2654435761u + 0x9E3779B9u) { if (!s) s = 1; next(); next(); }
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float uni() { return (next() >> 8) * (1.f / 16777216.f); }
    float gauss() {
        float u = std::max(uni(), 1e-7f), v = uni();
        return std::sqrt(-2.f * std::log(u)) * std::cos(6.2831853f * v);
    }
};

// Kubelka-Munk reflectance of a film (absorption amount k, scattering amount s)
// lying over a background of reflectance rg.
inline float km(float k, float s, float rg) {
    if (k <= 0.f && s <= 0.f) return rg;
    if (s < 1e-5f) return rg * std::exp(-2.f * k);
    float a = 1.f + k / s;
    float b2 = a * a - 1.f;
    float b = std::sqrt(b2 > 0.f ? b2 : 0.f);
    float x = b * s;
    float bc;  // b * coth(b * s)
    if (x < 1e-3f) bc = 1.f / s + b2 * s / 3.f;
    else if (x > 15.f) bc = b;
    else { float e = std::exp(2.f * x); bc = b * (e + 1.f) / (e - 1.f); }
    return (1.f - rg * (a - bc)) / (a - rg + bc);
}

inline float to_srgb(float v) {
    v = clampf(v, 0.f, 1.f);
    return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
}

}  // namespace

extern "C" {

struct Canvas {
    int W, H;
    float *dry;    // H*W*3  linear reflectance of dried paint
    float *K;      // H*W*3  wet absorption amount
    float *S;      // H*W*3  wet scattering amount
    float *V;      // H*W    wet film volume
    float *hgt;    // H*W    relief of dried paint
    float *tooth;  // H*W    ground texture, 0 (valley) .. 1 (peak)
};

struct Brush {
    int   type;          // 0 = soft hair (one reservoir), 1 = bristle (one reservoir per bristle),
                         // 2 = soft blender: carries no paint of its own, only moves what is wet
    float radius;        // px
    float hardness;      // 0 = airbrush-soft edge, 1 = crisp edge
    float aspect;        // footprint minor/major axis (1 = round)
    float angle;         // footprint rotation, radians
    int   follow;        // 1: angle is relative to the direction of travel
    float spacing;       // step between dabs, as a fraction of the radius
    float thick;         // film thickness the stroke builds up to at full pressure
    float pickup;        // 0..1 share of the wet film lifted (and so displaced) per pass
    float load;          // how far the paint lasts (<= 0: never runs out, never gets dirty)
    int   nbristle;
    float bristle_r;     // px
    float jitter;        // paint variation from bristle to bristle
    float tooth;         // 0..1 how much the ground texture breaks the stroke
    float press_radius;  // 0..1 how strongly pressure spreads the brush
    float press_thick;   // 0..1 how strongly pressure thickens the film
    uint32_t seed;
};

}  // extern "C"

namespace {

struct Reservoir {
    bool infinite;
    float k0[3], s0[3];   // the clean paint
    double V, K[3], S[3]; // what is on the brush right now (finite brushes)
};

thread_local std::vector<float> g_want;

// One touch of an elliptical footprint. All paint transfer happens here.
void dab(Canvas *c, float x, float y, float ax, float ay, float th, float hardness,
         float target, float pickup, float tooth_sens, float press, Reservoir &res) {
    float rmax = std::max(ax, ay);
    int x0 = std::max(0, (int)std::floor(x - rmax - 1.f));
    int x1 = std::min(c->W - 1, (int)std::ceil(x + rmax + 1.f));
    int y0 = std::max(0, (int)std::floor(y - rmax - 1.f));
    int y1 = std::min(c->H - 1, (int)std::ceil(y + rmax + 1.f));
    if (x1 < x0 || y1 < y0) return;
    float ct = std::cos(th), st = std::sin(th);
    float rmin = std::max(0.35f, std::min(ax, ay));
    // keep at least ~1.2 px of falloff so edges are antialiased
    float h = std::min(hardness, 1.f - 1.2f / rmin);
    if (h < 0.f) h = 0.f;
    float iax = 1.f / std::max(ax, 0.35f), iay = 1.f / std::max(ay, 0.35f);

    int bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    g_want.assign((size_t)bw * bh, 0.f);

    double liftV = 0, liftK[3] = {0, 0, 0}, liftS[3] = {0, 0, 0};
    double wantSum = 0;

    // pass 1: lift
    for (int py = y0; py <= y1; ++py) {
        for (int px = x0; px <= x1; ++px) {
            float dx = px + 0.5f - x, dy = py + 0.5f - y;
            float u = (dx * ct + dy * st) * iax;
            float v = (-dx * st + dy * ct) * iay;
            float d = std::sqrt(u * u + v * v);
            if (d >= 1.f) continue;
            float w = 1.f - smooth(h, 1.f, d);
            size_t i = (size_t)py * c->W + px;
            if (tooth_sens > 0.f) {
                float t = c->tooth[i];
                w *= clampf(1.f - tooth_sens * (1.f - t) * 2.2f * (1.25f - press), 0.f, 1.f);
            }
            if (w <= 0.f) continue;
            float Vc = c->V[i];
            float lift = pickup * w * Vc;
            float keep = Vc - lift;
            if (lift > 0.f) {
                float fr = keep / Vc;
                float *K = c->K + i * 3, *S = c->S + i * 3;
                for (int ch = 0; ch < 3; ++ch) {
                    liftK[ch] += K[ch] * (1.f - fr);
                    liftS[ch] += S[ch] * (1.f - fr);
                    K[ch] *= fr;
                    S[ch] *= fr;
                }
                c->V[i] = keep;
                liftV += lift;
            }
            float want = std::max(target * w, Vc) - keep;
            if (want > 0.f) {
                g_want[(size_t)(py - y0) * bw + (px - x0)] = want;
                wantSum += want;
            }
        }
    }

    // pass 2: lay down
    if (wantSum > 0) {
        float scale = 1.f;
        float kc[3], sc[3];
        if (res.infinite) {
            for (int ch = 0; ch < 3; ++ch) { kc[ch] = res.k0[ch]; sc[ch] = res.s0[ch]; }
        } else {
            if (res.V <= 1e-9) scale = 0.f;
            else {
                if (wantSum > res.V) scale = (float)(res.V / wantSum);
                for (int ch = 0; ch < 3; ++ch) {
                    kc[ch] = (float)(res.K[ch] / res.V);
                    sc[ch] = (float)(res.S[ch] / res.V);
                }
            }
        }
        if (scale > 0.f) {
            double given = 0;
            for (int py = y0; py <= y1; ++py) {
                for (int px = x0; px <= x1; ++px) {
                    float want = g_want[(size_t)(py - y0) * bw + (px - x0)];
                    if (want <= 0.f) continue;
                    float d = want * scale;
                    size_t i = (size_t)py * c->W + px;
                    float *K = c->K + i * 3, *S = c->S + i * 3;
                    for (int ch = 0; ch < 3; ++ch) { K[ch] += d * kc[ch]; S[ch] += d * sc[ch]; }
                    c->V[i] += d;
                    given += d;
                }
            }
            if (!res.infinite) {
                for (int ch = 0; ch < 3; ++ch) { res.K[ch] -= given * kc[ch]; res.S[ch] -= given * sc[ch]; }
                res.V -= given;
                if (res.V < 0) res.V = 0;
            }
        }
    }

    // what was lifted now sits on the brush and smears into the next touch
    if (!res.infinite && liftV > 0) {
        res.V += liftV;
        for (int ch = 0; ch < 3; ++ch) { res.K[ch] += liftK[ch]; res.S[ch] += liftS[ch]; }
    }
}

void init_reservoir(Reservoir &r, const float *K, const float *S, double volume, bool infinite) {
    r.infinite = infinite;
    for (int ch = 0; ch < 3; ++ch) {
        r.k0[ch] = K[ch];
        r.s0[ch] = S[ch];
        r.K[ch] = K[ch] * volume;
        r.S[ch] = S[ch] * volume;
    }
    r.V = volume;
}

struct Bristle {
    float ox, oy;   // position in the unit footprint
    float len;      // 0..1: short bristles only touch under pressure
    Reservoir res;
};

}  // namespace

extern "C" {

// Lay one stroke along a polyline of (x, y, pressure) samples.
void stroke(Canvas *c, const Brush *b, const float *K, const float *S, int n,
            const float *xs, const float *ys, const float *ps) {
    if (n < 1) return;
    Rng rng(b->seed);
    const float PI = 3.14159265f;
    bool infinite = b->load <= 0.f;

    Reservoir soft;
    std::vector<Bristle> br;
    if (b->type == 0) {
        double vol = (double)b->load * b->thick * PI * b->radius * b->radius * b->aspect;
        init_reservoir(soft, K, S, vol, infinite);
    } else if (b->type == 2) {
        // prime the blender with a copy of the wet paint under its first touch
        float r = b->radius;
        double v = 0, k[3] = {0, 0, 0}, sc[3] = {0, 0, 0};
        int x0 = std::max(0, (int)(xs[0] - r)), x1 = std::min(c->W - 1, (int)(xs[0] + r));
        int y0 = std::max(0, (int)(ys[0] - r)), y1 = std::min(c->H - 1, (int)(ys[0] + r));
        for (int py = y0; py <= y1; ++py)
            for (int px = x0; px <= x1; ++px) {
                float dx = px + 0.5f - xs[0], dy = py + 0.5f - ys[0];
                if (dx * dx + dy * dy > r * r) continue;
                size_t i = (size_t)py * c->W + px;
                v += c->V[i];
                for (int ch = 0; ch < 3; ++ch) { k[ch] += c->K[i * 3 + ch]; sc[ch] += c->S[i * 3 + ch]; }
            }
        soft.infinite = false;
        double f = std::max(0.f, b->load);
        soft.V = v * f;
        for (int ch = 0; ch < 3; ++ch) { soft.K[ch] = k[ch] * f; soft.S[ch] = sc[ch] * f; soft.k0[ch] = 0; soft.s0[ch] = 0; }
        infinite = false;
    } else {
        int nb = std::max(1, b->nbristle);
        br.resize(nb);
        for (int i = 0; i < nb; ++i) {
            // even-ish scatter over the disc (golden-angle spiral with jitter)
            float rr = std::sqrt((i + 0.5f) / nb);
            float aa = i * 2.39996323f + rng.uni() * 0.9f;
            rr = clampf(rr + (rng.uni() - 0.5f) * 0.25f, 0.f, 1.f);
            br[i].ox = rr * std::cos(aa);
            br[i].oy = rr * std::sin(aa);
            br[i].len = rng.uni();
            float kj[3], sj[3];
            float g = rng.gauss() * b->jitter;
            for (int ch = 0; ch < 3; ++ch) {
                kj[ch] = K[ch] * std::exp(g + 0.35f * b->jitter * rng.gauss());
                sj[ch] = S[ch];
            }
            double vol = (double)b->load * b->thick * 2.0 * b->bristle_r * b->radius * (0.6 + 0.8 * rng.uni());
            init_reservoir(br[i].res, kj, sj, vol, infinite);
        }
    }

    float step = b->type != 1 ? std::max(0.35f, b->spacing * b->radius)
                              : std::max(0.45f, 0.7f * b->bristle_r);
    float dirx = 1.f, diry = 0.f;
    if (n > 1) {
        float dx = xs[1] - xs[0], dy = ys[1] - ys[0];
        float l = std::sqrt(dx * dx + dy * dy);
        if (l > 1e-6f) { dirx = dx / l; diry = dy / l; }
    }

    auto touch = [&](float x, float y, float p) {
        if (p <= 0.f) return;
        float th = (b->follow ? std::atan2(diry, dirx) : 0.f) + b->angle;
        float r = b->radius * (1.f - b->press_radius + b->press_radius * p);
        float target = b->thick * (1.f - b->press_thick + b->press_thick * p);
        if (b->type != 1) {
            dab(c, x, y, r, r * b->aspect, th, b->hardness, b->type == 2 ? 0.f : target, b->pickup,
                b->tooth, p, soft);
        } else {
            float ct = std::cos(th), st = std::sin(th);
            float reach = p * 1.15f;
            for (auto &q : br) {
                if (q.len > reach) continue;
                float lx = q.ox * r, ly = q.oy * r * b->aspect;
                float bx = x + lx * ct - ly * st;
                float by = y + lx * st + ly * ct;
                float pb = clampf(p * (1.15f - 0.5f * q.len), 0.f, 1.f);
                dab(c, bx, by, b->bristle_r, b->bristle_r, 0.f, 0.35f,
                    target * (0.55f + 0.45f * pb), b->pickup, b->tooth, pb, q.res);
            }
        }
    };

    if (n == 1) { touch(xs[0], ys[0], ps[0]); return; }

    float carry = 0.f;  // distance already travelled since the last dab
    touch(xs[0], ys[0], ps[0]);
    for (int i = 0; i + 1 < n; ++i) {
        float dx = xs[i + 1] - xs[i], dy = ys[i + 1] - ys[i];
        float len = std::sqrt(dx * dx + dy * dy);
        if (len < 1e-6f) continue;
        float ux = dx / len, uy = dy / len;
        // ease the brush round corners instead of snapping
        dirx = dirx * 0.5f + ux * 0.5f;
        diry = diry * 0.5f + uy * 0.5f;
        float dl = std::sqrt(dirx * dirx + diry * diry);
        if (dl > 1e-6f) { dirx /= dl; diry /= dl; }
        float t = step - carry;
        while (t <= len) {
            float f = t / len;
            touch(xs[i] + dx * f, ys[i] + dy * f, ps[i] + (ps[i + 1] - ps[i]) * f);
            t += step;
        }
        carry = len - (t - step);
    }
}

// Many strokes with one brush: per-stroke paint, radius scale and seed.
void strokes(Canvas *c, const Brush *b, int nstrokes, const int *counts,
             const float *xs, const float *ys, const float *ps,
             const float *Ks, const float *Ss, const float *rscale) {
    Brush bb = *b;
    size_t off = 0;
    for (int i = 0; i < nstrokes; ++i) {
        bb.radius = b->radius * (rscale ? rscale[i] : 1.f);
        bb.seed = b->seed + 7919u * (uint32_t)(i + 1);
        stroke(c, &bb, Ks + (size_t)i * 3, Ss + (size_t)i * 3, counts[i], xs + off, ys + off, ps + off);
        off += counts[i];
    }
}

// Let the wet film dry: it becomes part of the dry layer.
void dry_canvas(Canvas *c) {
    size_t n = (size_t)c->W * c->H;
    for (size_t i = 0; i < n; ++i) {
        float v = c->V[i];
        if (v <= 1e-7f) continue;
        for (int ch = 0; ch < 3; ++ch) {
            c->dry[i * 3 + ch] = clampf(km(c->K[i * 3 + ch], c->S[i * 3 + ch], c->dry[i * 3 + ch]), 0.f, 1.f);
            c->K[i * 3 + ch] = 0.f;
            c->S[i * 3 + ch] = 0.f;
        }
        c->hgt[i] += v;
        c->V[i] = 0.f;
    }
}

// What the painting looks like now. out: H*W*3 floats, sRGB-encoded 0..1.
// relief > 0 adds the faint raking light that shows the body of the paint.
void render(const Canvas *c, float *out, float relief, float lx, float ly, float tooth_amt) {
    int W = c->W, H = c->H;
    auto height = [&](int x, int y) {
        x = std::min(std::max(x, 0), W - 1);
        y = std::min(std::max(y, 0), H - 1);
        size_t i = (size_t)y * W + x;
        return c->hgt[i] + c->V[i] + tooth_amt * c->tooth[i];
    };
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            size_t i = (size_t)y * W + x;
            float shade = 1.f;
            if (relief > 0.f) {
                float gx = (height(x + 1, y) - height(x - 1, y)) * 0.5f;
                float gy = (height(x, y + 1) - height(x, y - 1)) * 0.5f;
                shade = 1.f - relief * (gx * lx + gy * ly);
                shade = clampf(shade, 0.55f, 1.6f);
            }
            for (int ch = 0; ch < 3; ++ch) {
                float r = km(c->K[i * 3 + ch], c->S[i * 3 + ch], c->dry[i * 3 + ch]);
                out[i * 3 + ch] = to_srgb(r * shade);
            }
        }
    }
}

// Distance (px) from every pixel inside a mask to the nearest pixel outside it.
// Felzenszwalb & Huttenlocher's two-pass lower-envelope method.
void edt(const uint8_t *mask, int W, int H, float *out) {
    const float INF = 1e20f;
    int n = std::max(W, H);
    std::vector<float> f(n), d(n), z(n + 1);
    std::vector<int> v(n);
    auto pass = [&](int len) {
        int k = 0;
        v[0] = 0; z[0] = -INF; z[1] = INF;
        for (int q = 1; q < len; ++q) {
            float s;
            while (true) {
                int p = v[k];
                s = ((f[q] + (float)q * q) - (f[p] + (float)p * p)) / (2.f * q - 2.f * p);
                if (s <= z[k] && k > 0) --k; else break;
            }
            ++k; v[k] = q; z[k] = s; z[k + 1] = INF;
        }
        k = 0;
        for (int q = 0; q < len; ++q) {
            while (z[k + 1] < q) ++k;
            int p = v[k];
            d[q] = (float)(q - p) * (q - p) + f[p];
        }
    };
    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < H; ++y) f[y] = mask[(size_t)y * W + x] ? INF : 0.f;
        pass(H);
        for (int y = 0; y < H; ++y) out[(size_t)y * W + x] = d[y];
    }
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) f[x] = out[(size_t)y * W + x];
        pass(W);
        for (int x = 0; x < W; ++x) out[(size_t)y * W + x] = std::sqrt(std::min(d[x], 1e12f));
    }
}

// A small view of the whole canvas for the time-lapse: every `stride`-th pixel block,
// averaged from a few samples. out: ceil(H/stride) * ceil(W/stride) * 3, sRGB 0..1.
void preview(const Canvas *c, float *out, int stride) {
    int ow = (c->W + stride - 1) / stride, oh = (c->H + stride - 1) / stride;
    int sub = std::max(1, stride / 2);
    for (int oy = 0; oy < oh; ++oy) {
        for (int ox = 0; ox < ow; ++ox) {
            float acc[3] = {0, 0, 0};
            int cnt = 0;
            for (int y = oy * stride; y < std::min(c->H, (oy + 1) * stride); y += sub)
                for (int x = ox * stride; x < std::min(c->W, (ox + 1) * stride); x += sub) {
                    size_t i = (size_t)y * c->W + x;
                    for (int ch = 0; ch < 3; ++ch)
                        acc[ch] += km(c->K[i * 3 + ch], c->S[i * 3 + ch], c->dry[i * 3 + ch]);
                    ++cnt;
                }
            for (int ch = 0; ch < 3; ++ch)
                out[((size_t)oy * ow + ox) * 3 + ch] = to_srgb(acc[ch] / std::max(1, cnt));
        }
    }
}

}  // extern "C"
