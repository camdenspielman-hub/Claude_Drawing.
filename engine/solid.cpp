// Sculpted solids.
//
// A solid is built the way a modeller works in clay: lumps (ellipsoids, tapering rods) pressed
// together so they run into one another, and hollows pressed into them. Nothing here makes a
// picture. These routines answer the questions a draughtsman asks of the model on the table:
// how far is this point from the surface? where does this line of sight first touch it? which
// way does the surface face there? can that spot see the window? how shut in is it?
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

const int STRIDE = 16;      // doubles per primitive: type, op, group, mirror, k, then 11 parameters
const int MAXG = 8;

inline double smin(double a, double b, double k) {
    if (k <= 0) return std::min(a, b);
    double h = std::max(k - std::fabs(a - b), 0.0) / k;
    return std::min(a, b) - h * h * k * 0.25;
}
inline double smax(double a, double b, double k) { return -smin(-a, -b, k); }

inline uint32_t hash3(int x, int y, int z, uint32_t seed) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + (uint32_t)z * 2147483647u + seed * 362437u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
inline double lattice(int x, int y, int z, uint32_t seed) { return (hash3(x, y, z, seed) & 0xffffff) / 16777215.0; }
double noise(double x, double y, double z, uint32_t seed) {
    double fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    double tx = x - fx, ty = y - fy, tz = z - fz;
    tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty); tz = tz * tz * (3 - 2 * tz);
    double c[2][2];
    for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) {
            double a = lattice(ix, iy + j, iz + k, seed), b = lattice(ix + 1, iy + j, iz + k, seed);
            c[j][k] = a + (b - a) * tx;
        }
    double a = c[0][0] + (c[1][0] - c[0][0]) * ty, b = c[0][1] + (c[1][1] - c[0][1]) * ty;
    return a + (b - a) * tz;
}

// distance from p to one primitive
inline double prim_dist(const double *q, double x, double y, double z) {
    int type = (int)q[0];
    const double *p = q + 5;
    if (q[3] > 0.5) z = std::fabs(z);
    if (type == 0) {                         // ellipsoid: centre p[0..2], radii p[3..5]
        double dx = x - p[0], dy = y - p[1], dz = z - p[2];
        double ax = dx / p[3], ay = dy / p[4], az = dz / p[5];
        double k0 = std::sqrt(ax * ax + ay * ay + az * az);
        double bx = ax / p[3], by = ay / p[4], bz = az / p[5];
        double k1 = std::sqrt(bx * bx + by * by + bz * bz);
        if (k1 < 1e-9) return -std::min(p[3], std::min(p[4], p[5]));
        return k0 * (k0 - 1.0) / k1;
    }
    // tapering rod: from a = p[0..2] (radius p[6]) to b = p[3..5] (radius p[7]); optionally
    // flattened along one axis (p[8] = axis or -1, p[9] = how thin: 1 = round)
    double ax = p[0], ay = p[1], az = p[2], bx = p[3], by = p[4], bz = p[5];
    double s = 1.0;
    int axis = (int)p[8];
    if (axis >= 0) {
        s = p[9];
        double inv = 1.0 / s;
        if (axis == 0) { double m = 0.5 * (ax + bx); x = m + (x - m) * inv; ax = m + (ax - m) * inv; bx = m + (bx - m) * inv; }
        else if (axis == 1) { double m = 0.5 * (ay + by); y = m + (y - m) * inv; ay = m + (ay - m) * inv; by = m + (by - m) * inv; }
        else { double m = 0.5 * (az + bz); z = m + (z - m) * inv; az = m + (az - m) * inv; bz = m + (bz - m) * inv; }
    }
    double ex = bx - ax, ey = by - ay, ez = bz - az;
    double px = x - ax, py = y - ay, pz = z - az;
    double ee = ex * ex + ey * ey + ez * ez;
    double h = ee > 1e-12 ? (px * ex + py * ey + pz * ez) / ee : 0.0;
    h = std::min(1.0, std::max(0.0, h));
    double cx = px - h * ex, cy = py - h * ey, cz = pz - h * ez;
    double d = std::sqrt(cx * cx + cy * cy + cz * cz) - (p[6] + (p[7] - p[6]) * h);
    return d * std::min(s, 1.0);
}

// A wound ball: a squashed sphere (centre p[0..2], squash p[3..5], radius p[6]) with belts of
// parallel turns of a strand (radius p[7]) lying on it in layers. The next (int)p[8] rows describe
// the belts: axis (3), offset along the axis, half-width, height of the layer it lies in.
inline double wound_dist(const double *q, double x, double y, double z) {
    const double *p = q + 5;
    double px = (x - p[0]) / p[3], py = (y - p[1]) / p[4], pz = (z - p[2]) / p[5];
    double rho = std::sqrt(px * px + py * py + pz * pz);
    double R = p[6], r = p[7];
    int nb = (int)p[8];
    double smin_scale = std::min(p[3], std::min(p[4], p[5]));
    if (rho < 1e-9) return -R;
    if (rho > R + 4 * r + 2.0) return (rho - R - 4 * r - 1.0) * smin_scale;     // far off: no need to look at the belts
    double dx = px / rho, dy = py / rho, dz = pz / rho;
    double pitch = 2 * r / R;                 // spacing of neighbouring turns, as a sine
    double h = 0;
    for (int b = 0; b < nb; b++) {
        const double *w = q + (b + 1) * STRIDE + 5;
        double s = dx * w[0] + dy * w[1] + dz * w[2] - w[3];
        if (std::fabs(s) > w[4]) continue;
        double e = s / pitch;
        e -= std::floor(e + 0.5);             // -0.5 .. 0.5 across one strand
        double prof = r * (0.40 + 0.60 * std::cos(3.14159265358979 * e));
        // the belt thins out at its two edges
        double edge = std::min(1.0, (w[4] - std::fabs(s)) / (0.5 * pitch));
        double hi = w[5] + prof * (0.55 + 0.45 * edge);
        if (hi > h) h = hi;
    }
    return (rho - R - h) * smin_scale * 0.5;
}

struct Model {
    const double *prims;
    int np;
    double amp, freq;
    uint32_t seed;
    double amp2, freq2;
};

inline double dist(const Model &m, double x, double y, double z, int *group) {
    double acc[MAXG];
    bool used[MAXG];
    for (int g = 0; g < MAXG; g++) { acc[g] = 1e9; used[g] = false; }
    for (int i = 0; i < m.np; i++) {
        const double *q = m.prims + i * STRIDE;
        int g = (int)q[2];
        double d;
        if ((int)q[0] == 3) { d = wound_dist(q, x, y, z); i += (int)q[5 + 8]; }
        else d = prim_dist(q, x, y, z);
        if (q[1] < 0.5) acc[g] = used[g] ? smin(acc[g], d, q[4]) : d;
        else if (used[g]) acc[g] = smax(acc[g], -d, q[4]);
        used[g] = used[g] || q[1] < 0.5;
    }
    double best = 1e9;
    int bg = 0;
    for (int g = 0; g < MAXG; g++)
        if (used[g] && acc[g] < best) { best = acc[g]; bg = g; }
    if (m.amp > 0 && best < 4 * m.amp + 0.3)
        best -= m.amp * (noise(x * m.freq, y * m.freq, z * m.freq, m.seed) - 0.5) * 2;
    if (m.amp2 > 0 && best < 4 * m.amp2 + 0.1)
        best -= m.amp2 * (noise(x * m.freq2, y * m.freq2, z * m.freq2, m.seed + 17) - 0.5) * 2;
    if (group) *group = bg;
    return best;
}

inline Model model(const double *prims, int np, const double *rough) {
    return Model{prims, np, rough[0], rough[1], (uint32_t)rough[2], rough[3], rough[4]};
}

}  // namespace

extern "C" {

void solid_eval(const double *prims, int np, const double *rough, const double *P, int n, double *d, int *g) {
    Model m = model(prims, np, rough);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) d[i] = dist(m, P[3 * i], P[3 * i + 1], P[3 * i + 2], g ? g + i : nullptr);
}

// Where each line of sight (origin O, unit direction D) first touches the solid: t, or -1.
// near[i] is how close the line came, as a fraction of the distance travelled (for misses).
void solid_trace(const double *prims, int np, const double *rough, const double *O, const double *D, int n,
                 const double *centre, double radius, int maxsteps, double eps, double *t, double *near_) {
    Model m = model(prims, np, rough);
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; i++) {
        const double *o = O + 3 * i, *dd = D + 3 * i;
        t[i] = -1;
        if (near_) near_[i] = 1e9;
        double ox = o[0] - centre[0], oy = o[1] - centre[1], oz = o[2] - centre[2];
        double b = ox * dd[0] + oy * dd[1] + oz * dd[2];
        double c = ox * ox + oy * oy + oz * oz - radius * radius;
        double disc = b * b - c;
        if (disc <= 0) continue;
        double sq = std::sqrt(disc);
        double tt = std::max(-b - sq, 1e-4), tend = -b + sq;
        double closest = 1e9;
        for (int s = 0; s < maxsteps && tt < tend; s++) {
            double d = dist(m, o[0] + tt * dd[0], o[1] + tt * dd[1], o[2] + tt * dd[2], nullptr);
            closest = std::min(closest, d / tt);
            if (d < eps) { t[i] = tt; break; }
            tt += std::max(d * 0.8, eps * 0.5);
        }
        if (near_) near_[i] = closest;
    }
}

void solid_normal(const double *prims, int np, const double *rough, const double *P, int n, double e, double *N) {
    Model m = model(prims, np, rough);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
        double a = dist(m, x + e, y - e, z - e, nullptr), b = dist(m, x - e, y - e, z + e, nullptr);
        double c = dist(m, x - e, y + e, z - e, nullptr), d = dist(m, x + e, y + e, z + e, nullptr);
        double nx = a - b - c + d, ny = -a - b + c + d, nz = -a + b - c + d;
        double l = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (l < 1e-12) { nx = 0; ny = 1; nz = 0; l = 1; }
        N[3 * i] = nx / l; N[3 * i + 1] = ny / l; N[3 * i + 2] = nz / l;
    }
}

// How much of the window a point sees past the rest of the solid: 0 (none) .. 1 (all of it).
// `soft` sets how quickly a shadow's edge spreads with distance from what casts it.
void solid_shadow(const double *prims, int np, const double *rough, const double *P, int n, const double *L,
                  double maxdist, double soft, double *out) {
    Model m = model(prims, np, rough);
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; i++) {
        double res = 1.0, t = 0.03;
        for (int s = 0; s < 48 && t < maxdist; s++) {
            double h = dist(m, P[3 * i] + L[0] * t, P[3 * i + 1] + L[1] * t, P[3 * i + 2] + L[2] * t, nullptr);
            if (h < 0.0015) { res = 0; break; }
            res = std::min(res, soft * h / t);
            t += std::min(std::max(h * 0.8, 0.02), 0.5);
        }
        res = std::max(0.0, std::min(1.0, res));
        out[i] = res * res * (3 - 2 * res);
    }
}

// How open a point is to the room: 1 on a free surface, toward 0 in a deep hollow or crevice.
void solid_open(const double *prims, int np, const double *rough, const double *P, const double *N, int n,
                double reach, double *out) {
    Model m = model(prims, np, rough);
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; i++) {
        double occ = 0, w = 1.0, wsum = 0;
        for (int s = 1; s <= 6; s++) {
            double h = reach * s / 6.0;
            double d = dist(m, P[3 * i] + N[3 * i] * h, P[3 * i + 1] + N[3 * i + 1] * h, P[3 * i + 2] + N[3 * i + 2] * h, nullptr);
            occ += w * std::max(0.0, h - d) / h;
            wsum += w;
            w *= 0.8;
        }
        out[i] = std::max(0.0, 1.0 - occ / wsum);
    }
}

}  // extern "C"
