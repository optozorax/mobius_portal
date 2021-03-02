#version 100

precision highp float;

varying vec2 uv;
varying vec2 uv_screen;
uniform vec3 angles;

uniform mat4 first;
uniform mat4 first_inv;
uniform mat4 second;
uniform mat4 second_inv;

uniform float add_gray_after_teleportation;

uniform int teleport_light;

uniform sampler2D Texture;
uniform sampler2D Mobius;

#define PI acos(-1.)
#define PI2 (PI/2)

int texture_counti = 30;
float texture_count = 30.;

float round(float a) {
    return floor(a + 0.5);
}

struct Ray
{
    vec3 o;     // origin
    vec3 d;     // direction
};

struct crd3 {
    vec3 i;
    vec3 j;
    vec3 k;
    vec3 pos;
};

mat3 transpose(mat3 matrix) {
    vec3 row0 = matrix[0];
    vec3 row1 = matrix[1];
    vec3 row2 = matrix[2];
    mat3 result = mat3(
        vec3(row0.x, row1.x, row2.x),
        vec3(row0.y, row1.y, row2.y),
        vec3(row0.z, row1.z, row2.z)
    );
    return result;
}

float det(mat2 matrix) {
    return matrix[0].x * matrix[1].y - matrix[0].y * matrix[1].x;
}

mat3 inverse(mat3 matrix) {
    vec3 row0 = matrix[0];
    vec3 row1 = matrix[1];
    vec3 row2 = matrix[2];

    vec3 minors0 = vec3(
        det(mat2(row1.y, row1.z, row2.y, row2.z)),
        det(mat2(row1.z, row1.x, row2.z, row2.x)),
        det(mat2(row1.x, row1.y, row2.x, row2.y))
    );
    vec3 minors1 = vec3(
        det(mat2(row2.y, row2.z, row0.y, row0.z)),
        det(mat2(row2.z, row2.x, row0.z, row0.x)),
        det(mat2(row2.x, row2.y, row0.x, row0.y))
    );
    vec3 minors2 = vec3(
        det(mat2(row0.y, row0.z, row1.y, row1.z)),
        det(mat2(row0.z, row0.x, row1.z, row1.x)),
        det(mat2(row0.x, row0.y, row1.x, row1.y))
    );

    mat3 adj = transpose(mat3(minors0, minors1, minors2));

    return (1.0 / dot(row0, minors0)) * adj;
}

const crd3 crdDefault = crd3(vec3(1., 0., 0.), vec3(0., 1., 0.), vec3(0., 0., 1.), vec3(0.));

float project(vec3 a, vec3 to) {
    return dot(a, to) / dot(to, to);
}

vec3 projection(vec3 a, vec3 to) {
    return to * project(a, to);
}

struct QuadraticEquationResult {
    bool is_some;
    float x1;
    float x2;
};

float to_line(Ray line, vec3 pos) {
    return dot(pos - line.o, line.d);
}

vec3 from_line(Ray line, float pos) {
    return line.o + line.d * pos;
}

QuadraticEquationResult solve_quadratic_equation(float a, float b, float c) {
    float d = b*b - 4.*a*c;
    if (d < 0.0 || abs(a) < 1e-6) {
        return QuadraticEquationResult(false, 0., 0.);
    } else {
        d = sqrt(d);
        a = 1. / (2. * a);
        b = -b;
        return QuadraticEquationResult(true, (b - d) * a, (b + d) * a);
    }
}

struct Angles3 {
    float alpha;
    float beta;
};

struct SphereLineIntersection {
    bool is_some;
    Angles3 o;
    Angles3 od;
    vec3 vo;
    vec3 vod;
};

struct Sphere {
    vec3 pos;
    float r;
};

Angles3 from_sphere(Sphere s, vec3 a) {
    a -= s.pos;
    return Angles3(atan(a.z, a.x) + PI, atan(sqrt(a.x*a.x + a.z*a.z), a.y));
}

SphereLineIntersection intersect_sphere_line(Sphere s, Ray r) {
    vec3 p = r.o - s.pos;
    float a = dot(r.d, r.d);
    float b = 2. * dot(p, r.d);
    float c = dot(p, p) - s.r * s.r;
    QuadraticEquationResult result = solve_quadratic_equation(a, b, c);
    if (result.is_some) {
        vec3 o = r.o + r.d * result.x1;
        vec3 od = r.o + r.d * result.x2;
        return SphereLineIntersection(true, from_sphere(s, o), from_sphere(s, od), o, od);
    } else {
        return SphereLineIntersection(false, Angles3(0., 0.), Angles3(0., 0.), vec3(0.), vec3(0.));
    }
}

int simplemod(int a, int mod) {
    if (a >= mod) {
        a -= mod;
    }
    if (a < 0) {
        a = mod + a;
    }
    return a;
}

float get_best_init_value_ipos(int alpha1i, int beta1i, int alpha2i, int beta2i) {
    int xi = alpha1i * texture_counti + alpha2i;
    int yi = beta1i * texture_counti + beta2i;

    float x = (float(xi) + 0.5) / (texture_count * texture_count);
    float y = (float(yi) + 0.5) / (texture_count * texture_count);

    vec3 color = texture2D(Mobius, vec2(x, y)).rgb;

    float value = (color.r + color.g * 256. + color.b * 256. * 256.) / (256. * 256.) * 2. * PI;

    return value;
}

ivec4 get_angles(Angles3 o, Angles3 od) {
    float alpha1 = o.alpha / (2. * PI);
    float beta1 = o.beta / PI;
    float alpha2 = od.alpha / (2. * PI);
    float beta2 = od.beta / PI;

    int alpha1i = simplemod(int(floor(alpha1 * texture_count)), texture_counti);
    int beta1i = simplemod(int(floor(beta1 * texture_count)), texture_counti);
    int alpha2i = simplemod(int(floor(alpha2 * texture_count)), texture_counti);
    int beta2i = simplemod(int(floor(beta2 * texture_count)), texture_counti);

    return ivec4(alpha1i, beta1i, alpha2i, beta2i);
}

float get_best_init_value_angles(Angles3 o, Angles3 od) {
    ivec4 result = get_angles(o, od);
    return get_best_init_value_ipos(result.x, result.y, result.z, result.w);
}

vec2 get_best_init_value_ray(Ray r) {
    SphereLineIntersection hits = intersect_sphere_line(Sphere(vec3(0.), 1.55), r);
    if (hits.is_some) {
        return vec2(
            get_best_init_value_angles(hits.o, hits.od),
            get_best_init_value_angles(hits.od, hits.o)
        );
    } else {
        return vec2(-1., -1.);
    }
}

crd3 orthonormalize(crd3 a) {
    a.i = normalize(a.i);
    a.j = normalize(a.j);
    a.k = normalize(a.k);
    a.j = a.j - projection(a.j, a.i);
    a.k = a.k - projection(a.k, a.i) - projection(a.k, a.j);
    return a;
}

vec3 projectDir(crd3 crd, vec3 d) {
    // i*result.x + j*result.y + k*result.z = d
    return inverse(mat3(crd.i, crd.j, crd.k))*d;
}

vec3 projectCrd(crd3 crd, vec3 o) {
    // i*result.x + j*result.y + k*result.z + pos = o
    return projectDir(crd, o-crd.pos);
}

vec3 unprojectDir(crd3 crd, vec3 d) {
    return crd.i * d.x + crd.j * d.y + crd.k * d.z;
}

vec3 unprojectCrd(crd3 crd, vec3 d) {
    return crd.i * d.x + crd.j * d.y + crd.k * d.z + crd.pos;
}

vec2 two_lines_nearest_points(Ray a, Ray b) {
    vec3 n = cross(a.d, b.d);
    vec3 n1 = cross(a.d, n);
    vec3 n2 = cross(b.d, n);
    return vec2(
        dot(b.o-a.o, n2)/dot(a.d, n2),
        dot(a.o-b.o, n1)/dot(b.d, n1)
    );
}

vec3 mobiusO(float u) {
    return vec3(cos(u), 0, sin(u));
}

vec3 mobiusD(float u) {
    return vec3(cos(u/2.)*cos(u), sin(u/2.), cos(u/2.)*sin(u))/2.; // mobius
}

float mobiusStep(float u, Ray r) {
    float cosu = cos(u);
    float sinu = sin(u);
    float cosu2 = cos(u/2.);
    float sinu2 = sin(u/2.);

    Ray m = Ray(vec3(cosu, 0., sinu), vec3(cosu2 * cosu, sinu2, cosu2 * sinu)/2.);

    vec3 n = cross(m.d, r.d);
    vec3 n2 = cross(r.d, n);
    vec3 p = m.o - r.o;
    float t_mobius_middle = -dot(p, n2)/dot(m.d, n2);

    if (abs(t_mobius_middle) < 1.) {
        return abs(dot(p, n)) / length(n);
    } else {
        vec3 up = m.o + m.d;
        vec3 down = m.o - m.d;

        return min(
            length(up - from_line(r, to_line(r, up))), 
            length(down - from_line(r, to_line(r, down)))
        );
    }
}

vec2 mobius_step_ts(float u, Ray r) {
    return two_lines_nearest_points(Ray(mobiusO(u), mobiusD(u)), r);
}

vec3 mobius_d1(float v, float u) {
    float a = sin(u/2.);
    float b = cos(u/2.);
    float c = sin(u);
    float d = cos(u);
    return vec3(
        b*d/2., 
        b*c/2., 
        a/2.
    );
}

vec3 mobius_d2(float v, float u) {
    float a = sin(u/2.);
    float b = cos(u/2.);
    float c = sin(u);
    float d = cos(u);
    return vec3(
        -(0.25*v*a*d+0.5*v*c*b+c), 
        -(0.25*(v*a*c-2.*d*(v*b+2.))), 
        0.25*v*b
    );
}

struct MobiusIntersect {
    bool hit;
    float t;
    float u;
    float v;
    vec3 n;
};

float clampmod(float a, float max) {
    // a = max + mod(a, max);
    if (a < 0.) {
        a += max;
    }
    if (a > max) {
        a -= max;
    }
    return a;
}

float clampangle(float a) {
    return clampmod(a, 2. * PI);
}

struct SearchResult {
    float t;
    float u;
    float v;
};

SearchResult findBestApprox(float u, Ray r, SearchResult best) {
    float eps_der = 0.00005;
    float eps_newton = 0.0001;

    float step = mobiusStep(u, r);
    for (int k = 0; k < 10; k++) {
        if (step < eps_newton) {
            break;
        }
        float fx = step;
        float fx1 = mobiusStep(u + eps_der, r);
        float du = -fx/(fx1 - fx)*eps_der;
        u = clampangle(u + du);
        step = mobiusStep(u, r);
        if (best.t > 0. && abs(u-best.u) < 0.01) {
            return SearchResult(-1., 0., 0.);
        }
    }

    if (step < eps_newton) {
        vec2 ts = mobius_step_ts(u, r);
        return SearchResult(ts.y, u, ts.x);    
    } else {
        return SearchResult(-1., 0., 0.);
    }
}

SearchResult updateBestApprox(SearchResult best, SearchResult current) {
    if (current.t > 0. && (current.v > -1. && current.v < 1.)) {
        if (best.t < 0.) {
            best = current;
        } else {
            if (current.t < best.t) {
                best = current;
            }
        }
    }
    return best;
}

vec3 normalizeNormal(vec3 normal, Ray r) {
    normal = normalize(normal);
    if (dot(normal, r.d) > 0.) {
        normal *= -1.;
    }
    return normal;
}

struct MobiusRayMarchingIntersect {
    bool hit;
    float t;
    vec3 n;
};

mat2 Rot(float a) {
    float s=sin(a), c=cos(a);
    return mat2(c, -s, s, c);
}

float sdBox2d(vec2 p, vec2 s) {
    p = abs(p)-s;
    return length(max(p, 0.))+min(max(p.x, p.y), 0.);
}

// thank you https://www.shadertoy.com/view/3sycRV#
float GetDist(vec3 p) {
    float r1 = 1.;
    vec2 cp = vec2(length(p.xz) - 1., p.y);
    float a = atan(p.x, p.z) + PI/2.; // polar angle between -pi and pi
    cp *= Rot(a * .5);
    cp.y = abs(cp.y) - .2;
    float d1 = sdBox2d(cp, vec2(0.,.40));
    return d1 * 0.6;
}

vec3 GetNormal(vec3 p) {
    float d = GetDist(p);
    vec2 e = vec2(.001, 0);
    
    vec3 n = d - vec3(
        GetDist(p-e.xyy),
        GetDist(p-e.yxy),
        GetDist(p-e.yyx));
    
    return normalize(n);
}

#define MAX_STEPS 100
#define MAX_DIST 1000.
#define SURF_DIST .001

MobiusRayMarchingIntersect intersectMobiusRayMarching(Ray r) {
    float dO = 0.;
    
    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 p = r.o + r.d * dO;
        float dS = GetDist(p);
        dO += dS;
        if (dO > MAX_DIST || i > MAX_STEPS - 2) {
            return MobiusRayMarchingIntersect(false, 0., vec3(0.));
        }
        if (abs(dS) < SURF_DIST) {
            return MobiusRayMarchingIntersect(true, dO, GetNormal(r.o + r.d * dO));
        }
    }
}

bool intersect_mobius_sphere(Ray r) {
    vec3 op = -r.o;
    float b = dot(op, r.d);
    float det = b * b - dot(op, op) + 2.4055; // 1.55²
    return det >= 0.;
}

SearchResult findBest(Ray r) {
    SearchResult best = SearchResult(-1., 0., 0.);

    if (intersect_mobius_sphere(r)) {
        best = updateBestApprox(best, findBestApprox(0., r, best));
        best = updateBestApprox(best, findBestApprox(PI, r, best));
        for (int i = 0; i < 2; i++) {
            float u = float(i*2 + 1)/4. * 2. * PI;
            best = updateBestApprox(best, findBestApprox(u, r, best));
        }
        for (int i = 0; i < 4; i++) {
            float u = float(i*2 + 1)/8. * 2. * PI;
            best = updateBestApprox(best, findBestApprox(u, r, best));
        }
        if (best.t < 0.) {
            return best;
        }
        best = updateBestApprox(best, findBestApprox(float(8 - 1)/16. * 2. * PI + PI, r, best));
        best = updateBestApprox(best, findBestApprox(float(8 + 1)/16. * 2. * PI + PI, r, best));
    }
    return best;
}

struct SphereIntersect {
    bool hit;
    float t;
    vec3 n;
};

SphereIntersect intersectSphere(Ray r)
{
    vec3 sp = vec3(0, 0., 0.);
    float sr = 1.55;

    vec3 op = sp - r.o;
    float b = dot(op, r.d);
    float det = b * b - dot(op, op) + sr * sr;
    if (det < 0.) return SphereIntersect(false, 0., vec3(0.));

    det = sqrt(det);
    float t = b - det;
    if (t < 0.) t = b + det;
    if (t < 0.) return SphereIntersect(false, 0., vec3(0.));

    return SphereIntersect(true, t, (r.o + t * r.d - sp) / sr);
}

MobiusIntersect intersectMobius2(Ray r) {
    SearchResult best = findBest(r);
    if (best.t >= 0.) {
        vec3 normal = normalizeNormal(cross(mobius_d1(best.v, best.u), mobius_d2(best.v, best.u)), r);
        return MobiusIntersect(true, best.t, best.u, best.v, normal);
    } else {
        return MobiusIntersect(false, 0., 0., 0., vec3(0.));
    }
}

struct Plane {
    crd3 repr;
};

struct PlaneIntersect {
    bool hit;
    float t;
    float u;
    float v;
    vec3 n;
};

vec3 color(float r, float g, float b) {
    return vec3(r*r, g*g, b*b);
}

PlaneIntersect intersectPlane(Ray r, Plane p) {
    vec3 ro = projectCrd(p.repr, r.o);
    vec3 rd = projectDir(p.repr, r.d);

    float t = -ro.z/rd.z;
    if (t < 0.) {
        return PlaneIntersect(false, 0., 0., 0., vec3(0.));
    } else {
        vec3 pos = ro + rd * t; 
        return PlaneIntersect(true, t, pos.x, pos.y, normalizeNormal(p.repr.k, r));
    }
}

float deg2rad(float deg) {
    return deg/180. * PI;
}

vec3 addNormalToColor(vec3 color, vec3 normal, vec3 direction) {
    const float not_dark_count = 0.4;
    color *= (abs(dot(normalize(direction), normalize(normal))) + not_dark_count) / (1. + not_dark_count);
    return color;
}

vec3 gridColor(vec3 start, vec2 uv) {
    uv = uv - vec2(0.125, 0.125);
    const float fr = 3.14159*8.0;
    vec3 col = start;
    col += 0.4*smoothstep(-0.01,0.01,cos(uv.x*fr*0.5)*cos(uv.y*fr*0.5)); 
    float wi = smoothstep(-1.0,-0.98,cos(uv.x*fr))*smoothstep(-1.0,-0.98,cos(uv.y*fr));
    col *= wi;
    
    return col;
}

vec3 mulDir(mat4 matrix, vec3 vec) {
    return (matrix * vec4(vec, 0.)).xyz;
}

vec3 mulCrd(mat4 matrix, vec3 vec) {
    return (matrix * vec4(vec, 1.)).xyz;
}

struct CylinderIntersectResult {
    bool is_some;
    vec3 color;
};

CylinderIntersectResult intersect_cylinder(Ray ray) {
    vec3 A = vec3(0.);
    vec3 B = vec3(0., 1., 0.);

    vec3 V = ray.d;
    vec3 P = ray.o - A;
    vec3 D = B - A;

    float r = 0.5;

    float vv = dot(V, V);
    float pp = dot(P, P);
    float dd = dot(D, D);

    float dv = dot(D, V);
    float dp = dot(D, P);
    float pv = dot(P, V);

    float a = dd * vv -  dv * dv;
    float b = 2.0 * (dd * pv - dp * dv);
    float c = dd * pp - dp * dp - r*r * dd;

    QuadraticEquationResult result = solve_quadratic_equation(a, b, c);
    if (result.is_some) {
        vec3 o = ray.o + ray.d * result.x1;
        vec3 od = ray.o + ray.d * result.x2;
        if (abs(o.y) < 0.4 && abs(od.y) < 0.4) {
            return CylinderIntersectResult(true, vec3(0., clampmod(result.x2, 1.), clampmod(result.x1, 1.)));
        } else if (abs(o.y) < 0.4) {
            return CylinderIntersectResult(true, vec3(0., 0., clampmod(result.x1, 1.)));
        } else if (abs(od.y) < 0.4) {
            return CylinderIntersectResult(true, vec3(0., 0., clampmod(result.x2, 1.)));
        } else {
            return CylinderIntersectResult(false, vec3(0.));    
        }
    } else {
        return CylinderIntersectResult(false, vec3(0.));
    }
}

// level is [0,5], assumed to be a whole number
vec3 rainbow(float level)
{
    /*
        Target colors
        =============
        
        L  x   color
        0  0.0 vec4(1.0, 0.0, 0.0, 1.0);
        1  0.2 vec4(1.0, 0.5, 0.0, 1.0);
        2  0.4 vec4(1.0, 1.0, 0.0, 1.0);
        3  0.6 vec4(0.0, 0.5, 0.0, 1.0);
        4  0.8 vec4(0.0, 0.0, 1.0, 1.0);
        5  1.0 vec4(0.5, 0.0, 0.5, 1.0);
    */
    
    float r = float(level <= 2.0) + float(level > 4.0) * 0.5;
    float g = max(1.0 - abs(level - 2.0) * 0.5, 0.0);
    float b = (1.0 - (level - 4.0) * 0.5) * float(level >= 4.0);
    return vec3(r, g, b);
}

vec3 smoothRainbow (float x)
{
    float level1 = floor(x*6.0);
    float level2 = min(6.0, floor(x*6.0) + 1.0);
    
    vec3 a = rainbow(level1);
    vec3 b = rainbow(level2);
    
    return mix(a, b, fract(x*6.0));
}

vec3 intersectScene(Ray r) {
    Plane p = Plane(crdDefault);
    float size = 4.5;
    float scale = 1./(size * 2.);

    PlaneIntersect hitp;

    float gray = 1.;
    
    for (int i = 0; i < 100; ++i) {
        float current_t = 1e10;
        vec3 current_color = color(0.6, 0.6, 0.6);

        p.repr.pos.z = size;
        hitp = intersectPlane(r, p);
        if (hitp.hit && abs(hitp.u) < size && abs(hitp.v) < size && hitp.t < current_t) {
            current_color = addNormalToColor(gridColor(color(0.6, 0.6, 0.6), vec2(hitp.u, hitp.v) * scale), hitp.n, r.d);
            current_t = hitp.t;
        }

        p.repr.pos.z = -size;
        hitp = intersectPlane(r, p);
        if (hitp.hit && abs(hitp.u) < size && abs(hitp.v) < size && hitp.t < current_t) {
            current_color = addNormalToColor(gridColor(color(0.6, 0.6, 0.6), vec2(hitp.u, hitp.v) * scale), hitp.n, r.d);
            current_t = hitp.t;
        }

        p.repr.pos.z = 0.;
        p.repr.i = crdDefault.k;
        p.repr.k = crdDefault.i;

        p.repr.pos.x = size;
        hitp = intersectPlane(r, p);
        if (hitp.hit && abs(hitp.u) < size && abs(hitp.v) < size && hitp.t < current_t) {
            current_color = color(0.6, 0.6, 0.6);
            vec3 new_color = gridColor(current_color, vec2(hitp.u, hitp.v) * scale);
            // current_color = (current_color*2. + new_color)/3.;
            current_color = new_color;
            current_color = addNormalToColor(current_color, hitp.n, r.d);
            // current_color *= texture2D(Texture, (vec2(-hitp.u, -hitp.v) + vec2(size, size))/(size * 2.)).rgb;
            current_t = hitp.t;
        }

        p.repr.pos.x = -size;
        hitp = intersectPlane(r, p);
        if (hitp.hit && abs(hitp.u) < size && abs(hitp.v) < size && hitp.t < current_t) {
            current_color = addNormalToColor(gridColor(color(0.6, 0.6, 0.6), vec2(hitp.u, hitp.v) * scale), hitp.n, r.d);
            current_t = hitp.t;
        }

        p.repr.pos.x = 0.;
        p.repr.i = crdDefault.i;
        p.repr.j = crdDefault.k;
        p.repr.k = crdDefault.j;

        p.repr.pos.y = size;
        hitp = intersectPlane(r, p);
        if (hitp.hit && abs(hitp.u) < size && abs(hitp.v) < size && hitp.t < current_t) {
            current_color = addNormalToColor(gridColor(color(0.6, 0.6, 0.6), vec2(hitp.u, hitp.v) * scale), hitp.n, r.d);
            current_t = hitp.t;
        }

        p.repr.pos.y = -size;
        hitp = intersectPlane(r, p);
        if (hitp.hit && abs(hitp.u) < size && abs(hitp.v) < size && hitp.t < current_t) {
            current_color = addNormalToColor(gridColor(color(0.6, 0.6, 0.6), vec2(hitp.u, hitp.v) * scale), hitp.n, r.d);
            current_t = hitp.t;
        }

        p.repr.pos = vec3(0., 0., 0.);
        p.repr.j = crdDefault.j;
        p.repr.k = crdDefault.k;

        MobiusIntersect hit = intersectMobius2(Ray(mulCrd(first, r.o), mulDir(first, r.d)));
        MobiusIntersect hit2 = intersectMobius2(Ray(mulCrd(second, r.o), mulDir(second, r.d)));

        int portal_to_process = 0;

        if (hit.hit && hit2.hit && min(hit.t, hit2.t) < current_t) {
            if (hit.t < hit2.t) {
                portal_to_process = 1;
            } else {
                portal_to_process = 2;
            }
        } else if (hit.hit && hit.t < current_t) {
            portal_to_process = 1;
        } else if (hit2.hit && hit2.t < current_t) {
            portal_to_process = 2;
        } else {
            portal_to_process = 0;
        }

        if (portal_to_process == 1) {
            current_t = hit.t;
            if (abs(hit.v) > 0.80) {
                current_color = addNormalToColor(color(0.1, 0.15, 1.), hit.n, r.d);
            } else {
                if (teleport_light == 1) {
                    current_color = gridColor(color(0.6, 0.6, 0.6), vec2(hit.u, hit.v));
                } else {
                    r.o += r.d * hit.t;
                    r.o += r.d * 0.01;

                    r.o = mulCrd(second_inv, r.o);
                    r.d = mulDir(second_inv, r.d);

                    gray *= add_gray_after_teleportation;
                    continue;
                }
            }
        }

        if (portal_to_process == 2) {
            current_t = hit2.t;
            if (abs(hit2.v) > 0.80) {
                current_color = addNormalToColor(color(1., 0.55, 0.15), hit2.n, r.d);
            } else {
                if (teleport_light == 1) {
                    current_color = gridColor(color(0.6, 0.6, 0.6), vec2(hit2.u, hit2.v));
                } else {
                    r.o += r.d * hit2.t;
                    r.o += r.d * 0.01;

                    r.o = mulCrd(first_inv, r.o);
                    r.d = mulDir(first_inv, r.d);

                    gray *= add_gray_after_teleportation;
                    continue;
                }
            }
        }

        return current_color * gray;
    }
    return color(0., 1., 1.);
}

void main() {
    float viewAngle = deg2rad(80.);
    float h = tan(viewAngle / 2.);

    vec2 uv = uv_screen * h;

    float alpha = deg2rad(angles.x);
    float beta = deg2rad(angles.y);
    float radius = angles.z;

    vec3 lookAt = vec3(0., 0., 0.);
    vec3 pos = vec3(sin(PI/2. - beta) * cos(alpha), cos(PI/2. - beta), sin(PI/2. - beta) * sin(alpha)) * radius + lookAt;

    vec3 k = normalize(lookAt - pos);
    vec3 i = normalize(cross(vec3(0., 1., 0.), k));
    vec3 j = normalize(cross(k, i));
    
    Ray r = Ray(pos, normalize(i * uv.x + j * uv.y + k));
    gl_FragColor = vec4(sqrt(intersectScene(r)), 1.);
}