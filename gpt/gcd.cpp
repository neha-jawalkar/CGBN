#include <bits/stdc++.h>
#include <chrono>
#include <gmp.h>
#include <gmpxx.h>

using namespace std;

// ---------- helpers ----------

int ctz(uint64_t x, int bits_if_zero = 999999999)
{
    if (x == 0)
        return bits_if_zero;
    return __builtin_ctzll(x); // count trailing zeros
}

int ushiftamt(uint32_t x)
{
    // CUDA's bfind.shiftamt.u32
    if (x == 0)
        return -1;
    return 31 - __builtin_clz(x); // floor(log2(x))
}

struct SignedCoeffs
{
    long long alpha_a = 0;
    long long alpha_b = 0;
    long long beta_a = 0;
    long long beta_b = 0;
};

// ---------- gcd_reduce ----------

SignedCoeffs gcd_reduce(int a, int b)
{
    SignedCoeffs coeffs;
    coeffs.alpha_a = 1;
    coeffs.alpha_b = 0;
    coeffs.beta_a = -1;
    coeffs.beta_b = 1;

    int n = b - a;
    int c1 = 0;
    int count = 15;
    // cout << a << " " << b << "\n";

    while (true)
    {
        // keep the CUDA expression exactly as written
        int t = n ^ (n - 1);
        int c0 = 31 - ushiftamt(t);
        count -= c0;
        if (count < 0)
        {
            coeffs.alpha_a <<= c1;
            coeffs.alpha_b <<= c1;
            break;
        }

        int bshr = n >> c0;
        long long i = 1LL * bshr * (1LL * bshr * bshr + 62); // 6-bit inverse
        if (t > 0x3F)
        {
            i = i * (i * bshr + 2); // 12-bit inverse
            i = i * (i * bshr + 2); // 24-bit inverse
        }

        long long q = (1LL * a * i) & t;
        long long u = q & n;
        q = q - u - u;
        n = (bshr * q + a) >> c0;

        coeffs.alpha_a = (coeffs.alpha_a << (c0 + c1)) + q * coeffs.beta_a;
        coeffs.alpha_b = (coeffs.alpha_b << (c0 + c1)) + q * coeffs.beta_b;

        t = n ^ (n - 1);
        c1 = 31 - ushiftamt(t);
        count -= c1;
        if (count < 0)
        {
            coeffs.beta_a <<= c0;
            coeffs.beta_b <<= c0;
            break;
        }

        int ashr = n >> c1;
        i = 1LL * ashr * (1LL * ashr * ashr + 62); // 6-bit inverse
        if (t > 0x3F)
        {
            i = i * (i * ashr + 2); // 12-bit inverse
            i = i * (i * ashr + 2); // 24-bit inverse
        }

        q = (1LL * b * i) & t;
        u = q & n;
        q = q - u - u;
        n = (ashr * q + b) >> c1;

        coeffs.beta_a = (coeffs.beta_a << (c1 + c0)) + q * coeffs.alpha_a;
        coeffs.beta_b = (coeffs.beta_b << (c1 + c0)) + q * coeffs.alpha_b;
    }

    return coeffs;
}

// ---------- gcd_product ----------

long long gcd_product(long long sa, long long a, long long sb, long long b)
{
    long long ua = llabs(sa);
    long long ub = llabs(sb);
    long long sign = (sa ^ sb);
    long long r;
    if (sign >= 0)
    {
        r = ua * a + ub * b;
    }
    else
    {
        r = ua * a - ub * b;
        if (r < 0)
            r = -r;
    }
    return r;
}

// ---------- gcd (main) ----------

long long gcd_cuda_style(long long A, long long B)
{
    if (A == 0)
        return llabs(B);
    if (B == 0)
        return llabs(A);

    long long a = llabs(A);
    long long b = llabs(B);

    int BITS = 999999999;
    int ga = ctz(a, BITS);
    int gb = ctz(b, BITS);
    if (ga == BITS)
        return b;
    if (gb == BITS)
        return a;
    a >>= ga;
    b >>= gb;
    int gcd_shift = min(ga, gb);

    // Coarse phase (approximate)
    while (true)
    {
        if (a == b)
            break;
        long long a_high = a >> 32;
        long long b_high = b >> 32;
        if ((a_high | b_high) <= 0x3F)
            break;
        if (a > b)
        {
            long long d = a - b;
            d >>= ctz(d);
            a = d;
        }
        else
        {
            long long d = b - a;
            d >>= ctz(d);
            b = d;
        }
    }

    // Fine phase
    long long g;
    while (true)
    {
        if (a == 0)
        {
            g = b;
            break;
        }
        if (b == 0)
        {
            g = a;
            break;
        }

        int a_low = (int)(a & 0xFFFFFFFF);
        int b_low = (int)(b & 0xFFFFFFFF);

        if (a_low == 0 || (a_low & 1) == 0)
        {
            int z = ctz(a);
            a >>= z;
            continue;
        }
        if (b_low == 0 || (b_low & 1) == 0)
        {
            int z = ctz(b);
            b >>= z;
            continue;
        }

        SignedCoeffs reducer = gcd_reduce(a_low, b_low);

        long long a_temp = gcd_product(reducer.alpha_a, a, reducer.alpha_b, b);
        long long b_temp = gcd_product(reducer.beta_a, a, reducer.beta_b, b);

        if (a_temp != 0)
        {
            int a_shift = ctz(a_temp);
            a = a_temp >> a_shift;
        }
        else
        {
            a = 0;
        }

        if (b_temp != 0)
        {
            int b_shift = ctz(b_temp);
            b = b_temp >> b_shift;
        }
        else
        {
            b = 0;
        }

        if (reducer.beta_a == -1 && reducer.beta_b == 1)
        {
            swap(a, b);
        }
    }

    return g << gcd_shift;
}

// ------------ test ------------

// int main() {

//     int ITERS = 1000;

//     vector<pair<long long, long long>> pairs = {
//         {1234567890123456789LL, 9876543210LL}
//     };

//     for (auto &p : pairs) {
//         long long x = p.first;
//         long long y = p.second;
//         long long got = 0ULL;
//         for(int i = 0; i < ITERS; i++)
//         {
//             got = gcd_cuda_style(x, y);
//         }
//         long long expect = std::gcd(llabs(x), llabs(y));
//         cout << "gcd(" << x << ", " << y << ") = " << got
//              << ", expected " << expect
//              << ", OK? " << (got == expect) << "\n";
//     }
//     return 0;
// }

// int main()
// {
//     mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
//     uniform_int_distribution<unsigned long long> dist(1, ULLONG_MAX);

//     const int ITER = 1000;
//     vector<pair<unsigned long long, unsigned long long>> pairs;
//     pairs.reserve(ITER);
//     for (int i = 0; i < ITER; i++)
//     {
//         pairs.emplace_back(dist(rng), dist(rng));
//     }

//     auto start = chrono::high_resolution_clock::now();

//     for (int i = 0; i < ITER; i++)
//     {
//         auto [x, y] = pairs[i];
//         volatile auto got = gcd_cuda_style(x, y); // volatile prevents optimization
//     }

//     auto end = chrono::high_resolution_clock::now();
//     chrono::duration<double, milli> elapsed = end - start;

//     cout << "Ran " << ITER << " iterations in "
//          << elapsed.count() << " ms\n";

//     return 0;
// }

int main()
{
    mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<unsigned long long> dist(1, ULLONG_MAX);

    const int ITER = 10000;
    vector<pair<unsigned long long, unsigned long long>> pairs;
    pairs.reserve(ITER);
    for (int i = 0; i < ITER; i++)
    {
        pairs.emplace_back(dist(rng), dist(rng));
    }

    // ---- CUDA-style GCD timing ----
    auto start1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < ITER; i++)
    {
        auto [x, y] = pairs[i];
        volatile auto got = gcd_cuda_style(x, y);
    }
    auto end1 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed1 = end1 - start1;
    cout << "CUDA-style gcd: " << elapsed1.count() << " ms\n";

    // ---- std::gcd timing ----
    auto start2 = chrono::high_resolution_clock::now();
    for (int i = 0; i < ITER; i++)
    {
        auto [x, y] = pairs[i];
        volatile auto got = std::gcd(x, y);
    }
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed2 = end2 - start2;
    cout << "std::gcd:       " << elapsed2.count() << " ms\n";

    // ---- GMP gcd timing ----
    auto start3 = std::chrono::high_resolution_clock::now();
    for (auto [x, y] : pairs)
    {
        mpz_class gx, gy, g;
        mpz_import(gx.get_mpz_t(), 1, -1, sizeof(x), 0, 0, &x);
        mpz_import(gy.get_mpz_t(), 1, -1, sizeof(y), 0, 0, &y);
        g = gcd(gx, gy);
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed3 = end3 - start3;
    std::cout << "GMP gcd time: "
              << elapsed3.count()
              << " ms\n";

    return 0;
}