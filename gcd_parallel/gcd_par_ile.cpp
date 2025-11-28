#include <iostream>
#include <vector>
#include <gmp.h>
#include <chrono>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <cmath>

// Struct to hold a candidate reduction
struct Candidate {
    mpz_t r = {0};
    int a;
    int b;
    bool valid;

    Candidate() = default;

    Candidate(const mpz_t r_in, int a_in, int b_in, bool valid_in) : a(a_in), b(b_in), valid(valid_in) {
        mpz_init(r);
        mpz_set(r, r_in);
    }

    Candidate(const Candidate& other) : a(other.a), b(other.b), valid(other.valid) {
        mpz_init(r);
        mpz_set(r, other.r);
    }

    Candidate& operator=(const Candidate& other) {
        if (this != &other) {
            mpz_set(r, other.r);
            a = other.a;
            b = other.b;
            valid = other.valid;
        }
        return *this;
    }

    ~Candidate() = default;

    bool operator<(const Candidate& other) const {
        return mpz_cmp(r, other.r) < 0;
    }
};

size_t bit_length(const mpz_t x) {
    return mpz_sizeinbase(x, 2);
}

void par_ile(mpz_t new_u, mpz_t new_v, const mpz_t u, const mpz_t v, int m = 5) {
    size_t n = mpz_sizeinbase(u, 2);
    size_t p = mpz_sizeinbase(v, 2);
    size_t rho = n - p + 1;
    size_t lambda = 2 * m + rho + 1;
    size_t shift = (p > lambda) ? p - lambda : 0;

    mpz_t u1_mpz, v1_mpz;
    mpz_inits(u1_mpz, v1_mpz, NULL);
    mpz_fdiv_q_2exp(u1_mpz, u, shift);
    mpz_fdiv_q_2exp(v1_mpz, v, shift);

    uint64_t u1 = mpz_get_ui(u1_mpz);
    uint64_t v1 = mpz_get_ui(v1_mpz);

    if (v1 == 0) {
        mpz_set(new_u, v);
        mpz_mod(new_v, u, v);
        mpz_clears(u1_mpz, v1_mpz, NULL);
        return;
    }

    // Precompute magic number for division by v1
    __uint128_t magic = ((__uint128_t(1) << 63) / v1) + 1;

    uint64_t v1_k = v1 >> m;
    int best_a = 0, best_b = 0;
    uint64_t best_r = UINT64_MAX;
    bool found = false;

    for (int i = 1; i < (1 << m); ++i) {
        uint64_t mul = uint64_t(i) * u1;
        uint64_t q = (magic * mul) >> 63;
        uint64_t r = mul - q * v1;
        uint64_t s = v1 - r;

        if (r < v1_k && r < best_r) {
            best_r = r;
            best_a = i;
            best_b = -int(q);
            found = true;
        } else if (s < v1_k && s < best_r) {
            best_r = s;
            best_a = -i;
            best_b = int(q) + 1;
            found = true;
        }
    }

    if (!found) {
        mpz_mod(new_v, u, v);
        mpz_set(new_u, v);
    } else {
        mpz_t term1, term2;
        mpz_inits(term1, term2, NULL);
        mpz_mul_si(term1, u, best_a);
        mpz_mul_si(term2, v, best_b);
        mpz_add(new_v, term1, term2);
        if (mpz_sgn(new_v) < 0) mpz_neg(new_v, new_v);
        mpz_set(new_u, v);
        mpz_clears(term1, term2, NULL);
    }

    // mpz_clears(u1_mpz, v1_mpz, NULL);
}

void gcd_par_ile(mpz_t result, mpz_t u, mpz_t v, int m = 5) {
    mpz_t a, b;
    mpz_inits(a, b, NULL);
    int step = 0;
    while (mpz_cmp_ui(v, 0) != 0) {
        par_ile(a, b, u, v, m);
        mpz_set(u, a);
        mpz_set(v, b);
        gmp_printf("Step %d: u = %Zd, v = %Zd\n", ++step, u, v);
    }
    mpz_set(result, u);
}

void gcd_par_ile_extended(mpz_t g, mpz_t x_out, mpz_t y_out, const mpz_t a_in, const mpz_t b_in, int m = 5) {
    mpz_t a, b;
    mpz_inits(a, b, NULL);
    mpz_set(a, a_in);
    mpz_set(b, b_in);

    // Identity matrix N = [[1, 0], [0, 1]]
    mpz_t n00, n01, n10, n11;
    mpz_inits(n00, n01, n10, n11, NULL);
    mpz_set_ui(n00, 1); mpz_set_ui(n01, 0);
    mpz_set_ui(n10, 0); mpz_set_ui(n11, 1);

    while (mpz_cmp_ui(b, (1 << 20)) > 0) {
        mpz_t new_a, new_b;
        mpz_inits(new_a, new_b, NULL);
        par_ile(new_a, new_b, a, b, m);

        // Coefficients of the linear combination: new_b = a*a1 + b*b1
        mpz_t a1, b1;
        mpz_inits(a1, b1, NULL);
        // We'll reverse engineer a1, b1 from (new_b = a*a1 + b*b1)
        mpz_t tmp_gcd, tmp_x, tmp_y;
        mpz_inits(tmp_gcd, tmp_x, tmp_y, NULL);
        mpz_gcdext(tmp_gcd, tmp_x, tmp_y, a, b);
        mpz_gcdext(tmp_gcd, a1, b1, a, b); // Simplification; approximation of a1, b1

        // Multiply current matrix with:
        // [ 0  1 ]
        // [ a1 b1 ]
        mpz_t t00, t01, t10, t11;
        mpz_inits(t00, t01, t10, t11, NULL);
        mpz_set(t00, n10);  // n10
        mpz_set(t01, n11);  // n11

        mpz_mul(t10, a1, n00);  mpz_addmul(t10, b1, n10); // a1*n00 + b1*n10
        mpz_mul(t11, a1, n01);  mpz_addmul(t11, b1, n11); // a1*n01 + b1*n11

        mpz_set(n00, t00); mpz_set(n01, t01);
        mpz_set(n10, t10); mpz_set(n11, t11);

        mpz_set(a, new_a);
        mpz_set(b, new_b);

        mpz_clears(new_a, new_b, a1, b1, tmp_gcd, tmp_x, tmp_y, t00, t01, t10, t11, NULL);
    }

    // Use classic EEA for small numbers
    mpz_t g_, x_, y_;
    mpz_inits(g_, x_, y_, NULL);
    mpz_gcdext(g_, x_, y_, a, b);

    // Final x = x_ * n00 + y_ * n10
    // Final y = x_ * n01 + y_ * n11
    mpz_mul(x_out, x_, n00); mpz_addmul(x_out, y_, n10);
    mpz_mul(y_out, x_, n01); mpz_addmul(y_out, y_, n11);
    mpz_set(g, g_);

    mpz_clears(a, b, n00, n01, n10, n11, g_, x_, y_, NULL);
}


int main(int argc, char* argv[]) {
    int bits_u, bits_v;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_bits_u> <num_bits_v>" << std::endl;
        // Set default values if not provided
        bits_u = 1500;
        bits_v = 1500;
    }
    else {
        bits_u = std::stoi(argv[1]);
        bits_v = std::stoi(argv[2]);
    }

    if (bits_u <= 0 || bits_v <= 0) {
        std::cerr << "Bit lengths must be positive integers." << std::endl;
        return 1;
    }

    gmp_randstate_t rand_state;
    gmp_randinit_default(rand_state);
    gmp_randseed_ui(rand_state, time(NULL));

    mpz_t u, v, original_u, original_v, expected_gcd;
    mpz_inits(u, v, original_u, original_v, expected_gcd, NULL);
    mpz_urandomb(u, rand_state, bits_u);
    mpz_urandomb(v, rand_state, bits_v);

    if (mpz_cmp_ui(v, 0) == 0) mpz_set_ui(v, 1);
    if (mpz_cmp(u, v) < 0) {
        mpz_swap(u, v);
    }

    mpz_set(original_u, u);
    mpz_set(original_v, v);

    gmp_printf("Generated u (%d bits): %Zd\n", bits_u, u);
    gmp_printf("Generated v (%d bits): %Zd\n", bits_v, v);

    auto start = std::chrono::high_resolution_clock::now();
    mpz_t d;
    mpz_init(d);
    gcd_par_ile(d, u, v, 10);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    gmp_printf("\nGCD(u, v) = %Zd\n", d);
    // Time in ms
    std::cout << "\u23F1\uFE0F  Total time: " << elapsed.count() * 1000 << " ms" << std::endl;

    auto start_gcd = std::chrono::high_resolution_clock::now();
    mpz_gcd(expected_gcd, original_u, original_v);
    auto end_gcd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gcd = end_gcd - start_gcd;
    std::cout << "\u23F1\uFE0F  Time for mpz_gcd: " << elapsed_gcd.count() * 1000 << " ms" << std::endl;
    gmp_printf("Expected GCD (using mpz_gcd): %Zd\n", expected_gcd);
    if (mpz_cmp(d, expected_gcd) == 0) {
        std::cout << "GCD matches with mpz_gcd!" << std::endl;
    } else {
        std::cout << "GCD does not match with mpz_gcd!" << std::endl;
    }

    // // Run 100000 times
    // auto start_m = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 10000; ++i) {
    //     mpz_set(u, original_u);
    //     mpz_set(v, original_v);
    //     gcd_par_ile(d, u, v, 3);
    // }
    // auto end_m = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed_10k = end_m - start_m;
    // std::cout << "\u23F1\uFE0F  Time for 10000 iterations: " << elapsed_10k.count() * 1000 << " ms" << std::endl;
    // std::cout << "Average time per iteration: " << (elapsed_10k.count() * 1000) / 10000 << " ms" << std::endl;

    // mpz_t x, y, gcd;
    // mpz_inits(x, y, gcd, NULL);
    // gcd_par_ile_extended(gcd, x, y, original_u, original_v, 10);

    // gmp_printf("GCD: %Zd\n", gcd);
    // gmp_printf("x: %Zd\ny: %Zd\n", x, y);

    mpz_clears(u, v, original_u, original_v, expected_gcd, d, NULL);
    gmp_randclear(rand_state);

    return 0;
}
