#include <iostream>
#include <gmpxx.h>
#include <omp.h>
#include <vector>
#include <limits>
#include <tuple>
#include <cmath>
#include <chrono>
#include <limits>

struct Candidate {
    mpz_class r = 0;
    mpz_class a = 200;
    mpz_class b = 200;
    bool valid = false;

    bool operator<(const Candidate& other) const {
        // Compare absolute values of a
        return std::abs(a.get_si()) < std::abs(other.a.get_si());
    }
};

// Helper function to get bit length of an integer
size_t bit_length(const mpz_class& x) {
    return mpz_sizeinbase(x.get_mpz_t(), 2);
}

// Compute RILE and return (a, b, RILE)
std::tuple<mpz_class, mpz_class, mpz_class> par_ext_ile(const mpz_class& u, const mpz_class& v, int m = 5) {
    const mpz_class k = mpz_class(1) << m;
    size_t n = bit_length(u);
    size_t p = bit_length(v);
    size_t rho = n - p + 1;
    size_t lambda = 2 * m + rho + 1;
    size_t shift = p > lambda ? p - lambda : 0;

    mpz_class u1 = u >> shift;
    mpz_class v1 = v >> shift;

    size_t max_i = k.get_ui();
    std::vector<Candidate> candidates(max_i + 1);

    // Step 1: Fill candidate list in parallel
    auto start = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel for
    for (size_t i = 1; i <= max_i; ++i) {
        mpz_class qi = (i * u1) / v1;
        mpz_class ri = i * u1 - qi * v1;
        mpz_class si = v1 - ri;

        if (ri < v1 / k) {
            candidates[i] = {ri, mpz_class(i), -qi, true};
        } else if (si < v1 / k) {
            candidates[i] = {si, -mpz_class(i), qi + 1, true};
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "⏱️ Candidate generation time: " << elapsed.count() << " seconds" << std::endl;

    // Step 2: Time the tree reduction
    // Find the minimum candidate
    auto red_start = std::chrono::high_resolution_clock::now();

    auto min_it = std::min_element(candidates.begin(), candidates.end());
    Candidate min_candidate = *min_it;

    auto red_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> red_elapsed = red_end - red_start;

    // Output timing
    std::cout << "⏱️ Tree Reduction step time: " << red_elapsed.count() << " seconds" << std::endl;

    // Finalize result
    Candidate best = min_candidate;
    mpz_class rile = best.a * u + best.b * v;
    if (rile < 0) rile = -rile;

    return {best.a, best.b, rile};
}

// Main Extended GCD using Par-Ext-ILE loop
mpz_class parallel_ext_gcd(mpz_class u, mpz_class v, mpz_class& out_a, mpz_class& out_b, int m = 5) {
    std::vector<std::tuple<mpz_class, mpz_class>> matrices; // store (a, b)

    int num_iterations = 0;
    while (v >= (8 * (mpz_class(1) << (2 * m)))) {
        auto [a, b, r] = par_ext_ile(u, v, m);
        matrices.emplace_back(a, b);
        u = v;
        v = r;
        num_iterations++;
    }
    std::cout << "Number of iterations: " << num_iterations << std::endl;

    // Base case: Extended Euclidean Algorithm
    mpz_class d, a, b;
    mpz_gcdext(d.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t(), u.get_mpz_t(), v.get_mpz_t());

    // Back-propagate matrices to recover extended coefficients
    for (auto it = matrices.rbegin(); it != matrices.rend(); ++it) {
        auto [ma, mb] = *it;
        mpz_class new_a = ma * a + mb * b;
        mpz_class new_b = a;
        a = new_a;
        b = new_b;
    }

    out_a = a;
    out_b = b;
    return d;
}

// ---- MAIN ----
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_bits_u> <num_bits_v>" << std::endl;
        return 1;
    }

    int bits_u = std::stoi(argv[1]);
    int bits_v = std::stoi(argv[2]);

    if (bits_u <= 0 || bits_v <= 0) {
        std::cerr << "Bit lengths must be positive integers." << std::endl;
        return 1;
    }

    // Print number of threads
    int num_threads = omp_get_max_threads();
    std::cout << "Number of threads: " << num_threads << std::endl;

    // Initialize GMP random state
    gmp_randclass rand_gen(gmp_randinit_default);
    rand_gen.seed(time(NULL)); // Use system time for randomness

    // Generate random u, v with specified bit lengths
    mpz_class u = rand_gen.get_z_bits(bits_u);
    mpz_class v = rand_gen.get_z_bits(bits_v);

    if (v == 0) v = 1;
    if (u < v) std::swap(u, v);

    std::cout << "Generated random u (" << bits_u << " bits): " << u << std::endl;
    std::cout << "Generated random v (" << bits_v << " bits): " << v << std::endl;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    mpz_class a, b;
    mpz_class d = parallel_ext_gcd(u, v, a, b, 5);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\nGCD(u, v) = " << d << std::endl;
    std::cout << "Bezout coefficients:" << std::endl;
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;

    mpz_class check = a * u + b * v;
    std::cout << "a*u + b*v = " << check << std::endl;

    std::cout << "⏱️  Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}