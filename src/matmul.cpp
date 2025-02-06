#include <iostream>
#include <stdlib.h>
#include <vector>
#include <random>
#include <chrono>

int lazyMul1D(int n) {
	auto start = std::chrono::high_resolution_clock::now();
	std::vector<int> a; // do NOT declare size
	std::vector<int> b;
	std::vector<int> res(n * n);

	std::random_device rand_dev;
	std::mt19937 gen(rand_dev());

	std::uniform_int_distribution<int> val_dist(0, 100); // doesnt really matter

	for (int i = 0; i < n * n; i++) {
		a.push_back(val_dist(gen));
		b.push_back(val_dist(gen));
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int sum = 0;
			for (int k = 0; k < n; k++) {
				//res[i * n + j] += a[i * n + k] * b[k * n + j];
				sum += a[i * n + k] * b[k * n + j];
			}
			res[i * n + j] = sum;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	return duration.count();
}

int vectorMul1D(int n) {
	auto start = std::chrono::high_resolution_clock::now();
	std::vector<int> a(n * n);
	std::vector<int> b(n * n);
	std::vector<int> res(n * n);

	std::random_device rand_dev;
	std::mt19937 gen(rand_dev());

	std::uniform_int_distribution<int> val_dist(0, 100); // doesnt really matter

	for (int i = 0; i < n * n; i++) {
		a[i] = val_dist(gen);
		b[i] = val_dist(gen);
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int sum = 0;
			for (int k = 0; k < n; k++) {
				//res[i * n + j] += a[i * n + k] * b[k * n + j];
				sum += a[i * n + k] * b[k * n + j];
			}
			res[i * n + j] = sum;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	return duration.count();
}
int vectorHintedMul1D(int n) {
	auto start = std::chrono::high_resolution_clock::now();
	std::vector<int> a(n * n);
	std::vector<int> b(n * n);
	std::vector<int> res(n * n);

	std::random_device rand_dev;
	std::mt19937 gen(rand_dev());

	std::uniform_int_distribution<int> val_dist(0, 100); // doesnt really matter

	for (int i = 0; i < n * n; i++) {
		a[i] = val_dist(gen);
		b[i] = val_dist(gen);
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int sum = 0;
			const int *a_row = &a[i * n];
			const int *b_col = &b[j];
			for (int k = 0; k < n; k++) {
				sum += a_row[k] * b_col[k * n];
			}
			res[i * n + j] = sum;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	return duration.count();
}

int vectorMul2D(int n) {
	auto start = std::chrono::high_resolution_clock::now();
	std::vector<std::vector<int>> a(n, std::vector<int>(n));
	std::vector<std::vector<int>> b(n, std::vector<int>(n));
	std::vector<std::vector<int>> res(n, std::vector<int>(n));

	std::random_device rand_dev;
	std::mt19937 gen(rand_dev());

	std::uniform_int_distribution<int> val_dist(0, 100); // doesnt really matter

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i][j] = val_dist(gen);
			b[i][j] = val_dist(gen);
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int sum = 0;
			for (int k = 0; k < n; k++) {
				//res[i * n + j] += a[i * n + k] * b[k * n + j];
				sum += a[i][k] * b[k][j];
			}
			res[i][j] = sum;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	return duration.count();
}
int vectorTransposeMul2D(int n) {
	auto start = std::chrono::high_resolution_clock::now();
	std::vector<std::vector<int>> a(n, std::vector<int>(n));
	std::vector<std::vector<int>> b(n, std::vector<int>(n));
	std::vector<std::vector<int>> res(n, std::vector<int>(n));

	std::random_device rand_dev;
	std::mt19937 gen(rand_dev());

	std::uniform_int_distribution<int> val_dist(0, 100); // doesnt really matter

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i][j] = val_dist(gen);
			b[i][j] = val_dist(gen);
		}
	}
	
	// transpose
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int temp = b[i][j];
			b[i][j] = b[j][i];
			b[j][i] = temp;
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int sum = 0;
			for (int k = 0; k < n; k++) {
				//res[i * n + j] += a[i * n + k] * b[k * n + j];
				sum += a[i][k] * b[j][k];
			}
			res[i][j] = sum;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	return duration.count();
}

// I want this program to be a very "intuitive" approach to how students might write matrix allocations and multiplications. Methods won't be 1-1. For example, nested loops for filling the 2d version of std::vector could've been filled with a single loop, potentially reducing comparison operations etc (though possibly at the cost of some arithmatic. Damn, maybe I'll actually test both... Anyways, you get the point. I'll also compare to Zig for fun for whatever the best method ends up being.

int main(int argc, char ** argv) {
	if (argc < 2) {
		std::cout << "Please give array size as power of 2. i.e. 4 => 2 ** 4" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	int N = 1 << atoi(argv[1]);

	int lazy_time = lazyMul1D(N);
	std::cout << "Lazy Allocation 1D std::vector<int> in us: " << lazy_time << std::endl;
	int v1d_time = vectorMul1D(N);
	std::cout << "Pre- Allocation 1D std::vector<int> in us: " << v1d_time << std::endl;
	int v1d_hinted_time = vectorHintedMul1D(N);
	std::cout << "PreAlloc Hinted 1D std::vector<int> in us: " << v1d_hinted_time << std::endl;
	int v2d_time = vectorMul2D(N);
	std::cout << "Pre- Allocation 2D std::vector<int> in us: " << v2d_time << std::endl;
	int v2d_transpose_time = vectorTransposeMul2D(N);
	std::cout << "PreAlloc, Trans 2D std::vector<int> in us: " << v2d_transpose_time << std::endl;

	//fmt::print("Lazy 1d {:.2f}

	std::exit(EXIT_SUCCESS);
}
