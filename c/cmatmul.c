#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int ompNaive1D(int n) {
	#pragma omp parallel
	{
		#pragma omp single
		printf("Num threads for OMP: %d\n", omp_get_num_threads());
	
	}

	int *a = malloc(n * n * sizeof(int));
	int *b = malloc(n * n * sizeof(int));
	int *c = malloc(n * n * sizeof(int));

	#pragma omp parallel for
	for (int i = 0; i < n * n; i++) {
		a[i] = 1;
		b[i] = 1;
	}

	double start_time = omp_get_wtime();

	int i, j, temp, k;
	#pragma omp parallel for private(j, temp, k) 
	for (i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			temp = 0;
			for (k = 0; k < n; k++){
				temp += a[i * n + k] *  b[k * n + j];
			}
			c[i * n + j] = temp;
		}
	}
	double end_time = omp_get_wtime();

	return (int) ((end_time - start_time) * 1000000.0);
}

int ompTrans1D(int n) {
	int *a = malloc(n * n * sizeof(int));
	int *b = malloc(n * n * sizeof(int));
	int *c = malloc(n * n * sizeof(int));

	#pragma omp parallel for
	for (int i = 0; i < n * n; i++) {
		a[i] = 1;
		b[i] = 1;
	}

	double start_time = omp_get_wtime();

	// transpose B
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < 0; j++) {
			int temp = b[i * n + j];
			b[i * n + j] = b[j * n + i];
			b[j * n + i] = temp;
		}
	}

	int i, j, k, temp;
	#pragma omp parallel for private(j,temp,k)
	for (i = 0; i < n; i++){
		for(j = 0; j < n; j++){
			temp = 0;
			for (k = 0; k < n; k++){
				temp += a[i * n + k] *  b[j * n + k];
			}
			c[i * n + j] = temp;
		}
	}

	// transpose B
	#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < 0; j++) {
			int temp = b[i * n + j];
			b[i * n + j] = b[j * n + i];
			b[j * n + i] = temp;
		}
	}
	double end_time = omp_get_wtime();

	return (int) ((end_time - start_time) * 1000000.0);
}
int transpose1D(int n) {
	int *a = malloc(n * n * sizeof(int));
	int *b = malloc(n * n * sizeof(int));
	int *c = malloc(n * n * sizeof(int));

	for (int i = 0; i < n * n; i++) {
		a[i] = 1;
		b[i] = 1;
	}

	double start_time = omp_get_wtime();

	// transpose B
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < 0; j++) {
			int temp = b[i * n + j];
			b[i * n + j] = b[j * n + i];
			b[j * n + i] = temp;
		}
	}

	for (int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			int temp = 0;
			for (int k = 0; k < n; k++){
				temp += a[i * n + k] *  b[j * n + k];
			}
			c[i * n + j] = temp;
		}
	}

	// transpose B
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < 0; j++) {
			int temp = b[i * n + j];
			b[i * n + j] = b[j * n + i];
			b[j * n + i] = temp;
		}
	}
	double end_time = omp_get_wtime();

	return (int) ((end_time - start_time) * 1000000.0);
}

int naive1D(int n) {
	int *a = malloc(n * n * sizeof(int));
	int *b = malloc(n * n * sizeof(int));
	int *c = malloc(n * n * sizeof(int));

	for (int i = 0; i < n * n; i++) {
		a[i] = 1;
		b[i] = 1;
	}

	double start_time = omp_get_wtime();

	for (int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			int temp = 0;
			for (int k = 0; k < n; k++){
				temp += a[i * n + k] *  b[k * n + j];
			}
			c[i * n + j] = temp;
		}
	}
	double end_time = omp_get_wtime();

	return (int) ((end_time - start_time) * 1000000.0);
}

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "Usage: ./this.exe N, where N is the matrix size expressed as 2 ** N.\n");
		return EXIT_FAILURE;
	}

	int n_exp = atoi(argv[1]);
	int N = 1 << n_exp;

	//omp_set_num_threads(12);

	int naive_us = naive1D(N);
	printf("Naive 1d: %10dus\n", naive_us);
	int transpose_us = transpose1D(N);
	printf("Trans 1d: %10dus\n", transpose_us);
	int omp_naive_us = ompNaive1D(N);
	printf("OMP N 1d: %10dus\n", omp_naive_us);
	int omp_trans_1d = ompTrans1D(N);
	printf("OMP T 1d: %10dus\n", omp_trans_1d);

	return EXIT_SUCCESS;
}
