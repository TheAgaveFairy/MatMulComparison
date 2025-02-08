#!/usr/bin/env python3

import sys
import time # time
import numpy as np

def npTwoDimSmart(N):
    prep_time = time.time()
    a = np.ones(N * N, dtype = int).reshape(N, N)
    b = np.ones(N * N, dtype = int).reshape(N, N)
    start_mul = time.time()
    c = (a @ b).flatten()
    end_mul = time.time()
        
    time_to_run = (end_mul - start_mul) * 1_000_000 #s to us
    return ("npTwoDimSmart", round(time_to_run,2))

def npOneDimTranspose(N):
    prep_time = time.time()

    a = np.ones(N * N, dtype=int)
    b = np.ones(N * N, dtype=int)
    c = np.empty(N * N, dtype=int)

    start_mul = time.time()

    # transpose
    b = b.reshape(N, N).T.flatten()

    for i in range(N):
        for j in range(N):
            c[i * N + j] = np.dot(a[i*N : i* N + N], b[j * N : j * N + N])

    b = b.reshape(N, N).T.flatten()
    
    end_mul = time.time()
        
    time_to_run = (end_mul - start_mul) * 1_000_000 #s to us
    return ("npOneDimTranspose", round(time_to_run,2))


def npOneDimNaive(N):
    prep_time = time.time()

    a = np.ones(N * N, dtype=int)
    b = np.ones(N * N, dtype=int)
    c = np.empty(N * N, dtype=int)

    start_mul = time.time()

    for i in range(N):
        for j in range(N):
            temp = 0
            for k in range(N):
                temp += a[i * N + k] * b[k * N + j]
            c[i * N + j] = temp
    
    end_mul = time.time()
        
    time_to_run = (end_mul - start_mul) * 1_000_000 #s to us
    return ("npOneDimNaive", round(time_to_run,2))


def oneDimTranspose(N: int) -> (int, int):
    prep_time = time.time()

    capacity = N * N
    a = [0] * capacity
    b = [0] * capacity
    c = [0] * capacity

    for i in range(N * N):
        a[i] = 1
        b[i] = 1 # if we do b[i] = 1, then the len() won't work as the size isn't updating

    start_mul = time.time()

    # transpose
    for i in range(N):
        for j in range(N):
            b[i*N + j], b[j *N + i] = b[j *N + i], b[i*N + j]

    for i in range(N):
        for j in range(N):
            temp_sum = 0
            for k in range(N):
                temp_sum += a[i * N + k] * b[j * N + k]
            c[i * N + j] = temp_sum
    
    # transpose
    for i in range(N):
        for j in range(N):
            b[i*N + j], b[j *N + i] = b[j *N + i], b[i*N + j]
    
    end_mul = time.time()
        
    time_to_run = (end_mul - start_mul) * 1_000_000 #s to us
    return ("oneDimTranspose", round(time_to_run,2)) # __name__ i guess isn't a thing yet


def oneDimNaiveList(N: int) -> (int, int):

    prep_time = time.time()

    capacity = N * N
    a = [0] * capacity
    b = [0] * capacity
    c = [0] * capacity

    for i in range(N * N):
        a[i] = 1
        b[i] = 1 # if we do b[i] = 1, then the len() won't work as the size isn't updating

    start_mul = time.time()

    for i in range(N):
        for j in range(N):
            temp_sum = 0
            for k in range(N):
                temp_sum += a[i * N + k] * b[k * N + j]
            c[i * N + j] = temp_sum
    
    end_mul = time.time()
        
    time_to_run = (end_mul - start_mul) * 1_000_000 #s to us
    return ("oneDimensionalNaiveList", round(time_to_run,2)) # __name__ i guess isn't a thing yet


def main():
    args = sys.argv
    if len(args) < 2:
        print("Usage: python3 matmul.py N, where N is the size of the matrix (2 ** N) to test")
        exit(1)
    N = int(args[1])
    N = 1 << N
    print("Matrix of size", N)

    print(oneDimNaiveList(N))
    print(oneDimTranspose(N))
    print(npOneDimNaive(N))

    print(npTwoDimSmart(N))
    print(npOneDimTranspose(N))

if __name__ == "__main__":
    main()
