#!/usr/bin/env bash
gcc -fopenmp -g -O3 cmatmul.c -o cmopt.out
echo './cmopt.out created'
