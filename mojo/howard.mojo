from random import rand
from sys import info, simdwidthof
import time

import benchmark
from algorithm import parallelize, vectorize
import memory  # memset_zero, UnsafePointer

from algorithm import Static2DTileUnitFunc as Tile2DFunc

alias tile_n = 64
alias tile_k = 4

alias type = DType.float32
alias nelts = get_simd_width()

alias dim = 4096


fn get_simd_width() -> Int:
    return 2 * simdwidthof[type]()


struct Matrix[rows: Int, cols: Int]:
    var data: memory.UnsafePointer[Scalar[type]]

    fn __init__(out self):
        self.data = memory.UnsafePointer[Scalar[type]].alloc(rows * cols)
        rand(self.data, rows * cols)

    @implicit
    fn __init__(out self, data: memory.UnsafePointer[Scalar[type]]):
        self.data = data

    fn __copyinit__(out self, existing: Self):
        self.data = existing.data
        # self.rows = existing.rows
        # self.cols = existing.cols

    fn __del__(owned self):
        self.data.free()

    """
    fn zero(self) -> Self:
        memory.memset_zero(self.data, self.rows * self.cols * dtype_sizeof[type])()
        return self
    """

    @staticmethod
    fn rand() -> Self:
        var data = memory.UnsafePointer[Scalar[type]].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load(y, x)

    fn __setitem__(mut self, y: Int, x: Int, val: Scalar[type]):
        return self.store(y, x, val)

    fn load[nelts: Int = 1](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    """
    fn load_tr[nelts: Int = 1](self, y: Int, x: Int) -> SIMD[type, nelts]:
        # transposed SIMD load
        return strided_load[nelts, type](self.data + x * dtype_sizeof[type](), self.cols)
    """

    fn store[nelts: Int = 1](self, y: Int, x: Int, val: SIMD[type, nelts]):
        self.data.store(y * self.cols + x, val)


fn matmul_naive(mut C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for n in range(C.cols):
            for k in range(A.cols):
                C[m, n] += A[m, k] * B[k, n]


fn bench_naive():
    var a = Matrix[dim, dim].rand()
    var b = Matrix[dim, dim].rand()
    var c = Matrix[dim, dim].rand()

    var start_time = time.perf_counter_ns()
    matmul_naive(c, a, b)
    var end_time = time.perf_counter_ns()
    print("Naive:", (end_time - start_time) / 1000.0, "us")


fn matmul_vectorized(mut C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[dot, nelts, size = C.cols]()


fn bench_vectorized():
    var a = Matrix[dim, dim].rand()
    var b = Matrix[dim, dim].rand()
    var c = Matrix[dim, dim].rand()

    var start_time = time.perf_counter_ns()
    matmul_vectorized(c, a, b)
    var end_time = time.perf_counter_ns()
    print("vectorized", (end_time - start_time) / 1000.0, "us")


fn matmul_parallelized(mut C: Matrix, A: Matrix, B: Matrix):
    var num_workers = C.rows

    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[dot, nelts, size = C.cols]()

    parallelize[calc_row](C.rows, num_workers)


fn bench_parallelized():
    var a = Matrix[dim, dim].rand()
    var b = Matrix[dim, dim].rand()
    var c = Matrix[dim, dim].rand()

    var start_mul = time.perf_counter_ns()
    matmul_parallelized(c, a, b)
    var end_mul = time.perf_counter_ns()
    print("Parallelized:", (end_mul - start_mul) / 1000.0, "us")


# Perform 2D tiling on the iteration space defined by end_x and end_y
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# Use the above tile function to perform tiled matmul
# Also parallelize with num_workers threads
fn matmul_tiled(mut C: Matrix, A: Matrix, B: Matrix):
    var num_workers = C.rows

    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(
                        m,
                        n + x,
                        C.load[nelts](m, n + x)
                        + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize[dot, nelts, size=tile_x]()

        tile[calc_tile, tile_n, tile_k](C.cols, B.rows)

    parallelize[calc_row](C.rows, num_workers)


fn bench_tiled():
    var a = Matrix[dim, dim].rand()
    var b = Matrix[dim, dim].rand()
    var c = Matrix[dim, dim].rand()

    var start_time = time.perf_counter_ns()
    matmul_tiled(c, a, b)
    var end_time = time.perf_counter_ns()
    print("tiled", (end_time - start_time) / 1000.0, "us")


def main():
    """
    var report = benchmark.benchmark.run[bench_naive]()
    print("simple as")
    report.print_full("ns")

    report = benchmark.benchmark.run[bench_vectorized]()
    print("vectorized")
    report.print_full("ns")

    report = benchmark.benchmark.run[bench_parallelized]()
    print("parallelized vectorized")
    report.print_full("ns")

    report = benchmark.benchmark.run[bench_tiled]()
    print("tiled parallelized vectorized")
    report.print_full("ns")
    """
    bench_naive()
    bench_vectorize()
    bench_parallelized()
    bench_tiled()
