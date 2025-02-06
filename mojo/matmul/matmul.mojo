import random # rand we want
import sys # info, simdwidthof, argv
import memory # UnsafePointer
from python import Python, PythonObject
import collections # List
import time # perf_counter_ns

struct TestResult:
    var time: Int
    var calling_fn_name: String

    fn __init__(out self, time: Int, calling_fn_name: String):
        self.time = time
        self.calling_fn_name = calling_fn_name

    fn print(self):
        print(self.calling_fn_name + ":\n\t" + String(self.time) + "us")

fn oneDimensionalNaiveList(N: Int) -> TestResult:
    """
    var a = memory.UnsafePointer[Scalar[DType.int32]].alloc(N * N)
    random.rand(a, N * N, max=10)
    """

    var capacity = N * N
    var a = collections.List[Int](capacity=capacity)
    var b = collections.List[Int](capacity=capacity)
    var c = collections.List[Int](capacity=capacity)
    
    for i in range(N * N):
        a.append(1)
        b.append(1) # if we do b[i] = 1, then the len() won't work as the size isn't updating

    var start_mul = time.perf_counter_ns()

    for i in range(N):
        for j in range(N):
            var temp_sum = 0
            for k in range(N):
                temp_sum += a[i * N + k] * b[k * N + j]
            c[i * N + j] = temp_sum
    
    var end_mul = time.perf_counter_ns()

    if False:
        for i in range(N):
            for j in range(N):
                print(c[i * N + j], ",", end=' ')
            print()
        
    var time_to_run = (end_mul - start_mul) // 1000 #ns to us
    return TestResult(time_to_run, "oneDimensionalNaiveList") # __name__ i guess isn't a thing yet



def main():
    args = sys.argv()
    if len(args) < 2:
        print("Usage: mojo matmul.mojo N, where N is the size of the matrix (2 ** N) to test")
    N = Int(args[1])
    N = 1 << N
    print("Matrix of size", N)

    one_dim_naive_list = oneDimensionalNaiveList(N)
    one_dim_naive_list.print()
