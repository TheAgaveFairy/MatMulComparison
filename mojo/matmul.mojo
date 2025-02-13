import random # rand we want
import sys # info, simdwidthof, argv
import memory # UnsafePointer
from python import Python, PythonObject
import collections # List
import time # perf_counter_ns

struct TestResult:
    var t_prep: Int
    var t_run: Int
    var calling_fn_name: String

    fn __init__(out self, t_prep: Int, t_run: Int, calling_fn_name: String):
        self.t_prep = t_prep
        self.t_run = t_run
        self.calling_fn_name = calling_fn_name

    fn print(self):
        print(self.calling_fn_name + ":\n\t" + String(self.t_prep) + "us prepping" + "\n\t" + String(self.t_run) + "us running")

fn checkMatrix(n: Int, mat: collections.List[Int]) -> Bool:
    for i in range(n * n):
        if mat[i] != n:
            print("ERROR MULTIPLYING:", mat[i])
            return False
    return True

fn oneDimensionalNaiveUnsafePointerRandomValues(N: Int) -> TestResult:
    var prep_mul = time.perf_counter_ns()
    var capacity = N * N
    var a = memory.UnsafePointer[Scalar[DType.int32]].alloc(N * N)
    var b = memory.UnsafePointer[Scalar[DType.int32]].alloc(N * N)
    var c = memory.UnsafePointer[Scalar[DType.int32]].alloc(N * N)
    random.rand(a, N * N, max=10)
    random.rand(b, N * N, max=10)

    var start_mul = time.perf_counter_ns()

    for i in range(N):
        for j in range(N):
            var temp_sum = 0
            for k in range(N):
                c[i * N + j] += a[i * N + k] * b[k * N + j]
            #c[i * N + j] = temp_sum
    
    var end_mul = time.perf_counter_ns()

    var time_to_prep = (start_mul - prep_mul) // 1000
    var time_to_run = (end_mul - start_mul) // 1000 #ns to us
    #checkMatrix(N, c)
    a.free()
    b.free()
    c.free()
    return TestResult(time_to_prep, time_to_run, "oneDimNaiveUnsafePointerRandVals") 
fn oneDimensionalNaiveList(N: Int) -> TestResult:
    """
    If we wanted to do more of a "malloc" style:
    var a = memory.UnsafePointer[Scalar[DType.int32]].alloc(N * N)
    Random.rand(a, N * N, max=10)
    .
    """
    var prep_mul = time.perf_counter_ns()

    var capacity = N * N
    var a = collections.List[Int](capacity=capacity)
    var b = collections.List[Int](capacity=capacity)
    var c = collections.List[Int](capacity=capacity)
   
    for i in range(N * N):
        a.append(1) #random.random_ui64(0,10))
        b.append(1) # if we do b[i] = 1, then the len() won't work as the size isn't updating

    var start_mul = time.perf_counter_ns()

    for i in range(N):
        for j in range(N):
            var temp_sum = 0
            for k in range(N):
                temp_sum += a[i * N + k] * b[k * N + j]
            c[i * N + j] = temp_sum
    
    var end_mul = time.perf_counter_ns()

    var time_to_prep = (start_mul - prep_mul) // 1000
    var time_to_run = (end_mul - start_mul) // 1000 #ns to us
    #checkMatrix(N, c)
    return TestResult(time_to_prep, time_to_run, "oneDimensionalNaiveList") # __name__ i guess isn't a thing yet


fn oneDimensionalTransList(N: Int) -> TestResult:
    var prep_mul = time.perf_counter_ns()
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
            b[i * N + j], b[j * N + i] = b[j * N + i], b[i * N + j]

    for i in range(N):
        for j in range(N):
            var temp_sum = 0
            for k in range(N):
                temp_sum += a[i * N + k] * b[j * N + k]
            c[i * N + j] = temp_sum

    for i in range(N):
        for j in range(N):
            b[i * N + j], b[j * N + i] = b[j * N + i], b[i * N + j]

    
    var end_mul = time.perf_counter_ns()

    var time_to_prep = (start_mul - prep_mul) // 1000
    var time_to_run = (end_mul - start_mul) // 1000 #ns to us
    #checkMatrix(N, c)
    return TestResult(time_to_prep, time_to_run, "oneDimensionalTransList") # __name__ i guess isn't a thing yet

def main():
    args = sys.argv()
    if len(args) < 2:
        print("Usage: mojo matmul.mojo N, where N is the size of the matrix (2 ** N) to test")
    N = Int(args[1])
    N = 1 << N
    print("Matrix of size", N,"x",N)

    one_dim_naive_list = oneDimensionalNaiveList(N)
    one_dim_naive_list.print()


    one_dim_trans_list = oneDimensionalTransList(N)
    one_dim_trans_list.print()

    one_dim_rand = oneDimensionalNaiveUnsafePointerRandomValues(N)
    one_dim_rand.print()
