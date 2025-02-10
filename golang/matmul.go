package main

import (
	"fmt"
	"os"
	"log"
	"strconv"
	"time"
)

type TestResult struct {
	CallingFn	string
	TimePrep	int64 // microseconds
	TimeRunning	int64 // microseconds
}

func NewTestResult(callingFn string, tPrep int64, tRun int64) TestResult {
	return TestResult{
		CallingFn:		callingFn,
		TimePrepared:	tPrep,
		TimeRunning:	tRun,
	}
}

func (tr TestResult) Display() {
	fmt.Printf("%s:\n\tTime to Allocate and Fill:\t%dus.\n\tTime to Multiply:\t\t%dus\n",
		tr.CallingFn, tr.TimePrepared, tr.TimeRunning)
}

func transposeMul(n int) int64 {
	prep_time := time.Now()
	a := make([]int, n * n)
	b := make([]int, n * n)
	c := make([]int, n * n)

	for i := 0; i < n * n; i++ {
		a[i] = 1
		b[i] = 1
	}

	start_time := time.Now()

	// transpose
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			temp := b[i * n + j]
			b[i * n + j] = b[j * n + i];
			b[j * n + i] = temp
		}
	}

	for i:= 0; i < n; i++ {
		for j := 0; j < n; j++ {
			temp := 0
			for k := 0; k < n; k++ {
				temp += a[i * n + k] * a[j * n + k]
			}
			c[i * n + j] = temp
		}
	}	
	// transpose
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			temp := b[i * n + j]
			b[i * n + j] = b[j * n + i];
			b[j * n + i] = temp
		}
	}

	return time.Since(start_time).Microseconds()
}
func naiveMul(n int) int64 {
	a := make([]int, n * n)
	b := make([]int, n * n)
	c := make([]int, n * n)

	for i := 0; i < n * n; i++ {
		a[i] = 1
		b[i] = 1
	}

	start_time := time.Now()

	for i:= 0; i < n; i++ {
		for j := 0; j < n; j++ {
			temp := 0
			for k := 0; k < n; k++ {
				temp += a[i * n + k] * a[k * n + j]
			}
			c[i * n + j] = temp
		}
	}	

	return time.Since(start_time).Microseconds()
}

func main() {
	argv := os.Args
	if len(argv) < 2 {
		log.Fatal("Usage: go run . N\nwhere N is the size of the matrix to test as 2 ** N.\ni.e. if N =8, matrix is 256 x 256.\n")
	}

	n_exp, err := strconv.Atoi(argv[1])
	if err != nil {log.Fatal("Conversion error, goodbye!")}
	N := 1 << n_exp

	naive_result := naiveMul(N)
	fmt.Printf("naive took %dus\n", naive_result)
	transpose_result := transposeMul(N)
	fmt.Printf("trans took %dus\n", transpose_result)
}
