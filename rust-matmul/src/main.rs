use std::{env, time::Instant};

#[derive(Debug)]
struct TestResult {
    calling_fn: String,
    time_prepared: u128,
    time_running: u128,
}

impl TestResult {
    fn new(calling_fn: &str, t_p: u128, t_r: u128) -> Self {
        TestResult {
            calling_fn: calling_fn.to_string(),
            time_prepared: t_p,
            time_running: t_r,
        }
    }

    fn display(&self) {
        println!(
            "{}:\n\tTime Prepping:\t{}us\n\tTime Multiplying:\t\t{}us\n",
            self.calling_fn, self.time_prepared, self.time_running,
        );
    }
}
fn trans_one_dim(n: usize) -> TestResult {
    let prep_start_time = Instant::now();
    // Initialize matrices
    let a: Vec<usize> = vec![1; n * n];
    let mut b: Vec<usize> = vec![1; n * n];
    let mut c: Vec<usize> = vec![0, (n * n)];

    // Create transposed version of b for better memory access
    for i in 0..n {
        for j in 0..n {
            b.swap(i * n + j, j * n + i);
        }
    }
    let prep_end_time = prep_start_time.elapsed();
    let mul_time = Instant::now();
    // Perform matrix multiplication
    for i in 0..n {
        for j in 0..n {
            // Get the dot product of row i from a and column j from b
            let row_start = i * n;
            let col_start = j * n;

            // Using iterator for dot product
            //c[i * n + j] = (0..n).map(|k| a[row_start + k] * b[col_start + k]).sum();
            c.push((0..n).map(|k| a[row_start + k] * b[col_start + k]).sum());
        }
    }
    // Create transposed version of b for better memory access
    for i in 0..n {
        for j in 0..n {
            b.swap(i * n + j, j * n + i);
        }
    }
    TestResult::new(
        "transOneDim",
        prep_end_time.as_micros(),
        mul_time.elapsed().as_micros(),
    )
}
fn naive_one_dim(n: usize) -> TestResult {
    let prep_start_time = Instant::now();
    // Initialize matrices
    let a: Vec<usize> = vec![1; n * n];
    let b: Vec<usize> = vec![1; n * n];
    let mut c: Vec<usize> = Vec::with_capacity(n * n);

    let prep_end_time = prep_start_time.elapsed();
    let mul_time = Instant::now();

    // Perform matrix multiplication
    for i in 0..n {
        for j in 0..n {
            let mut temp: usize = 0;
            for k in 0..n {
                temp += a[i * n + k] * b[k * n + j];
            }
            //c[i * n + j] = temp;
            c.push(temp);
        }
    }
    TestResult::new(
        "transOneDim",
        prep_end_time.as_micros(),
        mul_time.elapsed().as_micros(),
    )
}

fn main() {
    // Parse N from command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} N, where N is the size of the matrix to test as 2 ** N (i.e. N=10 means 1024x1024", args[0]);
        std::process::exit(1);
    }

    let n_exp: usize = match args[1].parse() {
        Ok(num) => num,
        Err(_) => {
            eprintln!("Error: N must be a positive integer");
            std::process::exit(1);
        }
    };

    let n: usize = 1 << n_exp;

    let naive_one_dim_res: TestResult = naive_one_dim(n);
    naive_one_dim_res.display();
    let trans_one_dim_res: TestResult = trans_one_dim(n);
    trans_one_dim_res.display();
}
