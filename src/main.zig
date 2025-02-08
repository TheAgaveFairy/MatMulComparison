// 0.14.0
const std = @import("std");
const printerr = std.debug.print;

const vector_len = std.simd.suggestVectorLength(usize).?;

const TestResult = struct {
    calling_fn: []const u8,
    time_prepared: i64,
    time_running: i64,

    pub fn new(calling: []const u8, t_p: i64, t_r: i64) @This() {
        const result: @This() = .{
            .calling_fn = calling,
            .time_prepared = t_p,
            .time_running = t_r,
        };
        return result;
    }

    pub fn display(self: @This()) void {
        printerr("{str}:\n\tTime to Allocate and Fill:\t{d}us.\n\tTime to Multiply:\t\t{d}us\n", .{ self.calling_fn, self.time_prepared, self.time_running });
    }
};
fn alreadyTransposedMul(allocator: std.mem.Allocator, n: usize) !TestResult {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const start_time = std.time.microTimestamp();

    var a = try allocator.alloc(usize, n * n);
    defer allocator.free(a);
    var b = try allocator.alloc(usize, n * n);
    defer allocator.free(b);
    var r = try allocator.alloc(usize, n * n);
    defer allocator.free(r);

    var x: usize = 0;
    while (x < n * n) : (x += 1) {
        a[x] = random.intRangeAtMost(usize, 0, 1000);
        b[x] = random.intRangeAtMost(usize, 0, 1000);
    }

    const mul_time = std.time.microTimestamp();
    for (0..n) |i| {
        for (0..n) |j| {
            var sum: usize = 0;
            for (0..n) |k| {
                sum += a[i * n + k] * b[j * n + k];
            }
            r[i * n + j] = sum;
        }
    }

    const end_time = std.time.microTimestamp();

    return TestResult.new(@src().fn_name, mul_time - start_time, end_time - mul_time);
}
fn transposeMul(allocator: std.mem.Allocator, n: usize) !TestResult {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const start_time = std.time.microTimestamp();

    var a = try allocator.alloc(usize, n * n);
    defer allocator.free(a);
    var b = try allocator.alloc(usize, n * n);
    defer allocator.free(b);
    var r = try allocator.alloc(usize, n * n);
    defer allocator.free(r);

    var x: usize = 0;
    while (x < n * n) : (x += 1) {
        a[x] = random.intRangeAtMost(usize, 0, 1000);
        b[x] = random.intRangeAtMost(usize, 0, 1000);
    }

    const mul_time = std.time.microTimestamp();
    //transpose B
    for (0..n) |i| {
        for (0..n) |j| {
            const temp = b[i * n + j];
            b[i * n + j] = b[j * n + i];
            b[j * n + i] = temp;
        }
    }
    //const trans_time = std.time.microTimestamp();
    //printerr("{d}us to transpose, this will effect other numbers so delete this line\n", .{trans_time - mul_time});
    for (0..n) |i| {
        for (0..n) |j| {
            var sum: usize = 0;
            for (0..n) |k| {
                sum += a[i * n + k] * b[j * n + k];
            }
            r[i * n + j] = sum;
        }
    }

    const end_time = std.time.microTimestamp();

    return TestResult.new(@src().fn_name, mul_time - start_time, end_time - mul_time);
}
fn simdMul(allocator: std.mem.Allocator, n: usize) !TestResult {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const start_time = std.time.microTimestamp();

    const VecType = @Vector(vector_len, usize);

    var a = try allocator.alloc(usize, n * n);
    defer allocator.free(a);
    var b = try allocator.alloc(usize, n * n);
    defer allocator.free(b);
    var r = try allocator.alloc(usize, n * n);
    defer allocator.free(r);

    var x: usize = 0;
    while (x < n * n) : (x += 1) {
        //a[i..i + vector_len] = @Vector(random.intRangeAtMost(usize, 0, 1000); // another day perhaps
        a[x] = random.intRangeAtMost(usize, 0, 1000);
        b[x] = random.intRangeAtMost(usize, 0, 1000);
    }

    const mul_time = std.time.microTimestamp();

    for (0..n) |i| {
        for (0..n) |j| {
            var sum: usize = 0;
            var k: usize = 0;
            while (k < n) : (k += vector_len) {
                const row_start = i * n + k;
                const row: VecType = @as(VecType, a[row_start..][0..vector_len].*);

                var col: VecType = undefined;
                for (0..vector_len) |v| col[v] = b[(k + v) * n + j];

                const mul = row * col;

                sum += @reduce(.Add, mul);
            }
            r[i * n + j] = sum;
        }
    }
    const end_time = std.time.microTimestamp();
    return TestResult.new(@src().fn_name, mul_time - start_time, end_time - mul_time);
}

/// Multiplies two n * n matrices and returns the time to alloc and multiply in us.
fn oneDMul(allocator: std.mem.Allocator, n: usize) !TestResult {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const start_time = std.time.microTimestamp();

    var a = try allocator.alloc(usize, n * n);
    defer allocator.free(a);
    var b = try allocator.alloc(usize, n * n);
    defer allocator.free(b);
    var r = try allocator.alloc(usize, n * n);
    defer allocator.free(r);

    for (0..n * n) |i| {
        a[i] = random.intRangeAtMost(usize, 0, 1000);
        b[i] = random.intRangeAtMost(usize, 0, 1000);
    }

    const mul_time = std.time.microTimestamp();

    for (0..n) |i| {
        for (0..n) |j| {
            var sum: usize = 0;
            for (0..n) |k| {
                sum += a[i * n + k] * b[k * n + j];
            }
            r[i * n + j] = sum;
        }
    }
    const end_time = std.time.microTimestamp();
    return TestResult.new(@src().fn_name, mul_time - start_time, end_time - mul_time);
}

pub fn main() !void {
    //try stdout.print("Run `zig build test` to run the tests.\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    const argc = std.os.argv.len;
    if (argc < 2) {
        printerr("Please supply N (2 ** N for matrix size). {} given.\n", .{argc});
        return error.NoArgsSupplied;
    }

    const c_string = std.os.argv[1];
    const N_exp_slice = std.mem.span(c_string);
    const N_exp = try std.fmt.parseInt(u5, N_exp_slice, 10);
    const N: u32 = @as(u32, 1) << N_exp;
    printerr("N = {d}. Thus, matrix is {d} x {d}.\n", .{ N_exp, N, N });

    const oneDResult = try oneDMul(alloc, N);
    oneDResult.display();
    const simd_result = try simdMul(alloc, N);
    simd_result.display();
    const transpose_result = try transposeMul(alloc, N);
    transpose_result.display();
    const already_transposed_result = try alreadyTransposedMul(alloc, N);
    already_transposed_result.display();
}
