#include <vector>
#include <unordered_map>
#include <map>
#include <iostream>
#include <random>
#include <chrono>

class MyHashMap {
	//private:
	public:
		static const int TABLE_SIZE = 1021;
		std::vector<std::vector<std::pair<int, int>>> data;
	
		int hash(int key) const {
			return key % TABLE_SIZE;
		}	
	public:
		MyHashMap() : data(TABLE_SIZE) {}

		void set(int key, int value) {
			int idx = hash(key);
			for (auto &pair : data[idx]) {
				//std::cout << "here" << std::endl;
				if (pair.first == key) {
					pair.second = value;
					return;
				}
			}
			data[idx].push_back({key, value});
		}

		std::optional<int> get(int key) const {
			int idx = hash(key);
			for (const auto &pair : data[idx]) {
				if (pair.first == key) {
					return pair.second;
				}
			}
			return std::nullopt;
		}
		
		void increment(int key) {
			if (auto value = get(key)) {
				set(key, *value + 1);
			} else {
				set(key, 1);
			}
		}

		std::string toString() const {
			std::string result;
			for (int i = 0; i < TABLE_SIZE; i++) {
				if (!data[i].empty()) {
					//result += "{" + std::to_string(i) + ": "; // bucket
					for (const auto &pair : data[i]) {
						result += "{" + std::to_string(pair.first) + ": " + std::to_string(pair.second) + "}, ";
					}
					result += "\n";
				}
			}
			return result;
		}

		void print() const {
			std::cout << toString();
		}
};

void printVector(const std::vector<int> &v) {
	std::cout << "[";
	for (size_t i = 0; i < v.size(); i++){
		std::cout << v[i];
		if (i < v.size() - 1) std::cout << ", ";
	}
	std::cout << "]\n";
}

std::vector<int> naiveSolution(std::vector<int> &a, std::vector<int> &b, std::vector<std::vector<int>> &queries) {
	std::vector<int> ans;
	for (const auto &q : queries) {
		if (q[0] == 0) {
			b[q[1]] += q[2];
		} else {
			int found = 0;
			int x = q[1];
			for (int val : b) {
				int need = x - val;
				int count = 0;
				for (int num : a) {
					if (num == need) count++;
				}
				found += count;
			}
			ans.push_back(found);
		}

	}
	return ans;
}

std::vector<int> mySolution(std::vector<int> &a, std::vector<int> &b, std::vector<std::vector<int>> &queries) {
	std::unordered_map<int, int> aC;
	for (int num : a) aC[num]++;

	std::vector<int> ans;
	for (const auto &q : queries) {
		if (q[0] == 0) {
			b[q[1]] += q[2];
		} else {
			int found = 0;
			int x = q[1];
			for (int val : b) {
				int need = x - val;
				if (aC.count(need)) {
				//if (aC.find(need) == aC.end()) {
					found += aC[need];
				}
			}
			ans.push_back(found);
		}

	}
	return ans;
}

std::vector<int> hmSolution(std::vector<int> &a, std::vector<int> &b, std::vector<std::vector<int>> &queries) {
	MyHashMap aC;
	for (int num : a) aC.increment(num);

	//aC.print();

	std::vector<int> ans;
	for (const auto &q : queries) {
		if (q[0] == 0) {
			b[q[1]] += q[2];
		} else {
			int found = 0;
			int x = q[1];
			for (int val : b) {
				int need = x - val;
				if (need > 0) {
					if (auto count = aC.get(need)) {
						found += *count;
					}
				}
			}
			ans.push_back(found);
		}

	}
	return ans;
}

void testSolns(int times_to_run = 20) {
	std::random_device rand_dev;
	std::mt19937 gen(rand_dev());

	//std::uniform_int_distribution<int> len_dist1(1, 1000); // len of a, q
	//std::uniform_int_distribution<int> len_dist2(1, 5000); // len of b
	
	int a_len = 1000;//len_dist1(gen);
	int b_len = 5000;//len_dist2(gen);
	int q_len = 1000;//len_dist1(gen);

	std::uniform_int_distribution<int> val_dist(0, 100000000); // 10 ** 8
	std::uniform_int_distribution<int> bool_dist(0,1);
	std::uniform_int_distribution<int> b_idx_dist(0, b_len - 1);

	std::vector<int> a(a_len);
	std::vector<int> b(b_len);
	std::vector<std::vector<int>> queries(q_len);

	for (int i = 0; i < a_len; i++) a[i] = val_dist(gen);
	for (int i = 0; i < b_len; i++) b[i] = val_dist(gen);
	for (int i = 0; i < q_len; i++) {
		if (bool_dist(gen)) {
			queries[i] = {1, val_dist(gen)};
		} else {
			queries[i] = {0, b_idx_dist(gen), val_dist(gen)};
		}
	}

	std::cout << "a_len: " << a_len << " b_len: " << b_len << " q_len: " << q_len << " running " << times_to_run << " times\n";

	auto start = std::chrono::high_resolution_clock::now();
	std::vector<int> b_copy = b;	
	for (int i = 0; i < times_to_run; i++) {
		auto result = hmSolution(a, b_copy, queries);
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << MyHashMap::TABLE_SIZE << "hm\t" << duration.count() << "us\n"; 

	start = std::chrono::high_resolution_clock::now();
	b_copy = b;
	for (int i = 0; i < times_to_run; i++) {
		auto result = mySolution(a, b_copy, queries);
	}
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "counter\t" << duration.count() << "us\n"; 
}

int main() {
	std::vector<int> a0 = {1,2,3};
	std::vector<int> b0 = {1,4};
	std::vector<std::vector<int>> q0 = {{1,5}, {0,0,2}, {1,5}};
	
	std::vector<int> a1 = {1,2,2};
	std::vector<int> b1 = {2,3};
	std::vector<std::vector<int>> q1 = {{1,4}, {0,0,1}, {1,5}};

	std::vector<int> b_copy = b0;
	printVector(hmSolution(a0, b_copy, q0));
	b_copy = b0;
	printVector(mySolution(a0, b_copy, q0));

	b_copy = b1;
	printVector(hmSolution(a1, b_copy, q1));
	b_copy = b1;
	printVector(mySolution(a1, b_copy, q1));
	

	testSolns(5);
	std::exit(0);
}

