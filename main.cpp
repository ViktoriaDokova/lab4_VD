// Variant 8: m = 3 fields.
// field 0: read 20%, write 10%; field 1: read 25%, write 10%;
// field 2: read 20%, write 10%; string: 5%.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std;

const int M = 3;
const int TOTAL_OPS = 1200000;  // divisible by 1, 2 and 3
const int REPEATS = 10;
const string DATA_DIR = "data";

class FieldStore {
public:
    int get(int i) const {
        lock_guard<mutex> lock(fields[i].m);
        return fields[i].value;
    }

    void set(int i, int value) {
        lock_guard<mutex> lock(fields[i].m);
        fields[i].value = value;
    }

    operator string() const {
        int v0, v1, v2;
        {
            // scoped_lock locks all three mutexes without risk of deadlock
            scoped_lock lock(fields[0].m, fields[1].m, fields[2].m);
            v0 = fields[0].value;
            v1 = fields[1].value;
            v2 = fields[2].value;
        }
        return to_string(v0) + " " + to_string(v1) + " " + to_string(v2);
    }

private:
    // Each field gets its own cache line (64 bytes), so threads working
    // with different fields don't slow each other down (false sharing).
    struct alignas(64) Field {
        mutable mutex m;
        int value = 0;
    };

    Field fields[M];
};

// ---------- test file generation ----------

struct Scenario {
    string name;
    string title;
    vector<double> weights;  // read0, write0, read1, write1, read2, write2, string
};

const vector<Scenario> scenarios = {
    {"a", "a) as in the task", {20, 10, 25, 10, 20, 10, 5}},
    {"b", "b) all equal", {1, 1, 1, 1, 1, 1, 1}},
    {"c", "c) very different", {2, 3, 2, 35, 3, 5, 50}},
};

string fileName(int scenario, int threads, int t) {
    return DATA_DIR + "/" + scenarios[scenario].name + "_" + to_string(threads) + "t_" + to_string(t) + ".txt";
}

void generateFile(const string& path, int count, const vector<double>& weights, int seed) {
    ofstream out(path);
    mt19937 rng(seed);
    discrete_distribution<int> pickOp(weights.begin(), weights.end());
    uniform_int_distribution<int> pickValue(0, 999);

    for (int i = 0; i < count; i++) {
        int k = pickOp(rng);
        // k = 0..5 -> read/write of field k / 2, k = 6 -> string
        if (k == 6)
            out << "string\n";
        else if (k % 2 == 0)
            out << "read " << k / 2 << "\n";
        else
            out << "write " << k / 2 << " " << pickValue(rng) << "\n";
    }
}

// Each thread gets its own file with TOTAL_OPS / threads operations.
// Only missing files are generated.
void generateMissingFiles() {
    filesystem::create_directories(DATA_DIR);
    int seed = 1;
    for (int s = 0; s < 3; s++)
        for (int threads = 1; threads <= 3; threads++)
            for (int t = 0; t < threads; t++, seed++)
                if (!filesystem::exists(fileName(s, threads, t)))
                    generateFile(fileName(s, threads, t), TOTAL_OPS / threads, scenarios[s].weights, seed);
}

// ---------- running and timing ----------

struct Op {
    char type;  // 'r', 'w' or 's'
    int field;
    int value;
};

vector<Op> loadOps(const string& path) {
    ifstream in(path);
    if (!in) {
        cout << "Cannot open " << path << endl;
        exit(1);
    }
    vector<Op> ops;
    string cmd;
    while (in >> cmd) {
        Op op{ cmd[0], 0, 0 };
        if (cmd == "read")
            in >> op.field;
        else if (cmd == "write")
            in >> op.field >> op.value;
        ops.push_back(op);
    }
    return ops;
}

// The result is returned via `sink` and printed as a checksum,
// so the compiler can't throw the reads away.
void worker(FieldStore& store, const vector<Op>& ops, long long& sink) {
    long long sum = 0;
    for (const Op& op : ops) {
        if (op.type == 'r')
            sum += store.get(op.field);
        else if (op.type == 'w')
            store.set(op.field, op.value);
        else
            sum += string(store).size();
    }
    sink = sum;
}

double runOnce(const vector<vector<Op>>& opsPerThread, long long& checksum) {
    FieldStore store;
    vector<long long> sinks(opsPerThread.size());
    vector<thread> threads;

    auto start = chrono::steady_clock::now();
    for (size_t t = 0; t < opsPerThread.size(); t++)
        threads.emplace_back(worker, ref(store), cref(opsPerThread[t]), ref(sinks[t]));
    for (thread& th : threads)
        th.join();
    auto finish = chrono::steady_clock::now();

    for (long long sink : sinks)
        checksum += sink;
    return chrono::duration<double, milli>(finish - start).count();
}

int main() {
    generateMissingFiles();

    long long checksum = 0;
    cout << "Average time over " << REPEATS << " runs, " << TOTAL_OPS << " operations per case\n\n";
    cout << fixed << setprecision(2);

    for (int s = 0; s < 3; s++) {
        for (int threads = 1; threads <= 3; threads++) {
            vector<vector<Op>> opsPerThread;
            for (int t = 0; t < threads; t++)
                opsPerThread.push_back(loadOps(fileName(s, threads, t)));

            runOnce(opsPerThread, checksum);  // warm-up, not counted
            double total = 0;
            for (int r = 0; r < REPEATS; r++)
                total += runOnce(opsPerThread, checksum);

            cout << scenarios[s].title << " threads: " << threads
                << "  avg time: " << total / REPEATS << " ms" << endl;
        }
    }

    cout << "\nChecksum: " << checksum << endl;
    return 0;
}