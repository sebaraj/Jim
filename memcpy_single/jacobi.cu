#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    return (itr != end);
}

int main(int argc, char* argv[]) {
    // Parse command line arguments. add anything else?
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const bool csv = get_arg(argv, argv + argc, "-csv");
}
