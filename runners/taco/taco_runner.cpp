
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/tensor.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace taco;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc <= 1) {
    std::cout << "this program requires arguments.";
    return 1;
  }

  // Dumb stuff to cast arguments to strings for easier parsing.
  string op = argv[1];
  if (op == "gentensor") {

  } else if (op == "evaluate") {
  }

  Tensor<long> A("A", {2048, 2048}, Format({Dense, Dense}));
  write("A.tns", A);
}
