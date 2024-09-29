
// #include "random"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/parser/parser.h"
// #include "taco/parser/schedule_parser.h"
#include "taco/tensor.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace taco;
using namespace std;

extern int compute();

vector<int> parseDims(const string &s) {
  vector<int> values;
  stringstream stream(s);
  string tok;

  while (getline(stream, tok, ',')) {
    values.push_back(stoi(tok));
  }

  return values;
}

vector<ModeFormatPack> parseFormats(const string &s) {
  vector<ModeFormatPack> values;
  stringstream stream(s);
  string tok;

  while (getline(stream, tok, ',')) {
    // Apparently we can't switch on strings in c++, marvelous right?
    if (tok == "s") {
      values.push_back(Sparse);
    } else if (tok == "d") {
      values.push_back(Dense);
    } else if (tok == "c") {
      values.push_back(Compressed);
    } else if (tok == "g") {
      values.push_back(Singleton);
    }
  }

  return values;
}

int main(int argc, char *argv[]) {
  if (argc <= 1) {
    cout << "this program requires arguments.\n";
    return 1;
  }

  // Dumb stuff to cast arguments to strings for easier parsing.
  string op = argv[1];
  // Apparently we can't switch on strings in c++, marvelous right?
  if (op == "gentensor") {
    if (argc != 5) {
      cout << "gentensor <name> <dims> <formats>\n";
      return 1;
    }
    string name = argv[2];
    vector<int> dims = parseDims(argv[3]);
    vector<ModeFormatPack> formats = parseFormats(argv[4]);

    // Seed RNG with the current time approximately.
    // srand(time(0));

    Tensor<double> T(name, dims, Format(formats));
    // Populate the tensor with some data.
    for (int dim = dims.size() - 1; dim > 0; dim--) {
      if (formats[dim] == Dense) {

      } else if (formats[dim] == Sparse || formats[dim] == Compressed) {
      } else if (formats[dim] == Singleton) {
      }
    }

    cout << "packing generated tensor into correct format\n";
    // Compress the tensor into the correct format.
    T.pack();
    cout << "packed generated tensor\n";

    // save and write the tensor out.
    stringstream ss;
    ss << name << ".tns";
    write(ss.str(), T);
    return 0;
  } else if (op == "evaluate") {
    if (argc != 5) {
      cout << "evaluate <expression> <schedule>";
      return 1;
    }

    const string expression = argv[2];
    const string schedule = argv[3];

    // vector<vector<string>> parsed = parser::ScheduleParser(schedule);
  } else {
    cout << "unknown command provided, exiting";
    return 1;
  }
}
