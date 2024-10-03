#include "ipoptContactSolver.h"

// https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix.
// My version assumes column major txt file
template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<typename M::Scalar> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime>>(values.data(), rows, values.size()/rows);
}

int main() {
  VectorXl nx = load_csv<VectorXl>("nx.txt");
  VectorXl nu = load_csv<VectorXl>("nu.txt");
  VectorXl plane_idx = load_csv<VectorXl>("plane_idx.txt");
  VectorXl phase_starts = load_csv<VectorXl>("phase_starts.txt");
  VectorXl constraint_sizes = load_csv<VectorXl>("constraint_sizes.txt");
  VectorXd lb = load_csv<VectorXd>("ipopt_lb.txt");
  VectorXd ub = load_csv<VectorXd>("ipopt_ub.txt");
  VectorXd cl = load_csv<VectorXd>("ipopt_cl.txt");
  VectorXd cu = load_csv<VectorXd>("ipopt_cu.txt");
  VectorXd warm_start = load_csv<VectorXd>("warm_start.txt");

  IPOPTContactSolver solver(nx(0), nu(0), phase_starts, constraint_sizes, lb, ub, cl, cu, warm_start, -1, plane_idx(0), 250);
  VectorXd soln = VectorXd::Zero(warm_start.size());
  solver.solve(soln);

  return 0;
}
