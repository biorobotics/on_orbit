#include "constrained_llsq.h"
#include <chrono>

ConstrainedLLSQ::ConstrainedLLSQ() {
  solver.settings()->setVerbosity(false);
  solver.settings()->setWarmStart(true);
  solver.settings()->setAdaptiveRhoInterval(50);

  /*
  solver.settings()->setAbsoluteTolerance(1.0);
  solver.settings()->setRelativeTolerance(0.);
  solver.settings()->setCheckTermination(1);
  */

  solver.settings()->setAbsoluteTolerance(1e-4);
  solver.settings()->setRelativeTolerance(0.);

  /*
  solver.settings()->setAbsoluteTolerance(1e-8);
  solver.settings()->setRelativeTolerance(1e-8);
  */
  solver.settings()->setMaxIteration(100000);

  prev_solve_time = 0;
  prev_setup_time = 0;
}

void ConstrainedLLSQ::reset(size_t num_decision_vars) {
  this->num_decision_vars = num_decision_vars;
  num_constraints = 0;
  lhs.clear();
  rhs.clear();
  constraint_mats.clear();
  lbs.clear();
  ubs.clear();
  Qs.clear();
  cs.clear();

  solver.clearSolver();
  solver.data()->clearHessianMatrix();
  solver.data()->clearLinearConstraintsMatrix();
}

void ConstrainedLLSQ::add_llsq_objective(const MatrixXd &A, const VectorXd &b, double weight) {
  lhs.push_back(sqrt(weight)*A);
  rhs.push_back(sqrt(weight)*b);
}

void ConstrainedLLSQ::add_quadratic_cost(const MatrixXd &Q, double weight) {
  Qs.push_back(weight*Q);
}

void ConstrainedLLSQ::add_linear_cost(const VectorXd &c, double weight) {
  cs.push_back(weight*c);
}

void ConstrainedLLSQ::add_constraint(const MatrixXd &C, const VectorXd &lb, const VectorXd &ub) {
  constraint_mats.push_back(C);
  lbs.push_back(lb);
  ubs.push_back(ub);
  num_constraints += lb.size();
}

bool ConstrainedLLSQ::solve(VectorXd &primal, VectorXd &dual, const VectorXd &primal_warm_start, const VectorXd &dual_warm_start, bool use_primal_warm_start, bool use_dual_warm_start) {
  auto start_time = std::chrono::high_resolution_clock::now();
  MatrixXd dense_hessian = MatrixXd::Zero(num_decision_vars, num_decision_vars);
  VectorXd gradient = VectorXd::Zero(num_decision_vars);
  size_t num_llsq = lhs.size();
  for (size_t i = 0; i < num_llsq; ++i) {
    dense_hessian += lhs[i].transpose()*lhs[i];
    gradient += -rhs[i].transpose()*lhs[i];
  }
  size_t num_quadratic_costs = Qs.size();
  for (size_t i = 0; i < num_quadratic_costs; ++i) {
    dense_hessian += Qs[i];
  }
  // Sometimes we get negative Hessian eigenvalues in O(-1e-15)
  // dense_hessian += 1e-10*MatrixXd::Identity(num_decision_vars, num_decision_vars); 

  size_t num_linear_costs = cs.size();
  for (size_t i = 0; i < num_linear_costs; ++i) {
    gradient += cs[i];
  }

  SparseMatrix<double> hessian(num_decision_vars, num_decision_vars);
  for (size_t row = 0; row < num_decision_vars; ++row) {
    for (size_t col = 0; col < num_decision_vars; ++col) {
      if (dense_hessian(row, col) != 0) {
        hessian.insert(row, col) = dense_hessian(row, col);
      }
    }
  }

  SparseMatrix<double> linearMatrix(num_constraints, num_decision_vars);
  linearMatrix.setZero();
  VectorXd lowerBound(num_constraints);
  VectorXd upperBound(num_constraints);

  size_t start_row = 0;
  for (size_t i = 0; i < constraint_mats.size(); ++i) {
    size_t rows = constraint_mats[i].rows();
    for (size_t row = 0; row < rows; ++row) {
      size_t total_row = row + start_row;
      lowerBound(total_row) = lbs[i](row);
      upperBound(total_row) = ubs[i](row);
      for (size_t col = 0; col < num_decision_vars; ++col) {
        if (constraint_mats[i](row, col) != 0) {
          linearMatrix.insert(total_row, col) = constraint_mats[i](row, col);
        }
      }
    }
    start_row += rows;
  }

  solver.data()->setNumberOfVariables(num_decision_vars);
  solver.data()->setNumberOfConstraints(num_constraints);
  if(!solver.data()->setHessianMatrix(hessian)) return false;
  if(!solver.data()->setGradient(gradient)) return false;
  if(!solver.data()->setLinearConstraintsMatrix(linearMatrix)) return false;
  if(!solver.data()->setLowerBound(lowerBound)) return false;
  if(!solver.data()->setUpperBound(upperBound)) return false;

  if(!solver.initSolver()) return false;

  if (use_primal_warm_start && primal_warm_start.size() == num_decision_vars) {
    solver.setPrimalVariable(primal_warm_start);
  }

  if (use_dual_warm_start && dual_warm_start.size() == num_constraints) {
    solver.setDualVariable(dual_warm_start);
  }

  auto stop_time = std::chrono::high_resolution_clock::now();
  prev_setup_time = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();

  // solve the QP problem
  start_time = std::chrono::high_resolution_clock::now();
  bool solved = solver.solve();
  stop_time = std::chrono::high_resolution_clock::now();
  prev_solve_time = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();

  primal = solver.getSolution();
  dual = solver.getDualSolution();

  num_iter = solver.workspace()->info->iter;

  max_vio = std::max(std::max((linearMatrix*primal - upperBound).maxCoeff(), (lowerBound - linearMatrix*primal).maxCoeff()), 0.0);

  if(!solved) {
    cout << "status: " << solver.workspace()->info->status << " status value: " << solver.workspace()->info->status_val << endl;
    double max_vio = std::max(std::max((linearMatrix*primal - upperBound).maxCoeff(), (lowerBound - linearMatrix*primal).maxCoeff()), 0.0);
    std::cout << "Max constraint violation: " << max_vio << std::endl;
    std::cout << "Polish status: " << solver.workspace()->info->status_polish << std::endl;
  }

  return solved;
}

double ConstrainedLLSQ::get_prev_solve_time() {
  return prev_solve_time;
}

double ConstrainedLLSQ::get_prev_setup_time() {
  return prev_setup_time;
}

int ConstrainedLLSQ::get_num_constraints() {
  return num_constraints;
}

int ConstrainedLLSQ::get_num_iter() {
  return num_iter;
}

int ConstrainedLLSQ::get_polish_status() {
  return solver.workspace()->info->status_polish;
}

double ConstrainedLLSQ::get_max_vio() {
  return max_vio;
}
