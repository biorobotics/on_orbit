#include "osqp_wrapper.h"

OsqpWrapper::OsqpWrapper(int num_decision_vars, int num_constraints) : num_decision_vars(num_decision_vars), num_constraints(num_constraints), hessian(num_decision_vars, num_decision_vars), constraint_matrix(num_constraints, num_decision_vars) {
  solver.settings()->setVerbosity(true);
  solver.settings()->setWarmStart(true);
  solver.settings()->setAdaptiveRhoInterval(50);
  // solver.settings()->setAbsoluteTolerance(1e-6);
  // solver.settings()->setRelativeTolerance(1e-6);

  solver.clearSolver();
  solver.data()->clearHessianMatrix();
  solver.data()->clearLinearConstraintsMatrix();

  solver.data()->setNumberOfVariables(num_decision_vars);
  solver.data()->setNumberOfConstraints(num_constraints);
}

void OsqpWrapper::reset_num_decision_vars(int num_decision_vars) {
  solver.clearSolver();
  solver.data()->clearHessianMatrix();
  solver.data()->clearLinearConstraintsMatrix();
  solver.data()->setNumberOfVariables(num_decision_vars);
  solver.data()->setNumberOfConstraints(num_constraints);
}

void OsqpWrapper::reset_num_constraints(int num_constraints) {
  solver.data()->clearLinearConstraintsMatrix();
  this->num_constraints = num_constraints;
  solver.data()->setNumberOfConstraints(num_constraints);
}

bool OsqpWrapper::update_hessian(const Ref<const VectorXi> &rows, const Ref<const VectorXi> &cols, const Ref<const VectorXd> &vals) {
  vector<Triplet<double>> triplets(vals.size());
  for (int i = 0; i < vals.size(); ++i) {
    triplets[i] = Triplet<double>(rows(i), cols(i), vals(i));
  }
  hessian.setFromTriplets(triplets.begin(), triplets.end());
  return solver.data()->setHessianMatrix(hessian);
}

bool OsqpWrapper::update_constraint_matrix(const Ref<const Matrix<long, Dynamic, 1>> &rows, const Ref<const Matrix<long, Dynamic, 1>> &cols, const Ref<const VectorXd> &vals) {
  vector<Triplet<double>> triplets(vals.size());
  for (int i = 0; i < vals.size(); ++i) {
    triplets[i] = Triplet<double>(rows(i), cols(i), vals(i));
  }
  constraint_matrix.setFromTriplets(triplets.begin(), triplets.end());
  return solver.data()->setLinearConstraintsMatrix(constraint_matrix); }

bool OsqpWrapper::update_constraint_lb(Ref<VectorXd> lb) {
  return solver.data()->setLowerBound(lb);
}

bool OsqpWrapper::update_constraint_ub(Ref<VectorXd> ub) {
  return solver.data()->setUpperBound(ub);
}

bool OsqpWrapper::update_gradient(Ref<VectorXd> gradient) {
  return solver.data()->setGradient(gradient);
}

bool OsqpWrapper::solve(Ref<VectorXd> primal) {
  if (!solver.initSolver()) return false;
  // solver.setPrimalVariable(VectorXd(primal));

  // solve the QP problem
  bool solved = solver.solve();
  if(!solved) {
    cout << "OSQP failed. Status: " << solver.workspace()->info->status_val << endl;
  }

  primal = solver.getSolution();

  return solved;
}
