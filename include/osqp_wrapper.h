#pragma once

#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class OsqpWrapper {
  public:
    OsqpWrapper(int num_decision_vars, int num_constraints);

    void reset_num_decision_vars(int num_decision_vars);
    void reset_num_constraints(int num_constraints);
    bool update_hessian(const Ref<const VectorXi> &rows, const Ref<const VectorXi> &cols, const Ref<const VectorXd> &vals);
    bool update_constraint_matrix(const Ref<const Matrix<long, Dynamic, 1>> &rows, const Ref<const Matrix<long, Dynamic, 1>> &cols, const Ref<const VectorXd> &vals);
    bool update_constraint_lb(Ref<VectorXd> lb);
    bool update_constraint_ub(Ref<VectorXd> ub);
    bool update_gradient(Ref<VectorXd> gradient);

    bool solve(Ref<VectorXd> primal);

  private:
    int num_decision_vars;
    int num_constraints;
    OsqpEigen::Solver solver;
    SparseMatrix<double> hessian;
    SparseMatrix<double> constraint_matrix;
};
