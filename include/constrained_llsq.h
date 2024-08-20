#ifndef CONSTRAINED_LLSQ_H
#define CONSTRAINED_LLSQ_H

#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class ConstrainedLLSQ {
  public:
    /*
     * CONSTRUCTOR: initialized internal variables
     */
    ConstrainedLLSQ();

    /*
     * reset: removes all equations from the current problem and begins a new
     * problem with the specified number of decision variables
     * ARGUMENTS
     * num_decision_vars: number of decision variables in new problem
     */
    void reset(size_t num_decision_vars);

    /*
     * add_llsq_objective: add a linear equation sqrt(weight)*Ax = sqrt(weight)*b whose residual we try
     * to minimize
     * ARGUMENTS
     * A: A
     * b: b
     * weight: weight, representing the priority of this equation
     * in the quadratic program
     * REQUIRES: A.rows() == b.rows(), A.cols() == num_decision_vars
     */
    void add_llsq_objective(const MatrixXd &A, const VectorXd &b, double weight);

    /*
     * add_linear_cost: add a linear cost weight*c.transpose()*x that we try to minimize
     * to minimize
     * ARGUMENTS
     * D: D
     * weight: weight, representing the priority of this cost
     * in the quadratic program
     * REQUIRES: Q.rows() == num_decision_vars
     */
    void add_linear_cost(const VectorXd &c, double weight);

    /*
     * add_quadratic_cost: add a quadratic cost weight*x'Qx that we try to minimize
     * to minimize
     * ARGUMENTS
     * Q: Q
     * weight: weight, representing the priority of this cost
     * in the quadratic program
     * REQUIRES: Q.rows() == Q.cols(), Q.rows() == num_decision_vars
     */
    void add_quadratic_cost(const MatrixXd &Q, double weight);

    /*
     * add constraint: adds a constraint of the form lb <= Cx <= ub
     * ARGUMENTS
     * C: C
     * lb: lb
     * ub: ub
     * REQUIRES: C.rows() == lb.rows() == ub.rows(), C.cols() == num_decision_vars
     */
    void add_constraint(const MatrixXd &C, const VectorXd &lb, const VectorXd &ub);

    /*
     * solve: stack all the linear equations together, turn the whole thing into a
     * quadratic program, and solve
     * ARGUMENTS
     * primal: populated with primal solution, if feasible
     * dual: populated with primal solution, if feasible
     * primal warm start: primal warm start
     * dual warm start: dual warm start
     * use_primal_warm_start: use primal warm start?
     * use_dual_warm_start: use dual warm start?
     * RETURN: true if a solution was feasible, false if not
     */
    bool solve(VectorXd &primal, VectorXd &dual, const VectorXd &primal_warm_start, const VectorXd &dual_warm_start, bool use_primal_warm_start, bool use_dual_warm_start);

    double get_prev_solve_time();
    double get_prev_setup_time();

    int get_num_constraints();

    int get_num_iter();

    int get_polish_status();

    double get_max_vio();

  private:
    // Stores sqrt(weight)*A matrices in llsq costs
    vector<MatrixXd> lhs;

    // Stores sqrt(weight)*b vectors in llsq costs
    vector<VectorXd> rhs;

    // Stores weight*Q matrices in quadratic costs
    vector<MatrixXd> Qs;

    // Stores weight*c vectors in linear costs
    vector<VectorXd> cs;

    // Stores C matrices in constraint equations lb <= Cx <= ub
    vector<MatrixXd> constraint_mats;

    // Stores lb vectors in constraint equations lb <= Cx <= ub
    vector<MatrixXd> lbs;

    // Stores ub vectors in constraint equations lb <= Cx <= ub
    vector<MatrixXd> ubs;

    size_t num_constraints;
    size_t num_decision_vars;

    OsqpEigen::Solver solver;

    double prev_solve_time;
    double prev_setup_time;
    
    int num_iter;

    double max_vio;
};

#endif
