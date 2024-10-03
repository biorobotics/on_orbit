#include "ipoptContactSolver.h"

IPOPTContactSolver::IPOPTContactSolver(int nx, int nu, VectorXlRef_const phase_starts, VectorXlRef_const constraint_sizes, VectorXdRef_const lb, VectorXdRef_const ub, VectorXdRef_const cl, VectorXdRef_const cu, VectorXdRef_const warm_start, int success_after_iter, int plane_idx, int max_iter) {
  nlp = new ContactNLP(nx, nu, phase_starts, constraint_sizes, lb, ub, cl, cu, warm_start, success_after_iter, plane_idx);

  // Create a new instance of IpoptApplication
  //  (use a SmartPtr, not raw)
  // We are using the factory, since this allows us to compile this
  // example with an Ipopt Windows DLL
  app = new Ipopt::IpoptApplication();
  
  // Change some options
  // Note: The following choices are only examples, they might not be
  //       suitable for your optimization problem.
  app->Options()->SetNumericValue("tol", 1.0);
  app->Options()->SetNumericValue("constr_viol_tol", 1e-4);
  app->Options()->SetNumericValue("compl_inf_tol", 1e-2);
  app->Options()->SetIntegerValue("max_iter", max_iter);
  app->Options()->SetStringValue("warm_start_init_point", "no");
  
  // Initialize the IpoptApplication and process the options
  Ipopt::ApplicationReturnStatus status;
  status = app->Initialize();
  if( status != Ipopt::Solve_Succeeded )
  {
     std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
     std::cout << "Status " << (int) status << std::endl;
     exit(1);
  }
}

bool IPOPTContactSolver::solve(VectorXdRef soln) {
  // Ask Ipopt to solve the problem
  Ipopt::ApplicationReturnStatus status;
  status = app->OptimizeTNLP(nlp);

  nlp->get_soln(soln);

  num_iter = nlp->get_num_iter();
  
  if( status == Ipopt::Solve_Succeeded || status == Ipopt::User_Requested_Stop)
  {
     std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
     return true;
  }
  else
  {
     std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
     return false;
  }
  
  // As the SmartPtrs go out of scope, the reference count
  // will be decremented and the objects will automatically
  // be deleted.
}

ContactNLP::ContactNLP(int nx, int nu, VectorXlRef_const phase_starts, 
                       VectorXlRef_const constraint_sizes, 
                       VectorXdRef_const lb, VectorXdRef_const ub, 
                       VectorXdRef_const cl, VectorXdRef_const cu, 
                       VectorXdRef_const warm_start, int success_after_iter, int plane_idx) : 
                       nx(nx), 
                       nu(nu), 
                       phase_starts(phase_starts), 
                       constraint_sizes(constraint_sizes), 
                       lb(lb), 
                       ub(ub), 
                       cl(cl), 
                       cu(cu), 
                       warm_start(warm_start), 
                       success_after_iter(success_after_iter) 
  {
  vars_per_step = nx + nu;
  num_decision_vars = vars_per_step*phase_starts(phase_starts.size() - 1);

  // Load the functions
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    std::string phase_str = std::to_string(phase_idx);

    cost_functions.push_back(std::make_shared<CasadiExternalFunction2>("cost_phase_" + phase_str, plane_idx));
    cost_gradients.push_back(std::make_shared<CasadiExternalFunction2>("cost_grad_phase_" + phase_str, plane_idx));
    cost_hessians.push_back(std::make_shared<CasadiExternalFunction2>("cost_hess_phase_" + phase_str, plane_idx));
    constraint_functions.push_back(std::make_shared<CasadiExternalFunction2>("constraint_phase_" + phase_str, plane_idx));
    constraint_jacobians.push_back(std::make_shared<CasadiExternalFunction2>("constraint_jacobian_phase_" + phase_str, plane_idx));
  }

  // Get how many nonzeros are in the jacobians and hessian.
  // Don't worry about the cost, cost gradient, or constraint functions. We assume these were densified before code generation
  J_nnz = 0;
  H_nnz = 0;
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    int phase_length = phase_starts(phase_idx + 1) - phase_starts(phase_idx);
    J_nnz += constraint_jacobians[phase_idx]->get_nnz()*phase_length;
    H_nnz += cost_hessians[phase_idx]->get_nnz()*phase_length;
  }

  // Populate sparsity structure for jacobians and hessians
  Jrows = VectorXl::Zero(J_nnz);
  Jcols = VectorXl::Zero(J_nnz);
  Hrows = VectorXl::Zero(H_nnz);
  Hcols = VectorXl::Zero(H_nnz);
  int Jstart_idx = 0;
  int Hstart_idx = 0;
  int c_idx = 0;
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    int J_nnz_step = constraint_jacobians[phase_idx]->get_nnz();
    int H_nnz_step = cost_hessians[phase_idx]->get_nnz();
    for (int step = phase_starts(phase_idx); step < phase_starts(phase_idx + 1); ++step) {
      int decision_var_idx = step*vars_per_step;
      constraint_jacobians[phase_idx]->get_sparsity_structure(Jrows.segment(Jstart_idx, J_nnz_step),
                                                              Jcols.segment(Jstart_idx, J_nnz_step),
                                                              c_idx, decision_var_idx);

      c_idx += constraint_sizes(phase_idx);
      Jstart_idx += J_nnz_step;
      
      cost_hessians[phase_idx]->get_sparsity_structure(Hrows.segment(Hstart_idx, H_nnz_step),
                                                       Hcols.segment(Hstart_idx, H_nnz_step),
                                                       decision_var_idx, decision_var_idx);
      Hstart_idx += H_nnz_step;
    }
  }

  num_constraints = 0;
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    int phase_length = phase_starts(phase_idx + 1) - phase_starts(phase_idx);
    num_constraints += constraint_sizes(phase_idx)*phase_length;
  }
  c = VectorXd::Zero(num_constraints);
}

bool ContactNLP::get_nlp_info(
   Ipopt::Index&          n,
   Ipopt::Index&          m,
   Ipopt::Index&          nnz_jac_g,
   Ipopt::Index&          nnz_h_lag,
   Ipopt::TNLP::IndexStyleEnum& index_style
) {
  n = num_decision_vars;
  m = num_constraints;
  nnz_jac_g = J_nnz;
  nnz_h_lag = H_nnz;
  index_style = TNLP::C_STYLE;
  return true;
}

/** Method to return the bounds for my problem */
bool ContactNLP::get_bounds_info(
   Ipopt::Index   n,
   Ipopt::Number* x_l,
   Ipopt::Number* x_u,
   Ipopt::Index   m,
   Ipopt::Number* g_l,
   Ipopt::Number* g_u
) {
  Map<VectorXd>(x_l, n) = lb;
  Map<VectorXd>(x_u, n) = ub;
  Map<VectorXd>(g_l, m) = cl;
  Map<VectorXd>(g_u, m) = cu;
  return true;
}

/** Method to return the starting point for the algorithm */
bool ContactNLP::get_starting_point(
   Ipopt::Index   n,
   bool    init_x,
   Ipopt::Number* x,
   bool    init_z,
   Ipopt::Number* z_L,
   Ipopt::Number* z_U,
   Ipopt::Index   m,
   bool    init_lambda,
   Ipopt::Number* lambda
) {
  if (init_x) {
    Map<VectorXd>(x, n) = warm_start;
  }
  if (init_z) {
    Map<VectorXd>(z_L, n).setZero();
    Map<VectorXd>(z_U, n).setZero();
  }
  if (init_lambda) {
    Map<VectorXd>(lambda, m).setZero();
  }
  return true;
}

/** Method to return the objective value */
bool ContactNLP::eval_f(
   Ipopt::Index         n,
   const Ipopt::Number* x,
   bool          new_x,
   Ipopt::Number&       obj_value
) {
  double objective = 0;
  double step_objective = 0;
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    for (int step = phase_starts(phase_idx); step < phase_starts(phase_idx + 1); ++step) {
      cost_functions[phase_idx]->call(&x[vars_per_step*step], &step_objective);
      objective += step_objective;
    }
  }
  obj_value = objective;
  return true;
}

/** Method to return the gradient of the objective */
bool ContactNLP::eval_grad_f(
   Ipopt::Index         n,
   const Ipopt::Number* x,
   bool          new_x,
   Ipopt::Number*       grad_f
) {
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    for (int step = phase_starts(phase_idx); step < phase_starts(phase_idx + 1); ++step) {
      cost_gradients[phase_idx]->call(&x[vars_per_step*step], &grad_f[vars_per_step*step]);
    }
  }
  return true;
}

/** Method to return the constraint residuals */
bool ContactNLP::eval_g(
   Ipopt::Index         n,
   const Ipopt::Number* x,
   bool          new_x,
   Ipopt::Index         m,
   Ipopt::Number*       g
) {
  int c_idx = 0;
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    for (int step = phase_starts(phase_idx); step < phase_starts(phase_idx + 1); ++step) {
      constraint_functions[phase_idx]->call(&x[vars_per_step*step], &g[c_idx]);
      c_idx += constraint_sizes(phase_idx);
    }
  }
  return true;
}

/** Method to return:
 *   1) The structure of the jacobian (if "values" is NULL)
 *   2) The values of the jacobian (if "values" is not NULL)
 */
bool ContactNLP::eval_jac_g(
   Ipopt::Index         n,
   const Ipopt::Number* x,
   bool          new_x,
   Ipopt::Index         m,
   Ipopt::Index         nele_jac,
   Ipopt::Index*        iRow,
   Ipopt::Index*        jCol,
   Ipopt::Number*       values
) {
  if (values == NULL) {
    Map<VectorXl>(iRow, nele_jac) = Jrows;
    Map<VectorXl>(jCol, nele_jac) = Jcols;
  } else {
    int J_idx = 0;
    for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
      int J_nnz_step = constraint_jacobians[phase_idx]->get_nnz();
      for (int step = phase_starts(phase_idx); step < phase_starts(phase_idx + 1); ++step) {
        constraint_jacobians[phase_idx]->call(&x[vars_per_step*step], &values[J_idx]);
        J_idx += J_nnz_step;
      }
    }
  }
  return true;
}

/** Method to return:
 *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
 *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
 */
bool ContactNLP::eval_h(
   Ipopt::Index         n,
   const Ipopt::Number* x,
   bool          new_x,
   Ipopt::Number        obj_factor,
   Ipopt::Index         m,
   const Ipopt::Number* lambda,
   bool          new_lambda,
   Ipopt::Index         nele_hess,
   Ipopt::Index*        iRow,
   Ipopt::Index*        jCol,
   Ipopt::Number*       values
) {
  if (values == NULL) {
    Map<VectorXl>(iRow, nele_hess) = Hrows;
    Map<VectorXl>(jCol, nele_hess) = Hcols;
  } else {
    if (obj_factor != 0) {
      int H_idx = 0;
      for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
        int H_nnz_step = cost_hessians[phase_idx]->get_nnz();
        for (int step = phase_starts(phase_idx); step < phase_starts(phase_idx + 1); ++step) {
          cost_hessians[phase_idx]->call(&x[vars_per_step*step], &values[H_idx]);
          H_idx += H_nnz_step;
        }
      }
    } else {
      Map<VectorXd>(values, nele_hess).setZero();
    }
  }
  return true;
}

/** This method is called when the algorithm is complete so the TNLP can store/write the solution */
void ContactNLP::finalize_solution(
   Ipopt::SolverReturn               status,
   Ipopt::Index                      n,
   const Ipopt::Number*              x,
   const Ipopt::Number*              z_L,
   const Ipopt::Number*              z_U,
   Ipopt::Index                      m,
   const Ipopt::Number*              g,
   const Ipopt::Number*              lambda,
   Ipopt::Number                     obj_value,
   const Ipopt::IpoptData*           ip_data,
   Ipopt::IpoptCalculatedQuantities* ip_cq
) {
  soln = Map<const VectorXd>(x, n);
}

bool ContactNLP::intermediate_callback(
   Ipopt::AlgorithmMode              mode,
   Ipopt::Index                      iter,
   Ipopt::Number                     obj_value,
   Ipopt::Number                     inf_pr,
   Ipopt::Number                     inf_du,
   Ipopt::Number                     mu,
   Ipopt::Number                     d_norm,
   Ipopt::Number                     regularization_size,
   Ipopt::Number                     alpha_du,
   Ipopt::Number                     alpha_pr,
   Ipopt::Index                      ls_trials,
   const Ipopt::IpoptData*           ip_data,
   Ipopt::IpoptCalculatedQuantities* ip_cq
)
// Return true to continue, false to terminate early
{
  if (iter != 0) {
    timer_stop = std::chrono::high_resolution_clock::now();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(timer_stop - timer_start).count();
    iter_durations.push_back(millis);
  }
  timer_start = std::chrono::high_resolution_clock::now();

  this->num_iter = iter;
  if (success_after_iter != -1) 
  {
    if (iter >= success_after_iter) 
    {
      return false;
    }
  } 
  /**
   * Stop after finding the first feasible point. Comment the below code to solve to optimility.
   */

  /**
  else if (mode != Ipopt::AlgorithmMode::RestorationPhaseMode) 
  {
    double max_vio = ip_cq->unscaled_curr_nlp_constraint_violation(Ipopt::ENormType::NORM_MAX);
    /**
    if (max_vio < 1e-4) {
      std::cout << "Found feasible point. Max constraint violation is " << max_vio << ". Stopping" << std::endl;
      return false;
    }
  }
  */

  return true;
}
