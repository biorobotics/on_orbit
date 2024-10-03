#include "ipoptContactCodeGenCaller.h"

IPOPTContactCodeGenCaller::IPOPTContactCodeGenCaller(int nx, int nu, VectorXiRef_const phase_starts, VectorXiRef_const constraint_sizes, int plane_idx) : nx(nx), nu(nu), phase_starts(phase_starts), constraint_sizes(constraint_sizes) {
  vars_per_step = nx + nu;

  // Load the functions
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    std::string phase_str = std::to_string(phase_idx);

    cost_functions.push_back(std::make_shared<CasadiExternalFunction>("cost_phase_" + phase_str, plane_idx));
    cost_gradients.push_back(std::make_shared<CasadiExternalFunction>("cost_grad_phase_" + phase_str, plane_idx));
    cost_hessians.push_back(std::make_shared<CasadiExternalFunction>("cost_hess_phase_" + phase_str, plane_idx));
    constraint_functions.push_back(std::make_shared<CasadiExternalFunction>("constraint_phase_" + phase_str, plane_idx));
    constraint_jacobians.push_back(std::make_shared<CasadiExternalFunction>("constraint_jacobian_phase_" + phase_str, plane_idx));
  }

  // Get how many nonzeros are in the jacobians and hessian.
  // Don't worry about the cost, cost gradient, or constraint functions. We assume these were densified before code generation
  int J_nnz = 0;
  int H_nnz = 0;
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    int phase_length = phase_starts(phase_idx + 1) - phase_starts(phase_idx);
    J_nnz += constraint_jacobians[phase_idx]->get_nnz()*phase_length;
    H_nnz += cost_hessians[phase_idx]->get_nnz()*phase_length;
  }

  // Populate sparsity structure for jacobians and hessians
  Jrows = VectorXi::Zero(J_nnz);
  Jcols = VectorXi::Zero(J_nnz);
  Hrows = VectorXi::Zero(H_nnz);
  Hcols = VectorXi::Zero(H_nnz);
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

  // Allocate vectors we'll use to return NLP function values
  grad = VectorXd::Zero(vars_per_step*(phase_starts(phase_starts.size() - 1)));
  Jvals = VectorXd::Zero(J_nnz);
  Hvals = VectorXd::Zero(H_nnz);
  int num_constraints = 0;
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    int phase_length = phase_starts(phase_idx + 1) - phase_starts(phase_idx);
    num_constraints += constraint_sizes(phase_idx)*phase_length;
  }
  c = VectorXd::Zero(num_constraints);
}

double IPOPTContactCodeGenCaller::get_objective(VectorXd iterate) {
  double objective = 0;
  double step_objective = 0;
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    for (int step = phase_starts(phase_idx); step < phase_starts(phase_idx + 1); ++step) {
      cost_functions[phase_idx]->call(&iterate(vars_per_step*step), &step_objective);
      objective += step_objective;
    }
  }
  return objective;
}

VectorXd IPOPTContactCodeGenCaller::get_grad(VectorXd iterate) {
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    for (int step = phase_starts(phase_idx); step < phase_starts(phase_idx + 1); ++step) {
      cost_gradients[phase_idx]->call(&iterate(vars_per_step*step), &grad(vars_per_step*step));
    }
  }
  return grad;
}

VectorXd IPOPTContactCodeGenCaller::get_c(VectorXd iterate) {
  int c_idx = 0;
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    for (int step = phase_starts(phase_idx); step < phase_starts(phase_idx + 1); ++step) {
      constraint_functions[phase_idx]->call(&iterate(vars_per_step*step), &c(c_idx));
      c_idx += constraint_sizes(phase_idx);
    }
  }
  return c;
}

VectorXd IPOPTContactCodeGenCaller::get_J(VectorXd iterate) {
  int J_idx = 0;
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    int J_nnz_step = constraint_jacobians[phase_idx]->get_nnz();
    for (int step = phase_starts(phase_idx); step < phase_starts(phase_idx + 1); ++step) {
      constraint_jacobians[phase_idx]->call(&iterate(vars_per_step*step), &Jvals(J_idx));
      J_idx += J_nnz_step;
    }
  }
  return Jvals;
}

VectorXd IPOPTContactCodeGenCaller::get_H(VectorXd iterate) {
  int H_idx = 0;
  for (int phase_idx = 0; phase_idx < phase_starts.size() - 1; ++phase_idx) {
    int H_nnz_step = cost_hessians[phase_idx]->get_nnz();
    for (int step = phase_starts(phase_idx); step < phase_starts(phase_idx + 1); ++step) {
      cost_hessians[phase_idx]->call(&iterate(vars_per_step*step), &Hvals(H_idx));
      H_idx += H_nnz_step;
    }
  }
  return Hvals;
}
