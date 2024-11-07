#pragma once

#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <dlfcn.h>
#include <memory>

#define HAVE_CSTDDEF
#include <IpTNLP.hpp>
#undef HAVE_CSTDDEF
#define HAVE_STDDEF
#include <IpTNLP.hpp>
#undef HAVE_STDDEF
#include "IpIpoptApplication.hpp"
#include "IpIpoptCalculatedQuantities.hpp"
// #include "IpTNLP.hpp"

#include <chrono>

using namespace std::chrono;

using namespace Eigen;
namespace ca = casadi;

typedef Matrix<Ipopt::Index, Dynamic, 1> VectorXl;
typedef const Ref<const Vector3d>& Vector3dRef_const;
typedef const Ref<const Vector4d>& Vector4dRef_const;
typedef const Ref<const Matrix3d>& Matrix3dRef_const;

typedef const Ref<const VectorXd>& VectorXdRef_const;
typedef const Ref<const VectorXl>& VectorXlRef_const;
typedef const Ref<const Matrix<double, Dynamic, Dynamic, RowMajor>>& MatrixXdRef_const_rowmajor;

typedef Ref<VectorXd> VectorXdRef;
typedef Ref<VectorXl> VectorXlRef;
typedef Ref<Matrix<double, Dynamic, Dynamic, RowMajor>> MatrixXdRef_rowmajor;

typedef long long int casadi_int;
typedef void (*signal_t)(void);
typedef casadi_int (*getint_t)(void);
typedef int (*work_t)(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
typedef const casadi_int* (*sparsity_t)(casadi_int ind);
typedef int (*eval_t)(const double** arg, double** res, casadi_int* iw, double* w, int mem);
typedef int (*casadi_checkout_t)(void);
typedef void (*casadi_release_t)(int);

class CasadiExternalFunction2 {
  public:
    CasadiExternalFunction2(const std::string &function_name, int plane_idx) {
      std::cout << "Opening " << function_name << std::endl;
      /* Load the dll */
      handle = dlopen(("codegen_for_ipopt/plane_idx" + std::to_string(plane_idx) + "/" + function_name + ".so").c_str(), RTLD_LAZY);
      if(handle==0){
        std::cout << "Cannot open " << function_name << ".so, error " << dlerror() << std::endl;
        exit(1);
      }

      /* Reset error */
      dlerror();

      /* Memory management -- increase reference counter */
      incref = (signal_t)dlsym(handle, (function_name + "_incref").c_str());
      if(dlerror()) dlerror(); // No such function, reset error flags

      /* Memory management -- decrease reference counter */
      decref = (signal_t)dlsym(handle, (function_name + "_decref").c_str());
      if(dlerror()) dlerror(); // No such function, reset error flags

      /* Thread-local memory management -- checkout memory */
      checkout = (casadi_checkout_t)dlsym(handle, (function_name + "_checkout").c_str());
      if(dlerror()) dlerror(); // No such function, reset error flags

      /* Thread-local memory management -- release memory */
      release = (casadi_release_t)dlsym(handle, (function_name + "_release").c_str());
      if(dlerror()) dlerror(); // No such function, reset error flags

      /* Ipopt::Number of inputs */
      n_in_fcn = (getint_t)dlsym(handle, (function_name + "_n_in").c_str());
      if (dlerror()) exit(1);
      casadi_int n_in = n_in_fcn();

      /* Ipopt::Number of outputs */
      n_out_fcn = (getint_t)dlsym(handle, (function_name + "_n_out").c_str());
      if (dlerror()) exit(1);
      casadi_int n_out = n_out_fcn();

      /* Get sizes of the required work vectors */
      sz_arg=n_in;
      sz_res=n_out;
      sz_iw=0;
      sz_w=0;
      work = (work_t)dlsym(handle, (function_name + "_work").c_str());
      if(dlerror()) dlerror(); // No such function, reset error flags
      if (work && work(&sz_arg, &sz_res, &sz_iw, &sz_w)) exit(1);
      // printf("Work vector sizes:\n");
      // printf("sz_arg = %lld, sz_res = %lld, sz_iw = %lld, sz_w = %lld\n\n",
      //        sz_arg, sz_res, sz_iw, sz_w);

      /* Input sparsities */
      sparsity_in = (sparsity_t)dlsym(handle, (function_name + "_sparsity_in").c_str());
      if (dlerror()) exit(1);

      /* Output sparsities */
      sparsity_out = (sparsity_t)dlsym(handle, (function_name + "_sparsity_out").c_str());
      if (dlerror()) exit(1);

      /* Print the sparsities of the inputs and outputs */
      casadi_int i;
      for(i=0; i<n_in + n_out; ++i){
        // Retrieve the sparsity pattern - CasADi uses column compressed storage (CCS)
        const casadi_int *sp_i;
        if (i<n_in) {
          // printf("Input %lld\n", i);
          sp_i = sparsity_in(i);
        } else {
          // printf("Output %lld\n", i-n_in);
          sp_i = sparsity_out(i-n_in);
        }
        if (sp_i==0) exit(1);
        nrow = *sp_i++; /* Ipopt::Number of rows */
        ncol = *sp_i++; /* Ipopt::Number of columns */
        colind = sp_i; /* Column offsets */
        row = sp_i + ncol+1; /* Row nonzero */
        nnz = sp_i[ncol]; /* Ipopt::Number of nonzeros */

        /* Print the pattern */
        // printf("  Dimension: %lld-by-%lld (%lld nonzeros)\n", nrow, ncol, nnz);
        // printf("  Nonzeros: {");
        // casadi_int rr,cc,el;
        // for(cc=0; cc<ncol; ++cc){                    /* loop over columns */
        //   for(el=colind[cc]; el<colind[cc+1]; ++el){ /* loop over the nonzeros entries of the column */
        //     if(el!=0) printf(", ");                  /* Separate the entries */
        //     rr = row[el];                            /* Get the row */
        //     printf("{%lld,%lld}",rr,cc);                 /* Print the nonzero */
        //   }
        // }
        // printf("}\n\n");
      }

      /* Function for numerical evaluation */
      eval = (eval_t)dlsym(handle, function_name.c_str());
      if(dlerror()){
        std::cout << "Failed to retrieve " << function_name << " function" << std::endl;
        exit(1);
      }

      iw.resize(sz_iw);
      w.resize(sz_w);
    }

    void get_sparsity_structure(VectorXlRef rows, VectorXlRef cols, int start_row, int start_col) {
      casadi_int rr,cc,el;
      for(cc=0; cc<ncol; ++cc){                    /* loop over columns */
        for(el=colind[cc]; el<colind[cc+1]; ++el){ /* loop over the nonzeros entries of the column */
          rr = row[el];                            /* Get the row */
          rows(el) = start_row + rr;
          cols(el) = start_col + cc;
        }
      }
    }

    int get_nnz() {
      return nnz;
    }

    void call(const double *input, double *output) {
      /* Allocate input/output buffers and work vectors*/
      const double *arg[sz_arg];
      double *res[sz_res];
      casadi_int iw[sz_iw];
      double w[sz_w];

      // Allocate memory (thread-safe)
      incref();

      /* Evaluate the function */
      arg[0] = input;
      res[0] = output;

      // Checkout thread-local memory (not thread-safe)
      // Note MAX_NUM_THREADS
      int mem = checkout();

      // Evaluation is thread-safe
      if (eval(arg, res, iw, w, mem)) exit(1);

      // Release thread-local (not thread-safe)
      release(mem);

      /* Free memory (thread-safe) */
      decref();
    }

    ~CasadiExternalFunction2() {
      /* Free the handle */
      dlclose(handle);
    }

    CasadiExternalFunction2 (const CasadiExternalFunction2 &) = delete;
    CasadiExternalFunction2 & operator = (const CasadiExternalFunction2 &) = delete;

  private:
    void* handle;
    signal_t incref;
    signal_t decref;
    casadi_checkout_t checkout;
    casadi_release_t release;
    getint_t n_in_fcn;
    getint_t n_out_fcn;
    casadi_int sz_arg;
    casadi_int sz_res;
    casadi_int sz_iw;
    casadi_int sz_w;
    work_t work;
    sparsity_t sparsity_in;
    sparsity_t sparsity_out;
    casadi_int nrow;
    casadi_int ncol;
    const casadi_int *colind;
    const casadi_int *row;
    casadi_int nnz;
    eval_t eval;

    std::vector<casadi_int> iw;
    std::vector<double> w;
};

class ContactNLP : public Ipopt::TNLP {
  public:
    /** Constructor */
    ContactNLP(int nx, int nu, VectorXlRef_const phase_starts, VectorXlRef_const constraint_sizes, VectorXdRef_const lb, VectorXdRef_const ub, VectorXdRef_const cl, VectorXdRef_const cu, VectorXdRef_const warm_start, int success_after_iter, int plane_idx);

    /**@name Overloaded from TNLP */
    //@{
    /** Method to return some info about the NLP */
    virtual bool get_nlp_info(
       Ipopt::Index&          n,
       Ipopt::Index&          m,
       Ipopt::Index&          nnz_jac_g,
       Ipopt::Index&          nnz_h_lag,
       Ipopt::TNLP::IndexStyleEnum& index_style
    );

    /** Method to return the bounds for my problem */
    virtual bool get_bounds_info(
       Ipopt::Index   n,
       Ipopt::Number* x_l,
       Ipopt::Number* x_u,
       Ipopt::Index   m,
       Ipopt::Number* g_l,
       Ipopt::Number* g_u
    );
    /** Method to return the starting point for the algorithm */
    virtual bool get_starting_point(
       Ipopt::Index   n,
       bool    init_x,
       Ipopt::Number* x,
       bool    init_z,
       Ipopt::Number* z_L,
       Ipopt::Number* z_U,
       Ipopt::Index   m,
       bool    init_lambda,
       Ipopt::Number* lambda
    );

    /** Method to return the objective value */
    virtual bool eval_f(
       Ipopt::Index         n,
       const Ipopt::Number* x,
       bool          new_x,
       Ipopt::Number&       obj_value
    );

    /** Method to return the gradient of the objective */
    virtual bool eval_grad_f(
       Ipopt::Index         n,
       const Ipopt::Number* x,
       bool          new_x,
       Ipopt::Number*       grad_f
    );

    /** Method to return the constraint residuals */
    virtual bool eval_g(
       Ipopt::Index         n,
       const Ipopt::Number* x,
       bool          new_x,
       Ipopt::Index         m,
       Ipopt::Number*       g
    );

    /** Method to return:
     *   1) The structure of the jacobian (if "values" is NULL)
     *   2) The values of the jacobian (if "values" is not NULL)
     */
    virtual bool eval_jac_g(
       Ipopt::Index         n,
       const Ipopt::Number* x,
       bool          new_x,
       Ipopt::Index         m,
       Ipopt::Index         nele_jac,
       Ipopt::Index*        iRow,
       Ipopt::Index*        jCol,
       Ipopt::Number*       values
    );

    /** Method to return:
     *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
     *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
     */
    virtual bool eval_h(
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
    );

    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    virtual void finalize_solution(
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
    );
    //@}

    bool intermediate_callback(
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
    );

    void get_soln(VectorXdRef soln) {
      soln = this->soln;
    }

    int get_num_iter() {
      return num_iter;
    }

    double get_obj_value() const {
      return obj_value;
    }

    const std::vector<double>& get_iter_durations() {
      return iter_durations;
    }

  private:
    int nx;
    int nu;
    int vars_per_step;
    int num_decision_vars;
    int J_nnz;
    int H_nnz;
    int num_constraints;
    double obj_value;
    VectorXl phase_starts;
    VectorXl constraint_sizes;
    VectorXd lb;
    VectorXd ub;
    VectorXd cl;
    VectorXd cu;
    VectorXd warm_start;
    // One index per step, including the terminal step
    std::vector<std::shared_ptr<CasadiExternalFunction2>> cost_functions;
    std::vector<std::shared_ptr<CasadiExternalFunction2>> cost_gradients;
    std::vector<std::shared_ptr<CasadiExternalFunction2>> cost_hessians;
    std::vector<std::shared_ptr<CasadiExternalFunction2>> constraint_functions;
    std::vector<std::shared_ptr<CasadiExternalFunction2>> constraint_jacobians;
    std::vector<std::shared_ptr<CasadiExternalFunction2>> dynamics_functions;
    std::vector<std::shared_ptr<CasadiExternalFunction2>> dynamics_jacobians;

    VectorXl Jrows;
    VectorXl Jcols;

    VectorXl Hrows;
    VectorXl Hcols;

    VectorXd soln;

    VectorXd c;
    
    int success_after_iter;
    int num_iter;
    std::vector<double> iter_durations;
    std::chrono::time_point<std::chrono::high_resolution_clock> timer_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> timer_stop;
};

class IPOPTContactSolver {
  public:
    IPOPTContactSolver(int nx, int nu, VectorXlRef_const phase_starts, VectorXlRef_const constraint_sizes, VectorXdRef_const lb, VectorXdRef_const ub, VectorXdRef_const cl, VectorXdRef_const cu, VectorXdRef_const warm_start, int success_after_iter, int plane_idx, int max_iter);

    bool solve(VectorXdRef soln);
    int get_num_iter() {
      return num_iter;
    }
    VectorXd get_iter_durations() {
      return Map<const VectorXd>(nlp->get_iter_durations().data(), nlp->get_iter_durations().size());
    }

    double get_obj_value() const {
      return nlp->get_obj_value();
    }
  private:
    Ipopt::SmartPtr<ContactNLP> nlp;
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app;
    int num_iter;
};
