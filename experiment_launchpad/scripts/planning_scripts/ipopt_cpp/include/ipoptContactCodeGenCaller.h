#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <dlfcn.h>
#include <memory>

using namespace Eigen;
namespace ca = casadi;

typedef const Ref<const Vector3d>& Vector3dRef_const;
typedef const Ref<const Vector4d>& Vector4dRef_const;
typedef const Ref<const Matrix3d>& Matrix3dRef_const;

typedef const Ref<const VectorXd>& VectorXdRef_const;
typedef const Ref<const VectorXi>& VectorXiRef_const;
typedef const Ref<const Matrix<double, Dynamic, Dynamic, RowMajor>>& MatrixXdRef_const_rowmajor;

typedef Ref<VectorXd> VectorXdRef;
typedef Ref<VectorXi> VectorXiRef;
typedef Ref<Matrix<double, Dynamic, Dynamic, RowMajor>> MatrixXdRef_rowmajor;

typedef long long int casadi_int;
typedef void (*signal_t)(void);
typedef casadi_int (*getint_t)(void);
typedef int (*work_t)(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
typedef const casadi_int* (*sparsity_t)(casadi_int ind);
typedef int (*eval_t)(const double** arg, double** res, casadi_int* iw, double* w, int mem);
typedef int (*casadi_checkout_t)(void);
typedef void (*casadi_release_t)(int);

class CasadiExternalFunction {
  public:
    CasadiExternalFunction(const std::string &function_name, int plane_idx) {
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

      /* Number of inputs */
      n_in_fcn = (getint_t)dlsym(handle, (function_name + "_n_in").c_str());
      if (dlerror()) exit(1);
      casadi_int n_in = n_in_fcn();

      /* Number of outputs */
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
      printf("Work vector sizes:\n");
      printf("sz_arg = %lld, sz_res = %lld, sz_iw = %lld, sz_w = %lld\n\n",
             sz_arg, sz_res, sz_iw, sz_w);

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
          printf("Input %lld\n", i);
          sp_i = sparsity_in(i);
        } else {
          printf("Output %lld\n", i-n_in);
          sp_i = sparsity_out(i-n_in);
        }
        if (sp_i==0) exit(1);
        nrow = *sp_i++; /* Number of rows */
        ncol = *sp_i++; /* Number of columns */
        colind = sp_i; /* Column offsets */
        row = sp_i + ncol+1; /* Row nonzero */
        nnz = sp_i[ncol]; /* Number of nonzeros */

        /* Print the pattern */
        printf("  Dimension: %lld-by-%lld (%lld nonzeros)\n", nrow, ncol, nnz);
        printf("  Nonzeros: {");
        casadi_int rr,cc,el;
        for(cc=0; cc<ncol; ++cc){                    /* loop over columns */
          for(el=colind[cc]; el<colind[cc+1]; ++el){ /* loop over the nonzeros entries of the column */
            if(el!=0) printf(", ");                  /* Separate the entries */
            rr = row[el];                            /* Get the row */
            printf("{%lld,%lld}",rr,cc);                 /* Print the nonzero */
          }
        }
        printf("}\n\n");
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

    void get_sparsity_structure(VectorXiRef rows, VectorXiRef cols, int start_row, int start_col) {
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

    void call(double *input, double *output) {
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

    ~CasadiExternalFunction() {
      /* Free the handle */
      dlclose(handle);
    }

    CasadiExternalFunction (const CasadiExternalFunction &) = delete;
    CasadiExternalFunction & operator = (const CasadiExternalFunction &) = delete;

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

class IPOPTContactCodeGenCaller {
  public:
    IPOPTContactCodeGenCaller(int nx, int nu, VectorXiRef_const phase_starts, VectorXiRef_const constraint_sizes, int plane_idx);

    double get_objective(VectorXd iterate);

    VectorXd get_grad(VectorXd iterate);

    VectorXd get_c(VectorXd iterate);

    VectorXd get_J(VectorXd iterate);

    VectorXd get_H(VectorXd iterate);

    VectorXi get_Jrows() {
      return Jrows;
    }
    VectorXi get_Jcols() {
      return Jcols;
    }
    VectorXi get_Hrows() {
      return Hrows;
    }
    VectorXi get_Hcols() {
      return Hcols;
    }
  private:
    int nx;
    int nu;
    int vars_per_step;
    VectorXi phase_starts;
    VectorXi constraint_sizes;
    // One index per step, including the terminal step
    std::vector<std::shared_ptr<CasadiExternalFunction>> cost_functions;
    std::vector<std::shared_ptr<CasadiExternalFunction>> cost_gradients;
    std::vector<std::shared_ptr<CasadiExternalFunction>> cost_hessians;
    std::vector<std::shared_ptr<CasadiExternalFunction>> constraint_functions;
    std::vector<std::shared_ptr<CasadiExternalFunction>> constraint_jacobians;
    std::vector<std::shared_ptr<CasadiExternalFunction>> dynamics_functions;
    std::vector<std::shared_ptr<CasadiExternalFunction>> dynamics_jacobians;

    VectorXi Jrows;
    VectorXi Jcols;

    VectorXi Hrows;
    VectorXi Hcols;

    VectorXd grad;
    VectorXd c;
    VectorXd Jvals;
    VectorXd Hvals;
};
