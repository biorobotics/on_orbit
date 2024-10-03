#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>
#include "ipoptContactCodeGenCaller.h"
#include "ipoptContactSolver.h"

namespace py = pybind11;

PYBIND11_MODULE(code_gen_caller_bindings, m) {
  py::class_<IPOPTContactCodeGenCaller>(m, "IPOPTContactCodeGenCaller")
      .def(py::init<int, int, VectorXiRef_const, VectorXiRef_const, int>())
      .def("get_objective", &IPOPTContactCodeGenCaller::get_objective)
      .def("get_grad", &IPOPTContactCodeGenCaller::get_grad)
      .def("get_c", &IPOPTContactCodeGenCaller::get_c)
      .def("get_J", &IPOPTContactCodeGenCaller::get_J)
      .def("get_H", &IPOPTContactCodeGenCaller::get_H)
      .def("get_Jrows", &IPOPTContactCodeGenCaller::get_Jrows)
      .def("get_Jcols", &IPOPTContactCodeGenCaller::get_Jcols)
      .def("get_Hrows", &IPOPTContactCodeGenCaller::get_Hrows)
      .def("get_Hcols", &IPOPTContactCodeGenCaller::get_Hcols)
      ;

  py::class_<IPOPTContactSolver>(m, "IPOPTContactSolver")
      .def(py::init<int, int, VectorXlRef_const, VectorXlRef_const, VectorXdRef_const, VectorXdRef_const, VectorXdRef_const, VectorXdRef_const, VectorXdRef_const, int, int, int>())
      .def("solve", &IPOPTContactSolver::solve)
      .def("get_num_iter", &IPOPTContactSolver::get_num_iter)
      .def("get_iter_durations", &IPOPTContactSolver::get_iter_durations)
      ;
}
