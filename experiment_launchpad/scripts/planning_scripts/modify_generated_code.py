def modify_generated_code(num_phases):
  print('Modifying generated code')
  c_file = open('c_generated_code/acados_solver_multiphase_ocp.c', 'r')
  lines = c_file.readlines()

  modified_line_idx = []
  modified_lines = []

  for phase_idx in range(num_phases):
    start_idx = None
    end_idx = None
    for line_number, line in enumerate(lines):
      if start_idx is None and 'Constraints phase ' + str(phase_idx) in line:
        start_idx = line_number
      if phase_idx == num_phases - 1 and 'TERMINAL node' in line:
        end_idx = line_number
        break
      elif 'Constraints phase ' + str(phase_idx + 1) in line:
        end_idx = line_number
        break

    # Look for the constraint function and Jacobian lines
    h_jac_idx = None
    h_idx = None
    for line_number in range(start_idx, end_idx):
      if 'capsule->nl_constr_h_fun_jac' in lines[line_number]:
        h_jac_idx = line_number
      if 'capsule->nl_constr_h_fun' in lines[line_number]:
        h_idx = line_number

    if h_jac_idx is None and h_idx is None:
      continue
    elif h_jac_idx is None:
      print('Error: we have a jacobian without a function')
    elif h_idx is None:
      print('Error: we have a function without a jacobian')

    h_jac_line = lines[h_jac_idx]
    append_idx = h_jac_line.find('jac') + 3
    h_jac_line_mod = h_jac_line[:append_idx] + '_' + str(phase_idx) + '[i_fun]);'

    h_line = lines[h_idx]
    append_idx = h_line.find('fun') + 3
    h_line_mod = h_line[:append_idx] + '_' + str(phase_idx) + '[i_fun]);'

    modified_line_idx.append(h_jac_idx)
    modified_line_idx.append(h_idx)

    modified_lines.append(h_jac_line_mod)
    modified_lines.append(h_line_mod)

  for idx, line_mod in zip(modified_line_idx, modified_lines):
    lines[idx] = line_mod

  c_file = open("c_generated_code/acados_solver_multiphase_ocp.c", "w")
  c_file.writelines(lines)
  c_file.close()
  print('Modified generated code')
