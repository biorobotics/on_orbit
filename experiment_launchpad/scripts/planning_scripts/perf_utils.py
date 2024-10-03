import os
import subprocess
import signal
import time

def start_perf_proc(output_file):
  pid = str(os.getpid())
  devnull = open('/dev/null', 'w')
  subprocess.Popen(['echo', '0', '>', '/proc/sys/kernel/nmi_watchdog'], stdout=devnull, shell=False)
  perf_proc = subprocess.Popen(['perf', 'stat', '-o', output_file, '-e', 'r5340c7', '-e', 'r5310c7', '-e', 'r5304c7', '-e', 'r5301c7', '-p', pid], stdout=devnull, shell=False)
  time.sleep(0.1)
  return perf_proc

def stop_perf_proc(output_file, perf_proc):
  perf_pid = perf_proc.pid
  os.kill(perf_pid, signal.SIGINT)
  num_sub = 1
  time.sleep(0.01)
  while perf_proc.poll() is None:
    print('Waiting for perf to stop')
    time.sleep(0.01)
    num_sub += 1

  '''
  if perf_proc.poll() is not None:
    # print("Perf correctly halted")
    pass
    devnull = open('/dev/null', 'w')
    subprocess.Popen(['echo', '1', '>', '/proc/sys/kernel/nmi_watchdog'], stdout=devnull, shell=False)
  else:
    devnull = open('/dev/null', 'w')
    subprocess.Popen(['echo', '1', '>', '/proc/sys/kernel/nmi_watchdog'], stdout=devnull, shell=False)
    print("Error halting perf")
    return -1
  '''
  devnull = open('/dev/null', 'w')
  subprocess.Popen(['echo', '1', '>', '/proc/sys/kernel/nmi_watchdog'], stdout=devnull, shell=False)
    
  perf_file = open(output_file, 'r')
  lines = perf_file.readlines()

  count = 0
  for i, coef in zip(range(5, 9), [8, 4, 2, 1]):
    line = lines[i].lstrip()
    space_idx = line.find(' ')
    line = line[:space_idx].replace(',', '')
    ops = int(line)*coef
    count += ops
  # The time.sleep(0.01) above gets measured as a FLOP. Subtract that out
  count -= num_sub

  return count

def start_perf_memory_proc(output_file):
  pid = str(os.getpid())
  devnull = open('/dev/null', 'w')
  perf_proc = subprocess.Popen(['perf', 'stat', '-o', output_file, '-e', 'r5383d0', '-p', pid], stdout=devnull, shell=False)
  time.sleep(0.1)
  return perf_proc

def stop_perf_memory_proc(output_file, perf_proc):
  perf_pid = perf_proc.pid
  os.kill(perf_pid, signal.SIGINT)
  time.sleep(0.01)
  if perf_proc.poll() is not None:
    print("Perf correctly halted")
  else:
    print("Error halting perf")
    
  perf_file = open(output_file, 'r')
  lines = perf_file.readlines()

  count = lines[5].lstrip
  for i, coef in zip(range(5, 9), [8, 4, 2, 1]):
    line = lines[i].lstrip()
    space_idx = line.find(' ')
    line = line[:space_idx].replace(',', '')
    ops = int(line)*coef
    count += ops

  return count
