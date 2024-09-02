# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
Enables quick testing of Pyomo problem solves, intending to set up a system
similar to how it is set up in the Pyomo dispatcher. This allows rapid testing
of different configurations for the rolling window optimization.
"""
import platform

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition


if platform.system() == 'Windows':
  SOLVER = 'glpk'
else:
  SOLVER = 'cbc'

# setup stuff
components = ['steam_source', 'elec_generator', 'steam_storage', 'elec_sink']
resources = ['steam', 'electricity']
time = np.linspace(0, 10, 11) # from @1 to @2 in @3 steps
dt = time[1] - time[0]
resource_map = {'steam_source': {'steam': 0},
                'elec_generator': {'steam': 0, 'electricity': 1},
                'steam_storage': {'steam': 0},
                'elec_sink': {'electricity': 0},
                }
activity = {}
for comp in components:
  activity[comp] = np.zeros((len(resources), len(time)), dtype=float)

# sizing specifications
storage_initial = 50 # kg of steam
storage_limit = 400 # kg of steam
steam_produced = 100 # kg/h of steam
gen_consume_limit = 110 # consumes at most 110 kg/h steam
sink_limit = 10000 # kWh/h = kW of electricity

def make_concrete_model():
  """
    Test writing a simple concrete model with terms typical to the pyomo dispatcher.
    @ In, None
    @ Out, m, pyo.ConcreteModel, instance of the model to solve
  """
  m = pyo.ConcreteModel()
  # indices
  C = np.arange(0, len(components), dtype=int) # indexes component
  R = np.arange(0, len(resources), dtype=int)  # indexes resources
  T = np.arange(0, len(time), dtype=int)       # indexes time
  # move onto model
  m.C = pyo.Set(initialize=C)
  m.R = pyo.Set(initialize=R)
  m.T = pyo.Set(initialize=T)
  # store some stuff for reference -> NOT NOTICED by Pyomo, we hope
  m.Times = time
  m.Components = components
  m.resource_index_map = resource_map
  m.Activity = activity
  #*******************
  #  set up optimization variables
  # -> for now we just do this manually
  # steam_source
  m.steam_source_index_map = pyo.Set(initialize=range(len(m.resource_index_map['steam_source'])))
  m.steam_source_production = pyo.Var(m.steam_source_index_map, m.T, initialize=0)
  # elec_generator
  m.elec_generator_index_map = pyo.Set(initialize=range(len(m.resource_index_map['elec_generator'])))
  m.elec_generator_production = pyo.Var(m.elec_generator_index_map, m.T, initialize=0)
  # steam_storage
  m.steam_storage_index_map = pyo.Set(initialize=range(len(m.resource_index_map['steam_storage'])))
  m.steam_storage_production = pyo.Var(m.steam_storage_index_map, m.T, initialize=0)
  # elec_sink
  m.elec_sink_index_map = pyo.Set(initialize=range(len(m.resource_index_map['elec_sink'])))
  m.elec_sink_production = pyo.Var(m.elec_sink_index_map, m.T, initialize=0)
  #*******************
  #  set up lower, upper bounds
  # -> for now we just do this manually
  # -> consuming is negative sign by convention!
  # -> producing is positive sign by convention!
  # steam source produces exactly 100 steam
  m.steam_source_lower_limit = pyo.Constraint(m.T, rule=lambda m, t: m.steam_source_production[0, t] >= steam_produced)
  m.steam_source_upper_limit = pyo.Constraint(m.T, rule=lambda m, t: m.steam_source_production[0, t] <= steam_produced)
  # elec generator can consume steam to produce electricity; 0 < consumed steam < 1000
  # -> this effectively limits electricity production, but we're defining capacity in steam terms for fun
  # -> therefore signs are negative, -1000 < consumed steam < 0!
  m.elec_generator_lower_limit = pyo.Constraint(m.T, rule=lambda m, t: m.elec_generator_production[0, t] >= -gen_consume_limit)
  m.elec_generator_upper_limit = pyo.Constraint(m.T, rule=lambda m, t: m.elec_generator_production[0, t] <= 0)
  # elec sink can take any amount of electricity
  # -> consuming, so -10000 < consumed elec < 0
  m.elec_sink_lower_limit = pyo.Constraint(m.T, rule=lambda m, t: m.elec_sink_production[0, t] >= -sink_limit)
  m.elec_sink_upper_limit = pyo.Constraint(m.T, rule=lambda m, t: m.elec_sink_production[0, t] <= 0)
  # storage is in LEVEL not ACTIVITY (e.g. kg not kg/s) -> lets say it can store X kg
  m.steam_storage_lower_limit = pyo.Constraint(m.T, rule=lambda m, t: m.steam_storage_production[0, t] >= 0)
  m.steam_storage_upper_limit = pyo.Constraint(m.T, rule=lambda m, t: m.steam_storage_production[0, t] <= storage_limit)
  #*******************
  # create transfer function
  # 2 steam make 1 electricity (sure, why not)
  m.elec_generator_transfer = pyo.Constraint(m.T, rule=_generator_transfer)
  #*******************
  # create conservation rules
  # steam
  m.steam_conservation = pyo.Constraint(m.T, rule=_conserve_steam)
  # electricity
  m.elec_conservation = pyo.Constraint(m.T, rule=_conserve_electricity)
  #*******************
  # create objective function
  m.OBJ = pyo.Objective(sense=pyo.maximize, rule=_economics)
  #######
  # return
  return m

#######
#
# Callback Functions
#
def _generator_transfer(m, t):
  """
    Constraint rule for electricity generation in generator
    @ In, m, pyo.ConcreteModel, model containing problem
    @ In, t, int, time indexer
    @ Out, constraint, bool, constraining evaluation
  """
  return - m.elec_generator_production[0, t] == 2.0 * m.elec_generator_production[1, t]

def _conserve_steam(m, t):
  """
    Constraint rule for conserving steam
    @ In, m, pyo.ConcreteModel, model containing problem
    @ In, t, int, time indexer
    @ Out, constraint, bool, constraining evaluation
  """
  # signs are tricky here, consumption is negative and production is positive
  # a positive delta in steam storage level means it absorbed steam, so it's a negative term
  storage_source = - (m.steam_storage_production[0, t] - (storage_initial if t == 0 else m.steam_storage_production[0, t-1])) / dt
  sources = storage_source + m.steam_source_production[0, t]
  sinks = m.elec_generator_production[0, t]
  return sources + sinks == 0

def _conserve_electricity(m, t):
  """
    Constraint rule for conserving electricity
    @ In, m, pyo.ConcreteModel, model containing problem
    @ In, t, int, time indexer
    @ Out, constraint, bool, constraining evaluation
  """
  sources = m.elec_generator_production[1, t]
  sinks = m.elec_sink_production[0, t]
  return sources + sinks == 0

def _economics(m):
  """
    Constraint rule for optimization target
    @ In, m, pyo.ConcreteModel, model containing problem
    @ Out, objective, float, constraining evaluation
  """
  opex = sum(m.elec_generator_production[0, t] for t in m.T) * 10 # will be negative b/c consumed
  sales = - sum((m.elec_sink_production[0, t] * (100 if t < 5 else 1)) for t in m.T) # net positive because consumed
  return opex + sales

#######
#
# Debug printing functions
#
def print_setup(m):
  """
    Debug printing for pre-solve model setup
    @ In, m, pyo.ConcreteModel, model containing problem
    @ Out, None
  """
  print('/' + '='*80)
  print('DEBUGG model pieces:')
  print('  -> objective:')
  print('     ', m.OBJ.pprint())
  print('  -> variables:')
  for var in m.component_objects(pyo.Var):
    print('     ', var.pprint())
  print('  -> constraints:')
  for constr in m.component_objects(pyo.Constraint):
    print('     ', constr.pprint())
  print('\\' + '='*80)
  print('')

def extract_soln(m):
  """
    Extracts final solution from model evaluation
    @ In, m, pyo.ConcreteModel, model
    @ Out, res, dict, results dictionary for dispatch
  """
  res = {}
  T = len(m.T)
  res['steam_src'] = np.zeros(T)
  res['steam_storage'] = np.zeros(T)
  res['elec_gen_s'] = np.zeros(T)
  res['elec_gen_e'] = np.zeros(T)
  res['elec_sink'] = np.zeros(T)

  res['opex'] = np.zeros(T)
  res['sales'] = np.zeros(T)
  res['objective'] = np.zeros(T)

  for t in m.T:
    res['steam_src'][t] = m.steam_source_production[0, t].value
    res['steam_storage'][t] = m.steam_storage_production[0, t].value
    res['elec_gen_s'][t] = m.elec_generator_production[0, t].value
    res['elec_gen_e'][t] = m.elec_generator_production[1, t].value
    res['elec_sink'][t] = m.elec_sink_production[0, t].value

    # Calculations copied from _economics function
    res['opex'][t] = m.elec_generator_production[0, t].value * 10
    res['sales'][t] = m.elec_sink_production[0, t].value * -1 * (100 if t < 5 else 1)
    res['objective'][t] = res['opex'][t] + res['sales'][t]

  return res

def plot_solution(m):
  """
    Plots solution from optimized model
    @ In, m, pyo.ConcreteModel, model
    @ Out, None
  """
  res = extract_soln(m)
  fig, axs = plt.subplots(3, 1, sharex=True)
  axs[0].set_ylabel(r'Steam rate (kg/h)')
  ax_0_rh = axs[0].twinx()
  ax_0_rh.set_ylabel(r'Steam quantity (kg)')
  axs[1].set_ylabel(r'Elec (kW)')
  axs[2].set_ylabel(r'Cashflow ($/h)')
  axs[2].set_xlabel('Time (h)')
  axs[0].plot(time, res['steam_src'], 'o-', label='Steam source')
  axs[0].plot(time, res['elec_gen_s'], 'o-', label='Elec generator steam production')
  ax_0_rh.plot(time, res['steam_storage'], 'o-', label='Steam storage', color='m')
  axs[0].legend(loc='upper left')
  ax_0_rh.legend(loc='lower right')
  axs[1].plot(time, res['elec_gen_e'], 'o-', label='Elec generator elec production')
  axs[1].plot(time, res['elec_sink'], 'o-', label='Elec sink')
  axs[1].legend()
  axs[2].plot(time, res['opex'], 'o-', label='Expenditures')
  axs[2].plot(time, res['sales'], 'o-', label='Sales')
  axs[2].plot(time, res['objective'], 'o-', label='Profit')
  axs[2].legend()
  plt.suptitle(f'Basic Optimization Results')
  plt.savefig(f'dispatch_basic.png')

def print_solution(m):
  """
    Debug printing for post-solve model setup
    @ In, m, pyo.ConcreteModel, model containing problem
    @ Out, None
  """
  print('')
  print('*'*80)
  print('solution:')
  print('  objective value:', m.OBJ())
  print('time | steam source | steam storage | elec gen (s, e) | elec sink')
  for t in m.T:
    print(f'{m.Times[t]:1.2e} | ' +
        f'{m.steam_source_production[0, t].value: 1.3e} | ' +
        f'{m.steam_storage_production[0, t].value: 1.3e} | ' +
        f'({m.elec_generator_production[0, t].value: 1.3e}, {m.elec_generator_production[1, t].value: 1.3e}) | ' +
        f'{m.elec_sink_production[0, t].value: 1.3e}'
        )
  print('*'*80)

def output_solution(m):
  """
    Writes post-solve model setup to two CSVs
    @ In, m, pyo.ConcreteModel, model containing problem
    @ Out, None
  """
  # Objective value output
  with open('objective_value_soln.csv', 'w', newline='') as obj_val_csv:
    obj_val_dict = {"objective value": [m.OBJ()]}
    obj_val_df = pd.DataFrame(obj_val_dict)
    obj_val_df.to_csv(obj_val_csv, index=False)

  # Dispatch variable ouput
  with open('dispatch_variables_soln.csv', 'w', newline='') as dispatch_csv:
    dispatch_dict = {'time': [], 
                     'steam source': [], 
                     'steam storage': [], 
                     'elec gen (s)': [], 
                     'elec gen (e)': [], 
                     'elec sink': []
                     }
    
    for t in m.T:
      dispatch_dict['time'].append(f'{m.Times[t]:1.2e}')
      dispatch_dict['steam source'].append(f'{m.steam_source_production[0, t].value: 1.3e}')
      dispatch_dict['steam storage'].append(f'{m.steam_storage_production[0, t].value: 1.3e}')
      dispatch_dict['elec gen (s)'].append(f'{m.elec_generator_production[0, t].value: 1.3e}')
      dispatch_dict['elec gen (e)'].append(f'{m.elec_generator_production[1, t].value: 1.3e}')
      dispatch_dict['elec sink'].append(f'{m.elec_sink_production[0, t].value: 1.3e}')

    dispatch_df = pd.DataFrame(dispatch_dict)
    dispatch_df.to_csv(dispatch_csv, index=False)
      

#######
#
# Solver.
#
def solve_model(m):
  """
    Solves the model.
    @ In, m, pyo.ConcreteModel, model containing problem
    @ Out, m, pyo.ConcreteModel, results
  """
  soln = pyo.SolverFactory(SOLVER).solve(m)
  return soln

if __name__ == '__main__':
  m = make_concrete_model()
  # print_setup(m)
  s = solve_model(m)
  # print_solution(m)
  output_solution(m)
  # plot_solution(m)
  # plt.show()
