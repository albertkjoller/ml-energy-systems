import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.assignment3.bus118 import BUS118

DATA_DIR = Path('data')

# Load data files from csv-files
pmax    = pd.read_csv(DATA_DIR / 'pgmax.csv')
pmin    = pd.read_csv(DATA_DIR / 'pgmin.csv')
ru      = pd.read_csv(DATA_DIR / 'ramp.csv')
UT      = pd.read_csv(DATA_DIR / 'lu.csv')
DT      = pd.read_csv(DATA_DIR / 'ld.csv')    
demand  = pd.read_csv(DATA_DIR / 'Demand.csv', sep=';')   
c_op    = pd.read_csv(DATA_DIR / 'cost_op.csv') 
c_st    = pd.read_csv(DATA_DIR / 'cost_st.csv') 
PTDF    = pd.read_csv(DATA_DIR / 'PTDF.csv', sep=';') 
busgen  = pd.read_csv(DATA_DIR / 'busgen.csv', sep=';')
busload = pd.read_csv(DATA_DIR / 'busload.csv', sep=';')
fmax    = pd.read_csv(DATA_DIR / 'fmax.csv')

# PTDF
Hg      = pd.DataFrame(np.dot(PTDF, busgen), index=[f'Line {i+1}' for i in range(PTDF.shape[0])], columns=[f'Gen {i+1}' for i in range(busgen.shape[1])])
Hl      = pd.DataFrame(np.dot(PTDF, busload), index=[f'Line {i+1}' for i in range(PTDF.shape[0])], columns=[f'Load {i+1}' for i in range(busload.shape[1])])

# Load load profile samples
samples = pd.read_csv(DATA_DIR / 'samples.csv', header=None)

# Set global system variables
N_g     = busgen.shape[1]   # the number of generator units
N_t     = demand.shape[0]   # next 24 hours
N_load  = busload.shape[1]  # the number of load buses
N_lines = PTDF.shape[0]     # the number of transmission lines


if __name__ == '__main__':

    # SPECIFY THE SAMPLES TO RUN
    start_sample_idx    = 10
    end_sample_idx      = 12

    # Import the system class
    system = BUS118(
        N_g=N_g, N_t=N_t, N_load=N_load, N_lines=N_lines,
        demand=demand,
        pmin=pmin, pmax=pmax,
        Hg=Hg, Hl=Hl, fmax=fmax,
        UT=UT, DT=DT, ru=ru,
        c_op=c_op, c_st=c_st,
    )

    # Create an internal for loop to run and save the data
    for i in tqdm(range(start_sample_idx, end_sample_idx, 1)):
        # Select a load profile
        load_profile = samples.iloc[i, :].values[np.newaxis, :]

        # Optimize system with specified load profile
        opt = system.optimize(load_profile, M=1000)

        # Prepare data format for results
        on_off      = -1 * np.ones((system.N_g, system.N_t))
        start_up    = -1 * np.ones((system.N_g, system.N_t))

        # Extract results
        for t in range(24):
            for g in range(N_g):
                # Generator dependent results
                on_off[g, t]    = system.b[g, t].x
                start_up[g, t]  = system.u[g, t].x

        # For some reason, some zeros get a minus in front - taking the absolute value does not change the value, only the visual outline of the matrix
        on_off      = pd.DataFrame(abs(on_off),     columns=[f'Hour {i}' for i in range(1, system.N_t+1)], index=[f'Gen {i}' for i in range(1, system.N_g+1)])
        start_up    = pd.DataFrame(abs(start_up),   columns=[f'Hour {i}' for i in range(1, system.N_t+1)], index=[f'Gen {i}' for i in range(1, system.N_g+1)])

        constraints = pd.DataFrame([{
            'constr_type': c.ConstrName.split('[')[0] if '[' in c.ConstrName else c.ConstrName, 
            'constr_level': '[' + c.ConstrName.split('[')[1] if '[' in c.ConstrName else '-', 
            'constraint_slack': c.Slack, 
            'rhs': c.getAttr('rhs')
        } for c in system.model.getConstrs()])


        grouped_constraints = {}
        for name in tqdm(constraints['constr_type'].unique()):
            grouped_constraints[name] = constraints.query('constr_type == @name').reset_index(drop=True)

        line_flow_limits = grouped_constraints['pos_line_flow_limit']
        slack_line_vals_pos = line_flow_limits['constraint_slack'] - line_flow_limits['rhs']

        grouped_constraints = {}
        for name in tqdm(constraints['constr_type'].unique()):
            grouped_constraints[name] = constraints.query('constr_type == @name').reset_index(drop=True)

        line_flow_limits = grouped_constraints['neg_line_flow_limit']
        slack_line_vals_neg = line_flow_limits['constraint_slack'] - line_flow_limits['rhs']


        # Save the results
        save_res = pd.concat([on_off.T, start_up.T], axis=1)
        save_res.to_csv(DATA_DIR / f'Processed/saved_binary_var_{start_sample_idx}_{end_sample_idx}.csv', index=False, header=False, mode='a')

        pos_val = slack_line_vals_neg.values.reshape(24, 186)
        neg_val = slack_line_vals_neg.values.reshape(24, 186)

        res_aux = np.concatenate((pos_val, neg_val), axis=1)
        df_test = pd.DataFrame(res_aux)

        df_test.to_csv(DATA_DIR / f'Processed/saved_active_constraints_{start_sample_idx}_{end_sample_idx}.csv', index=False, header=False, mode='a')
        