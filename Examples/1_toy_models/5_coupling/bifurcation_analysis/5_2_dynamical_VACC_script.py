import gifba
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

array_task_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
array_total_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
bifurc_points_total = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

# chaos script details:
a0_global_min = 0.01        # minimum value for rel_abund[0]
a0_global_max = 0.99        # maximum value for rel_abund[0]
start_pts = np.linspace(a0_global_min, a0_global_max, array_total_runs+1) # 10 evenly spaced starting points for a0, excluding 1.0
a0_min_task = start_pts[array_task_idx-1]
a0_max_task = start_pts[array_task_idx]
n_params = bifurc_points_total // array_total_runs

media_min = 9               # media minimum value for Ex_A
media_max = 12              # media maximum value for Ex_A
# media_step_size = 1
n_conditions = 6            # how many media conditions to run

update_interval = 1
iters = 10000
lag = 500
csv_log_path = f"/logs/chaos_log{array_task_idx}.tsv"

with open(csv_log_path, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["rel_abund", "init_media", "period", "iter", "bio1_fluxes", "bio2_fluxes", "b_fluxes"])

def save_tsv_row(path, rel_abund, init_media, period, iter_, bio1_fluxes, bio2_fluxes, b_fluxes):
    rel_abund = np.asarray(rel_abund, float).reshape(2)
    bio1_fluxes = np.asarray(bio1_fluxes, float).reshape(lag)
    bio2_fluxes = np.asarray(bio2_fluxes, float).reshape(lag)
    b_fluxes = np.asarray(b_fluxes, float).reshape(lag)

    fmt = lambda a: ",".join(f"{x:.12f}" for x in a)


    with open(path, "a") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            fmt(rel_abund),
            init_media,
            "" if period is None else int(period),
            "" if iter_ is None else int(iter_),
            fmt(bio1_fluxes),
            fmt(bio2_fluxes),
            fmt(b_fluxes)
        ])




# rel_abunds to test
a0_vals = np.linspace(a0_min_task, a0_max_task, n_params+1)[:-1] # n_params evenly spaced values for a0, excluding the max (which is 1.0)
periods = np.zeros(n_params)
media_vals = - np.linspace(media_min, media_max, n_conditions)
print(media_vals)
# media_vals = -np.arange(media_min, media_max+1, media_step_size).astype(float) # -9, -10, -11, -12

# fig, ax = plt.subplots(figsize=(15,12))
for idx, a0 in enumerate(a0_vals):
    if idx % update_interval == 0:
        print(f"Running parameter set {idx+1}/{n_params} with a0={a0:.4f}...")
    for media_val in media_vals:
        if idx == 0:
            continue # skip the first one, no 0 abund and final final is duplicate
        if a0 >= 1:
            continue
        rel_abund = [a0, 1-a0]
        step_size=1

        # load models and media:
        sim = "5_2_dynamical"
        models, media = gifba.utils.load_simple_models(sim)
        # print(media_val)
        media["EX_A(e)"] = media_val

        # initialize community
        community = gifba.gifbaObject(models, media, rel_abund=rel_abund)

        # run iterations
        media_flux, org_flux = community.run_gifba(iters=iters, method="pfba")

        save_tsv_row(csv_log_path, 
                    community.rel_abund.flatten(), 
                    media_val,
                    community.periodicity, 
                    community.iter_converged, 
                    community.org_fluxes.loc[0]["EX_Bio(e)"].tail(lag).to_numpy().flatten(),
                    community.org_fluxes.loc[1]["EX_Bio(e)"].tail(lag).to_numpy().flatten(),
                    community.env_fluxes["EX_B(e)"].tail(lag).to_numpy().flatten())

    

