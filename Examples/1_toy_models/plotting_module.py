import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_micom_tradeoff_pts(micom_results, iterpretation, tradeoffs, rel_abund, color, ms, lw, offsets):
    # interpretation is either "specific" or "relative" to indicate which MICOM results to plot
    # offset is a tuple of (x_offset, y_offset) for each text labels
    x = micom_results["Org1_growth"].copy()
    y = micom_results["Org2_growth"].copy()
    if iterpretation == "specific":
        x /= rel_abund[0]
        y /= rel_abund[1]

    plt.plot(x, y,
            color=color, 
            lw=lw)

    for idx, alpha_tradeoff in enumerate(tradeoffs):
        posx = micom_results[micom_results["tradeoff"] == alpha_tradeoff].reset_index()["Org1_growth"].values
        posy = micom_results[micom_results["tradeoff"] == alpha_tradeoff].reset_index()["Org2_growth"].values
  
        if iterpretation == "specific":
                posx /= rel_abund[0]
                posy /= rel_abund[1]

        # old: label = str(alpha_tradeoff) if alpha_tradeoff != 1.0 else r"$\alpha$=1.0"
        label = rf"$\alpha$={alpha_tradeoff}"

        plt.text(posx+offsets[idx][0], posy+offsets[idx][1], 
                label,
                fontsize=12, 
                zorder=3, 
                color=color)
        
        plt.scatter(posx, posy, 
                    s=ms, 
                    c=color, 
                    marker="d", 
                    zorder=3)