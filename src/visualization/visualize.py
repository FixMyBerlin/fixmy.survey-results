import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def likert_plot(df, exclude=['duration', 'rating', 'sessionID'], filename_prefix=""):
    for var in df.columns:
        if var in exclude:
            continue
        else:
            unique = df[var].unique()
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,5))
            # From raw value to percentage
            totals = df[var].value_counts(dropna=False).sort_index()
            unsafe_bars = (df[df["rating"] == 0][var].value_counts(dropna=False)/totals * 100).sort_index()
            almost_unsafe_bars = (df[df["rating"] == 1][var].value_counts(dropna=False)/totals * 100).sort_index()
            almost_safe_bars = (df[df["rating"] == 2][var].value_counts(dropna=False)/totals * 100).sort_index()
            safe_bars = (df[df["rating"] == 3][var].value_counts(dropna=False)/totals * 100).sort_index()
            # plot
            barWidth = 0.99
            names = totals.index
            r = list(range(unique.shape[0]))
            # Create grey Bars
            buttom = totals/totals * 100
            ax1.barh(r, buttom, color='#33cc33', edgecolor='white', height=barWidth)
            # Create blue Bars
            buttom = buttom - safe_bars
            ax1.barh(r, buttom, color='#336600', edgecolor='white', height=barWidth)
            # Create orange Bars
            buttom = buttom - almost_safe_bars
            ax1.barh(r, buttom, color='#cc0000', edgecolor="white", height=barWidth)
            # Create green Bars
            buttom = buttom - almost_unsafe_bars
            ax1.barh(r, buttom, color='#ff3300', edgecolor='white', height=barWidth)
            # Custom x axis
            plt.yticks(r, unsafe_bars.index)

            ax2.barh(r, totals)
            # ax2.tick_params(
            #     axis='x',          # changes apply to the x-axis
            #     which='both',      # both major and minor ticks are affected
            #     bottom=False,      # ticks along the bottom edge are off
            #     top=False,         # ticks along the top edge are off
            #     labelbottom=False) # labels along the bottom edge are off
            plt.title(var)
            plt.tight_layout()
            #plt.savefig("plots/likert/" + filename_prefix + "_" + var + ".png")
            plt.show()

def likert_plot_hypo(group1, group2, name):
    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,5))
    group1 = group1.dropna()
    group2 = group2.dropna()
    totals = np.array([group1.shape[0], group2.shape[0]])
    # From raw value to percentage
    unsafe_bars = np.array([group1[group1 == 0].shape[0], group2[group2 == 0].shape[0]])/totals * 100
    almost_unsafe_bars = np.array([group1[group1 == 1].shape[0], group2[group2 == 1].shape[0]])/totals * 100
    almost_safe_bars = np.array([group1[group1 == 2].shape[0], group2[group2 == 2].shape[0]])/totals * 100
    safe_bars = np.array([group1[group1 == 3].shape[0], group2[group2 == 3].shape[0]])/totals * 100
    # plot
    barWidth = 0.99
    r = [0,1]
    # Create grey Bars
    buttom = totals/totals * 100
    ax1.barh(r, buttom, color='#33cc33', edgecolor='white', height=barWidth)
    # Create blue Bars
    buttom = buttom - safe_bars
    ax1.barh(r, buttom, color='#336600', edgecolor='white', height=barWidth)
    # Create orange Bars
    buttom = buttom - almost_safe_bars
    ax1.barh(r, buttom, color='#cc0000', edgecolor="white", height=barWidth)
    # Create green Bars
    buttom = buttom - almost_unsafe_bars
    ax1.barh(r, buttom, color='#ff3300', edgecolor='white', height=barWidth)
    # Custom x axis
    plt.yticks(r, [group1.name, group2.name])
    
    ax2.barh(r, totals)
    # ax2.tick_params(
    # axis='x',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # top=False,         # ticks along the top edge are off
    # labelbottom=False) # labels along the bottom edge are off

    plt.title(name)
    plt.tight_layout()
    #plt.savefig("plots/likert/" + name + ".png")
    plt.show()


def likert_plot2(df, exclude=['duration', 'rating', 'sessionID'], compare="perspective", filename_prefix=""):
    #TODO same ordering of factors for all subplots
    exclude.append(compare)
    candidates = df[compare].unique()
    for var in df.columns:
        if var in exclude:
            continue
        else:
            f, axes = plt.subplots(1, len(candidates))
            for i, candidate in enumerate(candidates):
                candidate_df = df[df[compare] == candidate]
                unique = candidate_df[var].unique()
                # From raw value to percentage
                totals = candidate_df[var].value_counts(dropna=False)
                greenBars = candidate_df[var][candidate_df["rating"] == 0].value_counts(dropna=False)/totals * 100
                orangeBars = candidate_df[var][candidate_df["rating"] == 1].value_counts(dropna=False)/totals * 100
                blueBars = candidate_df[var][candidate_df["rating"] == 2].value_counts(dropna=False)/totals * 100
                yellowBars = candidate_df[var][candidate_df["rating"] == 3].value_counts(dropna=False)/totals * 100
                # plot
                barWidth = 0.99
                names = totals.index
                r = list(range(unique.shape[0]))
                # Create grey Bars
                buttom = totals/totals * 100
                axes[i].barh(r, buttom, color='#33cc33', edgecolor='white', height=barWidth)
                # Create blue Bars
                buttom = buttom - yellowBars
                axes[i].barh(r, buttom, color='#336600', edgecolor='white', height=barWidth)
                # Create orange Bars
                buttom = buttom - blueBars
                axes[i].barh(r, buttom, color='#cc0000', edgecolor="white", height=barWidth)
                # Create green Bars
                buttom = buttom - orangeBars
                axes[i].barh(r, buttom, color='#ff3300', edgecolor='white', height=barWidth)
                # Custom y axis
                plt.sca(axes[i])
                plt.yticks(r, unique)
                # ax1.title.set_text('First Plot') # TODO titles for candidate

            plt.title(var)
            plt.tight_layout()
            plt.savefig("plots/likert2/" + filename_prefix + "_" + var + ".png")
            plt.close()