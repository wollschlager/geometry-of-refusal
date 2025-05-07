import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

def apply_style():
    """Applies the standard plotting style."""
    sns.set_style("white")
    sns.set_context("paper", font_scale=1, rc={
            "lines.linewidth": 1.2,
            "xtick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.major.size": 0,
            "ytick.minor.size": 0
        })

    matplotlib.rcParams["mathtext.fontset"] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['figure.autolayout'] = True

    plt.rc('font', size=13)
    plt.rc('axes', titlesize=13)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('legend', title_fontsize=13)
    plt.rc('legend', fontsize=13)
    plt.rc('figure', titlesize=16)

    colors = sns.color_palette('colorblind')
    colors[0], colors[-1] = colors[-1], colors[0] 
    return colors 