import sys
sys.dont_write_bytecode = True
from plot_style import *

def main():
    output = output_path()
    data = read_data('figure_1_alcohol_baseline.csv')
    figure, axis = plt.subplots(figsize=(7.4,5.9), layout='constrained')
    positions = np.arange(len(ANALYTES))
    for strategy, offset, color, marker, label in [('Spectral strategy selector',-.14,BLUE,'o','Spectral strategy selector'),('Alcohol-assisted selector',.14,GOLD,'s','Alcohol-assisted selector')]:
        group = data[data.strategy.eq(strategy)].set_index('analyte').loc[ANALYTES]
        bounds = np.array([group.rmse_reduction_pct-group.conditional_ci_low_pct,group.conditional_ci_high_pct-group.rmse_reduction_pct])
        axis.errorbar(group.rmse_reduction_pct,positions+offset,xerr=bounds,fmt=marker,markersize=4.8,color=color,mfc=color if marker=='o' else 'white',mew=1.1,ecolor=color,elinewidth=1.25,capsize=2.4,label=label)
    axis.axvline(0,color=GRAY,linewidth=.85,linestyle='--')
    axis.set(yticks=positions,yticklabels=ANALYTES,xlim=(-25,90),xlabel='RMSE reduction versus fixed alcohol-strength baseline (%)')
    axis.invert_yaxis()
    axis.grid(axis='x',color='#F0F2F4',linewidth=.7)
    axis.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.015),ncol=2,fontsize=9.5,columnspacing=1.3,handletextpad=.5)
    export(figure,output,'figure_1_alcohol_baseline')

if __name__ == '__main__':
    main()
