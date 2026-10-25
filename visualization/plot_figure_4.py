import sys
sys.dont_write_bytecode = True
from plot_style import *

def main():
    output = output_path()
    data = read_data('figure_4_strategy_comparison.csv').sort_values('macro_nrmse')
    figure,axis = plt.subplots(figsize=(7.4,4.8),layout='constrained')
    positions = np.arange(len(data))
    colors = [BLUE if value=='Spectral strategy selector' else GOLD if value=='Original-library unanchored reference' else '#A2B3BF' for value in data.strategy]
    axis.hlines(positions,.365,data.macro_nrmse,color='#E2E7EB',linewidth=1.2,zorder=1)
    axis.scatter(data.macro_nrmse,positions,c=colors,s=44,zorder=3)
    for position,value in zip(positions,data.macro_nrmse):
        axis.text(value+.0007,position,f'{value:.4f}',va='center',fontsize=9)
    axis.set(yticks=positions,yticklabels=data.strategy.tolist(),xlim=(.366,.419),xlabel='Macro normalized RMSE')
    axis.invert_yaxis()
    axis.grid(axis='x',color='#F0F2F4',linewidth=.7)
    export(figure,output,'figure_4_strategy_comparison')

if __name__ == '__main__':
    main()
