import sys
sys.dont_write_bytecode = True
from plot_style import *

def main():
    output = output_path()
    data = read_data('figure_2_concentration_reliability.csv')
    is_high = data.high_concentration.astype(str).str.lower().eq('true')
    high = data[is_high].set_index('analyte').loc[ANALYTES]
    other = data[~is_high].set_index('analyte').loc[ANALYTES]
    figure, (coverage,misses) = plt.subplots(1,2,figsize=(7.4,5.85),sharey=True,layout='constrained',gridspec_kw={'width_ratios':[1.1,1]})
    positions = np.arange(len(ANALYTES))
    for row in positions:
        coverage.plot([other.coverage.iloc[row]*100,high.coverage.iloc[row]*100],[row,row],color=PALE,linewidth=1.7,zorder=1)
    coverage.scatter(other.coverage*100,positions,color=TEAL,s=30,label='At or below training Q75',zorder=2)
    coverage.scatter(high.coverage*100,positions,facecolors='white',edgecolors=GOLD,linewidths=1.3,marker='s',s=31,label='Above training Q75',zorder=3)
    coverage.axvline(90,color=GRAY,linestyle='--',linewidth=.85)
    coverage.set(xlim=(74,101),xticks=[75,80,85,90,95,100],yticks=positions,yticklabels=ANALYTES,xlabel='Marginal interval coverage (%)')
    coverage.invert_yaxis()
    coverage.legend(frameon=False,loc='lower left',bbox_to_anchor=(-.02,1.015),fontsize=9.2,handletextpad=.4)
    coverage.grid(axis='x',color='#F0F2F4',linewidth=.7)
    misses.barh(positions,high.truth_above_upper_rate*100,color=BLUE,height=.49,label='True value above upper bound')
    misses.barh(positions,high.truth_below_lower_rate*100,left=high.truth_above_upper_rate*100,color='#BCC8D0',height=.49,label='True value below lower bound')
    misses.set(xlim=(0,26),xticks=[0,5,10,15,20,25],xlabel='High-group interval misses (%)')
    misses.tick_params(axis='y',left=False)
    misses.legend(frameon=False,loc='lower left',bbox_to_anchor=(-.01,1.015),fontsize=8.9,handlelength=1.2,handletextpad=.4)
    misses.grid(axis='x',color='#F0F2F4',linewidth=.7)
    for letter,axis in zip(['a','b'],[coverage,misses]):
        axis.text(-.02,1.15,letter,transform=axis.transAxes,fontweight='bold',fontsize=13)
    export(figure,output,'figure_2_concentration_reliability')

if __name__ == '__main__':
    main()
