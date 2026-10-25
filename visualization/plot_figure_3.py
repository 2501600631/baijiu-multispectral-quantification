import sys
sys.dont_write_bytecode = True
from plot_style import *

def main():
    output = output_path()
    data = read_data('figure_3_retention_macro.csv')
    analytes = read_data('figure_3_retention_analytes.csv').set_index('analyte').loc[ANALYTES]
    figure,(aggregate,individual) = plt.subplots(2,1,figsize=(7.4,8.6),layout='constrained',gridspec_kw={'height_ratios':[.85,1.15]})
    observed = data[data.control.eq('Equal-count random retention')].sort_values('mean_actual_retention')
    aggregate.plot(observed.mean_actual_retention*100,observed.observed_macro_nrmse,color=BLUE,marker='o',markersize=4.5,linewidth=1.7,label='Observed selection',zorder=5)
    for control,color,style,label in [('Equal-count random retention',GRAY,'--','Equal-count random'),('Concentration-quartile-matched random retention',GOLD,'-.','Concentration-matched random')]:
        group = data[data.control.eq(control)].sort_values('mean_actual_retention')
        x = group.mean_actual_retention.to_numpy()*100
        aggregate.fill_between(x,group.random_policy_low.to_numpy(),group.random_policy_high.to_numpy(),color='#EDF0F2' if color==GRAY else '#F8EEDF',linewidth=0)
        aggregate.plot(x,group.random_policy_low,color=color,linestyle=':',linewidth=.6)
        aggregate.plot(x,group.random_policy_high,color=color,linestyle=':',linewidth=.6)
        aggregate.plot(x,group.random_mean_macro_nrmse,color=color,linestyle=style,linewidth=1.4,label=label)
    aggregate.set(xlim=(47,102),ylim=(.35,.425),xlabel='Mean actual retention across analytes (%)',ylabel='Macro normalized RMSE',xticks=[50,60,70,80,90,100])
    aggregate.grid(color='#F0F2F4',linewidth=.7)
    aggregate.legend(frameon=False,fontsize=9.5,loc='lower left',bbox_to_anchor=(.015,1.01),ncol=1)
    positions = np.arange(len(ANALYTES))
    low = 100*(1-analytes.random_policy_high_g_L/analytes.random_mean_rmse_g_L)
    high = 100*(1-analytes.random_policy_low_g_L/analytes.random_mean_rmse_g_L)
    individual.hlines(positions,low,high,color='#C8D2D9',linewidth=3.5)
    individual.scatter(np.zeros(len(ANALYTES)),positions,facecolors='white',edgecolors=GRAY,s=20,linewidths=.8,zorder=3)
    special = analytes.below_random_policy_low.astype(str).str.lower().eq('true')
    individual.scatter(analytes.gain_vs_random_pct,positions,c=[BLUE if value else '#536B7C' for value in special],s=[40 if value else 27 for value in special],zorder=4)
    individual.axvline(0,color=GRAY,linestyle='--',linewidth=.8)
    individual.set(yticks=positions,yticklabels=ANALYTES,xlim=(-15,17),xlabel='RMSE reduction versus concentration-matched random mean (%)')
    individual.invert_yaxis()
    individual.grid(axis='x',color='#F0F2F4',linewidth=.7)
    for letter,axis in zip(['a','b'],[aggregate,individual]):
        axis.set_title(letter,loc='left',fontweight='bold',pad=11,fontsize=13)
    export(figure,output,'figure_3_retention_controls')

if __name__ == '__main__':
    main()
