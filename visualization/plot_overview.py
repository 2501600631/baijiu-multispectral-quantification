import sys
sys.dont_write_bytecode = True
from plot_style import *

def main():
    output = output_path()
    from matplotlib.patches import FancyBboxPatch
    figure,axis = plt.subplots(figsize=(11,4.1),layout='constrained')
    axis.set(xlim=(0,1),ylim=(0,1))
    axis.axis('off')
    axis.text(.035,.92,'139 Baijiu bottles  |  11 chromatographic constituents',fontsize=17,fontweight='bold',color=BLUE,va='top')
    axis.text(.035,.78,'Raman, ultraviolet and near-infrared spectra',fontsize=12,color=GRAY)
    cards=[(.035,'10 / 11','Analytes improved over the\nfixed alcohol-strength baseline',BLUE),(.36,'11 / 11','High-concentration subgroups\nbelow nominal interval coverage',GOLD),(.685,'4.76%  →  1.00%','Retention gains before and after\nmatching concentration composition',TEAL)]
    for x,value,label,color in cards:
        axis.add_patch(FancyBboxPatch((x,.29),.27,.34,boxstyle='round,pad=.012,rounding_size=.014',facecolor='#F2F6F8',edgecolor='#DDE5E9',linewidth=.7))
        axis.text(x+.135,.555,value,fontsize=20,fontweight='bold',color=color,ha='center',va='center')
        axis.text(x+.135,.385,label,fontsize=10,ha='center',va='center',linespacing=1.55)
    axis.text(.035,.12,'Incremental prediction, concentration boundaries and selective review',fontsize=12.5,color='#303B43')
    export(figure,output,'analysis_overview')

if __name__ == '__main__':
    main()
