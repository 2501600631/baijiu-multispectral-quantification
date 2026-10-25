from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
BLUE = '#24658C'
GOLD = '#C27A24'
TEAL = '#16817C'
GRAY = '#75818B'
PALE = '#DAE2E7'
ANALYTES = ['Acetaldehyde','Methanol','1-Propanol','Ethyl acetate','Isobutanol','Acetal','1-Butanol','Isoamyl alcohol','Ethyl butyrate','Ethyl lactate','Ethyl hexanoate']

def setup():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,'axes.labelsize':10.5,'axes.titlesize':11,'axes.linewidth':.7,'xtick.labelsize':9.5,'ytick.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True,'savefig.facecolor':'white'})

def read_data(name):
    return pd.read_csv(BASE / 'data' / name, float_precision='round_trip')

def output_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE / 'rendered')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    return args.output

def export(figure, output, stem):
    figure.savefig(output / (stem+'.pdf'), bbox_inches='tight')
    figure.savefig(output / (stem+'.eps'), bbox_inches='tight')
    figure.savefig(output / (stem+'.svg'), bbox_inches='tight')
    figure.savefig(output / (stem+'.png'), dpi=320, bbox_inches='tight')
    figure.savefig(output / (stem+'.tiff'), dpi=600, bbox_inches='tight', pil_kwargs={'compression':'tiff_lzw'})
    plt.close(figure)

setup()
