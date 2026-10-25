from pathlib import Path
import argparse
import subprocess
import sys

def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=base/'rendered')
    args = parser.parse_args()
    for name in ['plot_figure_1.py','plot_figure_2.py','plot_figure_3.py','plot_figure_4.py','plot_overview.py']:
        subprocess.run([sys.executable,str(base/name),'--output',str(args.output.resolve())],check=True)
    print('All analysis figures generated: '+str(args.output.resolve()))

if __name__ == '__main__':
    main()
