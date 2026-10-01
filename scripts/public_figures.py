"""Redraw the seven manuscript figures into a separate fresh directory."""
import argparse, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--figures',nargs='+',choices=['1','2','3','4','5','6','S1'],default=['1','2','3','4','5','6','S1'])
    a=p.parse_args();data=a.data_root.resolve(strict=True);out=a.output_root.resolve()
    if out.exists():p.error('Use a new output directory; existing figure exports are never overwritten.')
    if out==data or out.is_relative_to(data) or data.is_relative_to(out):p.error('Output must not overlap the supplied inputs.')
    sys.dont_write_bytecode=True
    import revision_v11_figures as m
    m.ROOT=data;m.OUT=out;out.mkdir(parents=True,exist_ok=False)
    with m.plt.rc_context(m.RC):
        for key in a.figures:m.FUNCTIONS[key]()
    print(f'Fresh figure exports written to {out}')

if __name__=='__main__':main()
