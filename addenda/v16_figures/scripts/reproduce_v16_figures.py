"""Recreate the six plots from frozen processed values, without model execution."""
from pathlib import Path
import hashlib,json,subprocess,sys
import pymupdf as fitz
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def remove_layout(value):
    if isinstance(value,dict):
        return {k:remove_layout(v) for k,v in value.items() if k not in ['layout','display_x']}
    if isinstance(value,list):return [remove_layout(x) for x in value]
    return value

def main():
    for script in ['revision_v16_figures_123.py','revision_v16_figures_456.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
    figures=[]
    for number in range(1,7):
        output=ROOT/'figures/revision_v16'
        saved=json.loads((ROOT/f'reference/fig{number}_source_data.json').read_text(encoding='utf-8'))
        fresh=json.loads((output/f'fig{number}_source.json').read_text(encoding='utf-8'))
        assert remove_layout(fresh['source_data'])==remove_layout(saved['source_data']), ('Scientific values differ',number)
        with fitz.open(output/f'fig{number}.pdf') as pdf:
            assert len(pdf)==1 and not pdf[0].get_images()
            text=[span for block in pdf[0].get_text('dict')['blocks'] if 'lines' in block
                  for line in block['lines'] for span in line['spans']]
            assert min(s['size'] for s in text)>=8.99
            assert abs(pdf[0].rect.width*25.4/72-174)<.001
        for extension in ['png','tiff']:
            with Image.open(output/f'fig{number}.{extension}') as image:
                assert min(float(x) for x in image.info['dpi'])>=1199
        audit=subprocess.run([sys.executable,str(ROOT/'scripts/nature_qa_v11/audit_figure_collisions.py'),
            str(output/f'fig{number}.pdf'),'--json'],capture_output=True,text=True,encoding='utf-8',check=False)
        report=json.loads(audit.stdout)
        assert audit.returncode==0 and report['verdict']=='PASS', (number,report)
        (output/f'fig{number}_pdf_collisions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        figures.append({'figure':number,'scientific_source_data':'EXACT_MATCH',
            'layout_only_fields_excluded':['layout','display_x'],
            'collision_verdict':report['verdict'],'pdf_sha256':sha(output/f'fig{number}.pdf')})
    result={'status':'PASS_SIX_FROZEN_DATA_REDRAWS','figures':figures,
        'scope':'Only frozen plotting inputs are read. No optimizers, statistical estimation or raw data fetches run.'}
    (ROOT/'reproduction_receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
