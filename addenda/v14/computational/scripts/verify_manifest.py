"""Verify every v14 computational payload file; does not run the controller."""
import argparse,hashlib,json
from pathlib import Path,PurePosixPath
def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',nargs='?',default=str(Path(__file__).resolve().parents[1]))
    args=parser.parse_args();root=Path(args.root).resolve()
    manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
    expected={row['path'] for row in manifest['files']}|{'MANIFEST.json'}
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if expected!=actual:raise RuntimeError('Manifest inventory mismatch: '+str(sorted(expected^actual)))
    for row in manifest['files']:
        name=PurePosixPath(row['path'])
        if name.is_absolute() or '..' in name.parts or '\\' in row['path'] or ':' in row['path']:raise RuntimeError('Unsafe manifest path')
        p=root/row['path']
        if p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Byte mismatch: '+row['path'])
    print(json.dumps({'status':'PASS_COMPLETE_V14_MANIFEST','files':len(manifest['files']),
                      'bytes':sum(row['bytes'] for row in manifest['files']),
                      'scope':'Byte verification only; no replay, refit or raw-source reconstruction.'},indent=2))
if __name__=='__main__':main()
