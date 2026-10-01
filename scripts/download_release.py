"""Download and verify the complete release archive into a fresh directory.

The Git repository itself omits large data. This command downloads the complete
code-plus-derived-data attachment, not GitHub's automatically generated source ZIP.
"""
import argparse, hashlib, json, os, re, stat, urllib.request, zipfile
from pathlib import Path, PurePosixPath

VERSION='1.0.0'
ARCHIVE=f'rdia-rolling-storage-v{VERSION}.zip'
BASE=f'https://github.com/williamjay1/rdia-rolling-storage/releases/download/v{VERSION}/'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def fetch(url,path,maximum):
    request=urllib.request.Request(url,headers={'User-Agent':'RDIA-reproducibility/1.0'})
    total=0
    with urllib.request.urlopen(request,timeout=60) as r,path.open('xb') as f:
        while True:
            chunk=r.read(1024*1024)
            if not chunk:break
            total+=len(chunk)
            if total>maximum:raise RuntimeError('Download exceeds the declared safety limit; partial download retained.')
            f.write(chunk)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();out=a.output_root.resolve();repo=Path(__file__).resolve().parents[1]
    if out.exists():p.error('Use a new directory; existing data are never overwritten.')
    if out==repo or out.is_relative_to(repo) or repo.is_relative_to(out):p.error('Download directory must not overlap the source repository.')
    if os.name=='nt' and out.drive.upper()!='D:':p.error('Use D: for the author Windows host computation/download copy.')
    out.mkdir(parents=True,exist_ok=False)
    checks=out/'SHA256SUMS';archive=out/ARCHIVE
    fetch(BASE+'SHA256SUMS',checks,128*1024)
    rows={name:checksum for checksum,name in re.findall(r'^([0-9a-f]{64})\s+\*?([^\r\n]+)$',checks.read_text(encoding='utf-8'),re.M)}
    if ARCHIVE not in rows:raise RuntimeError('Release checksum file does not identify the complete archive.')
    fetch(BASE+ARCHIVE,archive,2*1024**3)
    digest=sha(archive)
    if digest!=rows[ARCHIVE]:raise RuntimeError('Archive checksum mismatch; no extraction attempted.')
    package=out/'package';package.mkdir()
    with zipfile.ZipFile(archive) as z:
        names=set();expanded=0
        for entry in z.infolist():
            part=PurePosixPath(entry.filename)
            if part.is_absolute() or '..' in part.parts or '\\' in entry.filename or ':' in entry.filename:raise RuntimeError('Unsafe ZIP entry.')
            if stat.S_ISLNK(entry.external_attr>>16):raise RuntimeError('ZIP symlink is not permitted.')
            if entry.filename in names:raise RuntimeError('Duplicate ZIP entry.')
            names.add(entry.filename);expanded+=entry.file_size
            if entry.file_size>512*1024**2 or expanded>3*1024**3:raise RuntimeError('Unexpected archive expansion.')
        if z.testzip() is not None:raise RuntimeError('ZIP CRC integrity failed.')
        z.extractall(package)
    manifest=json.loads((package/'release_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path=(package/row['path']).resolve()
        if not path.is_relative_to(package) or not path.is_file():raise RuntimeError('Manifest path is missing or escapes the package.')
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise RuntimeError(f'Extracted file differs: {row["path"]}')
    receipt={'archive':ARCHIVE,'archive_sha256':digest,'extracted_files_verified':len(manifest['files']),
             'package_root':str(package),'solver_execution':False,'model_refitting':False,'raw_source_reconstruction':False}
    (out/'download_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
