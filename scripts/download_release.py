"""Download and verify the complete release archive into a fresh directory.

The Git repository itself omits large data. This command downloads the complete
code-plus-derived-data attachment, not GitHub's automatically generated source ZIP.
"""
import argparse, hashlib, json, os, re, stat, urllib.request, zipfile
from pathlib import Path, PurePosixPath

DEFAULT_VERSION='1.1.1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def fetch(url,path,maximum):
    request=urllib.request.Request(url,headers={'User-Agent':'RDIA-reproducibility/1.1.1'})
    total=0
    with urllib.request.urlopen(request,timeout=60) as r,path.open('xb') as f:
        while True:
            chunk=r.read(1024*1024)
            if not chunk:break
            total+=len(chunk)
            if total>maximum:raise RuntimeError('Download exceeds the declared safety limit; partial download retained.')
            f.write(chunk)

def checksum_rows(text):
    rows={};names=set()
    for line in text.splitlines():
        if not line.strip():continue
        match=re.fullmatch(r'([0-9a-f]{64})[ \t]+\*?([^\r\n]+)',line)
        if match is None:raise RuntimeError('Malformed release checksum line.')
        digest,name=match.groups()
        if name in ('.','..') or '/' in name or '\\' in name or ':' in name or any(ord(c)<32 for c in name):
            raise RuntimeError('Unsafe release checksum filename.')
        if name.casefold() in names:raise RuntimeError('Duplicate release checksum filename.')
        names.add(name.casefold());rows[name]=digest
    return rows

def manifest_rows(manifest):
    rows=manifest.get('files')
    if not isinstance(rows,list) or not rows:raise RuntimeError('Missing or empty release manifest file list.')
    names=set()
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('path'),str):raise RuntimeError('Invalid manifest file record.')
        name=row['path'];part=PurePosixPath(name)
        if not name or part.is_absolute() or '..' in part.parts or '\\' in name or ':' in name or part.as_posix()!=name or any(ord(c)<32 for c in name):
            raise RuntimeError('Unsafe manifest path.')
        if name=='release_manifest.json':raise RuntimeError('Manifest must exclude its own self-referential hash.')
        if name.casefold() in names:raise RuntimeError('Duplicate manifest path.')
        names.add(name.casefold())
        if not isinstance(row.get('bytes'),int) or isinstance(row['bytes'],bool) or row['bytes']<0 or not isinstance(row.get('sha256'),str) or re.fullmatch(r'[0-9a-f]{64}',row['sha256']) is None:
            raise RuntimeError('Invalid manifest byte count or SHA-256.')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--version',default=DEFAULT_VERSION,help='Complete release version, e.g. 1.1.1 or the preserved 1.0.0; figure-only tags have no data attachment.')
    a=p.parse_args();out=a.output_root.resolve();repo=Path(__file__).resolve().parents[1]
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+',a.version):p.error('Use an actual numeric release version without a v prefix.')
    archive_name=f'rdia-rolling-storage-v{a.version}.zip'
    base=f'https://github.com/williamjay1/rdia-rolling-storage/releases/download/v{a.version}/'
    if out.exists():p.error('Use a new directory; existing data are never overwritten.')
    if out==repo or out.is_relative_to(repo) or repo.is_relative_to(out):p.error('Download directory must not overlap the source repository.')
    out.mkdir(parents=True,exist_ok=False)
    checks=out/'SHA256SUMS';archive=out/archive_name
    fetch(base+'SHA256SUMS',checks,128*1024)
    rows=checksum_rows(checks.read_text(encoding='utf-8'))
    if archive_name not in rows:raise RuntimeError('Release checksum file does not identify the complete archive.')
    fetch(base+archive_name,archive,2*1024**3)
    digest=sha(archive)
    if digest!=rows[archive_name]:raise RuntimeError('Archive checksum mismatch; no extraction attempted.')
    package=out/'package'
    with zipfile.ZipFile(archive) as z:
        names=set();expanded=0
        for entry in z.infolist():
            part=PurePosixPath(entry.filename)
            if not part.parts or part.is_absolute() or '..' in part.parts or '\\' in entry.filename or ':' in entry.filename or part.as_posix()!=entry.filename.rstrip('/') or any(ord(c)<32 for c in entry.filename):raise RuntimeError('Unsafe ZIP entry.')
            if stat.S_ISLNK(entry.external_attr>>16):raise RuntimeError('ZIP symlink is not permitted.')
            if part.as_posix().casefold() in names:raise RuntimeError('Duplicate ZIP entry.')
            names.add(part.as_posix().casefold());expanded+=entry.file_size
            if entry.file_size>512*1024**2 or expanded>3*1024**3:raise RuntimeError('Unexpected archive expansion.')
        if z.testzip() is not None:raise RuntimeError('ZIP CRC integrity failed.')
        package.mkdir()
        z.extractall(package)
    manifest=json.loads((package/'release_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version')!=a.version:raise RuntimeError('Extracted package version differs from the requested Release.')
    records=manifest_rows(manifest)
    declared=[row['path'] for row in records]
    actual={path.relative_to(package).as_posix() for path in package.rglob('*') if path.is_file()}
    if actual!=set(declared)|{'release_manifest.json'}:raise RuntimeError('Undeclared or missing file in the complete archive.')
    for row in records:
        path=(package/row['path']).resolve()
        if not path.is_relative_to(package) or not path.is_file():raise RuntimeError('Manifest path is missing or escapes the package.')
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise RuntimeError(f'Extracted file differs: {row["path"]}')
    receipt={'status':'PASS_COMPLETE_PUBLISHED_ARCHIVE_VERIFIED','version':a.version,
             'archive':archive_name,'archive_sha256':digest,'extracted_files_verified':len(records),
             'release_manifest_sha256':sha(package/'release_manifest.json'),
             'package_root':str(package),'solver_execution':False,'model_refitting':False,'raw_source_reconstruction':False}
    (out/'download_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
