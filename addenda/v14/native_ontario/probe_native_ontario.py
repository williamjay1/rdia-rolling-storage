"""Bounded read-only-source probe of IESO native zonal-price XML reports.

Owns only results/revision_v14/ontario_probe and a new timestamped F raw folder.
Does not fit, optimize, concatenate legacy HOEP, or publish data.
"""
from __future__ import annotations
import collections, datetime as dt, hashlib, json, os, re, shutil, stat
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
import requests

OUT = Path(os.environ["RDIA_NATIVE_OUTPUT_ROOT"]).resolve()
RAW_PARENT = Path(os.environ["RDIA_NATIVE_RAW_ROOT"]).resolve()
NOW = dt.datetime.now(dt.timezone.utc)
RAW = RAW_PARENT / NOW.strftime("%Y%m%dT%H%M%S_%fZ")
MAX_TOTAL = 10 * 1024**2
MAX_FILE = 4 * 1024**2
MANIFEST = []
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AOOR-native-clock-research-probe/1.0", "Accept": "*/*"})

class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            a=dict(attrs)
            if a.get("href"): self.links.append(a["href"])

def download(url, name):
    dest=RAW/name
    if dest.exists(): raise FileExistsError(dest)
    try:
        r=SESSION.get(url,timeout=(15,45));r.raise_for_status()
    except requests.RequestException:
        # Direct public-source fallback; no credential/proxy values logged.
        direct=requests.Session();direct.trust_env=False
        direct.headers.update(SESSION.headers)
        r=direct.get(url,timeout=(15,45));r.raise_for_status()
    data=r.content
    if len(data)>MAX_FILE or sum(x["bytes"] for x in MANIFEST)+len(data)>MAX_TOTAL:
        raise RuntimeError("Bounded probe download limit exceeded")
    with dest.open("xb") as f:f.write(data)
    dest.chmod(stat.S_IREAD)
    row={"url":url,"final_url":r.url,"path":str(dest),"bytes":len(data),
         "sha256":hashlib.sha256(data).hexdigest(),"content_type":r.headers.get("Content-Type"),
         "server_last_modified":r.headers.get("Last-Modified"),"retrieved_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
         "raw_source_read_only":True}
    MANIFEST.append(row)
    print(json.dumps({"downloaded":name,"bytes":len(data)}),flush=True)
    return dest

def inventory(path, family):
    parser=Links();parser.feed(path.read_text(encoding="utf-8"))
    names=sorted(set(h for h in parser.links if h.startswith("PUB_"+family) and h.endswith(".xml")))
    expression=re.compile(r"^PUB_"+re.escape(family)+r"_(\d{8})(\d{2})?(?:_v(\d+))?\.xml$")
    rows=[]
    for name in names:
        m=expression.match(name)
        if m:rows.append({"filename":name,"date":m[1],"hour":int(m[2]) if m[2] else None,
                          "version":int(m[3]) if m[3] else None,"dated_latest_alias":m[3] is None})
    dates=sorted(set(x["date"] for x in rows)); september=[x for x in rows if x["date"].startswith("202609")]
    perdate=[]
    for day in (dt.date(2026,9,1)+dt.timedelta(days=i) for i in range(30)):
        key=day.strftime("%Y%m%d"); selected=[x for x in september if x["date"]==key]
        versions=[x for x in selected if x["version"] is not None]
        perdate.append({"date":key,"xml_names":len(selected),"version_names":len(versions),
                        "version_numbers":sorted(set(x["version"] for x in versions)),
                        "hours":sorted(set(x["hour"] for x in selected if x["hour"] is not None))})
    summary={"family":family,"listed_xml_files_including_alias":len(names),"dated_xml_files":len(rows),
             "first_date":dates[0] if dates else None,"last_date":dates[-1] if dates else None,
             "september_missing_dates":[x["date"] for x in perdate if not x["xml_names"]],
             "september_daily_retention":perdate,
             "interpretation":"Listing presence is not value/forecast support completeness, receipt-time proof, or an independent holdout guarantee."}
    (OUT/(family+"_inventory.json")).write_text(json.dumps({"summary":summary,"files":rows},indent=2),encoding="utf-8")
    return summary,rows

def local(tag):return tag.rsplit("}",1)[-1]

def parse_xml(path):
    root=ET.fromstring(path.read_bytes())
    nodes=collections.Counter(local(x.tag) for x in root.iter())
    values=collections.defaultdict(list)
    for x in root.iter():
        if len(x)==0 and x.text and x.text.strip(): values[local(x.tag)].append(x.text.strip())
    (OUT/(path.stem+"_parsed.json")).write_text(json.dumps({"root_tag":root.tag,"root_attributes":root.attrib,
        "node_counts":nodes,"leaf_values":values},indent=2),encoding="utf-8")
    sample={"filename":path.name,"root_tag":root.tag,"root_attributes":root.attrib,"node_counts":dict(nodes),
            "metadata":{k:v for k,v in values.items() if k in ("CreatedAt","DateTime","CreatedDate","DeliveryDate","DeliveryHour","Hour","Version","DocRevision","Revision","Interval","OntarioZonalPrice","ReportName","Confidentiality")},
            "price_values_not_printed":True,"parsed":True}
    return sample,root

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    space={drive:{"free_bytes":shutil.disk_usage(drive+":/").free} for drive in ("D","E","F")}
    if space["F"]["free_bytes"]<MAX_TOTAL*2:raise RuntimeError("F has insufficient space")
    RAW.mkdir(parents=True,exist_ok=False)
    families=("PredispHourlyOntarioZonalPrice","RealtimeOntarioZonalPrice")
    listings={}; summaries={}; rows_by_family={}
    for family in families:
        path=download(f"https://reports-public.ieso.ca/public/{family}/",family+"_index.html")
        summaries[family],rows_by_family[family]=inventory(path,family)
    docs=[("https://reports-public.ieso.ca/docrefs/helpfile/PredispHourlyOntarioZonalPrice_h1.pdf","PredispHourlyOntarioZonalPrice_h1.pdf"),
          ("https://reports-public.ieso.ca/docrefs/helpfile/RealtimeOntarioZonalPrice_h1.pdf","RealtimeOntarioZonalPrice_h1.pdf"),
          ("https://ieso.ca/-/media/Files/IESO/Document-Library/market-renewal/MRP-changes-to-report-site-Quick-Take.pdf","MRP_changes_report_site.pdf"),
          ("https://ieso.ca/Terms-of-Use","IESO_website_terms.html"),
          ("https://ieso.ca/power-data/data-directory","IESO_data_directory.html")]
    failures=[]
    for url,name in docs:
        try:download(url,name)
        except Exception as error:failures.append({"url":url,"error_type":type(error).__name__,"message":str(error)})
    parsed=[];schemas=[]
    for family,rows in rows_by_family.items():
        dated=[r for r in rows if r["date"]=="20260901" and (r["hour"] is None or r["hour"]==1) and r["version"] is not None]
        if not dated:raise RuntimeError((family,"September1 version not retained"))
        dated=sorted(dated,key=lambda x:x["version"])
        selected=[dated[0]] if len(dated)==1 else [dated[0],dated[-1]]
        for item in selected:
            url=f"https://reports-public.ieso.ca/public/{family}/{item['filename']}"
            path=download(url,item["filename"]);sample,root=parse_xml(path);parsed.append(sample)
            for key,value in root.attrib.items():
                if local(key)=="schemaLocation":
                    parts=value.split()
                    for schema in parts[1::2]:
                        absolute=urljoin(url,schema)
                        if absolute not in [x["url"] for x in schemas]:
                            try:
                                xs=download(absolute,family+"_schema.xsd")
                                schema_root=ET.fromstring(xs.read_bytes())
                                schemas.append({"url":absolute,"path":str(xs),"xml_parsed":True,"root_tag":schema_root.tag})
                            except Exception as error:failures.append({"url":absolute,"error_type":type(error).__name__,"message":str(error)})
    import fitz
    pdfs=[]
    for path in sorted(RAW.glob("*.pdf")):
        doc=fitz.open(path);text="\n".join(page.get_text() for page in doc)
        dest=OUT/(path.stem+"_text.txt");dest.write_text(text,encoding="utf-8")
        pdfs.append({"raw_pdf":str(path),"pages":len(doc),"text_output":str(dest),"parsed":True})
    result={"status":"SCHEMA_AND_RETENTION_PROBE_EXECUTED","retrieved_utc":NOW.isoformat(),"raw_folder":str(RAW),
            "scope":"Two official listings, documentation and at most two XML versions per report; no optimization, old HOEP concatenation, or public redistribution.",
            "space_check":space,"maximum_download_bytes":MAX_TOTAL,"downloaded_bytes":sum(x["bytes"] for x in MANIFEST),
            "downloads":MANIFEST,"inventories":summaries,"xml_samples":parsed,"schema_files":schemas,"pdf_extraction":pdfs,"failures":failures,
            "independent_holdout_claim":False,"full_september_values_parsed":False}
    (OUT/"probe_execution.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"raw_folder":str(RAW),"files":len(MANIFEST),"bytes":result["downloaded_bytes"],
        "parsed_xml_files":len(parsed),"failures":failures}),flush=True)

if __name__=="__main__":main()
