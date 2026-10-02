"""Validate the four probed XML reports and summarize their native clocks.

Raw originals are never edited. Only small official schema/documentation additions
are downloaded into a fresh F folder; working schema copies and reports use D.
"""
from __future__ import annotations
import collections, datetime as dt, hashlib, json, re, shutil, stat, os
from pathlib import Path
from html.parser import HTMLParser
import requests
import fitz
from lxml import etree

OUT = Path(os.environ["RDIA_NATIVE_OUTPUT_ROOT"]).resolve()
BASE = json.loads((OUT / "probe_execution.json").read_text(encoding="utf-8"))
RAW = Path(BASE["raw_folder"])
EXTRA = RAW.parent / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_schema_followup")
SCHEMA = OUT / "schema_validation"
MANIFEST = []

def download(url, name):
    session = requests.Session()
    session.headers["User-Agent"] = "AOOR-native-clock-research-probe/1.0"
    try:
        response = session.get(url, timeout=(15, 45)); response.raise_for_status()
    except requests.RequestException:
        session.trust_env = False
        response = session.get(url, timeout=(15, 45)); response.raise_for_status()
    data = response.content
    if len(data) > 4 * 1024**2 or BASE["downloaded_bytes"] + sum(row["bytes"] for row in MANIFEST) + len(data) > 10 * 1024**2:
        raise RuntimeError("Original plus follow-up exceeds bounded probe budget")
    path = EXTRA / name
    with path.open("xb") as stream: stream.write(data)
    path.chmod(stat.S_IREAD)
    MANIFEST.append({"url":url,"final_url":response.url,"path":str(path),"bytes":len(data),
        "sha256":hashlib.sha256(data).hexdigest(),"retrieved_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "server_last_modified":response.headers.get("Last-Modified"),"raw_source_read_only":True})
    print(json.dumps({"downloaded":name,"bytes":len(data)}),flush=True)
    return path

class PlainText(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]; self.ignore=0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"): self.ignore += 1
    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.ignore: self.ignore -= 1
    def handle_data(self, data):
        if not self.ignore and data.strip(): self.parts.append(data.strip())

def main():
    space={drive:shutil.disk_usage(drive+":/").free for drive in "DEF"}
    if space["F"] < 20*1024**2: raise RuntimeError("F free space insufficient")
    EXTRA.mkdir(parents=True,exist_ok=False); SCHEMA.mkdir(exist_ok=True)
    common = download("https://reports-public.ieso.ca/docrefs/schema/Document_r1.xsd", "Document_r1.xsd")
    training = download("https://www.ieso.ca/-/media/Files/IESO/Document-Library/training/WB-Interjurisdictional-Energy-Trading.ashx", "IESO_interjurisdictional_energy_trading_2025.pdf")
    release = download("https://ieso.ca/-/media/Files/IESO/Document-Library/it-release-plan/it-FinalReleasePlan-R550.pdf", "IESO_release_55_final_plan_2026.pdf")
    for path in (training,release):
        document=fitz.open(path)
        (OUT/(path.stem+"_text.txt")).write_text("\n".join(page.get_text() for page in document),encoding="utf-8")
    shutil.copyfile(common, SCHEMA/"Document_r1.xsd")
    schemas={}
    for family in ("PredispHourlyOntarioZonalPrice", "RealtimeOntarioZonalPrice"):
        path=SCHEMA/(family+"_schema.xsd");shutil.copyfile(RAW/path.name,path)
        schemas[family]=etree.XMLSchema(etree.parse(str(path)))
    ns={"i":"http://www.ieso.ca/schema"}; samples=[]
    for path in sorted(RAW.glob("PUB_*.xml")):
        tree=etree.parse(str(path)); family=tree.getroot().get("docID")
        schemas[family].assertValid(tree)
        row={"file":path.name,"xsd_validation":True,"family":family,
             "created_at":tree.findtext(".//i:CreatedAt",namespaces=ns),
             "delivery_date":tree.findtext(".//i:DeliveryDate",namespaces=ns),
             "document_revision":tree.findtext(".//i:DocRevision",namespaces=ns)}
        if family.startswith("Predisp"):
            entries=tree.findall(".//i:HourlyPriceComponents",namespaces=ns)
            row.update({"target_hours":[int(item.findtext("i:PricingHour",namespaces=ns)) for item in entries],
                "price_nodes":len(entries),"nonempty_price_count":sum(bool((item.findtext("i:ZonalPrice",namespaces=ns) or "").strip()) for item in entries)})
        else:
            entries=tree.findall(".//i:DocBody/i:ZonalPrice",namespaces=ns)
            row.update({"delivery_hour":int(tree.findtext(".//i:DeliveryHour",namespaces=ns)),
                "interval_nodes":len(entries),"nonempty_interval_price_count":sum(bool((item.findtext("i:LmpCap",namespaces=ns) or "").strip()) for item in entries),
                "intervals":[int(item.findtext("i:Interval",namespaces=ns)) for item in entries],
                "nonempty_flag_counts":dict(collections.Counter(item.findtext("i:Flag",namespaces=ns) for item in entries if item.findtext("i:Flag",namespaces=ns)))})
        samples.append(row)
    inventory=[]
    for family in schemas:
        data=json.loads((OUT/(family+"_inventory.json")).read_text(encoding="utf-8")); rows=data["files"]
        grouped=collections.defaultdict(list)
        for row in rows: grouped[(row["date"],row["hour"])].append(row)
        units=[]
        for (date,hour),items in sorted(grouped.items()):
            versions=sorted(row["version"] for row in items if row["version"] is not None)
            units.append({"date":date,"hour":hour,"numbered_versions":len(versions),"min_version":min(versions) if versions else None,
                "max_version":max(versions) if versions else None,"missing_numbers_between_endpoints":[i for i in range(min(versions),max(versions)+1) if i not in versions] if versions else [],
                "alias_present":any(row["version"] is None for row in items)})
        september=[row for row in units if row["date"].startswith("202609")]
        summary={"family":family,"september_units":len(september),"september_numbered_versions":sum(row["numbered_versions"] for row in september),
             "september_aliases":sum(row["alias_present"] for row in september),
             "september_counts_by_unit":dict(collections.Counter(row["numbered_versions"] for row in september)),
             "september_units_with_gaps":[row for row in september if row["missing_numbers_between_endpoints"]],
             "oldest_date_with_multiple_numbered_versions":min((row["date"] for row in units if row["numbered_versions"]>1),default=None),
             "september_all_units_have_alias":all(row["alias_present"] for row in september)}
        inventory.append(summary)
        (OUT/(family+"_unit_version_retention.json")).write_text(json.dumps({"summary":summary,"units":units},indent=2),encoding="utf-8")
    for name in ("IESO_website_terms.html","IESO_data_directory.html"):
        parser=PlainText();parser.feed((RAW/name).read_text(encoding="utf-8"))
        (OUT/(Path(name).stem+"_text.txt")).write_text("\n".join(parser.parts),encoding="utf-8")
    original_unchanged=all(hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()==row["sha256"] for row in BASE["downloads"])
    result={"status":"FOUR_XML_VALIDATED_AGAINST_OFFICIAL_SCHEMA","space_check_free_bytes":space,
        "raw_followup_folder":str(EXTRA),"followup_downloads":MANIFEST,"total_raw_bytes_including_original":BASE["downloaded_bytes"]+sum(row["bytes"] for row in MANIFEST),
        "original_raw_manifest_hashes_unchanged":original_unchanged,"validated_samples":samples,"archive_summary":inventory,
        "full_september_values_parsed":False,"independent_holdout_claim":False,"optimization_executed":False}
    (OUT/"schema_validation_receipt.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()
