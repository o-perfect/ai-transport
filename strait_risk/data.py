"""Full daily PortWatch history per chokepoint, cached as CSV for a day."""

import pathlib
import time

import pandas as pd
import requests

URL = ("https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
       "Daily_Chokepoints_Data/FeatureServer/0/query")
CACHE = pathlib.Path(__file__).resolve().parent.parent / ".cache" / "strait_risk"
FIELDS = ["date", "n_total", "n_container", "n_tanker", "n_dry_bulk", "capacity"]

# Every chokepoint PortWatch covers (name -> portid).
CHOKEPOINTS = {
    "Suez Canal": "chokepoint1", "Panama Canal": "chokepoint2", "Bosporus Strait": "chokepoint3",
    "Bab el-Mandeb Strait": "chokepoint4", "Malacca Strait": "chokepoint5",
    "Strait of Hormuz": "chokepoint6", "Cape of Good Hope": "chokepoint7",
    "Gibraltar Strait": "chokepoint8", "Dover Strait": "chokepoint9", "Oresund Strait": "chokepoint10",
    "Taiwan Strait": "chokepoint11", "Korea Strait": "chokepoint12", "Tsugaru Strait": "chokepoint13",
    "Luzon Strait": "chokepoint14", "Lombok Strait": "chokepoint15", "Ombai Strait": "chokepoint16",
    "Bohai Strait": "chokepoint17", "Torres Strait": "chokepoint18", "Sunda Strait": "chokepoint19",
    "Makassar Strait": "chokepoint20", "Magellan Strait": "chokepoint21",
    "Yucatan Channel": "chokepoint22", "Windward Passage": "chokepoint23",
    "Mona Passage": "chokepoint24", "Balabac Strait": "chokepoint25", "Bering Strait": "chokepoint26",
    "Mindoro Strait": "chokepoint27", "Kerch Strait": "chokepoint28",
}

# Straits the HMM is fitted on: busy ones only. The Cape of Good Hope is kept
# out of the pooled model because it gains traffic when the Red Sea loses it;
# it is reported separately as the diversion signal.
POOL = {n: CHOKEPOINTS[n] for n in (
    "Suez Canal", "Panama Canal", "Bosporus Strait", "Bab el-Mandeb Strait",
    "Malacca Strait", "Strait of Hormuz", "Gibraltar Strait", "Taiwan Strait")}
DIVERSION = {"Cape of Good Hope": CHOKEPOINTS["Cape of Good Hope"]}

# Short names people (and the routing network) use.
ALIASES = {
    "suez": "Suez Canal", "panama": "Panama Canal", "bosporus": "Bosporus Strait",
    "bosphorus": "Bosporus Strait", "istanbul strait": "Bosporus Strait", "turkish straits": "Bosporus Strait",
    "bab el-mandeb": "Bab el-Mandeb Strait", "bab-el-mandeb": "Bab el-Mandeb Strait",
    "bab al-mandab": "Bab el-Mandeb Strait", "red sea": "Bab el-Mandeb Strait",
    "malacca": "Malacca Strait", "singapore strait": "Malacca Strait",
    "hormuz": "Strait of Hormuz", "cape": "Cape of Good Hope", "cape of good hope": "Cape of Good Hope",
    "gibraltar": "Gibraltar Strait", "dover": "Dover Strait", "english channel": "Dover Strait",
    "taiwan": "Taiwan Strait", "lombok": "Lombok Strait", "sunda": "Sunda Strait",
}


def resolve(name):
    """PortWatch chokepoint name for a full name, alias or unique partial match, else None."""
    if not name:
        return None
    key = name.strip().lower()
    by_lower = {n.lower(): n for n in CHOKEPOINTS}
    if key in by_lower:
        return by_lower[key]
    if key in ALIASES:
        return ALIASES[key]
    hits = {n for low, n in by_lower.items() if key in low or low in key}
    hits |= {n for a, n in ALIASES.items() if a in key}
    return hits.pop() if len(hits) == 1 else None

_session = requests.Session()


def load(portid, max_age_hours=24):
    """Daily rows for one chokepoint since 2019, indexed by a complete daily calendar."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{portid}.csv"
    if not path.exists() or time.time() - path.stat().st_mtime > max_age_hours * 3600:
        rows, offset = [], 0
        while True:
            r = _session.get(URL, params={
                "f": "json", "where": f"portid='{portid}'", "outFields": ",".join(FIELDS),
                "orderByFields": "date ASC", "resultOffset": offset,
                "resultRecordCount": 2000, "returnGeometry": "false"}, timeout=60)
            r.raise_for_status()
            batch = [f["attributes"] for f in r.json().get("features", [])]
            rows += batch
            if len(batch) < 2000:
                break
            offset += len(batch)
        pd.DataFrame(rows).to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated()]
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    df.attrs["data_through"] = str(df.index.max().date())
    return df
