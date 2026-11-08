"""World ports. Uses the ~3,700 ports bundled with the `searoute` package
(UN/LOCODE codes + coordinates). Swap in another source later if needed."""
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=1)
def load_ports() -> pd.DataFrame:
    from searoute.data.ports_dict import node_list

    rows = []
    for (lon, lat), props in node_list.items():
        rows.append({
            "locode": props.get("port", ""),
            "name": (props.get("name") or "").strip(),
            "country": (props.get("cty") or "").replace("_", " ").strip().title(),
            "lat": lat,
            "lon": lon,
        })
    df = pd.DataFrame(rows).drop_duplicates("locode").sort_values(["name", "country"])
    df["label"] = df["name"] + ", " + df["country"] + " (" + df["locode"] + ")"
    return df.reset_index(drop=True)


def get_port(locode: str) -> dict:
    df = load_ports()
    row = df[df["locode"] == locode]
    if row.empty:
        raise KeyError(f"Unknown port {locode}")
    return row.iloc[0].to_dict()
