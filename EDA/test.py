# %% [markdown]
# # iNaturalist Observations (project 41347) → CSV
# Filtered to:
# - quality_grade=research
# - taxon_id=116999
# - per_page=200
# Includes 403/429-safe fetching + resume checkpoints, and exports:
# - observations CSV (one row per observation)
# - photos CSV (one row per photo)
#
# NOTE: checkpoint filenames include the filters so they won't collide with other runs.

# %%
import os
import json
import math
import time
import random
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests
import pandas as pd
from tqdm.auto import tqdm


# %%
BASE_OBS_URL = (
    "https://api.inaturalist.org/v1/observations"
    "?project_id=41347"
    "&quality_grade=research"
    "&per_page=200"
    "&fields=id%2Curi%2Cofvs%2Ctaxon%2Cphotos%2Cgeojson%2Clocation%2Cpositional_accuracy%2Cplace_country_name%2Cplace_state_name%2Cplace_county_name%2Cplace_town_name"
    "&photo_licensed=true"
    "&licensed=true"
)

# Output filenames (include filters)
OUT_CSV = "project_41347_research_taxon116999_observations.csv"
OUT_PHOTOS_CSV = "project_41347_research_taxon116999_photos.csv"

# Checkpoints (include filters)
CHECKPOINT_JSONL = "inat_41347_research_taxon116999_obs.jsonl"
CHECKPOINT_META  = "inat_41347_research_taxon116999_obs.meta.json"

# Browser-like UA helps reduce bot blocks
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


# %%
def set_query_params(url: str, **params) -> str:
    u = urlparse(url)
    q = parse_qs(u.query, keep_blank_values=True)
    for k, v in params.items():
        q[k] = [str(v)]
    new_query = urlencode(q, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))


class RateLimiter:
    def __init__(self, requests_per_minute=35):
        self.min_interval = 60.0 / float(requests_per_minute)
        self._last = 0.0

    def wait(self):
        now = time.time()
        sleep_for = self.min_interval - (now - self._last)
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last = time.time()


def request_json_with_retries(
    session: requests.Session,
    url: str,
    limiter: RateLimiter | None = None,
    timeout=60,
    max_retries=10,
):
    last_err = None
    for attempt in range(max_retries):
        try:
            if limiter:
                limiter.wait()

            r = session.get(url, timeout=timeout)

            if 200 <= r.status_code < 300:
                return r.json()

            if r.status_code in (403, 429):
                retry_after = r.headers.get("Retry-After")
                if retry_after and str(retry_after).isdigit():
                    wait = int(retry_after)
                else:
                    wait = min(300, (2 ** attempt) * 2) + random.uniform(0, 1.5)
                time.sleep(wait)
                continue

            r.raise_for_status()

        except Exception as e:
            last_err = e
            wait = min(120, (2 ** attempt)) + random.uniform(0, 1.0)
            time.sleep(wait)

    raise RuntimeError(f"Failed after retries for URL: {url}\nLast error: {last_err}")


def fetch_all_observations(
    base_url: str,
    requests_per_minute: int = 35,
    checkpoint_jsonl: str = CHECKPOINT_JSONL,
    checkpoint_meta: str = CHECKPOINT_META,
    resume: bool = True,
    max_pages: int | None = None,
):
    limiter = RateLimiter(requests_per_minute=requests_per_minute)

    results = []
    seen_ids = set()
    start_page = 1

    if resume and os.path.exists(checkpoint_jsonl):
        with open(checkpoint_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                oid = obj.get("id")
                if oid is not None and oid not in seen_ids:
                    results.append(obj)
                    seen_ids.add(oid)

        if os.path.exists(checkpoint_meta):
            meta = json.load(open(checkpoint_meta, "r", encoding="utf-8"))
            start_page = int(meta.get("last_page", 0)) + 1

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

        # page 1 determines totals
        first_url = set_query_params(base_url, page=1)
        first = request_json_with_retries(session, first_url, limiter=limiter)

        total = int(first.get("total_results", 0))
        per_page = int(first.get("per_page", 200)) or 200
        total_pages = max(1, math.ceil(total / per_page))

        if max_pages is not None:
            total_pages = min(total_pages, int(max_pages))

        if start_page > total_pages:
            return results

        if not resume:
            results, seen_ids = [], set()
            start_page = 1
            if os.path.exists(checkpoint_jsonl):
                os.remove(checkpoint_jsonl)
            if os.path.exists(checkpoint_meta):
                os.remove(checkpoint_meta)

        def append_checkpoint(batch):
            if not batch:
                return
            with open(checkpoint_jsonl, "a", encoding="utf-8") as f:
                for obj in batch:
                    oid = obj.get("id")
                    if oid is not None and oid in seen_ids:
                        continue
                    f.write(json.dumps(obj) + "\n")
                    seen_ids.add(oid)
                    results.append(obj)

        if start_page == 1:
            batch = first.get("results", []) or []
            append_checkpoint(batch)
            json.dump(
                {"last_page": 1, "total_pages": total_pages, "total_results": total},
                open(checkpoint_meta, "w", encoding="utf-8"),
                indent=2,
            )
            page_iter_start = 2
        else:
            page_iter_start = start_page

        for page in tqdm(range(page_iter_start, total_pages + 1), desc="Fetching observation pages"):
            page_url = set_query_params(base_url, page=page)
            data = request_json_with_retries(session, page_url, limiter=limiter)
            batch = data.get("results", []) or []
            if not batch:
                break

            append_checkpoint(batch)
            json.dump(
                {"last_page": page, "total_pages": total_pages, "total_results": total},
                open(checkpoint_meta, "w", encoding="utf-8"),
                indent=2,
            )

    return results


# %%
def _norm(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = s.replace('"', "").replace("’", "'")
    s = re.sub(r"[?]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_ofv(ofvs, must_contain_keywords):
    kws = [_norm(k) for k in must_contain_keywords]
    for ofv in (ofvs or []):
        name = _norm(ofv.get("name", ""))
        if all(k in name for k in kws):
            return {"field_id": ofv.get("field_id"), "name": ofv.get("name"), "value": ofv.get("value")}
    return {"field_id": None, "name": None, "value": None}


def extract_photo_urls(photos):
    urls = []
    for p in (photos or []):
        if not isinstance(p, dict):
            continue
        for k, v in p.items():
            if isinstance(v, str) and "url" in k.lower():
                urls.append(v)
    seen, uniq = set(), []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def photo_rows_for_obs(obs_id, photos):
    rows = []
    for p in (photos or []):
        if not isinstance(p, dict):
            continue
        rows.append(
            {
                "observation_id": obs_id,
                "photo_id": p.get("id"),
                "license_code": p.get("license_code"),
                "url": p.get("url"),
                "square_url": p.get("square_url"),
                "small_url": p.get("small_url"),
                "medium_url": p.get("medium_url"),
                "large_url": p.get("large_url"),
                "original_url": p.get("original_url"),
            }
        )
    return rows


def taxon_path_ids(taxon: dict):
    if not isinstance(taxon, dict):
        return []
    ids = []
    anc = taxon.get("ancestor_ids")
    if isinstance(anc, list) and anc:
        ids.extend([int(x) for x in anc if x is not None])
    else:
        ancestry = taxon.get("ancestry")
        if isinstance(ancestry, str) and ancestry.strip():
            for part in ancestry.split("/"):
                if part.strip().isdigit():
                    ids.append(int(part.strip()))
    tid = taxon.get("id")
    if tid is not None:
        try:
            ids.append(int(tid))
        except Exception:
            pass
    seen, out = set(), []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# %%
TAXA_ENDPOINT = "https://api.inaturalist.org/v1/taxa"


def fetch_taxa_lookup(taxon_ids, chunk_size=500, requests_per_minute=35):
    taxon_ids = sorted({int(x) for x in taxon_ids if x is not None})
    lookup = {}
    if not taxon_ids:
        return lookup

    limiter = RateLimiter(requests_per_minute=requests_per_minute)

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

        for i in tqdm(range(0, len(taxon_ids), chunk_size), desc="Fetching taxa for taxonomy ranks"):
            chunk = taxon_ids[i : i + chunk_size]
            ids_str = ",".join(map(str, chunk))

            url1 = set_query_params(TAXA_ENDPOINT, id=ids_str, per_page=min(chunk_size, 500))
            try:
                data = request_json_with_retries(session, url1, limiter=limiter)
            except Exception:
                url2 = set_query_params(TAXA_ENDPOINT, ids=ids_str, per_page=min(chunk_size, 500))
                data = request_json_with_retries(session, url2, limiter=limiter)

            for t in (data.get("results", []) or []):
                tid = t.get("id")
                if tid is not None:
                    lookup[int(tid)] = t

    return lookup


def ranks_from_path(path_ids, taxa_lookup):
    rank_map = {}
    for tid in (path_ids or []):
        t = taxa_lookup.get(int(tid))
        if not t:
            continue
        r = t.get("rank")
        n = t.get("name")
        if r and n and r not in rank_map:
            rank_map[r] = {"id": int(tid), "name": n}
    return rank_map


# %%
# 1) Fetch observations (filtered) — resumable + 403 safe
obs_results = fetch_all_observations(
    BASE_OBS_URL,
    requests_per_minute=35,  # if you still see 403, try 25
    resume=True,
    checkpoint_jsonl=CHECKPOINT_JSONL,
    checkpoint_meta=CHECKPOINT_META,
)

print("Total observations fetched:", len(obs_results))

# %%
# 2) Collect all taxon IDs needed for lineage ranks
all_taxon_ids = set()
for obs in obs_results:
    for tid in taxon_path_ids(obs.get("taxon")):
        all_taxon_ids.add(tid)

print("Unique taxon IDs to fetch:", len(all_taxon_ids))

# %%
# 3) Fetch taxa lookup
taxa_lookup = fetch_taxa_lookup(all_taxon_ids, chunk_size=500, requests_per_minute=35)
print("Taxa fetched into lookup:", len(taxa_lookup))

# %%
# 4) Flatten to CSV
FIELD_ID_MEANT = ["id meant for", "organism being eaten"]
FIELD_SPECIAL_FEED = ["special types of feeding"]
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

rows, photo_rows = [], []

for obs in tqdm(obs_results, desc="Flattening observations"):
    obs_id = obs.get("id")
    taxon = obs.get("taxon") or {}

    ofv1 = find_ofv(obs.get("ofvs"), FIELD_ID_MEANT)
    ofv2 = find_ofv(obs.get("ofvs"), FIELD_SPECIAL_FEED)

    urls = extract_photo_urls(obs.get("photos"))
    photo_rows.extend(photo_rows_for_obs(obs_id, obs.get("photos")))

    rank_map = ranks_from_path(taxon_path_ids(taxon), taxa_lookup)

    row = {
        "observation_id": obs_id,
        "uri": obs.get("uri"),
        "quality_grade": obs.get("quality_grade"),
        "license_code": obs.get("license_code"),

        "taxon_id": taxon.get("id"),
        "taxon_name": taxon.get("name"),
        "taxon_rank": taxon.get("rank"),
        "taxon_preferred_common_name": taxon.get("preferred_common_name"),

        "ofv_id_meant_field_id": ofv1["field_id"],
        "ofv_id_meant_value": ofv1["value"],
        "ofv_special_feeding_field_id": ofv2["field_id"],
        "ofv_special_feeding_value": ofv2["value"],

        "photo_urls": "|".join(urls),
        "photo_count": len(obs.get("photos") or []),

        "location": obs.get("location"),
        "positional_accuracy": obs.get("positional_accuracy"),
        "place_country_name": obs.get("place_country_name"),
        "place_state_name": obs.get("place_state_name"),
        "place_county_name": obs.get("place_county_name"),
        "place_town_name": obs.get("place_town_name"),
    }

    for r in RANKS:
        row[f"{r}_id"] = rank_map.get(r, {}).get("id")
        row[f"{r}_name"] = rank_map.get(r, {}).get("name")

    rows.append(row)

df = pd.DataFrame(rows)
df_photos = pd.DataFrame(photo_rows)

print("Obs DF:", df.shape, "Photos DF:", df_photos.shape)
df.head(3)

# %%
# 5) Save CSVs
df.to_csv(OUT_CSV, index=False)
df_photos.to_csv(OUT_PHOTOS_CSV, index=False)

print("Saved:", OUT_CSV)
print("Saved:", OUT_PHOTOS_CSV)

# %%
# Quick checks
if "kingdom_name" in df.columns:
    print("Missing kingdom_name:", df["kingdom_name"].isna().mean())
print("Missing ofv_id_meant_value:", df["ofv_id_meant_value"].isna().mean())
print("Missing photo_urls:", (df["photo_urls"].fillna("") == "").mean())
# %%
