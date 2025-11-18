# download_ddragon_icons.py  (Python 3.8+)
import os, json, urllib.request, concurrent.futures

ROOT = os.path.join(os.getcwd(), "ddragon_icons")
LANG = "ko_KR"

def get(url):
    with urllib.request.urlopen(url) as r:
        return r.read()

def get_json(url):
    return json.loads(get(url).decode("utf-8"))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def download(url, out_path):
    if os.path.exists(out_path):
        return
    try:
        data = get(url)
        with open(out_path, "wb") as f:
            f.write(data)
    except Exception:
        pass

def main():
    versions = get_json("https://ddragon.leagueoflegends.com/api/versions.json")
    ver = versions[0]
    base_cdn = f"https://ddragon.leagueoflegends.com/cdn/{ver}"

    champ_json = get_json(f"{base_cdn}/data/{LANG}/champion.json")["data"]
    item_json  = get_json(f"{base_cdn}/data/{LANG}/item.json")["data"]

    out_champ = os.path.join(ROOT, ver, "champion")
    out_item  = os.path.join(ROOT, ver, "item")
    ensure_dir(out_champ); ensure_dir(out_item)

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for c in champ_json.values():
            cid = c["id"]
            url = f"{base_cdn}/img/champion/{cid}.png"
            path = os.path.join(out_champ, f"{cid}.png")
            tasks.append(ex.submit(download, url, path))
        for iid, meta in item_json.items():
            fname = meta.get("image", {}).get("full", f"{iid}.png")
            url = f"{base_cdn}/img/item/{fname}"
            path = os.path.join(out_item, f"{iid}.png")
            tasks.append(ex.submit(download, url, path))
        for t in concurrent.futures.as_completed(tasks):
            pass

    print(f"Done: {out_champ} / {out_item}")

if __name__ == "__main__":
    main()
