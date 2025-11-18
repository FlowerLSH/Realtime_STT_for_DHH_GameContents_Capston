import requests, csv, re

LOCALE = "en_US"
OUT_CSV = "champion_skills.csv"

def latest_ver():
    return requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=30).json()[0]

def split_variants(s):
    return [p.strip() for p in re.split(r"\s*/\s*", s) if p.strip()]

def load_all(ver, locale):
    url = f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/{locale}/championFull.json"
    return requests.get(url, timeout=60).json()["data"]

def main():
    ver = latest_ver()
    data = load_all(ver, LOCALE)

    champs = sorted((c["name"], c) for c in data.values())
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        for name, c in champs:
            p = split_variants(c.get("passive", {}).get("name", "")) if c.get("passive") else []
            spells = c.get("spells", [])
            q = split_variants(spells[0].get("name", "")) if len(spells) > 0 else []
            w = split_variants(spells[1].get("name", "")) if len(spells) > 1 else []
            e = split_variants(spells[2].get("name", "")) if len(spells) > 2 else []
            r = split_variants(spells[3].get("name", "")) if len(spells) > 3 else []
            row = [name] + p + q + w + e + r
            wr.writerow(row)

if __name__ == "__main__":
    main()
