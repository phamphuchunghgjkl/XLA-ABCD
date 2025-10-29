import csv
import sys
from pathlib import Path

CUTOFF = 25.5
INPUT_CSV = Path("thisinh.csv")   
OUTPUT_TXT = Path("ketqua.txt") 

def normalize_name(s: str) -> str:
    tokens = [t for t in s.strip().split() if t]
    return " ".join([t.lower().capitalize() for t in tokens])

def priority_points(region: str, ethnicity: str) -> float:
    p = 0.0
    reg = str(region).strip()
    if reg == "1":
        p += 1.5
    elif reg == "2":
        p += 1.0
    if str(ethnicity).strip().lower() != "kinh":
        p += 1.5
    return p

def main(input_path: Path = INPUT_CSV, output_path: Path = OUTPUT_TXT) -> int:
    if not input_path.exists():
        print(f"Lỗi: Không tìm thấy file '{input_path}'.", file=sys.stderr)
        return 1

    candidates = []
    with input_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Lỗi: CSV không có header.", file=sys.stderr)
            return 1
        expected = ["name", "score", "ethnicity", "region"]
        header_lower = [h.strip().lower() for h in reader.fieldnames]
        for h in expected:
            if h not in header_lower:
                print("Lỗi: Header phải gồm name,score,ethnicity,region.", file=sys.stderr)
                return 1

        idx = 1
        for row in reader:
            name_raw = row.get("name", "")
            score_raw = row.get("score", "0")
            ethnicity = row.get("ethnicity", "")
            region = row.get("region", "")
            try:
                base_score = float(score_raw)
            except Exception:
                base_score = 0.0
            total = base_score + priority_points(region, ethnicity)
            code = f"TS{idx:02d}"
            candidates.append({
                "code": code,
                "name": normalize_name(name_raw),
                "total": round(total + 1e-8, 1),  
                "status": "Do" if total >= CUTOFF else "Truot",
            })
            idx += 1
    candidates.sort(key=lambda x: (-x["total"], x["code"]))

    with output_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            line = f'{c["code"]} {c["name"]} {c["total"]:.1f} {c["status"]}'
            print(line)
            f.write(line + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
