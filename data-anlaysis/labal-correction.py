import argparse
import os
import shutil

def fix_labels(root, from_label, to_label, backup=True, ext=".txt", dry_run=False):
    root = os.path.abspath(root)
    print(f"Scanning folder: {root}")
    changed_files = []
    for dirpath, _, filenames in os.walk(root):
        print(f"Scanning folder: {dirpath}")
        for fname in filenames:
            if not fname.lower().endswith(ext):
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except Exception:
                continue
            new_lines = []
            changed = False
            for line in lines:
                s = line.rstrip("\r\n")
                if not s.strip():
                    new_lines.append(s)
                    continue
                parts = s.split()
                try:
                    cls_val = int(float(parts[0]))
                except Exception:
                    new_lines.append(s)
                    continue
                if cls_val == from_label:
                    parts[0] = str(to_label)
                    changed = True
                new_lines.append(" ".join(parts))
            if changed:
                changed_files.append(path)
                if not dry_run:
                    if backup:
                        bak = path + ".bak"
                        if not os.path.exists(bak):
                            shutil.copy2(path, bak)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("\n".join(new_lines) + ("\n" if new_lines and not new_lines[-1].endswith("\n") else ""))
    return changed_files

def main():
    p = argparse.ArgumentParser(description="Replace YOLO label class indices (e.g. change 4 -> 3).")
    p.add_argument("--root", "-r", default=".", help="Root folder to scan (recursively).")
    p.add_argument("--from-label", type=int, default=4, help="Class index to replace.")
    p.add_argument("--to-label", type=int, default=3, help="Class index to replace with.")
    p.add_argument("--no-backup", action="store_true", help="Don't create .bak backups.")
    p.add_argument("--ext", default=".txt", help="Label file extension (default .txt).")
    p.add_argument("--dry-run", action="store_true", help="Only list files that would be changed.")
    args = p.parse_args()
    changed = fix_labels(args.root, args.from_label, args.to_label, backup=not args.no_backup, ext=args.ext, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Dry-run: {len(changed)} file(s) would be updated.")
    else:
        print(f"Updated {len(changed)} file(s).")
    for pth in changed:
        print("  " + pth)

if __name__ == "__main__":
    main()