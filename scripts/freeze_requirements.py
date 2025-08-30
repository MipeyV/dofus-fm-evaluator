# scripts/freeze_requirements.py
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    req_file = root / "requirements.txt"

    print(f"[INFO] Génération de {req_file} ...")

    # Appelle pip freeze
    result = subprocess.run(
        ["pip", "freeze"],
        capture_output=True,
        text=True,
        check=True
    )

    # Sauvegarde dans requirements.txt en UTF-8
    req_file.write_text(result.stdout, encoding="utf-8")
    print("[OK] requirements.txt mis à jour en UTF-8.")

if __name__ == "__main__":
    main()