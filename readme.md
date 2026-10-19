# Extrakce folklórních akcí pomocí lokálních LLM (Bakalářská práce)

Tento repozitář obsahuje nástroje pro automatizovanou extrakci dat o kulturních a folklórních akcích z webových stránek (`.html`), plakátů (`.jpg`, `.png`) a dokumentů (`.pdf`) do strukturovaného formátu JSON pomocí lokálně běžících multimodálních modelů přes nástroj Ollama.

---

## 1. Požadavky na systém (Prerekvizity)

Před nastavením Python prostředí se ujisti, že máš v systému (Ubuntu/Debian) nainstalovanou **Ollamu** a balíček pro tvorbu virtuálního prostředí:

```bash
# 1. Instalace podpory pro Python virtuální prostředí
sudo apt update && sudo apt install -y python3-venv python3-pip

# 2. Instalace Ollamy (pokud ještě na novém PC není)
curl -fsSL https://ollama.com/install.sh | sh
```

*(Pro plné využití grafické karty NVIDIA doporučuji ověřit funkčnost ovladačů příkazem `nvidia-smi`).*

---

## 2. Stažení AI modelů

Stáhni do Ollamy testované modely (optimalizováno pro ~6 GB VRAM, např. RTX 3060):

```bash
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
ollama pull gemma3:4b
```

Kontrolu stažených modelů provedeš příkazem:
```bash
ollama list
```

---

## 3. Nastavení virtuálního prostředí (venv)

Otevři terminál ve složce s projektem (např. `cd ~/bakalarka`) a postupuj následovně:

```bash
# 1. Vytvoření virtuálního prostředí ve skryté složce .venv
python3 -m venv .venv

# 2. Aktivace virtuálního prostředí
source .venv/bin/activate

# 3. Aktualizace pipu a instalace potřebných knihoven
pip install --upgrade pip
pip install ollama pydantic beautifulsoup4 pymupdf
```
*(Alternativně můžeš použít `pip install -r requirements.txt`, pokud máš soubor ve složce).*

> **Tip:** Virtuální prostředí poznáš tak, že máš na začátku řádku v terminálu napsáno `(.venv)`. Při každém novém otevření terminálu ho stačí znovu zapnout přes `source .venv/bin/activate`.

---

## 4. Struktura projektu a spuštění

Ujisti se, že máš soubory ve složce uspořádané takto:

```text
bakalarka/
├── .venv/                   # Virtuální prostředí Pythonu
├── test_data/               # Složka se vstupními soubory (.pdf, .html, .jpg, .png)
├── benchmark.py             # Testovací a porovnávací skript
├── vysledky_porovnani.json  # Výstupní soubor (vytvoří se automaticky po běhu)
├── requirements.txt         # Seznam závislostí
└── README.md                # Tento návod
```

### Spuštění benchmarku:
```bash
source .venv/bin/activate
python benchmark.py
```

---

## 5. Diagnostika a sledování výkonu (pro účely měření v BP)

Během běhu skriptu můžeš v druhém okně terminálu sledovat zatížení hardwaru:

* **Rozdělení modelu mezi GPU a CPU:**
  ```bash
  ollama ps
  ```
  *(Sleduj sloupec `PROCESSOR` – na RTX 3060 6GB by měly modely `4b` ukazovat `100% GPU`).*

* **Sledování využití VRAM v reálném čase:**
  ```bash
  watch -n 1 nvidia-smi
  ```
