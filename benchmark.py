import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import pymupdf

from providers import (
    EINFRA_CONFIG,
    ModelOdpoved,
    OpenAICompatibleProvider,
    nacti_env,
    rozpoznaj_model,
    zavolej_poskytovatele,
)

nacti_env()

# Seznam všech modelů, které se mají otestovat
# Lokální modely (Ollama) se píšou přímo nebo s prefixem "ollama:"
# Cloudové modely z e-INFRA CZ se píšou s prefixem "einfra:" (seznam ID vypíše: python benchmark.py --list-einfra)
MODELY = [
    "minicpm-v:8b",
    #"gemma3:12b", too slow
    "qwen2.5vl:7b",
    "gemma4:e4b",
    "llava-phi3:latest",
    "minicpm-v:latest",
    "deepseek-r1:8b",
    "deepseek-r1:7b",
    "qwen3.5:4b",
    "qwen3.5:9b",
    "gemma3:4b",
    "qwen3.5:2b",
    "phi4-mini:latest",
    "granite3.2-vision:2b",
    # Modely z einfra
    "einfra:gemma4",
    "einfra:qwen3.8-27b",
    "einfra:qwen3.8-flash-next",
    "einfra:mistral-medium-3.5",
    "einfra:deepseek-v4.1-flash",
    "einfra:glm-5.3-flash",
    "einfra:kimi-k3",
    "einfra:gpt-oss-120b",
]

# 1. Definice schématu
class Akce(BaseModel):
    nazev: str = Field(description="Název folklórní nebo kulturní akce")
    datum: str = Field(description="Datum konání (např. 17.10.2026)")
    cas: Optional[str] = Field(description="Čas začátku HH:MM, pokud je uveden")
    misto: str = Field(description="Obec nebo místo konání v 1. pádě (nominativu)")

class VysledekExtrakce(BaseModel):
    nalezeno: bool = Field(description="True pokud vstup obsahuje alespoň jednu akci")
    akce: List[Akce]

# 2. Čištění HTML na čistý text
def vycisti_html(cesta: Path) -> str:
    soup = BeautifulSoup(cesta.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "svg", "nav", "footer", "head", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

# 3. Volání modelu přes vybraného poskytovatele (Ollama / e-INFRA API)
def zavolej_model(model: str, zprava: dict) -> ModelOdpoved:
    _, odpoved = zavolej_poskytovatele(model, zprava, VysledekExtrakce.model_json_schema())
    return odpoved


def _sestav_vysledek(
    sablona: dict,
    cas: float,
    vystupni_tokeny: int,
    vstupni_tokeny: int,
    dur: float,
    data: dict,
) -> dict:
    """Sestaví finální slovník výsledku včetně počtu vygenerovaných tokenů a rychlosti."""
    tps = round(vystupni_tokeny / dur, 1) if dur > 0 else 0
    return {
        **sablona,
        "uspech": True,
        "cas_s": round(cas, 2),
        "pocet_tokenu": vystupni_tokeny,
        "vstupni_tokeny": vstupni_tokeny,
        "tokens_per_sec": tps,
        "pocet_akci": len(data.get("akce", [])),
        "vystup": data,
    }


# 4. Hlavní testovací funkce
def otestuj_soubor(cesta: Path, model: str) -> dict:
    instrukce = (
        f"Jsi přesný extraktor dat. Zpracováváš dokument/stránku: '{cesta.stem}'.\n"
        "Vyextrahuj výhradně folklórní akce: hody, krojované plesy, fašanky, besedy u cimbálu, "
        "košty vín/pálenek, vystoupení dechových a cimbálových hudeb a folklórní festivaly. "
        "Ignoruj nefolklórní události (sport, dětské dny, divadla, brigády, volby, soukromé akce).\n"
        "Pravidla pro pole:\n"
        "- misto: Uveď název obce nebo města v 1. pádě (bez předložek 'v', 've', 'na'). "
        "Pokud u samotné akce není obec uvedena (nebo je tam jen obecné místo jako 'Kulturní dům' či 'Hřiště'), "
        "použij název obce z nadpisu stránky nebo z názvu dokumentu. Není-li obec ani tam, nastav null.\n"
        "- cas: Uveď pouze konkrétní čas ve formátu HH:MM. Není-li uveden číselně, nastav null. Nikdy sem nepiš popis akce.\n"
        "- datum: Přepiš datum z textu. Chybí-li rok, doplň rok z nadpisu nebo názvu dokumentu.\n"
        "Nikdy si nevymýšlej údaje, které ve vstupu nejsou. Vrať výsledek přesně podle JSON schématu."
    )

    try:
        poskytovatel, _ = rozpoznaj_model(model)
    except ValueError as e:
        return {"model": model, "soubor": cesta.name, "uspech": False, "chyba": str(e)}

    pripona = cesta.suffix.lower()
    vysledek_sablona = {
        "model": model,
        "poskytovatel": poskytovatel,
        "soubor": cesta.name,
        "uspech": False,
    }

    # A) Obrázky a B) HTML soubory
    if pripona in [".jpg", ".jpeg", ".png", ".webp", ".html", ".htm"]:
        if pripona in [".html", ".htm"]:
            cisty_text = vycisti_html(cesta)
            zprava = {"role": "user", "content": f"{instrukce}\n\nTEXT WEBU:\n{cisty_text[:12000]}"}
        else:
            zprava = {"role": "user", "content": instrukce, "images": [str(cesta)]}

        odp = zavolej_model(model, zprava)
        if "chyba" in odp.data:
            return {**vysledek_sablona, "chyba": odp.data["chyba"], "cas_s": round(odp.cas, 2)}

        return _sestav_vysledek(
            vysledek_sablona,
            odp.cas,
            odp.vystupni_tokeny,
            odp.vstupni_tokeny,
            odp.eval_dur,
            odp.data,
        )

    # C) PDF soubory
    elif pripona == ".pdf":
        doc = pymupdf.open(cesta)
        vsechny_akce = []
        celkovy_cas = 0.0
        celkem_vystup_tok = 0
        celkem_vstup_tok = 0
        celkem_dur = 0.0

        max_stran = len(doc)
        for i in range(max_stran):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=150)
            zprava = {
                "role": "user",
                "content": instrukce,
                "images": [pix.tobytes("png")],
            }

            odp = zavolej_model(model, zprava)
            if "chyba" in odp.data:
                doc.close()
                return {**vysledek_sablona, "chyba": odp.data["chyba"], "cas_s": round(celkovy_cas + odp.cas, 2)}

            celkovy_cas += odp.cas
            celkem_vystup_tok += odp.vystupni_tokeny
            celkem_vstup_tok += odp.vstupni_tokeny
            celkem_dur += odp.eval_dur
            vsechny_akce.extend(odp.data.get("akce", []))

        doc.close()
        vystup_data = {"nalezeno": len(vsechny_akce) > 0, "akce": vsechny_akce}
        return _sestav_vysledek(
            vysledek_sablona,
            celkovy_cas,
            celkem_vystup_tok,
            celkem_vstup_tok,
            celkem_dur,
            vystup_data,
        )

    else:
        return {**vysledek_sablona, "chyba": f"Nepodporovaný formát: {pripona}"}


def _vypis_einfra_modely() -> int:
    try:
        modely = OpenAICompatibleProvider(EINFRA_CONFIG).vypis_modely()
    except RuntimeError as e:
        print(f"Chyba: {e}", file=sys.stderr)
        return 1

    print("Dostupné modely na e-INFRA CZ (použij v MODELY s prefixem 'einfra:'):")
    for m in modely:
        print(f"  - einfra:{m}")
    return 0


def _parsuj_argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark extrakce folklórních akcí (Ollama + e-INFRA API)")
    parser.add_argument(
        "--list-einfra",
        action="store_true",
        help="Vypíše dostupné modely na e-INFRA CZ API a skončí.",
    )
    parser.add_argument(
        "--modely",
        nargs="+",
        help="Volitelný seznam modelů ke spuštění místo výchozího seznamu MODELY.",
    )
    return parser.parse_args()


# 5. Spuštění benchmarku pro všechny modely a soubory
if __name__ == "__main__":
    args = _parsuj_argumenty()
    if args.list_einfra:
        sys.exit(_vypis_einfra_modely())

    slozka = Path("test_data")
    slozka.mkdir(exist_ok=True)

    soubory = [s for s in slozka.iterdir() if s.is_file()]
    if not soubory:
        print("Vlož do složky 'test_data' soubory a spusť znovu.")
        sys.exit(0)

    vybrane_modely = args.modely or MODELY
    vysledky = []
    for model in vybrane_modely:
        print("\n====================================")
        print(f" Testuji model: {model}")
        print("====================================")
        for soubor in soubory:
            print(f"  -> Soubor: {soubor.name} ... ", end="", flush=True)
            res = otestuj_soubor(soubor, model)
            vysledky.append(res)
            if "chyba" in res:
                print(f"CHYBA: {res['chyba']} ({res.get('cas_s', 0)} s)")
            else:
                print(f"HOTOVO ({res['cas_s']} s, {res['pocet_tokenu']} tok, {res['tokens_per_sec']} tok/s) | Akcí: {res['pocet_akci']}")

    cas_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    vystupni_soubor = Path(f"vysledky_porovnani_{cas_stamp}.json")
    vystupni_soubor.write_text(json.dumps(vysledky, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKompletní výsledky uloženy do {vystupni_soubor}")