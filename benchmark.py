import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from collections import defaultdict
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from ollama import chat
import pymupdf

# Seznam všech modelů, které se mají otestovat
MODELY = [
    "qwen3.5:9b",
    "qwen3.5:4b",
    "phi4-mini:latest",
    "gemma3:4b",
    # "minicpm-v:8b", # Přidej další modely podle potřeby
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

# 3. Pomocná funkce s odchycením chyb multimodality
def zavolej_model(model: str, zprava: dict) -> tuple[dict, float, int, float]:
    start = time.perf_counter()
    try:
        odpoved = chat(
            model=model,
            messages=[zprava],
            format=VysledekExtrakce.model_json_schema(),
            think=False,
            options={
                #Vypina kreativitu, aby model generoval přesně podle schématu
                "temperature": 0.0,
                #Nastavuje velikost kontextového okna (kolik textu ze zadání a dokumentu model najednou "přečte"
                "num_ctx": 8192,
                #Omezuje maximální možnou délku vygenerované odpovědi na 2048 tokenů.
                "num_predict": 4096
            }
        )
    except Exception as e:
        cas = time.perf_counter() - start
        chyba_str = str(e).lower()
        # Odchycení textových modelů na obrázkových vstupech
        if "multimodal" in chyba_str or "images" in chyba_str or "vision" in chyba_str:
            return {"chyba": "Nepodporuje multimodální vstup (neumí číst obrázky/PDF)"}, cas, 0, 0
        else:
            return {"chyba": f"Chyba modelu: {str(e)}"}, cas, 0, 0

    cas = time.perf_counter() - start
    surovy = odpoved["message"]["content"].strip()
    
    try:
        data = json.loads(surovy)
    except json.JSONDecodeError:
        data = {"nalezeno": False, "akce": [], "chyba": "JSON Decode Error", "surovy_vystup": surovy}

    eval_count = odpoved.get("eval_count", 0)
    eval_dur = odpoved.get("eval_duration", 1) / 1e9
    return data, cas, eval_count, eval_dur

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

    pripona = cesta.suffix.lower()
    vysledek_sablona = {"model": model, "soubor": cesta.name, "uspech": False}

    # A) Obrázky
    if pripona in [".jpg", ".jpeg", ".png", ".webp"]:
        zprava = {"role": "user", "content": instrukce, "images": [str(cesta)]}
        data, cas, tok, dur = zavolej_model(model, zprava)
        if "chyba" in data:
            return {**vysledek_sablona, "chyba": data["chyba"], "cas_s": round(cas, 2)}
            
        tps = round(tok / dur, 1) if dur > 0 else 0
        return {**vysledek_sablona, "uspech": True, "cas_s": round(cas, 2), "tokens_per_sec": tps, "pocet_akci": len(data.get("akce", [])), "vystup": data}

    # B) HTML soubory
    elif pripona in [".html", ".htm"]:
        cisty_text = vycisti_html(cesta)
        zprava = {"role": "user", "content": f"{instrukce}\n\nTEXT WEBU:\n{cisty_text[:12000]}"}
        data, cas, tok, dur = zavolej_model(model, zprava)
        if "chyba" in data:
            return {**vysledek_sablona, "chyba": data["chyba"], "cas_s": round(cas, 2)}