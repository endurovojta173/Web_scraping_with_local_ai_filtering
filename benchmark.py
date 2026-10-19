import time
import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from ollama import chat
import pymupdf

# 1. Definice schématu
class Akce(BaseModel):
    nazev: str = Field(description="Název folklórní nebo kulturní akce")
    datum: str = Field(description="Datum konání (např. 17.10.2026)")
    cas: Optional[str] = Field(description="Čas začátku HH:MM, pokud je uveden")
    misto: str = Field(description="Obec nebo místo konání v 1. pádě (nominativu)")

class VysledekExtrakce(BaseModel):
    nalezeno: bool = Field(description="True pokud vstup obsahuje alespoň jednu akci")
    akce: List[Akce]

# 2. Čištění HTML na čistý text (odstraní kompletně Elementor/HTML balast)
def vycisti_html(cesta: Path) -> str:
    soup = BeautifulSoup(cesta.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "svg", "nav", "footer", "head", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

# 3. Pomocná funkce pro jedno volání modelu (s ošetřením textových modelů)
def zavolej_model(model: str, zprava: dict) -> tuple[dict, float, int, float]:
    start = time.perf_counter()
    try:
        odpoved = chat(
            model=model,
            messages=[zprava],
            format=VysledekExtrakce.model_json_schema(),
            think=False,
            options={
                "temperature": 0.0,
                "num_ctx": 8192,
                "num_predict": 2048
            }
        )
    except Exception as e:
        # Pokud model nepodporuje obrázky a zpráva je obsahuje, pošleme prázdný výsledek / náhradní text
        if "does not support multimodal" in str(e) and "images" in zprava:
            zprava = {"role": "user", "content": f"{zprava['content']}\n(Pozn: Stránka neobsahuje textovou vrstvu a tento model neumí číst obrázky)."}
            odpoved = chat(
                model=model,
                messages=[zprava],
                format=VysledekExtrakce.model_json_schema(),
                think=False,
                options={
                    "temperature": 0.0,
                    "num_ctx": 8192,
                    "num_predict": 2048
                }
            )
        else:
            raise e

    cas = time.perf_counter() - start
    surovy = odpoved["message"]["content"].strip()
    
    try:
        data = json.loads(surovy)
    except json.JSONDecodeError:
        data = {"nalezeno": False, "akce": [], "chyba_parsovani": surovy}

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

    # A) Obrázky
    if pripona in [".jpg", ".jpeg", ".png", ".webp"]:
        zprava = {"role": "user", "content": instrukce, "images": [str(cesta)]}
        data, cas, tok, dur = zavolej_model(model, zprava)
        tps = round(tok / dur, 1) if dur > 0 else 0
        return {"model": model, "soubor": cesta.name, "cas_s": round(cas, 2), "tokens_per_sec": tps, "pocet_akci": len(data.get("akce", [])), "vystup": data}

    # B) HTML soubory (čistý text)
    elif pripona in [".html", ".htm"]:
        cisty_text = vycisti_html(cesta)
        zprava = {"role": "user", "content": f"{instrukce}\n\nTEXT WEBU:\n{cisty_text[:12000]}"}
        data, cas, tok, dur = zavolej_model(model, zprava)
        tps = round(tok / dur, 1) if dur > 0 else 0
        return {"model": model, "soubor": cesta.name, "cas_s": round(cas, 2), "tokens_per_sec": tps, "pocet_akci": len(data.get("akce", [])), "vystup": data}

    # C) PDF soubory (stránka po stránce, aby nedošlo k přetečení limitu tokenů)
    elif pripona == ".pdf":
        doc = pymupdf.open(cesta)
        vsechny_akce = []
        celkovy_cas = 0.0
        celkem_tok = 0
        celkem_dur = 0.0

        # Pro rychlý benchmark omezíme na prvních 5 stran (pro celé PDF změň na len(doc))
        max_stran = min(len(doc), 5)
        for i in range(max_stran):
            page = doc.load_page(i)
            text_stranky = page.get_text("text").strip()
            print(f"  -> Zpracovávám stranu {i+1}/{max_stran}...")

            if len(text_stranky) > 50:
                zprava = {"role": "user", "content": f"{instrukce}\n\nTEXT STRÁNKY:\n{text_stranky}"}
            else:
                pix = page.get_pixmap(dpi=120)
                zprava = {"role": "user", "content": instrukce, "images": [pix.tobytes("png")]}

            data, cas, tok, dur = zavolej_model(model, zprava)
            celkovy_cas += cas
            celkem_tok += tok
            celkem_dur += dur
            vsechny_akce.extend(data.get("akce", []))

        doc.close()
        tps = round(celkem_tok / celkem_dur, 1) if celkem_dur > 0 else 0
        vystup_data = {"nalezeno": len(vsechny_akce) > 0, "akce": vsechny_akce}
        return {"model": model, "soubor": cesta.name, "cas_s": round(celkovy_cas, 2), "tokens_per_sec": tps, "pocet_akci": len(vsechny_akce), "vystup": vystup_data}

    else:
        return {"error": f"Nepodporovaný formát: {pripona}"}

# 5. Spuštění porovnání
if __name__ == "__main__":
    # Pokud už nechceš čekat na pomalý 9B model, nech v seznamu jen ["qwen3.5:4b"]
    modely = ["qwen3.5:2b","qwen3.5:4b", "qwen3.5:9b", "gemma3:4b","minicpm-v:8b", "phi4-mini", "granite3.2-vision:2b" ]
    slozka = Path("test_data")
    slozka.mkdir(exist_ok=True)
    
    soubory = [s for s in slozka.iterdir() if s.is_file()]
    if not soubory:
        print("Vlož do složky 'test_data' soubory a spusť znovu.")
        exit()

    vysledky = []
    for soubor in soubory:
        print(f"\n=== Testuji soubor: {soubor.name} ===")
        for m in modely:
            print(f"Běží {m}...")
            res = otestuj_soubor(soubor, m)
            if "error" in res:
                print(f"Přeskočeno: {res['error']}")
                continue
            vysledky.append(res)
            print(f"Hotovo za {res['cas_s']} s ({res['tokens_per_sec']} tok/s) | Nalezeno akcí: {res['pocet_akci']}")

    Path("vysledky_porovnani.json").write_text(
        json.dumps(vysledky, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nVýsledky uloženy do vysledky_porovnani.json")