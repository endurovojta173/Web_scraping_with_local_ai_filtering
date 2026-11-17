import base64
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
from ollama import chat as ollama_chat


@dataclass
class ModelOdpoved:
    """Jednotný výsledek volání lokálního i cloudového/API modelu."""

    data: dict
    cas: float
    vystupni_tokeny: int
    vstupni_tokeny: int
    eval_dur: float


@dataclass(frozen=True)
class OpenAICompatConfig:
    """Konfigurace poskytovatele kompatibilního s OpenAI Chat Completions API."""

    nazev: str
    default_base_url: str
    api_key_env_vars: tuple[str, ...]
    base_url_env_var: str
    timeout_env_var: str
    default_timeout: float = 180.0
    klic_napoveda: str = ""


EINFRA_CONFIG = OpenAICompatConfig(
    nazev="einfra",
    default_base_url="https://llm.ai.e-infra.cz/v1",
    api_key_env_vars=("EINFRA_AI_TOKEN", "EINFRA_API_KEY"),
    base_url_env_var="EINFRA_BASE_URL",
    timeout_env_var="EINFRA_TIMEOUT",
    default_timeout=180.0,
    klic_napoveda="Vygeneruj klíč na https://chat.ai.e-infra.cz (Nastavení -> Účet -> API klíče) a vlož ho do .env.",
)

OPENAI_COMPAT_CONFIGS: dict[str, OpenAICompatConfig] = {
    "einfra": EINFRA_CONFIG,
}

ZNAMI_POSKYTOVATELE = {"ollama", *OPENAI_COMPAT_CONFIGS.keys()}


def nacti_env(cesta: Optional[Path] = None) -> None:
    """Načte proměnné prostředí ze souboru .env, aniž by přepsal již nastavené hodnoty."""
    env_cesta = cesta or (Path(__file__).resolve().parent / ".env")
    if not env_cesta.is_file():
        return

    for radek in env_cesta.read_text(encoding="utf-8").splitlines():
        cisty = radek.strip()
        if not cisty or cisty.startswith("#"):
            continue
        if cisty.startswith("export "):
            cisty = cisty[7:].lstrip()
        if "=" not in cisty:
            continue

        klic, hodnota = cisty.split("=", 1)
        klic = klic.strip()
        hodnota = hodnota.strip()
        if len(hodnota) >= 2 and hodnota[0] == hodnota[-1] and hodnota[0] in ('"', "'"):
            hodnota = hodnota[1:-1]
        if klic:
            os.environ.setdefault(klic, hodnota)


def rozpoznaj_model(model_spec: str) -> tuple[str, str]:
    """Rozdělí specifikaci modelu na dvojici (poskytovatel, model_id).

    Příklady:
      - 'einfra:gpt-oss-120b' -> ('einfra', 'gpt-oss-120b')
      - 'einfra:qwen3:32b'    -> ('einfra', 'qwen3:32b')
      - 'qwen3.5:4b'          -> ('ollama', 'qwen3.5:4b')
      - 'ollama:qwen3.5:4b'   -> ('ollama', 'qwen3.5:4b')
    """
    if ":" in model_spec:
        prefix, zbytek = model_spec.split(":", 1)
        prefix_lower = prefix.lower()
        if prefix_lower in ZNAMI_POSKYTOVATELE:
            if not zbytek.strip():
                raise ValueError(f"Chybí název modelu za prefixem '{prefix}:'.")
            return prefix_lower, zbytek
    return "ollama", model_spec


def parsuj_json_odpoved(surovy: str) -> dict:
    """Převede surovou textovou odpověď modelu na dict; umí očistit <think> bloky a markdown fences."""
    text = surovy.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    ocisteny = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if ocisteny.startswith("```"):
        ocisteny = re.sub(r"^```(?:json)?\s*", "", ocisteny, flags=re.IGNORECASE)
        ocisteny = re.sub(r"\s*```$", "", ocisteny).strip()

    try:
        return json.loads(ocisteny)
    except json.JSONDecodeError:
        return {
            "nalezeno": False,
            "akce": [],
            "chyba": "JSON Decode Error",
            "surovy_vystup": surovy,
        }


def _je_chyba_multimodality(chyba_str: str) -> bool:
    low = chyba_str.lower()
    klicova_slova = (
        "multimodal",
        "images",
        "image_url",
        "image input",
        "vision",
        "support",
        "content must be a string",
        "does not allow images",
    )
    return any(slovo in low for slovo in klicova_slova)


def _obrazek_na_data_url(obrazek: str | Path | bytes | bytearray) -> str:
    if isinstance(obrazek, (bytes, bytearray)):
        mime = "image/png"
        data = bytes(obrazek)
    else:
        cesta = Path(obrazek)
        mime, _ = mimetypes.guess_type(cesta.name)
        mime = mime or "image/png"
        data = cesta.read_bytes()

    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _sestav_openai_zpravy(zprava: dict) -> list[dict[str, Any]]:
    obrazky = zprava.get("images") or []
    role = zprava.get("role", "user")
    text = zprava.get("content", "")
    if not obrazky:
        return [{"role": role, "content": text}]

    obsah: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for img in obrazky:
        obsah.append({"type": "image_url", "image_url": {"url": _obrazek_na_data_url(img)}})
    return [{"role": role, "content": obsah}]


def _extrahuj_text_odpovedi(odpoved_json: dict) -> str:
    choices = odpoved_json.get("choices") or []
    if not choices:
        return ""
    zprava = choices[0].get("message") or {}
    obsah = zprava.get("content")
    if isinstance(obsah, str):
        return obsah.strip()
    if isinstance(obsah, list):
        casti = [c.get("text", "") for c in obsah if isinstance(c, dict) and c.get("type") == "text"]
        return "\n".join(casti).strip()
    return ""


def _extrahuj_chybovou_zpravu(resp: httpx.Response) -> str:
    try:
        telo = resp.json()
        if isinstance(telo, dict):
            err = telo.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
            if telo.get("detail"):
                return str(telo["detail"])
            if err:
                return str(err)
    except ValueError:
        pass
    return resp.text.strip() or f"HTTP {resp.status_code}"


class OpenAICompatibleProvider:
    """Klient pro OpenAI-kompatibilní cloudové/API endpointy (např. e-INFRA CZ)."""

    def __init__(self, config: OpenAICompatConfig):
        self.config = config

    def _nacti_klic(self) -> Optional[str]:
        for env_var in self.config.api_key_env_vars:
            val = os.environ.get(env_var, "").strip()
            if val:
                return val
        return None

    def _base_url(self) -> str:
        url = os.environ.get(self.config.base_url_env_var, "").strip() or self.config.default_base_url
        return url.rstrip("/")

    def _timeout(self) -> float:
        raw = os.environ.get(self.config.timeout_env_var, "").strip()
        if not raw:
            return self.config.default_timeout
        try:
            return float(raw)
        except ValueError:
            return self.config.default_timeout

    def _hlavicky(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _chybejici_klic_zprava(self) -> str:
        env_names = " / ".join(self.config.api_key_env_vars)
        return f"Chybí API klíč ({env_names}). {self.config.klic_napoveda}".strip()

    def vypis_modely(self) -> list[str]:
        """Vrátí seznam dostupných ID modelů z endpointu /models."""
        api_key = self._nacti_klic()
        if not api_key:
            raise RuntimeError(self._chybejici_klic_zprava())

        url = f"{self._base_url()}/models"
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url, headers=self._hlavicky(api_key))
        except httpx.HTTPError as e:
            raise RuntimeError(f"Síťová chyba při načítání modelů: {e}") from e

        if resp.status_code >= 400:
            raise RuntimeError(f"Chyba při načítání modelů ({resp.status_code}): {_extrahuj_chybovou_zpravu(resp)}")

        try:
            telo = resp.json()
        except ValueError as e:
            raise RuntimeError("API nevrátilo platný JSON se seznamem modelů.") from e

        modely = [item["id"] for item in telo.get("data", []) if isinstance(item, dict) and "id" in item]
        return sorted(modely)

    def zavolej(self, model_id: str, zprava: dict, schema: dict) -> ModelOdpoved:
        api_key = self._nacti_klic()
        if not api_key:
            return ModelOdpoved(
                data={"chyba": self._chybejici_klic_zprava()},
                cas=0.0,
                vystupni_tokeny=0,
                vstupni_tokeny=0,
                eval_dur=0.0,
            )

        url = f"{self._base_url()}/chat/completions"
        hlavicky = self._hlavicky(api_key)
        openai_zpravy = _sestav_openai_zpravy(zprava)
        ma_obrazky = bool(zprava.get("images"))

        formaty: list[Optional[dict[str, Any]]] = [
            {
                "type": "json_schema",
                "json_schema": {"name": "VysledekExtrakce", "schema": schema},
            },
            {"type": "json_object"},
            None,
        ]

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=self._timeout()) as client:
                resp = None
                for idx, fmt in enumerate(formaty):
                    start = time.perf_counter()
                    payload: dict[str, Any] = {
                        "model": model_id,
                        "messages": openai_zpravy,
                        "temperature": 0.0,
                        "max_tokens": 4096,
                    }
                    if fmt is not None:
                        payload["response_format"] = fmt

                    resp = client.post(url, headers=hlavicky, json=payload)
                    if resp.status_code < 400:
                        break

                    err_msg = _extrahuj_chybovou_zpravu(resp)
                    je_schema_chyba = resp.status_code in (400, 422) and any(
                        w in err_msg.lower()
                        for w in ("response_format", "json_schema", "json_object", "structured", "schema")
                    )
                    if je_schema_chyba and idx < len(formaty) - 1:
                        continue
                    break
        except httpx.TimeoutException:
            cas = time.perf_counter() - start
            return ModelOdpoved({"chyba": f"Timeout API po {round(cas, 1)} s"}, cas, 0, 0, 0.0)
        except httpx.HTTPError as e:
            cas = time.perf_counter() - start
            return ModelOdpoved({"chyba": f"Síťová chyba API: {e}"}, cas, 0, 0, 0.0)

        cas = time.perf_counter() - start
        assert resp is not None

        if resp.status_code >= 400:
            err_msg = _extrahuj_chybovou_zpravu(resp)
            if resp.status_code in (401, 403):
                return ModelOdpoved({"chyba": f"Neplatný nebo zamítnutý API klíč ({resp.status_code}): {err_msg}"}, cas, 0, 0, 0.0)
            if resp.status_code == 404:
                return ModelOdpoved({"chyba": f"Model '{model_id}' nebyl na {self.config.nazev} nalezen: {err_msg}"}, cas, 0, 0, 0.0)
            if resp.status_code == 429:
                return ModelOdpoved({"chyba": f"Překročen rate limit API (429): {err_msg}"}, cas, 0, 0, 0.0)
            if ma_obrazky and _je_chyba_multimodality(err_msg):
                return ModelOdpoved({"chyba": "Nepodporuje multimodální vstup (neumí číst obrázky/PDF)"}, cas, 0, 0, 0.0)
            return ModelOdpoved({"chyba": f"Chyba API ({resp.status_code}): {err_msg}"}, cas, 0, 0, 0.0)

        try:
            odpoved_json = resp.json()
        except ValueError:
            return ModelOdpoved(
                {"chyba": "Neplatná JSON odpověď z API", "surovy_vystup": resp.text[:500]},
                cas,
                0,
                0,
                0.0,
            )

        surovy = _extrahuj_text_odpovedi(odpoved_json)
        data = parsuj_json_odpoved(surovy)

        usage = odpoved_json.get("usage") or {}
        vystupni_tokeny = int(usage.get("completion_tokens") or 0)
        vstupni_tokeny = int(usage.get("prompt_tokens") or 0)

        eval_dur_ns = odpoved_json.get("eval_duration")
        completion_time = usage.get("completion_time")
        if eval_dur_ns:
            eval_dur = float(eval_dur_ns) / 1e9
        elif completion_time:
            eval_dur = float(completion_time)
        else:
            eval_dur = cas

        return ModelOdpoved(
            data=data,
            cas=cas,
            vystupni_tokeny=vystupni_tokeny,
            vstupni_tokeny=vstupni_tokeny,
            eval_dur=eval_dur,
        )


def zavolej_ollama(model_id: str, zprava: dict, schema: dict) -> ModelOdpoved:
    """Zavolá lokální model přes Ollamu."""
    ma_obrazky = bool(zprava.get("images"))
    start = time.perf_counter()
    try:
        odpoved = ollama_chat(
            model=model_id,
            messages=[zprava],
            format=schema,
            think=False,
            options={
                "temperature": 0.0,
                "num_ctx": 8192,
                "num_predict": 4096,
            },
        )
    except Exception as e:
        cas = time.perf_counter() - start
        if ma_obrazky and _je_chyba_multimodality(str(e)):
            return ModelOdpoved(
                data={"chyba": "Nepodporuje multimodální vstup (neumí číst obrázky/PDF)"},
                cas=cas,
                vystupni_tokeny=0,
                vstupni_tokeny=0,
                eval_dur=0.0,
            )
        return ModelOdpoved(
            data={"chyba": f"Chyba modelu: {e}"},
            cas=cas,
            vystupni_tokeny=0,
            vstupni_tokeny=0,
            eval_dur=0.0,
        )

    cas = time.perf_counter() - start
    surovy = odpoved["message"]["content"].strip()
    data = parsuj_json_odpoved(surovy)

    eval_count = int(odpoved.get("eval_count") or 0)
    prompt_eval_count = int(odpoved.get("prompt_eval_count") or 0)
    eval_dur_raw = odpoved.get("eval_duration")
    eval_dur = (eval_dur_raw if eval_dur_raw is not None else 1) / 1e9

    return ModelOdpoved(
        data=data,
        cas=cas,
        vystupni_tokeny=eval_count,
        vstupni_tokeny=prompt_eval_count,
        eval_dur=eval_dur,
    )


def zavolej_poskytovatele(model_spec: str, zprava: dict, schema: dict) -> tuple[str, ModelOdpoved]:
    """Rozpozná poskytovatele z názvu modelu a provede volání."""
    poskytovatel, model_id = rozpoznaj_model(model_spec)
    if poskytovatel == "ollama":
        return poskytovatel, zavolej_ollama(model_id, zprava, schema)

    config = OPENAI_COMPAT_CONFIGS[poskytovatel]
    klient = OpenAICompatibleProvider(config)
    return poskytovatel, klient.zavolej(model_id, zprava, schema)
