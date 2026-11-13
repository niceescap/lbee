# lbee — harnais LLM ultra-minimaliste

`lbee` est un service **purement backend** (aucune UI) qui expose un endpoint
`POST /chat` compatible avec n'importe quelle application cliente. Il :

- appelle un modèle **OpenAI-compatible** (OpenAI, Groq, OpenRouter, vLLM,
  Ollama, LM Studio…) via le SDK officiel `openai` ;
- gère trois modes de **mémoire de session** (`one-shot`, `personal`,
  `shared-compressed`) ;
- exécute des **outils Python** écrits par vos soins (function calling) ;
- sait répondre en **streaming SSE** ;
- tient dans une image `python:3.12-slim` non-root, sans base de données ni
  framework lourd.

Aucune clé API n'est stockée dans `config.yml` ni dans l'image : elle est lue
dans la variable d'environnement `API_KEY`.

---

## 1. Démarrage en 3 commandes

```bash
git clone https://github.com/niceescap/lbee.git
cd lbee
cp .env.example .env      # puis éditez API_KEY
docker compose up -d --build
```

Vérification :

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Bonjour, qui es-tu ?","session_id":"demo"}'
```

---

## 2. Architecture

```
.
├── Dockerfile            # image minimale, non-root (UID/GID 1000)
├── docker-compose.yml    # env_file + volumes montés
├── config.yml            # configuration unique, commentée
├── system_prompt.md      # prompt système
├── requirements.txt
├── .env.example          # API_KEY uniquement
├── app/
│   ├── config.py         # chargement/validation Pydantic
│   ├── memory.py         # 3 modes de mémoire (thread-safe)
│   ├── harness.py        # chargement des outils + boucle agentique
│   ├── state.py          # invariants + singletons
│   └── main.py           # FastAPI : /chat, /health, /memory/reset
├── class-tools/
│   └── example_tool.py   # outil d'exemple (calculatrice)
└── kb/                   # persistance (montée en volume)
```

Flux d'un appel à `POST /chat` :

1. `app/config.py` charge `config.yml` au démarrage (Pydantic strict :
   toute clé inconnue est refusée).
2. `app/harness.py` construit le contexte : `system_prompt.md` → mémoire de
   session → message utilisateur.
3. Si les outils sont activés, boucle `chat.completions` jusqu'à
   `max_tool_iterations` : chaque `tool_calls` est exécuté, résultat renvoyé au
   modèle, jusqu'à une réponse finale sans appel d'outil.
4. La mémoire est mise à jour, puis (mode `shared-compressed`) compressée si le
   seuil est franchi.
5. `app/main.py` renvoie du JSON ou un flux SSE.

---

## 3. Configuration (`config.yml`)

| Section | Clés | Rôle |
|---|---|---|
| `llm` | `base_url`, `model`, `timeout`, `max_retries` | client OpenAI-compatible |
| `generation` | `temperature`, `top_p`, `max_tokens`, `extra_body` | paramètres envoyés |
| `memory` | `mode`, `max_history_messages`, `compression_threshold`, `keep_last_messages_after_compression` | politique de mémoire |
| `features` | `enable_tools`, `max_tool_iterations`, `enable_streaming` | interrupteurs |
| `paths` | `system_prompt`, `tools_dir`, `kb_dir` | emplacements (relatifs à `/app`) |

> **Piège `top_k`** : `top_k`, `repetition_penalty`, `min_p`… ne sont pas des
> paramètres OpenAI. Les mettre au premier niveau provoque un
> `400 Bad Request: Unrecognized request argument supplied: top_k`.
> Ils doivent passer par `generation.extra_body` (voir l'exemple commenté).

### Mémoire

| Mode | Comportement | Stockage |
|---|---|---|
| `one-shot` | aucun historique entre les appels | — |
| `personal` | historique par `session_id` | `kb/sessions/<id>.json` |
| `shared-compressed` | historique global unique, résumé automatiquement par le LLM au-delà de `compression_threshold` caractères | `kb/shared_memory.json` |

Le harnais sérialise les tours de conversation (`asyncio.Lock`) et écrit les
fichiers de façon atomique (fichier temporaire + `os.replace`), ce qui rend le
service sûr en multi-worker léger et après un crash.

---

## 4. Écrire un outil

Déposez un fichier dans `class-tools/` (monté en lecture seule). Le contrat est
minimal :

```python
# class-tools/weather.py
from typing import Any, Dict

DESCRIPTION = "Météo factice."

SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Renvoie la météo d'une ville.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

def execute(city: str) -> Dict[str, Any]:
    if not city:
        raise ValueError("city est requis")
    return {"city": city, "temperature_c": 21, "sky": "ensoleillé"}
```

Règles :

- `SCHEMA` doit être un schéma OpenAI **function calling** valide
  (`{"type": "function", "function": {...}}`) ;
- `execute(**kwargs)` peut être synchrone ou `async` ;
- les arguments superflus sont filtrés selon la signature réelle ;
- une exception est renvoyée au modèle comme message d'outil d'erreur
  (`TOOL_ERROR: ...`), la conversation continue ;
- toute erreur de chargement d'un outil est loguée sans tuer le service.

Après ajout, redémarrez le conteneur (`docker compose restart`) : les outils
sont découverts au démarrage et listés dans `GET /health`.

---

## 5. API

### `GET /health`

```json
{"status":"ok","model":"gpt-4o-mini","memory":"personal","tools":1,"streaming":false}
```

### `POST /chat`

```json
{"message": "Calcule 12 * (3 + 4)", "session_id": "demo"}
```

Réponse :

```json
{"response":"84","session_id":"demo","model":"gpt-4o-mini","memory_mode":"personal"}
```

Paramètres :

- `message` (requis) ;
- `session_id` (défaut `"default"`, nettoyé pour l'usage fichier) ;
- `stream` (optionnel) : force ou désactive le SSE, sinon valeur de
  `features.enable_streaming`.

### `POST /memory/reset?session_id=demo`

Réinitialise la mémoire de la session (ou la mémoire partagée selon le mode).

### Erreurs

| Code | Cause |
|---|---|
| 400 | corps de requête invalide (Pydantic) |
| 429 | limite de débit du fournisseur |
| 502 | réponse LLM invalide / boucle d'outils épuisée |
| 504 | LLM injoignable ou expiré |

---

## 6. Streaming

`features.enable_streaming: true` (ou `"stream": true` dans le corps) produit
un flux SSE minimaliste :

```bash
curl -N -s -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Raconte-moi une histoire courte","session_id":"demo","stream":true}'
```

```
data: {"event": "delta", "data": "Il était"}

data: {"event": "delta", "data": " une fois"}

data: {"event": "done", "data": "[DONE]"}
```

Les appels d'outils sont **exécutés avant** l'ouverture du flux : les fragments
`tool_calls` ne sont jamais diffusés, le client ne reçoit que du texte.

---

## 7. Volumes et sécurité

```yaml
volumes:
  - ./config.yml:/app/config.yml:ro
  - ./system_prompt.md:/app/system_prompt.md:ro
  - ./class-tools:/app/class-tools:ro
  - ./kb:/app/kb
```

- Le conteneur tourne en `appuser` **UID/GID 1000**, propriétaire de `/app` et
  `/app/kb` : aucun conflit de droits sur les volumes montés depuis l'hôte.
- `.env` n'est **jamais** commité (`.gitignore`) et n'entre pas dans l'image
  (`.dockerignore`).
- Ne commitez jamais votre clé : en cas de fuite, révoquez-la immédiatement.
- `docker-compose.yml` publie le port sur `127.0.0.1:8000` uniquement. Pour une
  exposition réseau, placez un reverse proxy (nginx/Caddy) avec TLS et
  authentification devant l'API : `lbee` n'a pas d'authentification native.

---

## 8. Développement local (sans Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export API_KEY=sk-...
export LBE_ROOT=$(pwd)      # pour résoudre config.yml et class-tools hors /app
uvicorn app.main:app --reload --port 8000
```

`LBE_ROOT` permet de faire tourner le harnais à la racine du dépôt au lieu de
`/app` (utile en local et pour les tests).