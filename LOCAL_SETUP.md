# 🖥️ Chatterbox-TTS Extended (app anglaise) — Lancement local

Guide d'installation et de lancement en local de l'app principale **`Chatter.py`**
(TTS anglais + fonctions avancées : candidats, validation Whisper, batch, post-traitement, Voice Conversion).

> Pour la synthèse **multilingue / française**, voir [`multilingual/README.md`](multilingual/README.md).
> Les deux apps sont indépendantes et peuvent tourner en même temps (ports différents).

---

## 1. Prérequis

- **Python 3.10** (recommandé — les dépendances sont épinglées pour cette version).
- **FFmpeg** sur le `PATH` (`brew install ffmpeg` sur macOS).
- GPU **NVIDIA/CUDA** (Linux/Windows) ou **Apple Silicon (MPS)** ou CPU — détecté automatiquement.

L'outil [`uv`](https://github.com/astral-sh/uv) est recommandé pour obtenir un Python 3.10 autonome
(évite les Python système parfois cassés) :

```bash
brew install uv        # macOS
# ou : curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Installation

Depuis la **racine du projet** :

```bash
# Environnement Python 3.10 isolé
uv venv --python 3.10 .venv
```

### 2a. macOS (Apple Silicon / MPS) ou CPU

Sur Mac il n'y a pas de CUDA : il faut **retirer la ligne de l'index CUDA** de `requirements.txt`
(sinon `torch==2.7.0+cu128` est introuvable), torch/torchaudio venant alors de PyPI.

```bash
# Installer les dépendances SANS l'index CUDA (grep retire la ligne à la volée)
grep -v 'extra-index-url' requirements.txt | .venv/bin/python -m pip install -r /dev/stdin

# librosa 0.10 a besoin de pkg_resources, retiré dans setuptools >= 81
.venv/bin/python -m pip install "setuptools<81"
```

### 2b. Linux / Windows (NVIDIA CUDA)

L'index CUDA est nécessaire, on installe donc tel quel :

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install "setuptools<81"
```

> Le **modèle** est téléchargé automatiquement au premier lancement d'une génération
> (mis en cache dans `~/.cache/huggingface`, pas de re-téléchargement ensuite).

> ℹ️ `requirements.txt` a été ajusté : `auto-editor` non épinglé (la version `27.1.1` a été retirée
> de PyPI) et `pydub` ajouté (importé par `Chatter.py`).

---

## 3. Lancement

Depuis la **racine du projet** :

```bash
# macOS : PYTORCH_ENABLE_MPS_FALLBACK=1 permet aux rares opérations non gérées par MPS
# de basculer proprement sur CPU au lieu de planter.
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python Chatter.py --port 7860
```

Puis ouvrir : **http://127.0.0.1:7860**

Options utiles :
```bash
.venv/bin/python Chatter.py --port 7860            # choisir le port
.venv/bin/python Chatter.py --host 0.0.0.0         # exposer sur le réseau local
.venv/bin/python Chatter.py --share                # lien public Gradio temporaire
```

---

## 4. Configuration (`settings.json`)

L'app charge ses réglages depuis `settings.json` (racine) au démarrage et les y sauvegarde.
Une config **équilibrée, optimisée pour Apple Silicon**, y est fournie :

| Paramètre | Valeur | Rôle |
|---|---|---|
| Candidats / chunk | **3** | Réduit artefacts/hallucinations |
| Max attempts | **2** | Retente si la validation échoue |
| Validation Whisper | **faster-whisper `small`** | Bon compromis précision/vitesse |
| Workers parallèles | **4** | Adapté à un M1 Pro |
| CFG Weight | **0.5** | Voix naturelle (1.0 = monotone) |
| Exaggeration | **0.5** | Émotion neutre |
| Temperature | **0.8** | Variété maîtrisée |
| Formats export | **WAV + MP3** | |

Pour aller **plus vite** : baissez les candidats à 1, ou cochez « Bypass Whisper ».
Pour **plus de qualité** : montez les candidats et le modèle Whisper (`medium`/`large`) — mais c'est plus lent.

---

## 5. Dépannage

| Problème | Solution |
|---|---|
| `torch==2.7.0+cu128 ... no wheels ... macosx` | Vous êtes sur Mac : retirez la ligne `--extra-index-url` (voir §2a). |
| `No module named 'pkg_resources'` | `.venv/bin/python -m pip install "setuptools<81"` |
| `No module named 'pydub'` | `.venv/bin/python -m pip install pydub` |
| `auto-editor==27.1.1 ... unsatisfiable` | Version retirée de PyPI ; le `requirements.txt` la laisse désormais non épinglée. |
| Erreur `pyexpat` / `libexpat` (venv) | Python système cassé : utilisez un Python `uv` (`uv venv --python 3.10 .venv`). |
| Génération lente | Normal sur CPU/MPS. Réduisez candidats/workers, ou utilisez un modèle Whisper plus petit. |

---

## 6. Performances observées

Testé sur **MacBook Pro M1 Pro (32 Go, MPS)** : device auto-sélectionné `mps`, l'app démarre
et génère correctement. Sans GPU CUDA, la génération est plus lente qu'un poste NVIDIA — d'où
la config équilibrée par défaut.
