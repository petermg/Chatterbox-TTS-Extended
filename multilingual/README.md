# 🌍 Chatterbox Multilingual — Lancement local

Interface Gradio dédiée à la synthèse vocale **multilingue** (23 langues, dont le **français**),
construite sur le modèle officiel `ChatterboxMultilingualTTS` du package [`chatterbox-tts`](https://pypi.org/project/chatterbox-tts/).

Elle est **volontairement isolée** de l'app anglaise `Chatter.py` :
- environnement virtuel séparé (`.venv-ml`) pour éviter les conflits de dépendances ;
- lancée depuis le dossier `multilingual/` pour éviter la collision de nom avec le package local `./chatterbox`.

> ℹ️ L'app anglaise (`Chatter.py`) et cette app multilingue sont **indépendantes** et peuvent tourner en même temps (ports différents).

---

## 1. Prérequis

- **Python 3.10** (recommandé). L'outil [`uv`](https://github.com/astral-sh/uv) est utilisé ici pour gérer un Python autonome et l'environnement.
- **FFmpeg** sur le `PATH` (`brew install ffmpeg` sur macOS).
- macOS Apple Silicon (MPS), Linux/Windows CUDA, ou CPU — le device est détecté automatiquement.

Installer `uv` si besoin :
```bash
brew install uv        # macOS
# ou : curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Installation (première fois)

Depuis la **racine du projet** :

```bash
# Environnement Python 3.10 isolé pour le multilingue
uv venv --python 3.10 .venv-ml

# Dépendances (setuptools<81 pour compatibilité librosa/pkg_resources)
.venv-ml/bin/python -m pip install "setuptools<81" chatterbox-tts
# (ou, avec uv : uv pip install --python .venv-ml "setuptools<81" chatterbox-tts)
```

> Le **modèle** (~plusieurs Go) est téléchargé automatiquement au **premier** lancement d'une génération,
> puis mis en cache dans `~/.cache/huggingface` (aucun re-téléchargement ensuite).

---

## 3. Lancement

Depuis la **racine du projet** :

```bash
cd multilingual
../.venv-ml/bin/python app_ml.py
```

Puis ouvrir : **http://127.0.0.1:7861**

> ⚠️ Toujours lancer depuis le dossier `multilingual/`. Lancer depuis la racine ferait masquer
> le package `chatterbox` officiel par le dossier local `./chatterbox` (app anglaise).

Pour changer de port, éditez la dernière ligne de `app_ml.py` (`demo.launch(server_port=7861)`).

---

## 4. Utilisation

| Champ | Rôle |
|---|---|
| **Texte** | Le texte à synthétiser (2–4 phrases conseillées par génération). |
| **Langue** | 23 langues ; **français (`fr`)** sélectionné par défaut. |
| **Voix de référence** *(optionnel)* | Upload/enregistrement de 6–20 s pour **cloner un timbre**. |
| **Exaggeration** | Intensité émotionnelle (0 = neutre, 2 = très expressif). |
| **CFG Weight** | Rythme/fidélité. **0.3** = plus vivant ; **0.5** = équilibré. |
| **Temperature** | Variété de la voix (0.8 par défaut). |
| **Seed** | 0 = aléatoire ; une valeur fixe = résultat reproductible. |

Les fichiers générés sont enregistrés dans **`multilingual/outputs/`** (format `wav`).

### Langues supportées
`ar` arabe · `da` danois · `de` allemand · `el` grec · `en` anglais · `es` espagnol · `fi` finnois ·
`fr` **français** · `he` hébreu · `hi` hindi · `it` italien · `ja` japonais · `ko` coréen · `ms` malais ·
`nl` néerlandais · `no` norvégien · `pl` polonais · `pt` portugais · `ru` russe · `sv` suédois ·
`sw` swahili · `tr` turc · `zh` chinois

---

## 5. Conseils français

- Écrivez les **nombres et dates en toutes lettres** (« vingt-trois » plutôt que « 23 ») pour une meilleure prononciation.
- Pour un ton plus naturel/expressif : **Exaggeration ≈ 0.8**, **CFG Weight ≈ 0.3**.
- Gardez des passages courts ; cette UI ne fait pas de découpage automatique des longs textes.

---

## 6. Dépannage

| Problème | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'chatterbox.mtl_tts'` | Vous avez lancé depuis la racine → le dossier local `./chatterbox` masque le package. Lancez depuis `multilingual/`. |
| `ModuleNotFoundError: No module named 'pkg_resources'` | `.venv-ml/bin/python -m pip install "setuptools<81"` |
| Port 7861 déjà utilisé | Changez `server_port` dans `app_ml.py`, ou arrêtez l'ancien process (`pkill -f app_ml.py`). |
| Génération lente | Normal sur CPU/MPS (~20–25 s / phrase courte sur Apple Silicon M1). Les GPU CUDA sont bien plus rapides. |
| 1ʳᵉ génération très longue | Téléchargement du modèle en cours (une seule fois). |

---

## 7. Performances observées

Testé sur **MacBook Pro M1 Pro (32 Go, MPS)** : génération d'une phrase française courte en ~20–25 s
(hors premier téléchargement du modèle). Device sélectionné automatiquement : `mps`.
