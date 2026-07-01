"""
Chatterbox Multilingual — interface de test (23 langues, dont le français).

Utilise le modèle officiel ResembleAI multilingue (chatterbox.mtl_tts).
À lancer depuis CE dossier pour éviter le masquage par le package local ./chatterbox :
    cd multilingual && python app_ml.py

Environnement dédié : ../.venv-ml
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import datetime
import random

import numpy as np
import torch
import torchaudio as ta
import gradio as gr

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# ---- Device (Apple Silicon MPS -> CUDA -> CPU) ----
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"🚀 Multilingual model running on: {DEVICE}")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        print("⏳ Chargement du modèle multilingue (téléchargement au 1er lancement)...")
        MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
        print("✅ Modèle chargé.")
    return MODEL


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Dropdown "fr - French" style, French first
_lang_items = sorted(SUPPORTED_LANGUAGES.items(), key=lambda kv: (kv[0] != "fr", kv[1]))
LANG_CHOICES = [(f"{name} ({code})", code) for code, name in _lang_items]


def synth(text, language_id, ref_audio, exaggeration, cfg_weight, temperature, seed):
    if not text or not text.strip():
        raise gr.Error("Entrez du texte à synthétiser.")

    model = get_model()

    if seed and int(seed) != 0:
        set_seed(int(seed))

    kwargs = dict(
        language_id=language_id,
        exaggeration=float(exaggeration),
        cfg_weight=float(cfg_weight),
        temperature=float(temperature),
    )
    if ref_audio:
        kwargs["audio_prompt_path"] = ref_audio

    wav = model.generate(text, **kwargs)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"{language_id}_{stamp}.wav")
    ta.save(out_path, wav, model.sr)
    print(f"💾 Sauvegardé: {out_path}")
    return out_path


with gr.Blocks(title="Chatterbox Multilingual (FR)") as demo:
    gr.Markdown(
        "# 🌍 Chatterbox Multilingual\n"
        "Synthèse vocale multilingue (23 langues). **Français sélectionné par défaut.**\n\n"
        "> ⚠️ Le tout premier clic télécharge le modèle (~plusieurs Go) — patientez quelques minutes."
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="Texte",
                lines=6,
                value="Bonjour ! Ceci est un test de synthèse vocale en français avec le modèle multilingue Chatterbox.",
            )
            language_id = gr.Dropdown(
                choices=LANG_CHOICES, value="fr", label="Langue"
            )
            ref_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Voix de référence (optionnel — clonage de timbre, 6–20 s)",
            )
            with gr.Row():
                exaggeration = gr.Slider(0.0, 2.0, value=0.5, step=0.05, label="Exaggeration (émotion)")
                cfg_weight = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="CFG Weight (0.3 = plus expressif)")
            with gr.Row():
                temperature = gr.Slider(0.05, 2.0, value=0.8, step=0.05, label="Temperature")
                seed = gr.Number(value=0, precision=0, label="Seed (0 = aléatoire)")
            btn = gr.Button("🎙️ Générer", variant="primary")
        with gr.Column():
            out = gr.Audio(label="Résultat", type="filepath")

    btn.click(
        synth,
        inputs=[text, language_id, ref_audio, exaggeration, cfg_weight, temperature, seed],
        outputs=[out],
    )


if __name__ == "__main__":
    demo.launch(server_port=7861)
