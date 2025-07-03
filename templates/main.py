import os
import openai
import gradio as gr
from dotenv import load_dotenv
import random

# --------------------- VC-SYNAPTICA ---------------------
class Synaptica:
    @staticmethod
    def log(message):
        print(f"[SYNAPTICA] {message}")

    @staticmethod
    def sync_context(intent, drift, result_hash):
        print(f"[SYNAPTICA] Synchronizacja: INTENT={intent}, DRIFT={drift}, HASH={result_hash[:6]}...")


# --------------------- VC-INTENTMIND ---------------------
class IntentMind:
    @staticmethod
    def analyze(prompt):
        Synaptica.log("INTENTMIND analizuje prompt...")
        system_prompt = (
            "Jesteś warstwą VC-INTENTMIND™. Zanalizuj intencję użytkownika. "
            "Zidentyfikuj ton, emocje i główne założenie promptu."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']


# --------------------- VC-TARGETLOOP ---------------------
class TargetLoop:
    @staticmethod
    def generate_image(prompt, retries=2):
        Synaptica.log("TARGETLOOP generuje obraz...")
        for attempt in range(retries):
            try:
                response = openai.Image.create(
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )
                image_url = response['data'][0]['url']
                Synaptica.log(f"Obraz wygenerowany przy podejściu {attempt+1}")
                return image_url
            except Exception as e:
                Synaptica.log(f"Błąd generacji: {e}")
        return None


# --------------------- VC-SELFSTATE ---------------------
class SelfState:
    @staticmethod
    def evaluate_alignment(prompt, intent_summary):
        Synaptica.log("SELFSTATE ocenia spójność semantyczną...")
        query = (
            f"Czy poniższy prompt i jego interpretacja są spójne stylistycznie i semantycznie?\n"
            f"PROMPT: {prompt}\n\nINTERPRETACJA: {intent_summary}\n\n"
            f"Zwróć poziom zgodności (0.0–1.0) i krótki komentarz."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": query}]
        )
        return response['choices'][0]['message']['content']


# --------------------- VC-EXECUTEXT ---------------------
class ExecuText:
    @staticmethod
    def comment_on_text_image(prompt):
        if any(x in prompt.lower() for x in ["napis", "tekst", "tytuł", "cytat"]):
            return "⚠️ Wykryto potencjalny komponent tekstowy. VC-EXECUTEXT weryfikuje składnię i styl..."
        return "✅ Brak widocznego komponentu tekstowego – pomijam kontrolę EXECUTEXT."


# --------------------- INTERFEJS UŻYTKOWNIKA ---------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def visualcore_mvp(prompt):
    try:
        # VC-INTENTMIND
        intent = IntentMind.analyze(prompt)

        # VC-TARGETLOOP
        image_url = TargetLoop.generate_image(prompt)

        if not image_url:
            return None, "❌ Nie udało się wygenerować obrazu.", intent, ""

        # VC-SELFSTATE
        drift_check = SelfState.evaluate_alignment(prompt, intent)

        # VC-SYNAPTICA
        Synaptica.sync_context(intent, drift_check, image_url)

        # VC-EXECUTEXT
        text_eval = ExecuText.comment_on_text_image(prompt)

        return image_url, intent, drift_check, text_eval

    except Exception as e:
        return None, "❌ Błąd systemowy", f"{e}", ""

# --------------------- UI ---------------------
app = gr.Interface(
    fn=visualcore_mvp,
    inputs=gr.Textbox(label="🎯 Prompt wejściowy", placeholder="Napisz co chcesz wygenerować..."),
    outputs=[
        gr.Image(label="🖼️ Wygenerowany Obraz"),
        gr.Textbox(label="🧠 INTENTMIND — Interpretacja Promptu"),
        gr.Textbox(label="🧪 SELFSTATE — Ocena Spójności"),
        gr.Textbox(label="🔠 EXECUTEXT — Ocena Tekstu")
    ],
    title="VisualCore™ MVP — Architektura Blokowa",
    description="Prototyp systemu VisualCore™: rozkład promptu na intencję, obraz, spójność i analizę tekstu."
)

if __name__ == "__main__":
    app.launch()
