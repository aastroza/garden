import os
from modal import App, Secret, gpu, method, Image

app = App(name="garden")

image = Image.debian_slim(python_version="3.11").pip_install(
    "outlines==0.0.46",
    "transformers",
    "sentencepiece",
    "datasets",
    "accelerate",
)

@app.cls(image=image, secrets=[Secret.from_dotenv()], gpu=gpu.H100(), timeout=300)
class Model:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct") -> None:
        import outlines

        self.model = outlines.models.transformers(
            model_name,
            device="cuda",
            model_kwargs={
                "token": os.environ["HF_TOKEN"],
                "trust_remote_code": True,
            },
        )

    @method()
    def generate(self, schema: str, prompt: str, temperature: float = 0.7, whitespace_pattern: str = None):
        import outlines
        sampler = outlines.samplers.multinomial(temperature=temperature)
        if whitespace_pattern:
            generator = outlines.generate.json(self.model, schema.strip(), sampler, whitespace_pattern=whitespace_pattern)
        else:
            generator = outlines.generate.json(self.model, schema.strip(), sampler)

        result = generator(prompt)

        return result