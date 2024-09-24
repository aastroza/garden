import outlines
import modal
import os

from pydantic import BaseModel, Field
from rich import print
from rich.panel import Panel

class NewSubject(BaseModel):
    rewritten_text: str = Field(..., title="The rewritten text after change a subject.")
    story_title: str = Field(..., title="The title of the story.")

# Make our app
app = modal.App("fiction-machine")

# What language model will we use?
llm_base = "mistralai/Mistral-Nemo-Base-2407"
llm_instruct = "microsoft/Phi-3.5-mini-instruct"

# Set up the outlines image
modal_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "outlines",
    "transformers",
    "accelerate",
    "sentencepiece",
    "bitsandbytes",
    "vllm"
)

@app.cls(gpu="H100", image=modal_image, secrets=[modal.Secret.from_dotenv()])
class Model:
    @modal.build()
    def download_model(self):
        import outlines
        outlines.models.transformers(
            llm_base,
            device="cuda",
            model_kwargs={
                "token": os.environ["HF_TOKEN"],
                "trust_remote_code": True,
            },
        )

    @modal.enter()
    def setup(self):
        import outlines
        self.model = outlines.models.transformers(
            llm_base,
            device="cuda",
        )

    @modal.method()
    def make_story(self, prompt: str, max_tokens: int = 20):
        generator = outlines.generate.text(
            self.model,
            sampler=outlines.samplers.multinomial(temperature=0.3),
        )
        return generator(prompt, max_tokens)

@app.cls(gpu="H100", image=modal_image, secrets=[modal.Secret.from_dotenv()])
class InstructModel:
    @modal.build()
    def download_model(self):
        import outlines
        outlines.models.transformers(
            llm_instruct,
            device="cuda",
            model_kwargs={
                "token": os.environ["HF_TOKEN"],
                "trust_remote_code": True,
            },
        )

    @modal.enter()
    def setup(self):
        import outlines
        self.model = outlines.models.transformers(
            llm_instruct,
            device="cuda",
        )

    @modal.method()
    def change_subject(self, prompt: str):
        generator = outlines.generate.json(
            self.model,
            NewSubject,
            sampler=outlines.samplers.multinomial(temperature=0.7),
        )
        return generator(prompt)

examples = [
    {"text": "a dog is running", "rewritten_text":"a cat is running"},
    {"text": "the boy is playing football", "rewritten_text": "the girl is playing football"},
    {"text": "the president gave a speech", "rewritten_text": "the governor gave a speech"},
    {"text": "the horse jumped over the fence", "rewritten_text": "the cow jumped over the fence"},
    {"text": "a soldier stood guard", "rewritten_text": "a police officer stood guard"}
]

@outlines.prompt
def subject_change_prompt(
    story: str,
    examples: list = examples,
):
    """
    <|system|>
    You are a world class storyteller. You have been asked to rewrite a story by just changing one of the subjects of the story.
    
    Examples
    --------

    {% for example in examples %}
    Text: {{ example.text }}
    Rewritten Text: {{ example.rewritten_text }}
    {% endfor %}
    <|end|>
    <|user|>
    {{ story }}
    <|end|>
    <|assistant|>
    """
    
# Local entrypoint
@app.local_entrypoint()
def main():
    for i in range(3):
        # Set up our story
        story = Model().make_story.remote("Once upon a time a dog called", 50)
        story_transformation = InstructModel().change_subject.remote(story)

        # Set up panel width
        panel_width = 60

        # Start printing out some information about the story
        print(Panel.fit(story,
                        width=panel_width,
                        title="Story"))
        
        # Print out the transformed story
        print(Panel.fit(story_transformation.rewritten_text,
                        width=panel_width,
                        title=f"Transformed Story: {story_transformation.story_title}"))