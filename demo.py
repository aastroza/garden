import modal
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int
    strength: int

schema = """
{
"title": "User",
"type": "object",
"properties": {
"name": {"type": "string"},
"last_name": {"type": "string"},
"id": {"type": "integer"}
},
"required": ["name", "last_name", "id"]
}
"""

Model = modal.Cls.lookup("garden", "Model")
m = Model()

prompt = """Generate a character for my awesome ninja game."""

result = m.generate.remote(schema, prompt)

print(result)
