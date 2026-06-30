from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_NAME = "BioMistral/BioMistral-7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=700,
    do_sample=False,
)

SYSTEM_PROMPT = """You are a clinical documentation assistant.
You are not a doctor.
You must not invent clinical facts."""

def generate_soap_hf(transcript: str) -> str:
    prompt = f"""{SYSTEM_PROMPT}

Convert the transcript into a SOAP note.
Rules:
1. Use only explicitly stated information.
2. If a section has no evidence, write "Not documented".
3. Do not add medications, tests, allergies, or diagnoses unless stated.
4. Include "Clinician review required" at the end.
5. If there are possible red flags, list them separately.

Transcript:
{transcript}

SOAP Note:"""

    output = generator(prompt)
    return output[0]["generated_text"]


if __name__ == "__main__":
    sample = """Doctor: Good morning. What brings you in today?
Patient: I have had a cough and fever for about three days."""
    print(generate_soap_hf(sample))

