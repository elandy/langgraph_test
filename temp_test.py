import os
import time

from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.environ["GOOGLE_API_KEY"]
)

model = "gemini-3-flash-preview"

prompt = """
Invent a completely ridiculous religion based on kitchen appliances.
Give it a name, rituals, and mythology.
"""


def generate_with_retry(
    *,
    model: str,
    contents: str,
    config: types.GenerateContentConfigOrDict | None,
    max_retries: int = 5,
    initial_delay: float = 1.0,
) ->  types.GenerateContentResponse | None:
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
                model=model,
                contents=contents,
                config=config,
            )

        except Exception as e:
            error_text = str(e).lower()

            is_transient = any(
                code in error_text
                for code in (
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "resource exhausted",
                    "internal",
                    "unavailable",
                    "deadline exceeded",
                )
            )

            if not is_transient or attempt == max_retries:
                raise

            print(f"Transient error: {e}\n Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")

            time.sleep(delay)
            delay *= 2
    return None

config_normal = types.GenerateContentConfig(
    temperature=0,
    top_p=0.8,
)

config_creative = types.GenerateContentConfig(
    temperature=1.2,
    top_p=0.95,
)

for config in [config_normal, config_creative]:
    print(f"\n{'=' * 60}")
    print(f"config = {config}")
    print(f"{'=' * 60}\n")

    response = generate_with_retry(
        model=model,
        contents=prompt,
        config=config,
    )
    if response is not None:
        print(response.text)