"""Validation and provenance for paired image/prompt inference inputs."""
import hashlib
from pathlib import Path
from typing import Sequence, TypedDict


class ImagePromptInput(TypedDict):
    sample_id: int
    image_path: str
    image_sha256: str
    prompt: str


def load_image_prompts(image_paths: Sequence[str], prompts: Sequence[str]) -> list[ImagePromptInput]:
    """Pair lists by index, preserve prompt text, and verify images before model loading."""
    from PIL import Image

    if not image_paths or len(image_paths) != len(prompts):
        raise ValueError('image_path and prompt must be nonempty lists with equal lengths.')
    rows: list[ImagePromptInput] = []
    for index, (image_path, prompt) in enumerate(zip(image_paths, prompts)):
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(f'image_path[{index}] must be a nonempty string.')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f'prompt[{index}] must be a nonempty string.')
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'image_path[{index}] does not exist: {path}')
        try:
            with Image.open(path) as image:
                image.load()
        except (OSError, ValueError) as error:
            raise ValueError(f'image_path[{index}] is not a readable image: {path}') from error
        rows.append({'sample_id': index, 'image_path': str(path), 'prompt': prompt,
                     'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def summarize_inputs(records: list[dict]) -> dict:
    """Report generation statistics for inputs without reference answers."""
    return {'count': len(records),
            'truncated': sum(row['finish_reason'] == 'length' for row in records),
            'generated_tokens': sum(row['generated_tokens'] for row in records)}
