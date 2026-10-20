import json
from pathlib import Path


def build_chunk_text(stance: dict) -> str:
    other_names = ", ".join(stance["outros_nomes"])
    benefits = ", ".join(stance["beneficios"])
    muscles = ", ".join(stance["musculos"])

    body = (
        f"{stance['descricao']} "
        f"Também conhecida como: {other_names}. "
        f"Nível: {stance['nivel']}. "
        f"Benefícios: {benefits}. "
        f"Músculos trabalhados: {muscles}."
    )

    return f"title: {stance['nome']} ({stance['caracteres']}) | text: {body}"


if __name__ == "__main__":
    path = Path(__file__).parents[2] / "data" / "stances.json"
    with path.open(encoding="utf-8") as f:
        stances = json.load(f)

    for stance in stances:
        print(build_chunk_text(stance))
        print()
