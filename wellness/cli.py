"""python -m wellness.cli --level 3 --llm mock "Can I afford a $30,000 car with $5,000 down?" """
import argparse

from .data_gen import generate_customer
from .graph import level3
from .levels import level1, level2
from .llm import get_llm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--level", type=int, choices=[1, 2, 3], default=3)
    p.add_argument("--llm", default="mock", help="mock | ollama[:model] | openai[:model]")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    fn = {1: level1, 2: level2, 3: level3}[a.level]
    print(fn(get_llm(a.llm), generate_customer(a.seed), a.question))


if __name__ == "__main__":
    main()
