import random
import secrets
import csv

def generate_seed() -> int:
    return random.randint(1, 2**63 - 1)

def generate_seed2() -> int:
    return secrets.randbelow(2**64)

def get_seeds_from_csv(file_path: str):
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                yield row[1] # Column2