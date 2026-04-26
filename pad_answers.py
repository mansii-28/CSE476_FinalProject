# pad_answers.py — run this to fill missing answers with empty strings
import json
from pathlib import Path

checkpoint = json.load(open("outputs/cse_476_answers_checkpoint.json", encoding="utf-8"))
questions  = json.load(open("data/test_data.json", encoding="utf-8"))

answers = [{"output": checkpoint.get(str(i), "")} for i in range(len(questions))]

with open("outputs/cse_476_final_project_answers.json", "w", encoding="utf-8") as f:
    json.dump(answers, f, ensure_ascii=False, indent=2)

answered = sum(1 for a in answers if a["output"])
print(f"Done. {answered}/{len(questions)} questions answered, {len(questions)-answered} padded with empty string.")