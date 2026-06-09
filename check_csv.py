#!/usr/bin/env python3
import csv

src = r'C:\Users\Admin\Downloads\expert_ioc_eval.csv'

with open(src, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Check unique values
q_types = set(r['question_type'] for r in rows if r.get('question_type'))
behaviors = set(r['system_behavior'] for r in rows if r.get('system_behavior'))
scores = set(r['ioc_score'] for r in rows if r.get('ioc_score'))

print('Unique question_type:', q_types)
print('Unique system_behavior:', behaviors)
print('Unique ioc_score:', scores)
print()
print('Sample rows:')
for i, r in enumerate(rows[:5], 1):
    print(f"  {i}. Score: {r['ioc_score']}, Type: {r['question_type']}")
    print(f"     Q: {r['question'][:60]}...")
