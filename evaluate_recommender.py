"""Reproducible offline evaluation for the retrieval baseline.

Run: python evaluate_recommender.py
"""
import csv
import os
from recommendation_engine import AyurvedicRecommender


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cases_path = os.path.join(base_dir, "data", "evaluation_queries.csv")
    recommender = AyurvedicRecommender()
    with open(cases_path, encoding="utf-8", newline="") as file:
        cases = list(csv.DictReader(file))
    top_1 = 0
    top_3 = 0
    reciprocal_rank = 0.0
    for case in cases:
        expected = case["expected_condition"].lower()
        names = [result["disease"].lower() for result in recommender.recommend(case["query"], top_k=3)]
        rank = next((index + 1 for index, name in enumerate(names) if expected in name), None)
        top_1 += rank == 1
        top_3 += rank is not None
        reciprocal_rank += 1 / rank if rank else 0
        print(f"{case['query']}: expected={case['expected_condition']}; retrieved={names}; rank={rank or 'not found'}")
    count = len(cases)
    print(f"\nPrecision@1: {top_1 / count:.2%}\nRecall@3: {top_3 / count:.2%}\nMRR@3: {reciprocal_rank / count:.2%}")


if __name__ == "__main__":
    main()
