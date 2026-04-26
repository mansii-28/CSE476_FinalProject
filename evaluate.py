import time
from api import call_model

def self_evaluate(question: str, prediction: str, expected_answer: str) -> bool:
    """Use the model to check if the prediction is correct."""
    system = "You are a strict grader. Reply with exactly True or False."
    prompt = f"Question: {question}\nPrediction: {prediction}\nExpected: {expected_answer}\nIs the prediction correct? Reply True or False."

    result = call_model(
        prompt=prompt,
        system=system,
        temperature=0.0,
    )
    
    # Check the model response
    reply = (result.get("text") or "").strip().lower()
    
    if reply == "true":
        return True
    elif reply == "false":
        return False
        
    # Basic string fallback if the model doesn't say true or false
    from utils import normalize_answer
    pred_clean = normalize_answer(prediction)
    exp_clean = normalize_answer(expected_answer)
    
    if exp_clean and exp_clean == pred_clean:
        return True
        
    return False

def evaluate_results(results: list[dict]) -> None:
    """Grade all the results and print the accuracy."""
    print("Starting evaluation...")
    
    domain_stats = {}
    total_correct = 0
    total_items = len(results)
    
    for result in results:
        domain = result.get("domain", "unknown")
        
        # Setup dictionary for this domain if it doesn't exist
        if domain not in domain_stats:
            domain_stats[domain] = {"correct": 0, "total": 0}
            
        domain_stats[domain]["total"] += 1
        
        # Check if correct
        if not result.get("prediction"):
            result["correct"] = False
        else:
            is_correct = self_evaluate(result["input"], result["prediction"], result["expected"])
            result["correct"] = is_correct

        if result["correct"]:
            domain_stats[domain]["correct"] += 1
            total_correct += 1
            
        time.sleep(0.1)

    # Print summary
    print("\n--- Evaluation Summary ---")
    for domain, stats in domain_stats.items():
        correct = stats["correct"]
        total = stats["total"]
        if total > 0:
            acc = (correct / total) * 100
        else:
            acc = 0.0
        print(f"Domain: {domain} - Accuracy: {acc}% ({correct}/{total})")
        
    if total_items > 0:
        overall_acc = (total_correct / total_items) * 100
    else:
        overall_acc = 0.0
        
    print(f"\nOVERALL Accuracy: {overall_acc}% ({total_correct}/{total_items})")