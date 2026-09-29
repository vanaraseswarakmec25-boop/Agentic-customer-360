import json


def evaluate_scenario(
    inferred_file="inferred_events.jsonl",
    ground_truth_file="scenario_01/ground_truth.json",
):
    print("\n==========================================")
    print("  SCORING INFERRED OUTPUTS VS GROUND TRUTH")
    print("==========================================\n")

    # 1. Load Ground Truth
    try:
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {ground_truth_file} not found.")
        return

    # Handle if ground truth is wrapped inside a key like {"events": [...]}
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get(
            "events", ground_truth.get("ground_truth", [])
        )

    # 2. Load Predictions
    predictions = []
    try:
        with open(inferred_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        predictions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        print(
            f"❌ Error: {inferred_file} not found. Run replay_scenario.py first!"
        )
        return

    total_gt = len(ground_truth)
    print(f"Total Ground Truth Samples: {total_gt}")
    print(f"Total Predictions Logged:   {len(predictions)}\n")

    if total_gt == 0:
        print("❌ Ground truth file is empty or not parsed correctly.")
        return

    matched_states = 0
    matched_actions = 0

    # Extract the last N predictions to evaluate against ground truth events
    eval_predictions = (
        predictions[-total_gt:]
        if len(predictions) >= total_gt
        else predictions
    )

    for idx, gt_item in enumerate(ground_truth):
        if not isinstance(gt_item, dict):
            continue

        expected_state = gt_item.get("expected_state") or gt_item.get(
            "inferred_customer_state"
        )
        expected_action = gt_item.get("expected_action") or gt_item.get(
            "action_decided"
        )

        if idx < len(eval_predictions):
            pred = eval_predictions[idx]
            if not isinstance(pred, dict):
                continue

            pred_state = pred.get("inferred_customer_state")
            pred_action = pred.get("action_decided")

            state_match = pred_state == expected_state
            action_match = pred_action == expected_action

            if state_match:
                matched_states += 1
            if action_match:
                matched_actions += 1

            print(f"Sample #{idx+1}:")
            print(
                f"   State  -> Pred: '{pred_state}' | GT: '{expected_state}' {'✅' if state_match else '❌'}"
            )
            print(
                f"   Action -> Pred: '{pred_action}' | GT: '{expected_action}' {'✅' if action_match else '❌'}\n"
            )

    state_acc = (matched_states / total_gt * 100) if total_gt > 0 else 0
    action_acc = (matched_actions / total_gt * 100) if total_gt > 0 else 0

    print("📊 ACCURACY SCORES:")
    print(
        f" - State Inference Accuracy : {state_acc:.2f}% ({matched_states}/{total_gt})"
    )
    print(
        f" - Action Decision Accuracy  : {action_acc:.2f}% ({matched_actions}/{total_gt})"
    )
    print("\n------------------------------------------\n")


if __name__ == "__main__":
    evaluate_scenario()