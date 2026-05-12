from ultralytics import YOLO
import json
from pathlib import Path
import os

def main():
    # 1. Define paths
    # Replace these with your actual model path and dataset YAML path
    model_path = "models/player_detector4/weights/best.pt" 
    dataset_yaml = "configs/dataset.yaml" # or wherever your dataset config is
    
    output_dir = Path("outputs/validation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        # Fallback for demonstration if the trained model isn't there
        model_path = "yolov8n.pt"
        print(f"Using {model_path} as fallback...")

    # 2. Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # 3. Run validation
    print("Running validation... This may take a few minutes.")
    # The val() method calculates mAP, precision, recall, etc.
    metrics = model.val(
        data=dataset_yaml, 
        split="test", # or "val" depending on what you want to evaluate on
        plots=True,   # Saves PR curves, confusion matrices, etc.
        project=str(output_dir),
        name="evaluation" # Results will be saved in outputs/validation_results/evaluation
    )
    
    # 4. Extract metrics
    # metrics.results_dict contains the raw numbers
    results = {
        "mAP50-95": metrics.box.map,    # Mean Average Precision @ IoU 0.50:0.95
        "mAP50": metrics.box.map50,     # Mean Average Precision @ IoU 0.50
        "mAP75": metrics.box.map75,     # Mean Average Precision @ IoU 0.75
        "precision": metrics.box.p.tolist() if hasattr(metrics.box.p, "tolist") else metrics.box.p,
        "recall": metrics.box.r.tolist() if hasattr(metrics.box.r, "tolist") else metrics.box.r,
        "fitness": metrics.fitness
    }
    
    # You can also get per-class metrics
    if hasattr(metrics.box, "maps"):
        results["per_class_mAP"] = metrics.box.maps.tolist() if hasattr(metrics.box.maps, "tolist") else metrics.box.maps
        
    # 5. Save metrics to JSON
    json_path = output_dir / "evaluation" / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n--- Validation Complete ---")
    print(f"mAP50-95: {results['mAP50-95']:.4f}")
    print(f"mAP50:    {results['mAP50']:.4f}")
    print(f"Metrics JSON saved to: {json_path}")
    print(f"Visual plots (PR curves, confusion matrix) saved in: {output_dir / 'evaluation'}")

if __name__ == "__main__":
    main()
