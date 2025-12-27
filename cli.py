import argparse
import json
import sys
from predictor import predict


def process_batch(texts, args):
    """Predict and print results for a batch of texts."""
    preds, confidences = predict(
        texts,
        model_dir=args.model_dir,
        threshold=args.threshold,
        device=args.device,
        max_length=args.max_length,
    )

    for t, p, c in zip(texts, preds, confidences):
        if args.json:
            print(json.dumps({"text": t, "is_vulgar": p, "confidence": c}))
        else:
            status = "Vulgar" if p else "Clean"
            print(f"{t} [{status} Confidence: {c * 100:.2f}%]")


def main():
    parser = argparse.ArgumentParser(description="CLI for vulgar language detection.")
    parser.add_argument("--text", type=str, help="Input text to classify")
    parser.add_argument("--file", type=str, help="Path to a file with one text per line")
    parser.add_argument("--model-dir", type=str, default="model", help="Directory of the trained model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for file processing")
    parser.add_argument("--json", action="store_true", help="Output results in JSON Lines format")
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Provide either --text or --file")

    if args.text:
        process_batch([args.text], args)
        return

    # File mode: Process in batches to be memory efficient
    batch = []
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    batch.append(clean_line)
                
                if len(batch) >= args.batch_size:
                    process_batch(batch, args)
                    batch = []
            
            # Final batch
            if batch:
                process_batch(batch, args)
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
