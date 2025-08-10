import argparse
import json
from predictor import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Input text to classify as vulgar or clean")
    parser.add_argument("--file", type=str, help="Path to a file with one text per line")
    parser.add_argument("--model-dir", type=str, default="model", help="Directory of the trained model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for labeling as vulgar")
    parser.add_argument("--device", type=str, default=None, help="Device to use: cuda, mps, or cpu")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length for tokenization")
    parser.add_argument("--json", action="store_true", help="Output results in JSON Lines format")
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Provide either --text or --file")

    if args.text:
        is_vulgar, confidence = predict(
            args.text,
            model_dir=args.model_dir,
            threshold=args.threshold,
            device=args.device,
            max_length=args.max_length,
        )
        if args.json:
            print(json.dumps({"text": args.text, "is_vulgar": is_vulgar, "confidence": confidence}))
        else:
            print(f"{args.text} [{'Vulgar' if is_vulgar else 'Clean'} Confidence: {confidence * 100:.2f}%]")
        return

    # File mode
    with open(args.file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    preds, confidences = predict(
        texts,
        model_dir=args.model_dir,
        threshold=args.threshold,
        device=args.device,
        max_length=args.max_length,
    )

    if args.json:
        for t, p, c in zip(texts, preds, confidences):
            print(json.dumps({"text": t, "is_vulgar": p, "confidence": c}))
    else:
        for t, p, c in zip(texts, preds, confidences):
            print(f"{t} [{'Vulgar' if p else 'Clean'} Confidence: {c * 100:.2f}%]")


if __name__ == "__main__":
    main()