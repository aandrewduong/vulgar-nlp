import argparse
from predictor import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text", type=str, required=True, help="Input text to classify as vulgar or clean"
    )
    args = parser.parse_args()

    is_vulgar, confidence = predict(args.text)
    print(f"{args.text} [{'Vulgar' if is_vulgar else 'Clean'} Confidence: {confidence * 100:.2f}%]")

if __name__ == "__main__":
    main()