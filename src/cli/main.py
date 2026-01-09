import argparse
import sys
from pathlib import Path
from src.core.face_swapper import FaceSwapper
from src.core.image_utils import load_image, save_image, validate_image_format
from src.core.paths import get_assets_path, get_models_path


def main():
    parser = argparse.ArgumentParser(
        description="Swap faces in images using geometric transformations"
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to input image (JPG or PNG)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output image (default: input_image_madurified.jpg)"
    )
    parser.add_argument(
        "-m", "--maduro-face",
        type=str,
        default=None,
        nargs="+",
        help="Path(s) to Maduro face template(s). Can specify multiple images. (default: assets/maduro_face*.jpg)"
    )
    parser.add_argument(
        "-p", "--predictor",
        type=str,
        default=None,
        help="Path to dlib shape predictor (default: models/shape_predictor_68_face_landmarks.dat)"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode (saves intermediate images)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.input_image).exists():
        print(f"Error: Input image not found: {args.input_image}", file=sys.stderr)
        sys.exit(1)
    
    if not validate_image_format(args.input_image):
        print(f"Error: Unsupported image format. Only JPG and PNG are supported.", file=sys.stderr)
        sys.exit(1)
    
    if args.output is None:
        input_path = Path(args.input_image)
        args.output = str(input_path.parent / f"{input_path.stem}_madurified{input_path.suffix}")
    
    if args.maduro_face is None:
        assets_dir = Path(get_assets_path("maduro_face.jpg")).parent
        maduro_files = list(assets_dir.glob("maduro_face*.jpg")) + list(assets_dir.glob("maduro_face*.png"))
        if len(maduro_files) == 0:
            default_path = get_assets_path("maduro_face.jpg")
            print(f"Error: No Maduro face images found in {assets_dir}", file=sys.stderr)
            print(f"Please place face image(s) at: {default_path} or maduro_face*.jpg in assets/", file=sys.stderr)
            sys.exit(1)
        args.maduro_face = [str(f) for f in maduro_files]
    else:
        args.maduro_face = args.maduro_face if isinstance(args.maduro_face, list) else [args.maduro_face]
    
    for face_path in args.maduro_face:
        if not Path(face_path).exists():
            print(f"Error: Maduro face image not found: {face_path}", file=sys.stderr)
            sys.exit(1)
    
    if args.predictor is None:
        args.predictor = get_models_path("shape_predictor_68_face_landmarks.dat")
    
    try:
        print("Loading images...")
        target_image = load_image(args.input_image)
        
        print("Initializing face swapper...")
        swapper = FaceSwapper(args.maduro_face, args.predictor, debug=args.debug)
        
        print("Processing image...")
        result = swapper.swap_face(target_image, debug_path=args.output if args.debug else None)
        
        if args.debug:
            print(f"Debug images saved to: {Path(args.output).parent}")
        
        print(f"Saving result to {args.output}...")
        save_image(result, args.output)
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

