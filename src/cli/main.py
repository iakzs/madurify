import argparse
import sys
from pathlib import Path
from src.core.face_swapper import FaceSwapper
from src.core.image_utils import load_image, save_image, validate_image_format
from src.core.paths import get_assets_path, get_models_path
from src.core.video import is_video_file, process_video


def process_single(swapper, input_path, output_path, debug=False):
    try:
        if not validate_image_format(str(input_path)):
            print(f"  Skipping {input_path.name}: unsupported format")
            return False

        target_image = load_image(str(input_path))
        debug_path = str(output_path) if debug else None
        result = swapper.swap_face(target_image, debug_path=debug_path)
        save_image(result, str(output_path))

        if debug:
            print(f"  Debug images saved to: {output_path.parent}")

        print(f"  Saved: {output_path.name}")
        return True

    except ValueError as e:
        print(f"  Skipping {input_path.name}: {e}")
        return False
    except Exception as e:
        print(f"  Error processing {input_path.name}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Swap faces in images using geometric transformations"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Path to input image, directory of images, or video file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path (file for single image/video, directory for batch)"
    )
    parser.add_argument(
        "--cam",
        action="store_true",
        help="Real-time webcam mode (no input file needed)"
    )
    parser.add_argument(
        "--cam-index",
        type=int,
        default=0,
        help="Camera device index for --cam (default: 0)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Downscale factor for detection in video/cam mode "
             "(default: 0.6; 1.0 = full resolution)"
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=3,
        help="In video/cam mode, re-detect faces every N frames (default: 3)"
    )
    parser.add_argument(
        "-m", "--maduro-face",
        type=str,
        default=None,
        nargs="+",
        help="Path(s) to Maduro face template(s). Can specify multiple images."
    )
    parser.add_argument(
        "-p", "--predictor",
        type=str,
        default=None,
        help="Path to dlib shape predictor"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode (saves intermediate images)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_madurified",
        help="Suffix for output files in batch mode (default: _madurified)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["jpg", "png"],
        help="Output format for batch mode (default: same as input)"
    )

    args = parser.parse_args()

    if not args.cam and args.input is None:
        parser.error("input is required unless --cam is used")

    input_path = Path(args.input) if args.input else None

    if not args.cam and not input_path.exists():
        print(f"Error: Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.maduro_face is None:
        assets_dir = Path(get_assets_path("maduro_face.jpg")).parent
        maduro_files = list(assets_dir.glob("maduro_face*.jpg")) + list(assets_dir.glob("maduro_face*.png"))
        if len(maduro_files) == 0:
            default_path = get_assets_path("maduro_face.jpg")
            print(f"Error: No Maduro face images found in {assets_dir}", file=sys.stderr)
            print(f"Place face image(s) at: {default_path} or maduro_face*.jpg in assets/", file=sys.stderr)
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

    print(f"Initializing face swapper with {len(args.maduro_face)} source image(s)...")
    try:
        swapper = FaceSwapper(args.maduro_face, args.predictor, debug=args.debug)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.cam:
        from src.cli.cam import run_camera
        run_camera(
            swapper,
            cam_index=args.cam_index,
            scale=args.scale if args.scale is not None else 0.6,
            detect_every=args.detect_every,
            snapshot_dir=str(input_path) if input_path else ".",
        )
        return

    if input_path.is_file() and is_video_file(input_path):
        if args.output is None:
            output = input_path.parent / f"{input_path.stem}{args.suffix}.mp4"
        else:
            output = Path(args.output)

        scale = args.scale if args.scale is not None else 0.6
        print(f"Processing video {input_path.name} (detection scale {scale}, "
              f"re-detect every {args.detect_every} frames)...")

        def _progress(done, total):
            if total > 0:
                pct = done * 100 // total
                print(f"\r  {done}/{total} frames ({pct}%)", end="", flush=True)
            else:
                print(f"\r  {done} frames", end="", flush=True)

        try:
            out = process_video(swapper, input_path, output, scale=scale,
                                detect_every=args.detect_every,
                                progress_cb=_progress)
        except ValueError as e:
            print(f"\nError: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"\n  Saved: {out}")
        return

    if input_path.is_file():
        if not validate_image_format(str(input_path)):
            print(f"Error: Unsupported format. Only JPG and PNG supported.", file=sys.stderr)
            sys.exit(1)

        if args.output is None:
            output = str(input_path.parent / f"{input_path.stem}{args.suffix}{input_path.suffix}")
        else:
            output = args.output

        print(f"Processing {input_path.name}...")
        success = process_single(swapper, input_path, Path(output), args.debug)
        if not success:
            sys.exit(1)

    else:
        valid_ext = {'.jpg', '.jpeg', '.png'}
        image_files = [f for f in sorted(input_path.iterdir())
                       if f.suffix.lower() in valid_ext and f.is_file()]

        if len(image_files) == 0:
            print(f"Error: No JPG or PNG images found in {input_path}", file=sys.stderr)
            sys.exit(1)

        out_dir = Path(args.output) if args.output else input_path / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        fail_count = 0

        print(f"Processing {len(image_files)} images from {input_path}...")
        for img_file in image_files:
            suffix = args.suffix if args.suffix else ""
            out_ext = f".{args.format}" if args.format else img_file.suffix
            out_name = f"{img_file.stem}{suffix}{out_ext}"
            out_path = out_dir / out_name

            if process_single(swapper, img_file, out_path, args.debug):
                success_count += 1
            else:
                fail_count += 1

        print(f"\nDone! {success_count} succeeded, {fail_count} failed")
        print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
