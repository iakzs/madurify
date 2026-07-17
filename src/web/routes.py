from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
import numpy as np
from PIL import Image
import io
import tempfile
from pathlib import Path
from src.core.face_swapper import FaceSwapper
from src.core.image_utils import validate_image_format
from src.core.video import is_video_file, process_video

router = APIRouter()

MAX_VIDEO_BYTES = 200 * 1024 * 1024  # 200 MB upload guard

_swapper = None
_maduro_face_path = None


def initialize_swapper(maduro_face_paths, predictor_path=None):
    global _swapper, _maduro_face_path
    if isinstance(maduro_face_paths, str):
        maduro_face_paths = [maduro_face_paths]
    _maduro_face_path = maduro_face_paths
    try:
        _swapper = FaceSwapper(maduro_face_paths, predictor_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize face swapper: {str(e)}")


@router.post("/process")
async def process_image(file: UploadFile = File(...), fmt: str = "jpeg"):
    if _swapper is None:
        raise HTTPException(status_code=500, detail="Face swapper not initialized")

    if not validate_image_format(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Only JPG and PNG are supported."
        )

    fmt = fmt.lower()
    if fmt not in ("jpeg", "jpg", "png"):
        fmt = "jpeg"

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_array = np.array(image)

        result = _swapper.swap_face(image_array)

        result_image = Image.fromarray(result.astype(np.uint8))
        output = io.BytesIO()

        save_fmt = "JPEG" if fmt in ("jpeg", "jpg") else "PNG"
        save_kwargs = {"quality": 95} if save_fmt == "JPEG" else {}
        result_image.save(output, format=save_fmt, **save_kwargs)
        output.seek(0)

        ext = "jpg" if save_fmt == "JPEG" else "png"
        media_type = f"image/{ext}" if ext == "png" else "image/jpeg"

        return Response(
            content=output.read(),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=madurified.{ext}"}
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "ok", "swapper_initialized": _swapper is not None}


@router.post("/process-video")
async def process_video_endpoint(file: UploadFile = File(...)):
    if _swapper is None:
        raise HTTPException(status_code=500, detail="Face swapper not initialized")

    if not file.filename or not is_video_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Supported: mp4, avi, mov, mkv, webm, m4v."
        )

    contents = await file.read()
    if len(contents) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Video too large (max 200 MB)")

    suffix = Path(file.filename).suffix.lower()
    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_in.write(contents)
    tmp_in.close()
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()

    input_path = Path(tmp_in.name)
    output_path = Path(tmp_out.name)

    def _cleanup():
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    try:
        await run_in_threadpool(
            process_video, _swapper, input_path, output_path, 0.6, 3, 0.65, True, None
        )
    except ValueError as e:
        _cleanup()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _cleanup()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="madurified.mp4",
        background=BackgroundTask(_cleanup),
    )

