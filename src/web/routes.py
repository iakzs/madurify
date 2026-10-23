from concurrent.futures import ThreadPoolExecutor
import threading
import uuid

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
from src.core.video import is_video_file, process_video

router = APIRouter()

MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

_swapper = None
_maduro_face_path = None
_video_jobs = {}
_video_jobs_lock = threading.Lock()
_video_executor = ThreadPoolExecutor(max_workers=1)


def initialize_swapper(maduro_face_paths, predictor_path=None):
    global _swapper, _maduro_face_path
    if isinstance(maduro_face_paths, str):
        maduro_face_paths = [maduro_face_paths]
    _maduro_face_path = maduro_face_paths
    try:
        _swapper = FaceSwapper(maduro_face_paths, predictor_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize face swapper: {str(e)}")


async def read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    contents = bytearray()
    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
        contents.extend(chunk)
        if len(contents) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large (max {max_bytes // (1024 * 1024)} MB)")
    return bytes(contents)


async def save_upload_limited(file: UploadFile, destination: Path, max_bytes: int) -> None:
    size = 0
    with destination.open("wb") as output:
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(status_code=413, detail=f"File too large (max {max_bytes // (1024 * 1024)} MB)")
            output.write(chunk)


@router.post("/process")
async def process_image(file: UploadFile = File(...), fmt: str = "jpeg"):
    if _swapper is None:
        raise HTTPException(status_code=500, detail="Face swapper not initialized")

    if not file.filename or Path(file.filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".gif"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Supported: JPG, PNG, GIF."
        )

    fmt = fmt.lower()
    if fmt not in ("jpeg", "jpg", "png"):
        fmt = "jpeg"

    try:
        contents = await read_upload_limited(file, MAX_IMAGE_BYTES)
        image = Image.open(io.BytesIO(contents))
        animated_gif = image.format == "GIF" and getattr(image, "is_animated", False)

        # Copy each frame before advancing the GIF decoder.
        frames, durations = [], []
        if animated_gif:
            for frame_index in range(image.n_frames):
                image.seek(frame_index)
                frames.append(image.convert("RGB").copy())
                durations.append(image.info.get("duration", 100))
        else:
            frames = [image.copy()]

        processed_frames = []
        swapped_any_frame = False
        for frame in frames:
            try:
                result = await run_in_threadpool(
                    _swapper.swap_face, np.array(frame.convert("RGB"))
                )
                processed_frames.append(Image.fromarray(result.astype(np.uint8)))
                swapped_any_frame = True
            except ValueError as e:
                if not animated_gif or "No face detected in target image" not in str(e):
                    raise
                # Animated clips often have transition frames or brief occlusions.
                # Keep those frames so a single detection miss doesn't fail the GIF.
                processed_frames.append(frame.convert("RGB").copy())

        if animated_gif and not swapped_any_frame:
            raise ValueError("No face detected in any GIF frame")

        output = io.BytesIO()
        if animated_gif:
            processed_frames[0].save(output, format="GIF", save_all=True,
                                     append_images=processed_frames[1:], duration=durations,
                                     loop=image.info.get("loop", 0), disposal=2, optimize=False)
            ext, media_type = "gif", "image/gif"
        else:
            result_image = processed_frames[0]
            save_fmt = "JPEG" if fmt in ("jpeg", "jpg") else "PNG"
            result_image.save(output, format=save_fmt, **({"quality": 95} if save_fmt == "JPEG" else {}))
            ext = "jpg" if save_fmt == "JPEG" else "png"
            media_type = "image/jpeg" if save_fmt == "JPEG" else "image/png"

        output.seek(0)
        return Response(
            content=output.read(),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=madurified.{ext}"}
        )

    except HTTPException:
        raise
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        await file.close()


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

    suffix = Path(file.filename).suffix.lower()
    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_in.close()
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()

    input_path = Path(tmp_in.name)
    output_path = Path(tmp_out.name)
    try:
        try:
            await save_upload_limited(file, input_path, MAX_VIDEO_BYTES)
        finally:
            await file.close()
    except HTTPException:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        await file.close()

    job_id = uuid.uuid4().hex
    with _video_jobs_lock:
        _video_jobs[job_id] = {
            "status": "queued", "progress": 0, "total": 0,
            "error": None, "input_path": input_path, "output_path": output_path,
        }

    _video_executor.submit(_run_video_job, job_id, input_path, output_path, _swapper)
    return {"job_id": job_id, "status_url": f"/process-video/{job_id}"}


def _run_video_job(job_id, input_path, output_path, swapper):
    def update_progress(done, total):
        with _video_jobs_lock:
            job = _video_jobs.get(job_id)
            if job is not None:
                job.update(status="processing", progress=done, total=total)

    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
        if job is not None:
            job["status"] = "processing"

    try:
        process_video(swapper, input_path, output_path, 0.6, 2, 0.85,
                      True, update_progress)
        with _video_jobs_lock:
            job = _video_jobs.get(job_id)
            if job is not None:
                job["status"] = "completed"
    except Exception as e:
        output_path.unlink(missing_ok=True)
        with _video_jobs_lock:
            job = _video_jobs.get(job_id)
            if job is not None:
                job.update(status="failed", error=str(e))
    finally:
        input_path.unlink(missing_ok=True)


@router.get("/process-video/{job_id}")
async def get_video_job(job_id: str):
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Video job not found")
        return {
            "status": job["status"],
            "progress": job["progress"],
            "total": job["total"],
            "error": job["error"],
        }


@router.get("/process-video/{job_id}/download")
async def download_video_job(job_id: str):
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Video job not found")
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail="Video processing is not complete")
        output_path = job["output_path"]
        input_path = job["input_path"]

    def cleanup_job():
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        with _video_jobs_lock:
            _video_jobs.pop(job_id, None)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="madurified.mp4",
        background=BackgroundTask(cleanup_job),
    )
