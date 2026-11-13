from concurrent.futures import ThreadPoolExecutor
import io
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from src.core.animation import process_animation
from src.core.face_swapper import FaceSwapper
from src.core.image_utils import is_animated, orient_image, save_animation
from src.core.video import is_video_file, process_video

router = APIRouter()

MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
JOB_TTL_SECONDS = 30 * 60
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
QUALITY_PRESETS = {
    "fast": {"scale": 0.55, "detect_every": 1, "max_edge": 960},
    "balanced": {"scale": 0.7, "detect_every": 1, "max_edge": 1280},
    "quality": {"scale": 0.85, "detect_every": 1, "max_edge": 1920},
}

_swapper = None
_maduro_face_path = None
_jobs = {}
_jobs_lock = threading.Lock()
_media_executor = ThreadPoolExecutor(max_workers=1)


def initialize_swapper(maduro_face_paths, predictor_path=None):
    global _swapper, _maduro_face_path
    if isinstance(maduro_face_paths, str):
        maduro_face_paths = [maduro_face_paths]
    _maduro_face_path = maduro_face_paths
    try:
        _swapper = FaceSwapper(maduro_face_paths, predictor_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize face swapper: {str(e)}")


def _preset(quality):
    return QUALITY_PRESETS.get((quality or "").lower(), QUALITY_PRESETS["balanced"])


def _fmt_from_name(name):
    ext = Path(name or "").suffix.lower()
    if ext == ".png":
        return "png"
    if ext == ".webp":
        return "webp"
    if ext == ".gif":
        return "gif"
    return "jpeg"


def _unlink_job_files(job):
    for key in ("input_path", "output_path"):
        path = job.get(key)
        if path is not None:
            Path(path).unlink(missing_ok=True)


def _purge_expired_jobs():
    now = time.time()
    expired = []
    with _jobs_lock:
        for job_id, job in list(_jobs.items()):
            if job["status"] in {"completed", "failed"} and now - job["created"] > JOB_TTL_SECONDS:
                expired.append((job_id, job))
        for job_id, _job in expired:
            _jobs.pop(job_id, None)
    for _job_id, job in expired:
        _unlink_job_files(job)


def _new_job(input_path, output_path, media_type, filename):
    _purge_expired_jobs()
    job_id = uuid.uuid4().hex
    job = {
        "status": "queued",
        "progress": 0,
        "total": 0,
        "error": None,
        "input_path": input_path,
        "output_path": output_path,
        "media_type": media_type,
        "filename": filename,
        "audio_dropped": False,
        "audio_note": None,
        "created": time.time(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    return job_id


def _job_payload(job_id):
    return {
        "job_id": job_id,
        "status_url": f"/jobs/{job_id}",
        "download_url": f"/jobs/{job_id}/download",
    }


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


def _encode_still(image, fmt):
    output = io.BytesIO()
    if fmt == "png":
        image.save(output, format="PNG")
        return output.getvalue(), "png", "image/png"
    if fmt == "webp":
        image.save(output, format="WEBP", quality=90)
        return output.getvalue(), "webp", "image/webp"
    if fmt == "gif":
        image.save(output, format="GIF")
        return output.getvalue(), "gif", "image/gif"
    image.save(output, format="JPEG", quality=95)
    return output.getvalue(), "jpg", "image/jpeg"


def _still_response(image, fmt):
    content, ext, media_type = _encode_still(image, fmt)
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename=madurified.{ext}"},
    )


def _swap_still(swapper, contents):
    with Image.open(io.BytesIO(contents)) as image:
        oriented = orient_image(image)
        rgb = np.array(oriented.convert("RGB"))
    result = swapper.swap_face(rgb)
    return Image.fromarray(result.astype(np.uint8))


def _swap_animation_bytes(swapper, contents, max_edge):
    with Image.open(io.BytesIO(contents)) as image:
        frames, durations, loop, kind = process_animation(
            swapper, image, max_edge=max_edge
        )
    output = io.BytesIO()
    save_animation(frames, output, durations, loop=loop, kind=kind)
    media_type = "image/webp" if kind == "WEBP" else "image/gif"
    ext = "webp" if kind == "WEBP" else "gif"
    return output.getvalue(), ext, media_type


@router.post("/process")
async def process_image(
    file: UploadFile = File(...),
    fmt: str | None = None,
    quality: str = Form("balanced"),
):
    if _swapper is None:
        raise HTTPException(status_code=500, detail="Face swapper not initialized")

    if not file.filename or Path(file.filename).suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Supported: JPG, PNG, GIF, WebP."
        )

    if fmt is None:
        fmt = _fmt_from_name(file.filename)
    fmt = fmt.lower()
    if fmt not in ("jpeg", "jpg", "png", "webp", "gif"):
        fmt = "jpeg"
    if fmt == "jpg":
        fmt = "jpeg"

    try:
        contents = await read_upload_limited(file, MAX_IMAGE_BYTES)
        with Image.open(io.BytesIO(contents)) as image:
            animated = is_animated(image)
        preset = _preset(quality)
        if animated:
            content, ext, media_type = await run_in_threadpool(
                _swap_animation_bytes, _swapper, contents, preset["max_edge"]
            )
            return Response(
                content=content,
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename=madurified.{ext}"},
            )
        result_image = await run_in_threadpool(_swap_still, _swapper, contents)
        return _still_response(result_image, fmt)
    except HTTPException:
        raise
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        await file.close()


@router.post("/process-animation")
async def process_animation_endpoint(
    file: UploadFile = File(...),
    fmt: str | None = None,
    quality: str = Form("balanced"),
):
    if _swapper is None:
        raise HTTPException(status_code=500, detail="Face swapper not initialized")

    if not file.filename or Path(file.filename).suffix.lower() not in {".gif", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Supported: GIF, WebP."
        )

    if fmt is None:
        fmt = _fmt_from_name(file.filename)
    fmt = "webp" if fmt.lower() == "webp" else "gif"

    try:
        contents = await read_upload_limited(file, MAX_IMAGE_BYTES)
        with Image.open(io.BytesIO(contents)) as image:
            animated = is_animated(image)
            source_format = image.format
    except HTTPException:
        raise
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        await file.close()

    preset = _preset(quality)
    if not animated:
        try:
            result_image = await run_in_threadpool(_swap_still, _swapper, contents)
            return _still_response(result_image, fmt)
        except (ValueError, OSError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    suffix = ".webp" if source_format == "WEBP" else ".gif"
    media_type = "image/webp" if suffix == ".webp" else "image/gif"
    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_in.write(contents)
    tmp_in.close()
    tmp_out = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_out.close()
    input_path = Path(tmp_in.name)
    output_path = Path(tmp_out.name)
    job_id = _new_job(input_path, output_path, media_type, f"madurified{suffix}")
    _media_executor.submit(
        _run_animation_job, job_id, input_path, output_path, _swapper, preset["max_edge"]
    )
    return JSONResponse(_job_payload(job_id))


def _run_animation_job(job_id, input_path, output_path, swapper, max_edge):
    def update_progress(done, total):
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update(status="processing", progress=done, total=total)

    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["status"] = "processing"

    try:
        with Image.open(input_path) as image:
            frames, durations, loop, kind = process_animation(
                swapper, image, max_edge=max_edge, progress_cb=update_progress
            )
        save_animation(frames, output_path, durations, loop=loop, kind=kind)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["status"] = "completed"
    except Exception as e:
        output_path.unlink(missing_ok=True)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update(status="failed", error=str(e))
    finally:
        input_path.unlink(missing_ok=True)


@router.get("/health")
async def health_check():
    return {"status": "ok", "swapper_initialized": _swapper is not None}


@router.post("/process-video")
async def process_video_endpoint(
    file: UploadFile = File(...),
    quality: str = Form("balanced"),
):
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
        await save_upload_limited(file, input_path, MAX_VIDEO_BYTES)
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

    preset = _preset(quality)
    job_id = _new_job(input_path, output_path, "video/mp4", "madurified.mp4")
    _media_executor.submit(
        _run_video_job, job_id, input_path, output_path, _swapper, preset
    )
    return _job_payload(job_id)


def _run_video_job(job_id, input_path, output_path, swapper, preset):
    def update_progress(done, total):
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update(status="processing", progress=done, total=total)

    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["status"] = "processing"

    info = {}
    try:
        process_video(
            swapper, input_path, output_path,
            scale=preset["scale"],
            detect_every=preset["detect_every"],
            keep_audio=True,
            progress_cb=update_progress,
            max_edge=preset["max_edge"],
            media_info=info,
        )
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update(
                    status="completed",
                    audio_dropped=info.get("audio_dropped", False),
                    audio_note=info.get("audio_note"),
                )
    except Exception as e:
        output_path.unlink(missing_ok=True)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update(status="failed", error=str(e))
    finally:
        input_path.unlink(missing_ok=True)


def _job_status(job_id):
    _purge_expired_jobs()
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return {
            "status": job["status"],
            "progress": job["progress"],
            "total": job["total"],
            "error": job["error"],
            "audio_dropped": job["audio_dropped"],
            "audio_note": job["audio_note"],
        }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    status = _job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.get("/jobs/{job_id}/download")
async def download_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail="Processing is not complete")
        output_path = job["output_path"]
        input_path = job["input_path"]
        media_type = job["media_type"]
        filename = job["filename"]

    if output_path is None or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Result expired")

    def cleanup_job():
        if input_path is not None:
            Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
        with _jobs_lock:
            _jobs.pop(job_id, None)

    return FileResponse(
        output_path,
        media_type=media_type,
        filename=filename,
        background=BackgroundTask(cleanup_job),
    )
