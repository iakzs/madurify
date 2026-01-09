from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
import numpy as np
from PIL import Image
import io
from pathlib import Path
from src.core.face_swapper import FaceSwapper
from src.core.image_utils import validate_image_format

router = APIRouter()

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
async def process_image(file: UploadFile = File(...)):
    if _swapper is None:
        raise HTTPException(status_code=500, detail="Face swapper not initialized")
    
    if not validate_image_format(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Only JPG and PNG are supported."
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        result = _swapper.swap_face(image_array)
        
        result_image = Image.fromarray(result.astype(np.uint8))
        output = io.BytesIO()
        result_image.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        return Response(
            content=output.read(),
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=madurified.jpg"}
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "ok", "swapper_initialized": _swapper is not None}

