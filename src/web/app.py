from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from .routes import router, initialize_swapper
from src.core.paths import get_assets_path, get_models_path

app = FastAPI(title="Madurify API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.on_event("startup")
async def startup_event():
    from pathlib import Path
    assets_dir = Path(get_assets_path("maduro_face.jpg")).parent
    maduro_files = list(assets_dir.glob("maduro_face*.jpg")) + list(assets_dir.glob("maduro_face*.png"))
    predictor_path = get_models_path("shape_predictor_68_face_landmarks.dat")
    
    if len(maduro_files) == 0:
        print(f"Warning: No Maduro face images found in {assets_dir}")
        print("Please place face image(s) named maduro_face*.jpg or maduro_face*.png in assets/")
    else:
        try:
            maduro_paths = [str(f) for f in maduro_files]
            initialize_swapper(
                maduro_paths,
                predictor_path if Path(predictor_path).exists() else None
            )
            print(f"Face swapper initialized with {len(maduro_paths)} source image(s)")
        except Exception as e:
            print(f"Warning: Could not initialize face swapper: {e}")

