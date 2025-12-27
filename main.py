from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from esrgan import super_resolve

app = FastAPI(title="ESRGAN API")

@app.post("/super-resolve")
async def super_resolve_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    sr_bytes = super_resolve(image_bytes)

    return Response(
        content=sr_bytes,
        media_type="image/png"
    )
