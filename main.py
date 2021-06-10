import os

from fastapi import FastAPI
from fastapi import File, UploadFile

from routers import k_means, optimization, preprocessing

app = FastAPI()
_STORAGE_PATH = './uploaded_files'


@app.post('/cluster/data/uploadfile/')
async def create_upload_file(file: UploadFile = File(...)):
    if file.filename:
        filename = file.filename
        file_path = os.path.join(_STORAGE_PATH, filename)
        open(file_path, 'wb').write(await file.read())

        return {'file_path': file_path, 'file_name': filename}

app.include_router(k_means.router)
app.include_router(optimization.router)
app.include_router(preprocessing.router)
