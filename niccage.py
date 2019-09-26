# Borrowed heavily from https://github.com/simonw/cougar-or-not
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from fastai.vision import open_image, get_transforms, models, load_learner
import uvicorn
import torch
import sys
import uvicorn
import aiohttp
import asyncio
from pathlib import Path
from io import BytesIO


app = Starlette(debug=True)
path = Path(".")
cage_learner = load_learner(path)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    value = predict_image_from_bytes(bytes)
    return templates.TemplateResponse(value, {"request": request})


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    value = predict_image_from_bytes(bytes)
    return templates.TemplateResponse(value, {"request": request})


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    losses = cage_learner.predict(img)
    if losses[0].obj == "niccage":
        return "yes.html"
    return "no.html"


@app.route("/")
def form(request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
