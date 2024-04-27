import logging

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic.main import BaseModel

from route import ir_service
from uvicorn import run

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)


class ServiceInfo(BaseModel):
    service_name: str
    version: str


my_info = ServiceInfo(service_name='IR Service', version='1.0')
service_info = APIRouter()

@service_info.get('/service_info', response_model=ServiceInfo)
async def service_info_api():
    return my_info


logger.info('Init {} version {}'.format(my_info.service_name, my_info.version))
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
app.include_router(service_info)
app.include_router(ir_service)

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn, specifying the port
    run(app, host="0.0.0.0", port=8000)