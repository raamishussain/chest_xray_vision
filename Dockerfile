FROM public.ecr.aws/lambda/python:3.9

COPY chest_xray_vision/ ${LAMBDA_TASK_ROOT}/chest_xray_vision
COPY requirements.txt .

RUN pip3 install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

CMD ["chest_xray_vision.app.handler"]