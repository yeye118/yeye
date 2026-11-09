FROM python:3.11-slim

WORKDIR /app/program

COPY requirements.txt /app/program/requirements.txt
RUN pip install --no-cache-dir -r /app/program/requirements.txt

COPY . /app/program/solution/

# Windows 打包常把 .sh 变成 CRLF，导致容器内 exec 报 "$'\r': command not found"
RUN sed -i 's/\r$//' /app/program/solution/run.sh && chmod +x /app/program/solution/run.sh

WORKDIR /app/program/solution

ENV DATA_ROOT=/input

ENTRYPOINT ["/app/program/solution/run.sh"]
CMD ["--task", "all"]
