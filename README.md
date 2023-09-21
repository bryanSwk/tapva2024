
# TAP VA Assessment 2024 - (Siow Wei Kang Bryan)

This project demonstrates how to perform inference using the [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) model through FastAPI endpoints. The application is containerized with Docker and includes a Streamlit interface for easy interaction.

Examples:

<img src="./assets/github-example.png" width="1000">

## Installation

1. Clone the repository:

```
git clone https://github.com/bryanSwk/tapva2024.git
cd tapva2024
```

2. Create the environment:

```
conda create -n FastSAM python=3.9
conda activate FastSAM
```

3. Install Torch from: [Link](https://pytorch.org/get-started/locally/)

4. Install requirements:

```
pip install -r requirements.txt
```

5. Install CLIP:
```
pip install git+https://github.com/openai/CLIP.git
```
## Usage/Examples

Download weights using:

```
./scripts/dl-weights.sh
```

if transfer.sh link expires:

```
ViT-B-32.pt: https://drive.google.com/file/d/1ucNN-iEx4i-omhzuuQMMlSOnndM01CyG/view?usp=drive_link
FastSAM-x.pt: https://drive.google.com/file/d/1qhxab2Qpj08AaqrJMmCby1oEI8QqzDze/view?usp=drive_link

Download and place in tapva2024/weights
```

### IMPORTANT!: Before building inference container, ensure that `scripts/entrypoint/api-entrypoint.sh` has the `LF` as EOL Sequence as git tends to convert it to `CRLF`.

Build inference container:
```
docker build `
    -t "fastapi:latest" `
    -f docker/fastapi.Dockerfile `
    --platform linux/amd64 .
```
Run FastAPI container:
```
docker run --rm -it -p 4000:4000 `
    --name fastapi-server `
    --gpus all `
    -v "$(Get-Location)\weights:/home/user/tap-va/weights" `
    fastapi:latest
```
Start Streamlit App:
```
streamlit run app.py
```

## Unit Tests

To run unit tests, spin up an inference container or run it locally:
Run pytest in root dir afterwards.

```
python ./inference_fastapi.py
pytest
```

There are 3 parts to the pytest:

- Health
- Functional
- Format

This should be the valid output for the correct configuration:

```
================================================= 10 passed in 7.16s ==================================================
```

## Optimization

<img src="./assets/inference-timings.png" width="1000">

During loading of the model,