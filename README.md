
# TAP VA Assessment 2024 - (Siow Wei Kang Bryan)

This project demonstrates how to perform inference using the [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) model through FastAPI endpoints. The application is containerized with Docker and includes a Streamlit interface for easy interaction.


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

TODO: upload weights to s3/gcp
```
curl 
```
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
