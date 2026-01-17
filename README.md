# Running Guide for Linux

## Table of Contents

- [Install Docker](#install-docker)
- [Configure Docker](#configure-docker)
- [Build Image](#build-image)
- [Validate & Push](#validate--push)
- [How the Organizers Run the Image](#how-the-organizers-run-the-image)

---

## Install Docker

Download and install Docker following the guide at: <https://docs.docker.com/get-started/get-docker/>

Check the installed Docker version:

```bash
sudo docker --version
```

Check the Docker service status:

```bash
sudo systemctl status docker
```

Check running containers:

```bash
docker ps
```

Start Docker service (if not running):

```bash
sudo systemctl start docker
```

---

## Configure Docker

### 1. Clone Repository

```bash
git clone <>

cd sport
```

### 2. Configure Docker Daemon

Open the Docker configuration file:

```bash
sudo nano /etc/docker/daemon.json
```

Add the following content to the file:

```json
{
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  },
  "insecure-registries": ["222.255.250.24:8002"], // Add the domain here
  "max-concurrent-uploads": 1
}
```

> **Note:**
> - `insecure-registries`: Add the organizer's registry domain
> - `max-concurrent-uploads`: Set to 1 to avoid RAM overflow when pushing

### 3. Restart Docker

```bash
sudo systemctl restart docker
```

### 4. Login to Registry

```bash
docker login 222.255.250.24:8002 -u <team_id>
```

> **Note:** Use the account provided by the organizers to log in.

---

## Build Image

Build Docker image with cache from base image:

```bash
docker build --cache-from 222.255.250.24:8001/data-storm/pytorch:2.8.0-cuda12.8-cudnn9-devel \
  -t DOMAIN/TEAM_ID/submission .
```

> **Note:** Use the `--cache-from` tag to optimize push speed.

---

## Validate & Push

### Validate Submission

Run the validation script before pushing:

```bash
bash validate_submission.sh
```

### Push Image

```bash
docker push 222.255.250.24:8002/<team_id>/submission
```

---

## How the Organizers Run the Image

The code that the organizers will use to run participants' images:

```bash
docker run --rm --network none --gpus '"device=1"' \
  -v /path/to/input:/data/input:ro \
  -v /path/to/results:/data/output \
  -e INPUT_PATH=/data/input \
  -e OUTPUT_PATH=/data/output \
  DOMAIN/your_team/submission
```

| Parameter | Description |
|-----------|-------------|
| `--rm` | Automatically remove container after execution |
| `--network none` | Disable network connection |
| `--gpus '"device=1"'` | Use GPU device 1 |
| `-v /path/to/input:/data/input:ro` | Mount input directory (read-only) |
| `-v /path/to/results:/data/output` | Mount output directory |
| `-e INPUT_PATH` | Environment variable for input path |
| `-e OUTPUT_PATH` | Environment variable for output path |