# Phase 1 image: Ubuntu 22.04 with its SYSTEM python3.10.
#
# This is deliberately NOT `ros:humble`, even though phase 2 needs ROS 2 Humble:
#   * ros:humble is itself FROM ubuntu:22.04 with the same glibc and the same Python 3.10,
#     so nothing ABI-relevant differs between this image and the phase-2 one.
#   * It is ~2 GB instead of ~300 MB, which is slow to rebuild over Docker Desktop.
#   * Most importantly: with rclpy absent, the rule "the core package never imports ROS"
#     is enforced by the package not existing. An accidental `import rclpy` fails here.
#
# No virtualenv: Humble's rclpy lives in the system interpreter's site-packages, so a venv
# would need --system-site-packages contortions in phase 2. Install into system Python.
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user at UID/GID 1000 to match the default WSL2 user, so files created in the
# container are owned correctly on the host. Check `id -u` in your distro; if it is not
# 1000, override these build args rather than fighting the permissions.
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# MPLBACKEND=Agg: the container is headless. Plots are written into the run folder as PNG,
# never displayed. PYTHONPATH=/workspace instead of an editable install, so nothing writes
# .egg-info into the bind-mounted source tree and `python3 -m crop_mot` just works.
ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

USER ${USERNAME}
WORKDIR /workspace
CMD ["bash"]
