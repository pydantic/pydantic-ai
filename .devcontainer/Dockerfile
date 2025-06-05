FROM python:3.12-bookworm

ARG GIT_USERNAME
ARG GIT_EMAIL
ARG USERNAME=coder
ARG USER_UID=1000
ARG USER_GID=1000
ARG WORKSPACE=/home/$USERNAME/workspace
ARG DEBIAN_FRONTEND=noninteractive


ENV SHELL=/bin/bash
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN apt -y update \
    && apt -y upgrade

RUN apt -y install inetutils-ping vim-nox htop git curl wget net-tools 	bind9-dnsutils locales
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG=en_US.UTF-8  
ENV LANGUAGE=en_US:en  
ENV LC_ALL=en_US.UTF-8     

RUN apt -y autoremove \
    && apt -y clean \
    && rm -Rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - 

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME:$USERNAME" | chpasswd \
    && adduser $USERNAME sudo

RUN mkdir -p $WORKSPACE \
    && chown -R $USERNAME:$USERNAME $WORKSPACE \
    && chmod -R u=rwX,go=rX $WORKSPACE

RUN pip install --no-cache-dir uv

USER $USERNAME

RUN git config --global user.email "$GIT_EMAIL"
RUN git config --global user.name "$GIT_USERNAME"