ARG SKYWRIGHT_IMAGE=skywright:0.1.0-cpu
FROM ${SKYWRIGHT_IMAGE}
COPY --chown=skywright:skywright training.py ./
CMD ["training:setup", "--epochs", "2"]
