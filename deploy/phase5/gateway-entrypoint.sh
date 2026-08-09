#!/bin/sh
set -eu

# The base image can use system trust outside OpenShift. The deployment sets
# VALIDATION_TLS_CA_BUNDLE to this Service CA path unconditionally, making a
# missing or malformed injected bundle a startup error. Preserve the same
# behavior for direct image runs when a non-empty bundle is mounted.
validation_ca_file=/var/run/secrets/watermark-validation-ca/service-ca.crt
if [ -s "$validation_ca_file" ]; then
    export VALIDATION_TLS_CA_BUNDLE="$validation_ca_file"
fi

exec "$@"
