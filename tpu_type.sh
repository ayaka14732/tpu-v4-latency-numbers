#!/usr/bin/env bash
set -euo pipefail

accelerator_type=$(curl --noproxy '*' -fsS --connect-timeout 2 --max-time 5 \
    -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type)
if [[ -z "${accelerator_type//[[:space:]]/}" ]]; then
    printf 'No TPU accelerator type returned\n' >&2
    exit 1
fi

endpoints=$(curl --noproxy '*' -fsS --connect-timeout 2 --max-time 5 \
    -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints)

multi_host=$(printf '%s\n' "$endpoints" | awk -F ',' '
    {
        for (i = 1; i <= NF; i++) {
            if ($i ~ /[^[:space:]]/) count++
        }
    }
    END {
        if (count == 0) {
            print "No TPU worker endpoints returned" > "/dev/stderr"
            exit 1
        }
        print (count > 1 ? "true" : "false")
    }
')

printf 'accelerator_type=%s\nmulti_host=%s\n' "$accelerator_type" "$multi_host"
