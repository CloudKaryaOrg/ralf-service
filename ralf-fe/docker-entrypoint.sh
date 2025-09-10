#!/bin/sh
# Export secrets from api_keys.sh as environment variables

if [ -f /run/secrets/api_keys.sh ]; then
# set -a
	. /run/secrets/api_keys.sh
# set +a
fi

exec streamlit run ralf/app.py --server.port=8501 --server.address=0.0.0.0