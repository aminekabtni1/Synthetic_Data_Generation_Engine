"""Vercel entrypoint — exposes the FastAPI app as a serverless function."""
from syndata.interface.app import app

# Vercel expects `app` to be the ASGI app. No extra handler needed.
# For Mangum compatibility (if Vercel uses AWS Lambda adapter), this also works:
# from mangum import Mangum
# handler = Mangum(app)
