import logging
from flask import request, jsonify

logger = logging.getLogger(__name__)

def register_routes(app, get_model):
    @app.route("/")
    def api_ready():
        logger.info("Healthcheck received on /")
        return jsonify({'status': "TBA classifier API ready."})