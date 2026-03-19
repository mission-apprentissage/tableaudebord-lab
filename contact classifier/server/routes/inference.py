import logging
from flask import request, jsonify

logger = logging.getLogger(__name__)

def register_routes(app, get_model):
    """Register inference routes."""

    @app.route('/model/score', methods=['POST'])
    def score():
        if not request.is_json:
            logger.warning("Non-JSON request received on /score")
            return jsonify({'error': 'Request must be JSON'}), 400

        request_data = request.get_json()
        logger.info("Received /model/score data: %s", request_data)
        version = request_data.get('version')
        model = get_model(version)
        logger.info("Model loaded: %s", model.version)
        data = request_data.get('data')
        logger.info("Data sended: %s", data)
        result = model.score(data)
        logger.info("Scores computed: %s", result)
        return jsonify(result), 200
