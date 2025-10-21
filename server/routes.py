import logging
from flask import request, jsonify

logger = logging.getLogger(__name__)

def register_routes(app, get_model):
    @app.route("/")
    def api_ready():
        logger.info("Healthcheck received on /")
        return jsonify({'status': "TBA classifier API ready."})

    @app.route("/model/load", methods=['GET'])
    def load_model():
        version = request.args.get('version')
        if not version:
            log = "'version' argument missing."
            logger.warning(log)
            return jsonify({'error': log}), 400

        logger.debug("Received /model/load: %s", version)
        model = get_model(version=version)
        logger.info("Model version ready: %s", model.version)
        return jsonify({'model': model.version}), 200

    @app.route("/model/version", methods=['GET'])
    def model_version():
        model = get_model()
        version = model.version if model else None
        logger.info("Model version loaded: %s", version)
        return jsonify({'model': version}), 200

        @app.route('/model/score', methods=['POST'])

    def score():
        if not request.is_json:
            logger.warning("Non-JSON request received on /score")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        version = data.get('version')
        texts = data.get('texts')
        logger.debug("Received /model/score data: %s", data)

        if not isinstance(texts, list):
            logger.warning("Invalid /model/score payload: 'texts' is not a list")
            return jsonify({'error': '"texts" must be a list.'}), 400

        model = get_model(version)
        result = model.score(texts)
        logger.info("Labels computed for texts")
        return jsonify(result), 200