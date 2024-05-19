from flask import Flask, request, jsonify, render_template
from rate_text import RateText

app = Flask(__name__)


@app.route('/api/rate', methods=['POST'])
def hello_world():  # put application's code here
    """Handles incoming requests to the API.

    Args:
        request (flask.Request): The incoming request object.

    Returns:
        flask.Response: The JSON response to be sent back to the client.
    """

    try:
        data = request.get_json()
        if not data:
            raise ValueError("No data received")

        text = data['text']
        print("text:", text)

        rate = RateText(text).detect_evaluate()

        response = {
            'message': 'Data received successfully',
            'data': rate
        }
        return jsonify(response), 200

    except (ValueError, KeyError) as e:
        # Handle specific exceptions for better error reporting
        print(f"Error processing request: {e}")
        response = {
            'message': f"Error: {str(e)}"
        }
        return jsonify(response), 400

    except Exception as e:
        # Catch-all for unexpected errors
        print(f"An unexpected error occurred: {e}")
        response = {
            'message': 'Internal server error'
        }
        return jsonify(response), 500


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
