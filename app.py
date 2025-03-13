import json
from ariadne import graphql_sync, make_executable_schema, gql, load_schema_from_path, ObjectType, ScalarType
from flask import request,jsonify,Flask
import sys

from api import settings
from api import resolve_fitApplyFourierCorrection

app = Flask(__name__)
json_scalar = ScalarType('JSON')
@json_scalar.serializer
def serialize_json(value):
    return value
@json_scalar.value_parser
def parse_json_value(value):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
@json_scalar.literal_parser
def parse_json_literal(ast):
    # Assuming AST is a string representing a JSON object
    try:
        return json.loads(ast.value)
    except (json.JSONDecodeError, TypeError):
        return None

query = ObjectType("Query")
query.set_field('fitApplyFourierCorrection', resolve_fitApplyFourierCorrection)
type_defs = load_schema_from_path(settings.GRAPHQL_SCHEMA)
schema = make_executable_schema(
    type_defs,query
)

@app.route("/health", methods=["GET"])
def health():
    return 'OK', 200

# GraphQL endpoint
@app.route('/graphql', methods=["POST"])
def graphql():
    data = request.get_json()
    success, result = graphql_sync(schema, data)
    status_code = 200 if success else 400 
    return jsonify(result), status_code


if __name__ == '__main__':
    app.run(host=settings.SERVER_HOST,port=settings.SERVER_PORT,debug=True)