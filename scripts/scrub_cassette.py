"""One-time script to apply serializer redaction to an existing cassette."""

import sys
sys.path.insert(0, 'tests')

from json_body_serializer import deserialize, serialize

cassette_path = sys.argv[1] if len(sys.argv) > 1 else (
    'tests/models/cassettes/test_multimodal_tool_returns/'
    'test_multimodal_tool_return_matrix[direct-uploaded_file-image-bedrock_nova].yaml'
)

with open(cassette_path) as f:
    content = f.read()

# Round-trip through deserialize/serialize to apply redaction
cassette_dict = deserialize(content)
redacted = serialize(cassette_dict)

with open(cassette_path, 'w') as f:
    f.write(redacted)

# Verify
if 'SCRUBBED' in redacted and 'ASIAVEMKNXYDQ6ZF4HY7' not in redacted:
    print(f'Successfully redacted credentials in {cassette_path}')
else:
    print('WARNING: Redaction may not have worked correctly')
    sys.exit(1)
