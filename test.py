import re
import json

texts = [
    r'{"a": "hello \escape"}',     # single invalid escape
    r'{"a": "hello \\escape"}',    # literal backslash followed by escape
    r'{"a": "hello \n"}',          # valid escape
    r'{"a": "hello \\n"}',         # literal backslash followed by n
    r'{"a": "hello \\\escape"}',   # three backslashes
]

for i, text in enumerate(texts):
    print(f"Test {i+1}:")
    print("Original:", repr(text))
    # Replace any backslash that is neither preceded by a backslash nor followed by a valid escape char.
    res = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r'\\\\', text)
    print("Result of sub:", repr(res))
    try:
        json.loads(res)
        print("Success")
    except Exception as e:
        print("Error:", e)
    print("-" * 20)
