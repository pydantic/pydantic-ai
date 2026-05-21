"""v1: vendor_details / vendor_id fields."""
from pydantic_ai.messages import ModelResponse, TextPart


def trigger():
    r = ModelResponse(parts=[TextPart(content='x')])
    # DEPRECATION: F2_vendor_details
    _ = r.vendor_details
    _ = r.vendor_id
    return r


EXPECT = '`vendor_details` is deprecated, use `provider_details` instead'

if __name__ == '__main__':
    trigger()
