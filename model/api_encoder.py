import base64

def encode_api_key(api_key: str) -> str:
    """Encode an API key using Base64."""
    encoded_bytes = base64.b64encode(api_key.encode("utf-8"))
    return encoded_bytes.decode("utf-8")

def decode_api_key(encoded_key: str) -> str:
    """Decode an API key from its Base64 encoded form."""
    decoded_bytes = base64.b64decode(encoded_key.encode("utf-8"))
    return decoded_bytes.decode("utf-8")


# Example usage:
if __name__ == "__main__":
    my_api_key = "b865f8ab64997428792213c1280ae895"
    encoded_key = encode_api_key(my_api_key)
    decoded_key = decode_api_key(encoded_key)

    print("Original Key:", my_api_key)
    print("Encoded Key: ", encoded_key)
    print("Decoded Key: ", decoded_key)
