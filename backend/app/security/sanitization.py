"""
Security sanitization utilities using the bleach library.
"""

import bleach


def sanitize_user_input(text: str) -> str:
    """
    Sanitize user input using bleach to prevent XSS attacks.

    Args:
        text: The input text to sanitize

    Returns:
        str: The sanitized text with potentially dangerous HTML/script tags removed
    """
    if not text:
        return text

    # Define allowed tags and attributes for basic formatting
    allowed_tags = ["b", "i", "u", "em", "strong", "p", "br", "span"]

    allowed_attributes = {
        "span": ["class"],
        "p": ["class"],
    }

    # Sanitize the text
    sanitized = bleach.clean(
        text, tags=allowed_tags, attributes=allowed_attributes, strip=True
    )

    return str(sanitized)
