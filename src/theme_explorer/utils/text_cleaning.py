"""Provides text cleaning functions to preprocess data."""

import string

import ftfy
import pandas as pd


def remove_punctuation(series: pd.Series) -> pd.Series:
    """Remove and standardise punctuation from text in a pandas Series.

    Converts common punctuation marks to spaces and removes others entirely.
    Preserves underscores, ampersands, and forward slashes by converting them
    to spaces rather than removing them.

    Args:
        series: A pandas Series containing text data to clean.

    Returns:
        A pandas Series with punctuation removed/standardised.

    Example:
        >>> import pandas as pd
        >>> text = pd.Series(["Hello, world!", "Test-case_here"])
        >>> remove_punctuation(text)
        0    Hello  world
        1    Test case here
        dtype: object
    """
    # Define punctuation to remove (excluding _, &, /)
    punctuation_to_remove = (
        (string.punctuation + "''").replace("_", "").replace("&", "").replace("/", "")
    )

    # Convert specific punctuation to spaces, then remove remaining punctuation
    result = (
        series.str.replace("-", " ", regex=False)
        .str.replace("—", " ", regex=False)  # em dash
        .str.replace(".", " ", regex=False)
        .str.replace("?", " ", regex=False)
        .str.replace("!", " ", regex=False)
        .str.replace("_", " ", regex=False)
        .str.replace("/", " ", regex=False)
        .str.translate(str.maketrans("", "", punctuation_to_remove))
        .str.replace("\n", " ", regex=False)
    )

    return result


def remove_numbers(series: pd.Series) -> pd.Series:
    """Remove numbers and alphanumeric codes from text in a pandas Series.

    Removes both standalone numbers and numbers that appear within text.
    Uses two different regex patterns to ensure comprehensive number removal.

    Args:
        series: A pandas Series containing text data to clean.

    Returns:
        A pandas Series with numbers removed.

    Example:
        >>> import pandas as pd
        >>> text = pd.Series(["Product 123 costs $45", "Code ABC123DEF"])
        >>> remove_numbers(text)
        0    Product  costs $
        1    Code ABCDEF
        dtype: object
    """
    # Remove standalone numbers (word boundaries)
    result = series.str.replace(r"\b[0-9]+\b", "", regex=True)

    # Remove any remaining digits
    result = result.str.replace(r"\d+", "", regex=True)

    return result


def standardise_spaces(series: pd.Series) -> pd.Series:
    """Standardise whitespace in text to single spaces with no leading/trailing whitespace.

    Collapses multiple consecutive whitespace characters (spaces, tabs, newlines)
    into single spaces and removes leading and trailing whitespace.

    Args:
        series: A pandas Series containing text data to standardise.

    Returns:
        A pandas Series with standardised whitespace.

    Example:
        >>> import pandas as pd
        >>> text = pd.Series(["  hello    world  ", "text\t\twith\n\nspaces"])
        >>> standardise_spaces(text)
        0    hello world
        1    text with spaces
        dtype: object
    """
    # Replace multiple whitespace characters with single spaces
    result = series.str.replace(r"\s+", " ", regex=True)

    # Remove leading and trailing whitespace
    result = result.str.strip()

    return result


def clean_encoding_issues(text: str) -> str:
    """Handle issues with character encoding and convert to ASCII.

    Data may have come from different encodings, which can create artifacts.
    This function attempts to fix encoding issues using ftfy and then returns
    the string as ASCII only.

    Note:
        This does not handle accents (e.g. é). Accented characters will be dropped.

    Args:
        text: Text string to attempt to clean up.

    Returns:
        String containing only ASCII characters.

    Example:
        >>> clean_encoding_issues("Héllo wörld")
        'Hllo wrld'
    """
    # Fix common encoding issues
    text = ftfy.fix_text(text)

    # Convert to ASCII, ignoring non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode("ascii")

    return text


def flag_spam_content(series: pd.Series) -> pd.Series:
    """Flag text that contains potential spam indicators.

    Identifies text that may be spam based on common spam characteristics:
    - Excessive capitalisation (more than 50% of letters are uppercase)
    - Presence of URLs (http/https links)
    - Excessive punctuation repetition (3+ consecutive punctuation marks)

    Args:
        series: A pandas Series containing text data to analyse.

    Returns:
        A pandas Series of boolean values indicating spam likelihood (True = likely spam).

    Example:
        >>> import pandas as pd
        >>> text = pd.Series(["URGENT!!! Click here", "normal text", "Visit https://example.com"])
        >>> flag_spam_content(text)
        0     True
        1    False
        2     True
        dtype: bool
    """
    # Check for excessive capitalisation (>50% uppercase letters)
    has_excessive_caps = series.str.len() > 0
    for i in series.index:
        if pd.notna(series.iloc[i]) and series.iloc[i].strip():
            letters = [c for c in series.iloc[i] if c.isalpha()]
            if letters:
                uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
                has_excessive_caps.iloc[i] = uppercase_ratio > 0.5
            else:
                has_excessive_caps.iloc[i] = False
        else:
            has_excessive_caps.iloc[i] = False

    # Check for URLs
    url_pattern = r"https?://\S+|www\.\S+"
    has_urls = series.str.contains(url_pattern, case=False, na=False, regex=True)

    # Check for excessive punctuation repetition
    excessive_punct_pattern = r"[!?.,]{3,}"
    has_excessive_punct = series.str.contains(
        excessive_punct_pattern, na=False, regex=True
    )

    # Combine all spam indicators
    result = has_excessive_caps | has_urls | has_excessive_punct

    return result


def mask_personal_info(series: pd.Series) -> pd.Series:
    """Mask potentially personally identifying information in text.

    Replaces the following types of PII with generic placeholders:
    - UK phone numbers (formats like 0xxxx xxxxxx, +44 xxxx xxxxxx)
    - Email addresses
    - Common names (basic detection using capitalisation patterns)

    Args:
        series: A pandas Series containing text data to mask.

    Returns:
        A pandas Series with PII replaced by generic placeholders.

    Example:
        >>> import pandas as pd
        >>> text = pd.Series(["Call me on 07700 123456", "Email john@example.com", "Hello Sarah"])
        >>> mask_personal_info(text)
        0    Call me on [PHONE]
        1    Email [EMAIL]
        2    Hello [NAME]
        dtype: object
    """
    # Mask UK phone numbers (various formats)
    phone_patterns = [
        r"\+44\s?[0-9]\s?[0-9]{4}\s?[0-9]{6}",  # +44 7 1234 567890
        r"\+44\s?\([0-9]\)\s?[0-9]{4}\s?[0-9]{6}",  # +44 (7) 1234 567890
        r"0[0-9]{4}\s?[0-9]{6}",  # 07700 123456
        r"0[0-9]{10}",  # 07700123456
        r"\([0-9]{5}\)\s?[0-9]{6}",  # (07700) 123456
    ]

    result = series.copy()
    for pattern in phone_patterns:
        result = result.str.replace(pattern, "[PHONE]", regex=True, case=False)

    # Mask email addresses
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    result = result.str.replace(email_pattern, "[EMAIL]", regex=True)

    # Mask potential names (basic detection: capitalised words not at sentence start)
    name_pattern = r"(?<!^)(?<!\. )\b[A-Z][a-z]{2,}\b"
    result = result.str.replace(name_pattern, "[NAME]", regex=True)

    return result


def clean_text(search_terms: pd.Series, mask_pii: bool = True) -> pd.Series:
    """Clean feedback text using comprehensive text preprocessing.

    Applies multiple cleaning steps to produce standardised search queries:
    - Fills missing values with empty strings
    - Optionally masks personally identifying information
    - Fixes character encoding issues (ASCII only)
    - Converts to lowercase
    - Removes punctuation
    - Standardises whitespace

    Note:
        Number removal is currently commented out but available if needed.

    Args:
        search_terms: A pandas Series containing search query text to be cleaned.
        mask_pii: Whether to mask personally identifying information. Defaults to True.

    Returns:
        A pandas Series containing cleaned search queries with:
        - Only ASCII characters
        - All text in lowercase
        - No punctuation
        - Standardised whitespace (single spaces, no leading/trailing)
        - Optionally masked PII

    Example:
        >>> import pandas as pd
        >>> queries = pd.Series(["Hello, World!", "Call me on 07700 900123", None])
        >>> clean_queries(queries)
        0    hello world
        1    call me on phone
        2
        dtype: object
    """
    # Handle missing values
    search_terms = search_terms.fillna("")

    # Optionally mask PII before other cleaning steps
    if mask_pii:
        result = mask_personal_info(search_terms)
    else:
        result = search_terms.copy()

    # Apply encoding fixes
    result = result.apply(clean_encoding_issues)

    # Convert to lowercase
    result = result.str.lower()

    # Remove punctuation
    result = remove_punctuation(result)

    # Optionally remove numbers (currently disabled)
    # result = remove_numbers(result)

    # Standardise whitespace
    result = standardise_spaces(result)

    return result
