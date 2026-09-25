import re
import unicodedata

# Precompiled regexes
RE_URL = re.compile(r'https?://|www\.|(?:\.(?:com|org|net|in|fr|co|gov|edu|biz|info))\b', re.IGNORECASE)
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
RE_SPACES = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\b\d+\b')
RE_HONORIFICS = re.compile(r'\b(?:m/s|m\s*s|shri|shree|sri|smt|mr|mrs|ms|dr|prof)\b', re.IGNORECASE)
RE_LEGAL = re.compile(
    r'\b(?:private limited|pvt\s+ltd|pvt|ltd|limited|llc|l\.l\.c\.|inc|incorporated|corp|corporation|co|company|llp|gmbh|sarl|sa|sas|eurl|cie|holding|holdings|enterprises|enterprise|services|service|solutions|solution)\b',
    re.IGNORECASE
)

# Address road & unit standardization
ROAD_REPLACEMENTS = [
    (re.compile(r'\bavenue\b', re.IGNORECASE), 'ave'),
    (re.compile(r'\broad\b', re.IGNORECASE), 'rd'),
    (re.compile(r'\bstreet\b', re.IGNORECASE), 'st'),
    (re.compile(r'\bdrive\b', re.IGNORECASE), 'dr'),
    (re.compile(r'\blane\b', re.IGNORECASE), 'ln'),
    (re.compile(r'\bboulevard\b', re.IGNORECASE), 'blvd'),
    (re.compile(r'\bparkway\b', re.IGNORECASE), 'pkwy'),
    (re.compile(r'\bhighway\b', re.IGNORECASE), 'hwy'),
    (re.compile(r'\bcourt\b', re.IGNORECASE), 'ct'),
    (re.compile(r'\bapartment\b', re.IGNORECASE), 'apt'),
    (re.compile(r'\bsuite\b', re.IGNORECASE), 'ste'),
    (re.compile(r'\bfloor\b', re.IGNORECASE), 'fl'),
    (re.compile(r'\bhouse\s*no\b|\bh\.?no\b|\bdoor\s*no\b|\bplot\s*no\b|\bflat\s*no\b', re.IGNORECASE), ''),
]

def clean_text(s):
    """
    Clean string by normalizing unicode accents/diacritics, removing URLs,
    DBAs, honorifics, non-alphanumerics, and normalising whitespace.
    """
    if not s:
        return ""
    # Strip diacritics / accents (e.g. French accents, German umlauts)
    s = unicodedata.normalize('NFKD', str(s)).encode('ASCII', 'ignore').decode('utf-8')
    s = s.lower()
    s = RE_URL.sub('', s)
    if ' dba ' in s:
        s = s.split(' dba ')[-1]
    s = RE_HONORIFICS.sub('', s)
    s = RE_NON_ALPHANUM.sub(' ', s)
    return RE_SPACES.sub(' ', s).strip()

def get_skeleton_name(c_name):
    """Strip legal suffixes and generic enterprise terms to obtain the core name skeleton."""
    if not c_name:
        return ""
    s = RE_LEGAL.sub('', c_name)
    s = RE_SPACES.sub(' ', s).strip()
    return s if s else c_name

def normalize_address(addr):
    """Normalize address components (accents, road abbreviations, units, etc.)."""
    if not addr:
        return ""
    s = unicodedata.normalize('NFKD', str(addr)).encode('ASCII', 'ignore').decode('utf-8')
    s = s.lower()
    for pattern, repl in ROAD_REPLACEMENTS:
        s = pattern.sub(repl, s)
    s = RE_NON_ALPHANUM.sub(' ', s)
    return RE_SPACES.sub(' ', s).strip()

def extract_numbers(text):
    """Extract list of normalized digit strings (leading zeros stripped) from text."""
    if not text:
        return []
    raw = RE_NUMBERS.findall(str(text))
    res = []
    for r in raw:
        st = r.lstrip('0')
        res.append(st if st else '0')
    return res

def extract_tokens(text):
    """Extract unique set of meaningful tokens."""
    if not text:
        return set()
    return {t for t in text.split() if len(t) >= 2}
