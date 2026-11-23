import re
def sentences(text):
    text=' '.join(text.split())
    return [s.strip() for s in re.findall(r'.+?(?:[.!?][”"]?(?=\s+[A-Z“"]|$)|$)',text) if s.strip()]
