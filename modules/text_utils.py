import re
STOP = set("a an the and or but if of to in on at for with by from as is are was were be been being it its this that these those i you he she we they my your our their me them so than then also just very can could would should will do does did have has had not no yes".split())
FILLERS = ["um", "uh", "erm", "ah", "like", "you know", "basically", "actually", "sort of", "kind of", "i mean", "so yeah"]

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9\+\#\.\-]+", text.lower())

def content_words(text):
    return [t for t in tokenize(text) if t not in STOP and len(t) > 1]

def sentences(text):
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
    return s or ([text.strip()] if text.strip() else [])
