import re

def clean_html(raw_html):
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", raw_html)
    return cleantext.replace("&nbsp;", "").replace("&nbspt", "")


def text_brushing(text):
    text = text.replace("--", "—").replace("\r", "").replace("\n", " \n ")
    return text
