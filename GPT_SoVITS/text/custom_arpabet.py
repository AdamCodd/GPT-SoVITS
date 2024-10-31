import re
class CustomArpabetProcessor:
    def segment(LangSegment, text):
        text = CustomArpabetProcessor.preprocess(text)
        parts = re.split(r'({.*?})', text)
        result = []

        for part in parts:
            part : str
            if not len(part):
                continue
            sp = part.strip()
            if not (sp.startswith('{') and sp.endswith('}')):
                result.append(' '.join(t['text'] for t in LangSegment.getTexts(
                    part)))
            else:
                result.append(part.strip())
        return ''.join(result)

    def preprocess(text):
        text = re.sub(r'\s+', ' ', text)
        return text

    def g2p(symbols, text, language_module):
        text = CustomArpabetProcessor.preprocess(text)
        parts = re.split(r'({.*?})', text)
        result = []

        for part in parts:
            part : str
            if not len(part):
                continue
            sp = part.strip()
            if not (sp.startswith('{') and sp.endswith('}')):
                part = language_module.text_normalize(part)
                phones = language_module.g2p(part)
                result.extend(phones)
            else:
                part = part[1:-1]
                part = part.upper()
                phones = part.split()
                for ph in phones:
                    if ph not in symbols:
                        warn(f"Unknown symbol {ph} detected in arpabet sequence")
                phones = ['UNK' if ph not in symbols else ph for ph in phones]
                result.extend(phones)
        return result
