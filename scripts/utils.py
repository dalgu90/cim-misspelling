import re
import string


anon_pattern = re.compile('\[\*\*.*?\*\*\]')

def remove_anonymized(t):
    """ Remove anonymized terms (wrapped by [** and **]) """
    ret = ''
    start, end = 0, len(t)
    for m in anon_pattern.finditer(t):
        end = m.start()
        ret += t[start:end]
        start = m.end()
    end = len(t)
    ret += t[start:end]
    return ret

def sanitize_text(t):
    """ Case-by-case text sanitizing """
    l = t.split('\n')
    for s in l:
        if s.startswith('.....'):
            l.remove(s)
    return '\n'.join(l)

def clean_text(text):
    """ Text cleaning by Linyuan """
    sl = []
    for s in text.strip().split('\n'):
        # s = re.sub(r'\[\*\*.*?\*\*\]', ' ', s)
        if not re.search('[a-zA-Z]', s):
            s = ''
        else:
            s = re.sub(r'\?{3,}', ' ', s)
            s = re.sub(r'-{3,}', ' ', s)
            s = re.sub(r'(- ){3,}', ' ', s)
            s = re.sub(r'#{3,}', ' ', s)
            s = re.sub(r'\*{3,}', ' ', s)
            s = re.sub(r'~{3,}', ' ', s)
            s = re.sub(r'`{3,}', ' ', s)
            s = re.sub(r'_{3,}', ' ', s)
            s = re.sub(r'\.{2,}', '.', s)
            s = re.sub(r'{+', '(', s)
            s = re.sub(r'\}+', ')', s)
            s = re.sub(r'\'{2,}', '\'', s)
            s = re.sub(r'!{2,}', '!', s)
            s = re.sub(r'<{2,}', '<', s)
            s = re.sub(r'>{2,}', '>', s)
            s = re.sub(r'\+{5,}', '+++++', s)
            s = ' '.join(s.strip().split())
        if s.startswith("===") and s.endswith("==="):
            s = s.strip("=")
            s = re.sub(r'={3,}', ' ', s)
            s = ' '.join(s.strip().split())
            if sl and sl[-1]:
                sl.append("")
            sl.append(s)
            sl.append("")
        else:
            s = re.sub(r'={3,}', ' ', s)
            s = ' '.join(s.strip().split())
            # if sl and sl[-1] and s:
            if sl and sl[-1] and s and s[0] in string.ascii_letters: # condition to be appended changed
                sl[-1] += ' ' + s
            elif (sl and sl[-1]) or s:
                sl.append(s)
    # sl = [s for s in sl if s]
    # sl2 = []
    # for s in sl:
        # ct = Counter(s)
        # score = sum(ct[ch] for ch in string.ascii_letters + string.digits) + ct[' '] * 0.25
        # if score / len(s) >= 0.6:
            # sl2.append(s)
    # return '\n'.join(sl2)
    return '\n'.join(sl)
