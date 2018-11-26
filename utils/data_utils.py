from nltk import Tree

word_tags = {
    'NP': ['NN', 'NP', 'NNP', 'NNS', 'NNPS', 'NX', 'WHNP'],
    'PP': ['PRP', 'PRP$', 'WP', 'WP$', 'PRP', 'PRP$'],
    'VP': ['VP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD'],
    'ADJP': ['ADJP', 'JJ', 'JJR', 'JJS'],
    'ADVP': ['ADVP', 'WHADVP', 'RB', 'RBR', 'RBS', 'WRB'],
    'DET': ['DT', 'EX', 'PDT', 'WDT'],
    'ADP': ['IN'],
    'CONJP': ['CC'],
    'NUM': ['CD'],
    'PUNC': ['!', '#', '$', '``', "''", '(', ')', ',', '-LRB-', '-RRB-', '.', ':', '?', 'LST', 'PRN'],
    'PRT': ['PRT', 'PP', 'TO', 'POS', 'RP']}

pharse_tags = {
    'NP': ['NP', '@NP', 'NX', '@NX', 'WHNP', '@WHNP'],
    'VP': ['VP', '@VP'],
    'ADJP': ['ADJP', '@ADJP', 'WHADJP', '@WHADJP'],
    'ADVP': ['ADVP', '@ADVP', 'WHADVP', '@WHADVP'],
    'S': ['S', '@S', 'SBAR', '@SBAR', 'SQ', '@SQ', 'SINV', '@SINV'],
    'ROOT': ['ROOT'],
    'CONJP': ['CONJP', '@CONJP'],
    'NUM': ['QP', '@QP'],
    'PUNC': ['LST', '@LST'],
    'PRT': ['PRT', '@PRT', 'PP', '@PP', 'WHPP', '@WHPP']}

root_merged_phrase_tags = {
    'NP': ['NP', '@NP', 'NX', '@NX', 'WHNP', '@WHNP'],
    'VP': ['VP', '@VP'],
    'ADJP': ['ADJP', '@ADJP', 'WHADJP', '@WHADJP'],
    'ADVP': ['ADVP', '@ADVP', 'WHADVP', '@WHADVP'],
    'S': ['S', '@S', 'SBAR', '@SBAR', 'SQ', '@SQ', 'SINV', '@SINV', 'ROOT'],
    'CONJP': ['CONJP', '@CONJP'],
    'NUM': ['QP', '@QP'],
    'PUNC': ['LST', '@LST'],
    'PRT': ['PRT', '@PRT', 'PP', '@PP', 'WHPP', '@WHPP']}


def parse(s):
    result = []
    for t in s.split():
        if t.startswith("("):
            result.append("(")
            result.append(t[1:])
        elif t.endswith(")"):
            n = sum([1 for x in t if x == ")"])
            result.append(t[:-n])
            result.extend([")"] * n)
        else:
            result.append(t)

    return result


def preprocess_WORD(s):
    return Tree.fromstring(s).leaves()


def preprocess_WORD_TAG(s):
    pos = [t for w, t in Tree.fromstring(s).pos()]

    new_pos = []
    for t in pos:
        added = False
        for k in word_tags.keys():
            if t in word_tags[k]:
                new_pos.append(k)
                added = True
        if not added:
            new_pos.append('X')
    return new_pos


def preprocess_PARSE(s):
    s = parse(s)

    new_s = []
    for i, t in enumerate(s):
        if (t == '(' and s[i + 3] != ')') or (t == ')' and s[i - 3] != '('):
            new_s.append(t)
        elif i >= 2 and i < len(s) - 2 and s[i - 2] == '(' and s[i + 1] == ')':
            new_s.append(t)

    s = ['reduce' if t == ')' else 'shift' for t in new_s if t != '(']
    return s


def preprocess_CONS_TAG_root_divided(s):
    s = parse(s)
    new_s = []
    for i, t in enumerate(s):
        if t == ")":
            cnt = 0
            cnt_max = 0
            for j in range(i, -1, -1):
                if s[j] == "(":
                    cnt -= 1
                elif s[j] == ")":
                    cnt += 1
                    if cnt > cnt_max:
                        cnt_max = cnt

                if cnt == 0 and cnt_max > 1:
                    new_s.append(s[j + 1])
                    break

    new_new_s = []
    for t in new_s:
        added = False
        for k in pharse_tags.keys():
            if t in pharse_tags[k]:
                new_new_s.append(k)
                added = True
        if not added:
            new_new_s.append('X')

    return new_new_s


def preprocess_CONS_TAG_root_merged(s):
    s = parse(s)
    new_s = []
    for i, t in enumerate(s):
        if t == ")":
            cnt = 0
            cnt_max = 0
            for j in range(i, -1, -1):
                if s[j] == "(":
                    cnt -= 1
                elif s[j] == ")":
                    cnt += 1
                    if cnt > cnt_max:
                        cnt_max = cnt

                if cnt == 0 and cnt_max > 1:
                    new_s.append(s[j + 1])
                    break

    new_new_s = []
    for t in new_s:
        added = False
        for k in root_merged_phrase_tags.keys():
            if t in root_merged_phrase_tags[k]:
                new_new_s.append(k)
                added = True
        if not added:
            new_new_s.append('X')

    return new_new_s
