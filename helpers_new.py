import os
import re
from pathlib import Path
from symspellpy import Verbosity

from gector_utils import load_sym_spell

import spacy
import lemminflect

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = nlp.tokenizer.tokens_from_list

VOCAB_DIR = Path(__file__).resolve().parent / "data"
PAD = "@@PADDING@@"
UNK = "@@UNKNOWN@@"
START_TOKEN = "$START"
SEQ_DELIMETERS = {"tokens": " ",
                  "labels": "SEPL|||SEPR",
                  "operations": "SEPL__SEPR"}
REPLACEMENTS = {
    "''": '"',
    '--': '—',
    '`': "'",
    "'ve": "' ve",
}

sym_spell_general = load_sym_spell('frequency_dictionary_en_82_765.txt', max_edit_distance=int(os.environ.get('SPELL_DISTANCE', 2)))

def get_verb_form_dicts():
    path_to_dict = os.path.join(VOCAB_DIR, "verb-form-vocab.txt")
    encode, decode = {}, {}
    with open(path_to_dict, encoding="utf-8") as f:
        for line in f:
            words, tags = line.split(":")
            word1, word2 = words.split("_")
            tag1, tag2 = tags.split("_")
            decode_key = f"{word1}_{tag1}_{tag2.strip()}"
            if decode_key not in decode:
                encode[words] = tags
                decode[decode_key] = word2
    return encode, decode


ENCODE_VERB_DICT, DECODE_VERB_DICT = get_verb_form_dicts()


def check_casetype(source_token, target_token):
    if source_token.lower() != target_token.lower():
        return None
    if source_token.lower() == target_token:
        return "$TRANSFORM_CASE_LOWER"
    elif source_token.capitalize() == target_token:
        return "$TRANSFORM_CASE_CAPITAL"
    elif source_token.upper() == target_token:
        return "$TRANSFORM_CASE_UPPER"
    elif source_token[1:].capitalize() == target_token[1:] and source_token[0] == target_token[0]:
        return "$TRANSFORM_CASE_CAPITAL_1"
    elif source_token[:-1].upper() == target_token[:-1] and source_token[-1] == target_token[-1]:
        return "$TRANSFORM_CASE_UPPER_-1"
    else:
        return None


def correct_using_spellchecker(input_word, symspell_object=sym_spell_general):
    if not input_word.isalpha():
        return input_word

    case_type = check_casetype(input_word.lower(), input_word)

    suggestions = [s.term for s in
                   symspell_object.lookup(input_word, Verbosity.TOP, max_edit_distance=int(os.environ.get('SPELL_DISTANCE', 2)),
                                            transfer_casing=case_type is None)]

    if int(os.environ.get('PRINT_N_SPELLCHECKER_SUGGESTIONS', 0)) > 0:
        suggestions2 = [s.term for s in
                   symspell_object.lookup(input_word, Verbosity.CLOSEST, max_edit_distance=int(os.environ.get('SPELL_DISTANCE', 2)),
                                            transfer_casing=case_type is None)]
        print('N_SPELLCHECKER_SUGGESTIONS:', len(suggestions2))

    if len(suggestions) == 0:
        suggestion = None
    else:
        suggestion = suggestions[0]
        if case_type is not None:
            suggestion = convert_using_case(suggestion, case_type)

    if suggestion is not None:
        print('spellcheck correcting', input_word, 'to', suggestion)
        return suggestion
    else:
        print('spellcheck failed for word:', input_word)
        return input_word


def get_target_sent_by_edits(source_tokens, edits, USE_CUDA):
    round_2_edits = []

    # start with non-LM edits
    target_tokens = source_tokens[:]
    shift_idx = 0
    for edit in edits:
        start, end, label, _ = edit
        target_pos = start + shift_idx
        source_token = target_tokens[target_pos] \
            if len(target_tokens) > target_pos >= 0 else ''

        assert label != '', 'are you using the old predict.py code?' \
                            'Check that the get_token_action() method in your predict.py file ' \
                            'does not replace the $DELETE tag with an empty string'

        if label == "$DELETE":
            del target_tokens[target_pos]
            shift_idx -= 1
        elif start == end:
            if label == '$APPEND':
                round_2_edits.append((edit[0] + shift_idx, edit[1] + shift_idx, label, edit[3]))
            else:
                assert label.startswith('$APPEND_') or label in ['$MERGE_SPACE', '$MERGE_HYPHEN'], \
                    f'nope, got {label} instead. Are you using the old predict.py code? ' \
                    'Check that the get_token_action() method in your predict.py file ' \
                    'does not replace the $DELETE tag with an empty string'

                word = label.replace("$APPEND_", "")
                target_tokens[target_pos: target_pos] = [word]
                shift_idx += 1
        elif label.startswith("$TRANSFORM_"):
            word = apply_reverse_transformation(source_token, label)
            if word is None \
                    or (label == '$TRANSFORM_SPLIT_SEGMENT' and int(os.environ.get('SEGMENT_ABLATE', 0)) > 0) \
                    or (label == '$TRANSFORM_SPLIT_NONWORD' and int(os.environ.get('SPLIT_NONWORD_ABLATE', 0)) > 0) \
                    or ((label.startswith('$TRANSFORM_VERB') or label.startswith('$TRANSFORM_AGREEMENT')) and int(os.environ.get('ABLATE_GTS_VERB_PLURAL', 0)) > 0):
                word = source_token
            target_tokens[target_pos] = word
        elif start == end - 1:
            if label in ['$SPELL', '$SPELL_2', '$REPLACE'] or label.startswith('$INFLECT_'):
                round_2_edits.append((edit[0] + shift_idx, edit[1] + shift_idx, label, edit[3]))
            elif label.startswith('$REPLACE_'):
                word = label.replace("$REPLACE_", "")
                target_tokens[target_pos] = word
            else:
                raise Exception(f"Unknown tag {label}")
        elif label.startswith("$MERGE_"):
            round_2_edits.append((edit[0] + shift_idx, edit[1] + shift_idx, label, edit[3]))

    round_3_edits = []

    source_tokens2 = target_tokens[:]
    shift_idx = 0
    for edit in round_2_edits:
        start, end, label, _ = edit
        target_pos = start + shift_idx
        if start == end:
            assert label == '$APPEND'
            raise Exception('Not supported anymore')
        elif start == end - 1:
            assert label in ['$REPLACE', '$SPELL', '$SPELL_2'] or label.startswith('$INFLECT_'), f'nope, got {label} instead'
            if label in ['$SPELL', '$SPELL_2']:
                if int(os.environ.get('SPELL2_NOOP', 0)) > 0:
                    word = source_tokens2[start]
                else:
                    if int(os.environ.get('SPELL2_USE_LM_VOCAB', 0)) > 0:
                        raise Exception('Not supported anymore')
                    else:
                        word = correct_using_spellchecker(source_tokens2[start])
            elif label.startswith('$INFLECT_'):
                if (label.startswith('$INFLECT') and int(os.environ.get('ABLATE_INFLECT', 0)) > 0):
                    word = source_tokens2[start]
                else:
                    pos = label.replace("$INFLECT_", "")
                    doc = nlp(target_tokens)
                    word = doc[target_pos]._.inflect(pos)
                    print(f'inflecting {target_tokens[target_pos]} to {word} ({pos})')
            elif label == '$REPLACE':
                raise Exception('Not supported anymore')

            target_tokens[target_pos] = word
        elif label.startswith("$MERGE_"):
            round_3_edits.append((edit[0] + shift_idx, edit[1] + shift_idx, label, edit[3]))

    shift_idx = 0
    for edit in round_3_edits:
        start, end, label, _ = edit
        target_pos = start + shift_idx
        assert label.startswith("$MERGE_")
        target_tokens[target_pos + 1: target_pos + 1] = [label]
        shift_idx += 1

    return replace_merge_transforms(target_tokens)


def replace_merge_transforms(tokens):
    if all(not x.startswith("$MERGE_") for x in tokens):
        return tokens

    target_line = " ".join(tokens)
    target_line = target_line.replace(" $MERGE_HYPHEN ", "-")
    target_line = target_line.replace(" $MERGE_SPACE ", "")
    return target_line.split()


def convert_using_case(token, smart_action):
    if not smart_action.startswith("$TRANSFORM_CASE_"):
        return token
    if smart_action.endswith("LOWER"):
        return token.lower()
    elif smart_action.endswith("UPPER"):
        return token.upper()
    elif smart_action.endswith("CAPITAL"):
        return token.capitalize()
    elif smart_action.endswith("CAPITAL_1"):
        return token[0] + token[1:].capitalize()
    elif smart_action.endswith("UPPER_-1"):
        return token[:-1].upper() + token[-1]
    else:
        return token


def convert_using_verb(token, smart_action):
    key_word = "$TRANSFORM_VERB_"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    encoding_part = f"{token}_{smart_action[len(key_word):]}"
    decoded_target_word = decode_verb_form(encoding_part)
    return decoded_target_word


def convert_using_split(token, smart_action):
    if smart_action == '$TRANSFORM_SPLIT_HYPHEN':
        target_words = token.split("-")
    elif smart_action == '$TRANSFORM_SPLIT_NONWORD':
        target_words = list(filter(None, re.split('(\W)', token)))
    elif smart_action == '$TRANSFORM_SPLIT_SEGMENT':
        raise Exception('Not supported anymore')
    else:
        raise Exception(f"Unknown action type {smart_action}")
    return " ".join(target_words)


def convert_using_plural(token, smart_action):
    if smart_action.endswith("PLURAL"):
        return token + "s"
    elif smart_action.endswith("SINGULAR"):
        return token[:-1]
    else:
        raise Exception(f"Unknown action type {smart_action}")


def apply_reverse_transformation(source_token, transform):
    if transform.startswith("$TRANSFORM"):
        # deal with equal
        if transform == "$KEEP":
            return source_token
        # deal with case
        if transform.startswith("$TRANSFORM_CASE"):
            return convert_using_case(source_token, transform)
        # deal with verb
        if transform.startswith("$TRANSFORM_VERB"):
            return convert_using_verb(source_token, transform)
        # deal with split
        if transform.startswith("$TRANSFORM_SPLIT"):
            return convert_using_split(source_token, transform)
        # deal with single/plural
        if transform.startswith("$TRANSFORM_AGREEMENT"):
            return convert_using_plural(source_token, transform)
        # raise exception if not find correct type
        raise Exception(f"Unknown action type {transform}")
    else:
        return source_token


def read_parallel_lines(fn1, fn2):
    lines1 = read_lines(fn1, skip_strip=True)
    lines2 = read_lines(fn2, skip_strip=True)
    assert len(lines1) == len(lines2)
    out_lines1, out_lines2 = [], []
    for line1, line2 in zip(lines1, lines2):
        if not line1.strip() or not line2.strip():
            continue
        else:
            out_lines1.append(line1)
            out_lines2.append(line2)
    return out_lines1, out_lines2


def read_lines(fn, skip_strip=False):
    if not os.path.exists(fn):
        return []
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [s.strip() for s in lines if s.strip() or skip_strip]


def write_lines(fn, lines, mode='w'):
    if mode == 'w' and os.path.exists(fn):
        os.remove(fn)
    with open(fn, encoding='utf-8', mode=mode) as f:
        f.writelines(['%s\n' % s for s in lines])


def decode_verb_form(original):
    return DECODE_VERB_DICT.get(original)


def encode_verb_form(original_word, corrected_word):
    decoding_request = original_word + "_" + corrected_word
    decoding_response = ENCODE_VERB_DICT.get(decoding_request, "").strip()
    if original_word and decoding_response:
        answer = decoding_response
    else:
        answer = None
    return answer


def get_weights_name(transformer_name, lowercase):
    if transformer_name == 'bert' and lowercase:
        return 'bert-base-uncased'
    if transformer_name == 'bert' and not lowercase:
        return 'bert-base-cased'
    if transformer_name == 'bert-large' and not lowercase:
        return 'bert-large-cased'
    if transformer_name == 'distilbert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'distilbert-base-uncased'
    if transformer_name == 'albert':
        if not lowercase:
            print('Warning! This model was trained only on uncased sentences.')
        return 'albert-base-v1'
    if lowercase:
        print('Warning! This model was trained only on cased sentences.')
    if transformer_name == 'roberta':
        return 'roberta-base'
    if transformer_name == 'roberta-large':
        return 'roberta-large'
    if transformer_name == 'gpt2':
        return 'gpt2'
    if transformer_name == 'transformerxl':
        return 'transfo-xl-wt103'
    if transformer_name == 'xlnet':
        return 'xlnet-base-cased'
    if transformer_name == 'xlnet-large':
        return 'xlnet-large-cased'


def remove_double_tokens(sent):
    tokens = sent.split(' ')
    deleted_idx = []
    for i in range(len(tokens) -1):
        if tokens[i] == tokens[i + 1]:
            deleted_idx.append(i + 1)
    if deleted_idx:
        tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]
    return ' '.join(tokens)


def normalize(sent):
    sent = remove_double_tokens(sent)
    for fr, to in REPLACEMENTS.items():
        sent = sent.replace(fr, to)
    return sent.lower()
