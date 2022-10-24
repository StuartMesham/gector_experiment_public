import os
import torch
from transformers import AutoTokenizer
from symspellpy import SymSpell


def gector_tokenize(tokenizer, input_text, max_bpe_length=512, max_bpe_pieces=5, return_word_start_indices=False, return_tensors=None, device='cpu'):
    tokenized_inputs = tokenizer(
        input_text,
        padding=False,
        truncation=False,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
        add_special_tokens=False,
    )

    new_input_ids = []
    new_word_ids = []

    n_sequences = len(input_text)
    for i in range(n_sequences):
        input_ids = tokenized_inputs[i].ids
        word_ids = tokenized_inputs[i].word_ids

        temp_max_bpe_pieces = max_bpe_pieces

        first_iteration = True

        while first_iteration or (len(input_ids) > max_bpe_length and temp_max_bpe_pieces > 1):
            current_word_len = 0
            current_word = None

            i = 0
            while i < len(input_ids):
                if current_word != word_ids[i]:
                    current_word_len = 1
                    current_word = word_ids[i]

                if current_word_len > temp_max_bpe_pieces:
                    del input_ids[i]
                    del word_ids[i]
                else:
                    i += 1
                    current_word_len += 1

            temp_max_bpe_pieces -= 1
            first_iteration = False

        new_input_ids.append(input_ids)
        new_word_ids.append(word_ids)

    output_dict = {
        'input_ids': new_input_ids,
        'word_ids': new_word_ids,
    }

    if return_tensors is None:
        output_dict['input_ids'] = new_input_ids
    else:
        assert return_tensors == 'pt'
        # add padding
        max_len = max([len(l) for l in new_input_ids])
        output_dict['input_ids'] = torch.tensor([l + [tokenizer.pad_token_id]*(max_len - len(l)) for l in new_input_ids], device=device)
        output_dict['attention_mask'] = torch.tensor([[1]*len(l) + [0]*(max_len - len(l)) for l in new_input_ids], device=device)

    if return_word_start_indices:
        new_word_start_indices = []
        for word_ids in new_word_ids:
            word_start_indices = []
            current_word = None
            for i, wi in enumerate(word_ids):
                if wi != current_word:
                    word_start_indices.append(i)
                    current_word = wi
            new_word_start_indices.append(word_start_indices)

        output_dict['word_start_indices'] = new_word_start_indices

    return output_dict


def load_sym_spell(frequency_dictionary, max_edit_distance=2):
    sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance)
    pickle_file = f'{frequency_dictionary}_maxdist_{max_edit_distance}.pkl'
    if os.path.isfile(pickle_file):
        sym_spell.load_pickle(pickle_file)
    else:
        assert os.path.isfile(frequency_dictionary), f'{frequency_dictionary} not found'
        sym_spell.load_dictionary(frequency_dictionary, 0, 1)
        sym_spell.save_pickle(pickle_file)
    return sym_spell


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        'roberta-base',
        use_fast=True,
        add_prefix_space=True,
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ['$START']})
    words = '$START Today supercalifragilisticexpialidocious supercalifragilisticexpialidocious supercalifragilisticexpialidocious supercalifragilisticexpialidocious supercalifragilisticexpialidocious supercalifragilisticexpialidocious supercalifragilisticexpialidocious supercalifragilisticexpialidocious supercalifragilisticexpialidocious supercalifragilisticexpialidocious supercalifragilisticexpialidocious supercalifragilisticexpialidocious supercalifragilisticexpialidocious I <mask> all the way back home to tokenize the inputs to my model.'.split(' ')
    tokenized_inputs = gector_tokenize(tokenizer, [words, 'Hi there'.split()])
    print(tokenized_inputs)
