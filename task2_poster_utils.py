import string

import numpy as np


def aggregate_question_words(tokens, relevance):
    """Aggregate GPT-2 subword relevance into question-level words."""
    relevance = np.asarray(relevance, dtype=float)
    if len(tokens) != len(relevance):
        raise ValueError("tokens and relevance must have the same length")

    start, end = _question_token_span(tokens)
    words = []
    values = []

    current_word = ""
    current_value = 0.0
    for token, value in zip(tokens[start:end], relevance[start:end]):
        if token == "Ċ":
            break

        piece, starts_new_word = _decode_gpt2_piece(token)
        if not piece:
            continue

        if _is_punctuation(piece):
            if current_word:
                current_value += float(value)
            continue

        if starts_new_word and current_word:
            words.append(_display_word(current_word))
            values.append(current_value)
            current_word = piece
            current_value = float(value)
        else:
            current_word += piece
            current_value += float(value)

    if current_word:
        words.append(_display_word(current_word))
        values.append(current_value)

    return words, np.array(values, dtype=float)


def _question_token_span(tokens):
    try:
        question_idx = tokens.index("Question")
    except ValueError as exc:
        raise ValueError("Question token not found") from exc

    start = question_idx + 1
    if start < len(tokens) and tokens[start] == ":":
        start += 1

    end = len(tokens)
    for idx in range(start, len(tokens)):
        if tokens[idx] in {"Ċ", "Answer"}:
            end = idx
            break
    return start, end


def _decode_gpt2_piece(token):
    if token.startswith("Ġ"):
        return token[1:], True
    return token, False


def _is_punctuation(piece):
    return all(ch in string.punctuation for ch in piece)


def _display_word(word):
    return word.strip(string.punctuation + " ")
