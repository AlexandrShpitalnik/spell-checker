import math
import numpy as np


class Language_models_new:
    def __init__(self, laplasian_single, laplasian_pair, constant_penalty=None):
        self.single_words_stats = {}
        self.words_pairs_stats = {}
        self.all_seen_words_counter = 0
        self.unique_seen_words_counter = None
        self.laplasian_lambda_single = laplasian_single  # smoothing coefficient
        self.laplasian_lambda_pair = laplasian_pair  # smoothing coefficient
        self.unknown_tag = -1
        self.constant_penalty = constant_penalty

    def __count_new_word(self, word):
        self.all_seen_words_counter += 1
        if word in self.single_words_stats:
            self.single_words_stats[word] += 1
        else:
            self.single_words_stats[word] = 1

    def __count_new_pair(self, word1, word2):
        if word1 in self.words_pairs_stats:
            next_words_stats, _ = self.words_pairs_stats[word1]
            self.words_pairs_stats[word1][1] += 1
            if word2 in next_words_stats:
                next_words_stats[word2] += 1
            else:
                next_words_stats[word2] = 1
        else:
            self.words_pairs_stats[word1] = [{word2: 1}, 1]  # next_words_stats; next_words_counter

    def count_new_seq(self, seq):
        """add new seq to model"""
        for i in range(len(seq) - 1):
            self.__count_new_word(seq[i])
            self.__count_new_pair(seq[i], seq[i+1])
        if seq:
            self.__count_new_word(seq[-1])

    def calculate_statistics(self):
        """stop collecting statistics and prepare for correcting words
           words counters will be turned into words weights"""
        unique_seen_words = len(self.single_words_stats.keys())
        laplasian_additive = self.laplasian_lambda_single * unique_seen_words
        for key in self.single_words_stats.keys():
            self.single_words_stats[key] = math.log((self.single_words_stats[key] + self.laplasian_lambda_single)
                                                    / (self.all_seen_words_counter + laplasian_additive))
        self.single_words_stats[self.unknown_tag] = math.log(self.laplasian_lambda_single
                                                             / (self.all_seen_words_counter + laplasian_additive)) # for unknown words

        for value in self.words_pairs_stats.values():
            next_words_stats, this_word_seen_counter = value
            laplasian_additive = self.laplasian_lambda_pair * len(next_words_stats.keys())
            for key in next_words_stats:
                next_words_stats[key] = math.log((next_words_stats[key] + self.laplasian_lambda_pair)
                                                 / (laplasian_additive + this_word_seen_counter))
            if self.constant_penalty:
                next_words_stats[self.unknown_tag] = self.constant_penalty
            else:
                next_words_stats[self.unknown_tag] = math.log(self.laplasian_lambda_pair
                                                          / (laplasian_additive + this_word_seen_counter))

    def get_word_proba(self, word):
        return self.single_words_stats[word] if word in self.single_words_stats \
            else self.single_words_stats[self.unknown_tag]

    def get_word_pair_proba(self, word1, word2):
        if word1 in self.words_pairs_stats:
            next_w_stats = self.words_pairs_stats[word1][0]
            return next_w_stats[word2] if word2 in next_w_stats else next_w_stats[self.unknown_tag]
        else:
            return 0

    def is_in_dict(self, word):
        return word in self.single_words_stats


class Error_model_new:
    def __init__(self):
        self.error_counter = 0
        self.error_dict = {}

    @staticmethod
    def to_levenshtein_format(word1, word2):
        # build levenshtein matrix
        n, m = len(word1), len(word2)
        matrix = np.zeros((m+1, n+1))
        matrix[0] = np.array([i for i in range(n+1)])
        for i in range(1, m + 1):
            matrix[i][0] = i
            previous_row, current_row = matrix[i-1], matrix[i]
            for j in range(1, n + 1):
                add_cost, delete_cost, change_cost = previous_row[j] + 1, current_row[j-1] + 1, previous_row[j-1]
                if word1[j-1] != word2[i-1]:
                    change_cost += 1
                matrix[i][j] = min(add_cost, delete_cost, change_cost)

        # to new format
        res_word1, res_word2 = word1, word2
        i, j = m, n
        var_idx = np.zeros(3)
        while matrix[i][j] != 0:
            var_idx[2] = matrix[i-1][j] if i > 0 else math.inf
            var_idx[1] = matrix[i][j-1] if j > 0 else math.inf
            var_idx[0] = matrix[i-1][j-1] if (i > 0 and j > 0) else math.inf

            cheapest_move_id = np.argmin(var_idx)

            if cheapest_move_id == 2:
                res_word1 = res_word1[:j] + '_' + res_word1[j:]
                i -= 1
            if cheapest_move_id == 0:
                i -= 1
                j -= 1
            if cheapest_move_id == 1:
                res_word2 = res_word2[:i] + '_' + res_word2[i:]
                j -= 1

        return '^' + res_word1, '^' + res_word2

    def count_errors(self, orig, fix):
        for i in range(len(orig)):
            if orig[i] != fix[i]:
                self.error_counter += 1

                if orig[i] in self.error_dict:
                    cor_char_dict = self.error_dict[orig[i]]
                    if fix[i] in cor_char_dict:
                        cor_char_dict[fix[i]] += 1
                    else:
                        cor_char_dict[fix[i]] = 1
                else:
                    self.error_dict[orig[i]] = {fix[i]: 1}

    def count_new_seq(self, orig_words, fix_words):
        # len(orig_words) == len(fix_words)
        for i in range(len(orig_words)):
            orig_i_formatted, fix_i_formatted = self.to_levenshtein_format(orig_words[i], fix_words[i])
            self.count_errors(orig_i_formatted, fix_i_formatted)

    def calculate_statistics(self):
        """stop collecting statistics and prepare for correcting words
           words counters will be turned into words weights"""
        for val in self.error_dict.values():
            for key in val.keys():
                val[key] = math.log(val[key] / self.error_counter)

    def get_letter_errors_log_probas(self, letter):
        """for letter return dict with log(proba) of the error"""
        if letter in self.error_dict:
            return self.error_dict[letter]
        else:
            return {}


class TrieNode:
    def __init__(self):
        self.next_nodes = {}

    def add_by_letter(self, letter):
        if letter not in self.next_nodes:
            self.next_nodes[letter] = TrieNode()
        return self.next_nodes[letter]


class Trie:
    def __init__(self, get_error_proba_func, get_word_proba_func, is_in_dict_func):
        self.get_letter_errors_log_probas = get_error_proba_func
        self.get_word_log_proba = get_word_proba_func
        self.start_node = TrieNode()
        self.is_in_dict = is_in_dict_func

    def add_word(self, word):
        cur_node = self.start_node
        for letter in word:
            cur_node = cur_node.add_by_letter(letter)

    def count_new_seq(self, seq):
        for word in seq:
            self.add_word(word)

    def get_word_candidates(self, orig, n_candidates=5, lang_weight=0.5, error_weight=0.5, max_penalty=-35,
                            only_dict=False):
        """get n fixed candidates for orig word
           max_penalty: max penalty for candidates search"""
        running_candidates = [['', 0, self.start_node]]  # [[word_start, next_letter_pos, node]]
        running_penalties = [0]
        res_candidates = []  # [[penalty, word]]
        res_words = set()

        while len(res_candidates) < n_candidates and len(running_candidates) > 0:
            id_cur = np.argmax(running_penalties)
            cur_cnd = running_candidates[id_cur]
            cur_penalty = running_penalties[id_cur]
            cur_word_start = cur_cnd[0]
            cur_next_letter_pos = cur_cnd[1]
            cur_node = cur_cnd[2]
            rest_penalty = max_penalty - cur_penalty  # rest_penalty < 0

            if cur_next_letter_pos == -1:  # add new word to res
                running_penalties.pop(id_cur)
                running_candidates.pop(id_cur)
                res_candidates.append([cur_penalty, cur_word_start])
                res_words.add(cur_word_start)
                continue

            # 'fix' missing letter error / add extra letter
            errros_probas = self.get_letter_errors_log_probas('_')
            for next_letter, log_proba in errros_probas.items():
                if next_letter in cur_node.next_nodes and log_proba > rest_penalty:  # log(proba) > 0
                    new_penalty = cur_penalty + log_proba
                    new_word_start = cur_word_start + next_letter
                    new_node = cur_node.next_nodes[next_letter]

                    running_candidates.append([new_word_start, cur_next_letter_pos, new_node])
                    running_penalties.append(new_penalty)

            if cur_next_letter_pos == len(orig):
                if (not only_dict or self.is_in_dict(cur_word_start)) and cur_word_start not in res_words:
                    running_penalties[id_cur] = error_weight*running_penalties[id_cur] + \
                                                lang_weight*self.get_word_log_proba(cur_word_start)
                    running_candidates[id_cur][1] = -1
                else:
                    running_candidates.pop(id_cur)
                    running_penalties.pop(id_cur)
                continue

            # 'fix' extra letter error / remove extra letter
            errros_probas = self.get_letter_errors_log_probas(orig[cur_next_letter_pos])
            if '_' in errros_probas and errros_probas['_'] > rest_penalty:
                log_proba = errros_probas['_']
                new_penalty = cur_penalty + log_proba
                new_next_letter_pos = cur_next_letter_pos + 1

                running_candidates.append([cur_word_start, new_next_letter_pos, cur_node])
                running_penalties.append(new_penalty)

            # 'fix' standard error / change letter
            errros_probas = self.get_letter_errors_log_probas(orig[cur_next_letter_pos])
            for next_letter, log_proba in errros_probas.items():
                if next_letter in cur_node.next_nodes and log_proba > rest_penalty:  # log(proba) > 0
                    new_penalty = cur_penalty + log_proba
                    new_word_start = cur_word_start + next_letter
                    new_next_letter_pos = cur_next_letter_pos + 1
                    new_node = cur_node.next_nodes[next_letter]

                    running_candidates.append([new_word_start, new_next_letter_pos, new_node])
                    running_penalties.append(new_penalty)

            # don't correct word
            next_letter = orig[cur_next_letter_pos]
            new_next_letter_pos = cur_next_letter_pos + 1
            if next_letter in cur_node.next_nodes:
                new_node = cur_node.next_nodes[next_letter]
                new_word_start = cur_word_start + next_letter
                running_candidates[id_cur] = [new_word_start, new_next_letter_pos, new_node]
            else:
                running_penalties.pop(id_cur)
                running_candidates.pop(id_cur)

        return res_candidates


class Models:
    def __init__(self, lang_laplasian_single, lang_laplasian_pair, constant_penalty=None):
        self.language_model = Language_models_new(laplasian_single=lang_laplasian_single,
                                                  laplasian_pair=lang_laplasian_pair, constant_penalty=constant_penalty)
        self.error_model = Error_model_new()
        self.trie = Trie(get_error_proba_func=self.error_model.get_letter_errors_log_probas,
                         get_word_proba_func=self.language_model.get_word_proba,
                         is_in_dict_func=self.language_model.is_in_dict)
        self.cur_dict = {}
        self.counter = 0
        self.min_trash_token = 25
        self.correct_letters = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')

    def split_by_words(self, query_string):
        tokens = query_string.split(' ')
        res_tokens = []
        for token in tokens:
            if token != '':
                token = token.lower()
                token = ''.join([letter for letter in token if letter in self.correct_letters])
                if len(token) >= self.min_trash_token or len(token) == 0:  # trash query
                    return []
                res_tokens.append(token)
        return res_tokens

    def build_model(self, filename):
        with open(filename) as f:
            for line in f:
                parts = line.split('\t')
                if len(parts) == 2:  # fixed error
                    orig_str, fixed_str = parts
                    orig_words, fixed_words = self.split_by_words(orig_str), self.split_by_words(fixed_str)
                    if len(orig_words) == len(fixed_words):  # don't count split / joined words
                        self.counter += 1
                        self.error_model.count_new_seq(orig_words, fixed_words)
                else:  # correct query
                    fixed_words = self.split_by_words(parts[0])
                self.language_model.count_new_seq(fixed_words)
                self.trie.count_new_seq(fixed_words)
        self.language_model.calculate_statistics()
        self.error_model.calculate_statistics()
        print(self.counter)

    def check_query(self, orig, n_candidates=5, lang_weight=0.5, error_weight=0.5, max_penalty=-25,
                    seq_single_weight=0.5, seq_pair_weight=0.5, debug=False, only_dict=False):
        """only_dict: select candidates from dictionary"""
        orig_words = self.split_by_words(orig)
        candidates_by_lvl = []
        running_variants = []  # [prev_lvl, [ids_by_lvl]]
        running_penalties = []

        for word in orig_words:
            candidates_by_lvl.append(self.trie.get_word_candidates(word, n_candidates,
                                                                   lang_weight, error_weight, max_penalty, only_dict))
            # [[penalty, word]]
            if debug:
                print(candidates_by_lvl[-1])
        n_lvl = len(candidates_by_lvl)

        for idx, item in enumerate(candidates_by_lvl[0]):
            penalty = item[0]
            running_variants.append([0, [idx]])
            running_penalties.append(penalty*seq_single_weight)

        while True:
            id_cur = np.argmax(running_penalties)
            prev_lvl = running_variants[id_cur][0]
            cur_lvl = prev_lvl + 1
            prev_word_id = running_variants[id_cur][1][-1]
            prev_word = candidates_by_lvl[prev_lvl][prev_word_id][1]
            penalty = running_penalties[id_cur]
            if cur_lvl == n_lvl:
                res = running_variants[id_cur][1]
                break

            for new_id in range(len(candidates_by_lvl[cur_lvl])):
                new_word = candidates_by_lvl[cur_lvl][new_id][1]
                new_word_penalty = candidates_by_lvl[cur_lvl][new_id][0] * seq_single_weight
                word_pair_penalty = self.language_model.get_word_pair_proba(prev_word, new_word)*seq_pair_weight

                new_word_ids_seq = running_variants[id_cur][1][:] + [new_id]
                new_penalty = penalty + new_word_penalty + word_pair_penalty
                if new_id == len(candidates_by_lvl[cur_lvl])-1:
                    running_penalties[id_cur] = new_penalty
                    running_variants[id_cur][1].append(new_id)
                    running_variants[id_cur][0] += 1
                    pass
                else:
                    running_penalties.append(new_penalty)
                    running_variants.append([cur_lvl, new_word_ids_seq])

        words = []
        for lvl, idx in enumerate(res):
            words.append(candidates_by_lvl[lvl][idx][1])
        return ' '.join(words)
