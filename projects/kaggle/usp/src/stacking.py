import pandas as pd
import numpy as np
from tqdm import tqdm
from lightgbm.sklearn import LGBMRegressor
from sklearn.linear_model import BayesianRidge
import time
from gezi.common import *


def run_stacking(train_data, test_data):
    '''
    Input:
        train_data: [id, anchor, target, context, score, bert_proba, fold]
        test_data: [id, anchor, target, context, bert_proba]
    
    Return:
        X: ['id', 'score', 'bert_proba', 'tree_proba', 'stacking_proba']
        X_test: ['id', 'bert_proba', 'tree_proba', 'stacking_proba']
    '''
    
    FOLD = train_data['fold'].nunique()
    
    
    def edit_distance(x, y):
        len1 = len(x)
        len2 = len(y)
        dp = np.zeros((len1 + 1, len2 + 1))
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                delta = 0 if x[i - 1] == y[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
        return dp[len1][len2]

    def common_prefix_len(x, y):
        min_len = min(len(x), len(y))
        res = 0
        for i in range(min_len):
            if x[i] != y[i]:
                break
            res += 1
        return res

    def common_suffix_len(x, y):
        min_len = min(len(x), len(y))
        res = 0
        for i in range(1, min_len + 1):
            if x[-i] != y[-i]:
                break
            res += 1
        return res

    def LCStr(x, y):
        len1 = len(x)
        len2 = len(y)
        record = [[0 for i in range(len2 + 1)] for j in range(len1 + 1)]
        maxLen, p = 0, 0
        for i in range(len1):
            for j in range(len2):
                if x[i] == y[j]:
                    record[i + 1][j + 1] = record[i][j] + 1
                    if record[i + 1][j + 1] > maxLen:
                        maxLen = record[i + 1][j + 1]
                        p = i + 1
        return x[p - maxLen : p]

    def head_common_index(x, y):
        for i, j in enumerate(y):
            if j == x[0]:
                return i
        return -1

    def combine(x, y):
        res = []
        for i in x:
            for j in y:
                res.append(i + '_' + j)
        return res
    
    def feat_eng_core(train_data, test_data):
        train_data['tag'] = 0
        test_data['tag'] = 1
        df = pd.concat([train_data, test_data], axis=0, ignore_index=True)

        df['context_code'] = df['context'].apply(lambda x: x[0])
        df['context_num'] = df['context'].apply(lambda x: x[1:]).astype('int16')
        
        for f in tqdm(['anchor', 'context']):
            df[f'{f}_count'] = df[f].map(df[f].value_counts())

        df['anchor_context_count'] = df.groupby(['anchor', 'context'])['anchor'].transform('count')
        df['anchor_in_context_count_prop'] = df['anchor_context_count'] / df['context_count']

        df['anchor_in_context_nunique'] = df.groupby('context')['anchor'].transform('nunique')
        df['anchor_in_context_nunique_count_ratio'] = df['anchor_in_context_nunique'] / df['context_count']

        df['context_in_anchor_nunique'] = df.groupby('anchor')['context'].transform('nunique')
        df['context_in_anchor_nunique_count_ratio'] = df['context_in_anchor_nunique'] / df['anchor_count']

        df['anchor_context_code_count'] = df.groupby(['anchor', 'context_code'])['anchor'].transform('count')
        df['anchor_in_context_code_nunique'] = df.groupby('context_code')['anchor'].transform('nunique')
        df['context_code_in_anchor_nunique'] = df.groupby('anchor')['context_code'].transform('nunique')
        
        for f in tqdm(['anchor', 'target']):
            df[f'{f}_char_list'] = df[f].apply(lambda x: list(x.replace(' ', '')))
            df[f'{f}_word_list'] = df[f].apply(lambda x: x.split(' '))

            df[f'{f}_char_set'] = df[f].apply(lambda x: set(x.replace(' ', '')))
            df[f'{f}_word_set'] = df[f].apply(lambda x: set(x.split(' ')))

            df[f'{f}_char_len'] = df[f'{f}_char_list'].apply(len)
            df[f'{f}_word_len'] = df[f'{f}_word_list'].apply(len)

        for f in tqdm(['char_len', 'word_len']):
            df[f'{f}_diff'] = abs(df[f'anchor_{f}'] - df[f'target_{f}'])
            df[f'{f}_diff_in_anchor_prop'] = df[f'{f}_diff'] / df[f'anchor_{f}']
            df[f'{f}_diff_in_target_prop'] = df[f'{f}_diff'] / df[f'target_{f}']
            df[f'{f}_ratio'] = df[f'anchor_{f}'] / df[f'target_{f}']

        for f in tqdm(['char', 'word']):
            df[f'{f}_inter'] = df[[f'anchor_{f}_set', f'target_{f}_set']].apply(
                lambda x: len(x[f'anchor_{f}_set'] & x[f'target_{f}_set']), axis=1
            )
            df[f'{f}_union'] = df[[f'anchor_{f}_set', f'target_{f}_set']].apply(
                lambda x: len(x[f'anchor_{f}_set'] | x[f'target_{f}_set']), axis=1
            )
            df[f'{f}_IoU'] = df[f'{f}_inter'] / df[f'{f}_union']
        
        df['char_edit_dist'] = df[['anchor_char_list', 'target_char_list']].apply(
            lambda x: edit_distance(x['anchor_char_list'], x['target_char_list']), axis=1)
        df['word_edit_dist'] = df[['anchor_word_list', 'target_word_list']].apply(
            lambda x: edit_distance(x['anchor_word_list'], x['target_word_list']), axis=1)
        
        for f in tqdm(['char', 'word']):
            df[f'{f}_common_prefix_len'] = df[[f'anchor_{f}_list', f'target_{f}_list']].apply(
                lambda x: common_prefix_len(x[f'anchor_{f}_list'], x[f'target_{f}_list']), axis=1)
            df[f'{f}_common_prefix'] = df[[f'anchor_{f}_list', f'{f}_common_prefix_len']].apply(
                lambda x: ' '.join(x[f'anchor_{f}_list'][:x[f'{f}_common_prefix_len']])
                          if x[f'{f}_common_prefix_len'] > 0 else '', axis=1
            )

            df[f'{f}_common_suffix_len'] = df[[f'anchor_{f}_list', f'target_{f}_list']].apply(
                lambda x: common_suffix_len(x[f'anchor_{f}_list'], x[f'target_{f}_list']), axis=1)
            df[f'{f}_common_suffix'] = df[[f'anchor_{f}_list', f'{f}_common_suffix_len']].apply(
                lambda x: ' '.join(x[f'anchor_{f}_list'][-x[f'{f}_common_suffix_len']:])
                          if x[f'{f}_common_suffix_len'] > 0 else '', axis=1
            )

            df[f'{f}_common_prefix_suffix_count'] = df.groupby(
                [f'{f}_common_prefix', f'{f}_common_suffix'])['anchor'].transform('count')

            df[f'{f}_LCStr'] = df[[f'anchor_{f}_list', f'target_{f}_list']].apply(
                lambda x: LCStr(x[f'anchor_{f}_list'], x[f'target_{f}_list']), axis=1)
            df[f'{f}_LCStr_len'] = df[f'{f}_LCStr'].apply(len)
            df[f'{f}_LCStr'] = df[f'{f}_LCStr'].apply(''.join)
            df[f'{f}_LCStr_count'] = df[f'{f}_LCStr'].map(df[f'{f}_LCStr'].value_counts())

            df[f'anchor_in_target_{f}_head_common_index'] = df[[f'anchor_{f}_list', f'target_{f}_list']].apply(
                lambda x: head_common_index(x[f'anchor_{f}_list'], x[f'target_{f}_list']), axis=1)
            df[f'target_in_anchor_{f}_head_common_index'] = df[[f'anchor_{f}_list', f'target_{f}_list']].apply(
                lambda x: head_common_index(x[f'target_{f}_list'], x[f'anchor_{f}_list']), axis=1)
        
        cate_cols = ['anchor', 'context', 'context_code']
        for f in tqdm(['anchor', 'target']):
            df[f'{f}_first_word'] = df[f'{f}_word_list'].apply(lambda x: x[0])
            df[f'{f}_last_word'] = df[f'{f}_word_list'].apply(lambda x: x[-1])

            df[f'{f}_first_word_count'] = df[f'{f}_first_word'].map(df[f'{f}_first_word'].value_counts())
            df[f'{f}_last_word_count'] = df[f'{f}_last_word'].map(df[f'{f}_last_word'].value_counts())

            cate_cols.extend([
                f'{f}_first_word',
                f'{f}_last_word'
            ])

        for f in tqdm(['word']):
            df[f'first_{f}_count'] = df.groupby([f'anchor_first_{f}', f'target_first_{f}'])['anchor'].transform('count')
            df[f'first_{f}_count_in_anchor_prop'] = df[f'first_{f}_count'] / df[f'anchor_first_{f}_count']
            df[f'first_{f}_count_in_target_prop'] = df[f'first_{f}_count'] / df[f'target_first_{f}_count']

            df[f'last_{f}_count'] = df.groupby([f'anchor_last_{f}', f'target_last_{f}'])['anchor'].transform('count')
            df[f'last_{f}_count_in_anchor_prop'] = df[f'last_{f}_count'] / df[f'anchor_last_{f}_count']
            df[f'last_{f}_count_in_target_prop'] = df[f'last_{f}_count'] / df[f'target_last_{f}_count']
        
        df['anchor_isin_target'] = df[['anchor', 'target']].apply(
            lambda x: x['anchor'] in x['target'], axis=1).astype('int8')
        df['target_isin_anchor'] = df[['anchor', 'target']].apply(
            lambda x: x['target'] in x['anchor'], axis=1).astype('int8')

        df['anchor_word_isin_target'] = df[['anchor_word_set', 'target_word_set']].apply(
            lambda x: x['anchor_word_set'].issubset(x['target_word_set']), axis=1).astype('int8')
        df['target_word_isin_anchor'] = df[['anchor_word_set', 'target_word_set']].apply(
            lambda x: x['target_word_set'].issubset(x['anchor_word_set']), axis=1).astype('int8')

        df['first_word_issame'] = (df['anchor_first_word'] == df['target_first_word']).astype('int8')
        df['last_word_issame'] = (df['anchor_last_word'] == df['target_last_word']).astype('int8')
        
        word_cnt_dict = {}
        lists = df['anchor_word_list'].values.tolist() + df['target_word_list'].values.tolist()
        for l in tqdm(lists):
            for w in l:
                if w not in word_cnt_dict.keys():
                    word_cnt_dict[w] = 0
                word_cnt_dict[w] += 1

        df['anchor_word_cnt_list'] = df['anchor_word_list'].apply(lambda x: [word_cnt_dict[c] for c in x])
        df['anchor_word_cnt_max'] = df['anchor_word_cnt_list'].apply(np.max)
        df['anchor_word_cnt_min'] = df['anchor_word_cnt_list'].apply(np.min)
        df['anchor_word_cnt_mean'] = df['anchor_word_cnt_list'].apply(np.mean)
        df['anchor_word_cnt_sum'] = df['anchor_word_cnt_list'].apply(np.sum)
        df['anchor_word_cnt_std'] = df['anchor_word_cnt_list'].apply(np.std)

        df['target_word_cnt_list'] = df['target_word_list'].apply(lambda x: [word_cnt_dict[c] for c in x])
        df['target_word_cnt_max'] = df['target_word_cnt_list'].apply(np.max)
        df['target_word_cnt_min'] = df['target_word_cnt_list'].apply(np.min)
        df['target_word_cnt_mean'] = df['target_word_cnt_list'].apply(np.mean)
        df['target_word_cnt_sum'] = df['target_word_cnt_list'].apply(np.sum)
        df['target_word_cnt_std'] = df['target_word_cnt_list'].apply(np.std)

        df['word_cnt_sum_ratio'] = df['anchor_word_cnt_sum'] / df['target_word_cnt_sum']

        del word_cnt_dict, lists
        
        df['word_combine_list'] = df[['anchor_word_list', 'target_word_list']].apply(
            lambda x: combine(x['anchor_word_list'], x['target_word_list']), axis=1)

        word_combine_cnt_dict = {}
        lists = df['word_combine_list'].values.tolist()
        for l in tqdm(lists):
            for w in l:
                if w not in word_combine_cnt_dict.keys():
                    word_combine_cnt_dict[w] = 0
                word_combine_cnt_dict[w] += 1

        df['word_combine_cnt_list'] = df['word_combine_list'].apply(lambda x: [word_combine_cnt_dict[c] for c in x])
        df['word_combine_cnt_max'] = df['word_combine_cnt_list'].apply(np.max)
        df['word_combine_cnt_min'] = df['word_combine_cnt_list'].apply(np.min)
        df['word_combine_cnt_median'] = df['word_combine_cnt_list'].apply(np.median)
        df['word_combine_cnt_mean'] = df['word_combine_cnt_list'].apply(np.mean)
        df['word_combine_cnt_sum'] = df['word_combine_cnt_list'].apply(np.sum)
        df['word_combine_cnt_std'] = df['word_combine_cnt_list'].apply(np.std)

        del word_combine_cnt_dict, lists
        
        for f in tqdm([
            'anchor_char_list', 'anchor_word_list', 'target_char_list', 'target_word_list',
            'anchor_char_set', 'anchor_word_set', 'target_char_set', 'target_word_set',
            'anchor_word_cnt_list', 'target_word_cnt_list', 'word_combine_list', 'word_combine_cnt_list',
            'char_common_prefix', 'char_common_suffix', 'char_LCStr',
            'word_common_prefix', 'word_common_suffix', 'word_LCStr',
        ]):
            del df[f]
        
        for f in tqdm(cate_cols):
            uniq = df[df[f].notna()][f].unique()
            df[f] = df[f].map(dict(zip(uniq, range(len(uniq)))))
        
        X = df[df['tag'] == 0].reset_index(drop=True)
        X_test = df[df['tag'] == 1].reset_index(drop=True)
        
        for f in tqdm([
            ['anchor_first_word'],
            ['anchor_last_word'],
            ['target_first_word'],
            ['target_last_word'],
            ['anchor_first_word', 'target_first_word'],
            ['anchor_last_word', 'target_last_word'],
            ['anchor_first_word', 'anchor_last_word'],
            ['target_first_word', 'target_last_word'],
        ]):
            enc_field = f'{"_".join(f)}_target_mean'
            X[enc_field] = 0
            X_test[enc_field] = 0
            for i in range(FOLD):
                val_idx = X[X['fold'] == i].index
                trn_idx = X[X['fold'] != i].index

                trn_x = X.iloc[trn_idx].reset_index(drop=True)
                val_x = X[f].iloc[val_idx].reset_index(drop=True)
                test_copy = X_test[f].copy()

                trn_x[enc_field] = trn_x.groupby(f)['score'].transform('mean')
                m = trn_x[f + [enc_field]].drop_duplicates(f).reset_index(drop=True)
                val_x = val_x.merge(m, on=f, how='left')
                val_x[enc_field] = val_x[enc_field].fillna(X['score'].mean())
                X.loc[val_idx, enc_field] = val_x[enc_field].values
                test_copy = test_copy.merge(m, on=f, how='left')
                test_copy[enc_field] = test_copy[enc_field].fillna(X['score'].mean())
                X_test[enc_field] += test_copy[enc_field].values / FOLD
        
        return X, X_test
    
    
    X, X_test = feat_eng_core(train_data, test_data)
    cols = [
        f for f in X.columns if f not in ['id', 'anchor', 'target', 'score', 'tag', 'fold', 'bert_proba']
    ]
    X['tree_proba'] = 0
    X_test['tree_proba'] = 0
    n_jobs = 2 if gezi.in_kaggle() else 10
    ic(n_jobs)
    clf = LGBMRegressor(
        learning_rate=0.1,
        num_leaves=31,
        n_estimators=3000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2022,
        n_jobs=2,
        metric='None'
    )
    for i in range(FOLD):
        print(f'lgbm --------------------- {i} fold ---------------------')
        t = time.time()
        val_idx = X[X['fold'] == i].index
        trn_idx = X[X['fold'] != i].index
        ic(val_idx, trn_idx)
        trn_x = X.iloc[trn_idx].reset_index(drop=True)
        val_x = X.iloc[val_idx].reset_index(drop=True)
        ic(trn_x.shape, val_x.shape)
        clf.fit(
            trn_x[cols], trn_x['score'],
            eval_set=[(val_x[cols], val_x['score'])],
            eval_metric='l1',
            early_stopping_rounds=300,
            verbose=300
        )
        X.loc[val_idx, 'tree_proba'] = clf.predict(val_x[cols])
        X_test['tree_proba'] += clf.predict(X_test[cols]) / FOLD
        print(f'runtime: {time.time() - t}\n')
    
    cols = ['bert_proba', 'tree_proba']
    X['stacking_proba'] = 0
    X_test['stacking_proba'] = 0
    clf = BayesianRidge()
    for i in range(FOLD):
        print(f'ridge --------------------- {i} fold ---------------------')
        t = time.time()
        val_idx = X[X['fold'] == i].index
        trn_idx = X[X['fold'] != i].index
        trn_x = X.iloc[trn_idx].reset_index(drop=True)
        val_x = X.iloc[val_idx].reset_index(drop=True)
        clf.fit(trn_x[cols], trn_x['score'])
        X.loc[val_idx, 'stacking_proba'] += clf.predict(val_x[cols])
        X_test['stacking_proba'] += clf.predict(X_test[cols]) / FOLD
        print(f'runtime: {time.time() - t}\n')
    
    return X[['id', 'score', 'bert_proba', 'tree_proba', 'stacking_proba']], X_test[['id', 'bert_proba', 'tree_proba', 'stacking_proba']]

