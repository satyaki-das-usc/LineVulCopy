import pandas as pd

def boost_with_dupes(X: pd.Series, Y: pd.Series, dup_cnt=10):
    print(type(Y))
    # Y_list = Y.tolist()
    # unq_targets = set(Y_list)

    # dupe_targets = []

    # for tg in unq_targets:
    #     cnt = Y_list.count(tg)
    #     if cnt < 2:
    #         dupe_targets.append(tg)
    
    # X_dupes = []
    # Y_dupes = []

    # for idx, entry in enumerate(X):
    #     if Y[idx] in dupe_targets:
    #         X_dupes.extend([entry] * dup_cnt)
    #         Y_dupes.extend([Y[idx]] * dup_cnt)
    
    # new_X = X.append(pd.Series(X_dupes))
    # new_Y = Y.append(pd.Series(Y_dupes))

    return 0, 0
    
    return new_X, new_Y