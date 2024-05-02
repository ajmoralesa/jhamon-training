def ikt_todf(my_dict):

    import pandas as pd

    # From dict to df
    datadf = pd.DataFrame.from_dict({(p, n, s, r, v): my_dict[p][n][s][r][v]
                                     for p in my_dict.keys()
                                     for n in my_dict[p].keys()
                                     for s in my_dict[p][n].keys()
                                     for r in my_dict[p][n][s].keys()
                                     for v in my_dict[p][n][s][r].keys()}).stack(level=[0, 1, 2, 3, 4]).to_frame()

    datadf.reset_index(inplace=True)
    datadf.columns = ['timepoint', 'par',
                      'trses', 'set', 'rep', 'var', 'value']

    # create 'tr_group' variable
    NH, IK, CO = alocate_training_group()
    datadf.loc[datadf['par'].isin(NH), 'tr_group'] = 'NH'
    datadf.loc[datadf['par'].isin(IK), 'tr_group'] = 'IK'
    datadf.loc[datadf['par'].isin(CO), 'tr_group'] = 'CO'

    # add column 'all_labels'
    colu = ['par', 'trses', 'set', 'rep', 'var', 'tr_group']
    datadf['all_labels'] = datadf[colu].apply(lambda x: ' '.join(x), axis=1)
    datadf.reset_index(inplace=True)

    return datadf


def nht_todf(my_dict):

    import pandas as pd

    # from dict to df
    datadf = pd.DataFrame.from_dict({(p, n, s, r, v): my_dict[p][n][s][r][1][v]
                                     for p in my_dict.keys()
                                     for n in my_dict[p].keys()
                                     for s in my_dict[p][n].keys()
                                     for r in my_dict[p][n][s].keys()
                                     for v in my_dict[p][n][s][r][1].keys()}).stack(level=[0, 1, 2, 3, 4]).to_frame()

    datadf.reset_index(inplace=True)
    datadf.columns = ['timepoint', 'par',
                      'trses', 'set', 'rep', 'var', 'value']

    # create 'tr_group' variable
    NH, IK, CO = alocate_training_group()
    datadf.loc[datadf['par'].isin(NH), 'tr_group'] = 'NH'
    datadf.loc[datadf['par'].isin(IK), 'tr_group'] = 'IK'
    datadf.loc[datadf['par'].isin(CO), 'tr_group'] = 'CO'

    colu = ['par', 'trses', 'set', 'rep', 'var', 'tr_group']
    datadf['all_labels'] = datadf[colu].apply(lambda x: ' '.join(x), axis=1)
    datadf.reset_index(inplace=True)

    return datadf


def alocate_training_group():
    """Create list of training groups for participant allocation

    Returns:
        list -- Three list objects of participant names
    """

    NH = ['jhamon01', 'jhamon02', 'jhamon03', 'jhamon04', 'jhamon05', 'jhamon06',
          'jhamon09', 'jhamon10', 'jhamon11', 'jhamon12', 'jhamon14', 'jhamon15', 'jhamon16']

    IK = ['jhamon18', 'jhamon20', 'jhamon22', 'jhamon23', 'jhamon24', 'jhamon25', 'jhamon26',
          'jhamon28', 'jhamon29', 'jhamon30', 'jhamon31', 'jhamon32', 'jhamon33', 'jhamon34']

    CO = ['jhamon07', 'jhamon08', 'jhamon17', 'jhamon19', 'jhamon21', 'jhamon27', 'jhamon35',
          'jhamon36', 'jhamon37', 'jhamon38', 'jhamon40', 'jhamon42']

    return NH, IK, CO


def remove_empty_nested_dicts(input_dict):
    """
    Remove empty nested dictionaries from a given dictionary.

    Args:
        input_dict (dict): The input dictionary to process.

    Returns:
        dict: A modified dictionary with empty nested dictionaries removed.

    Example:
        >>> input_dict = {
        ...     "a": {"x": {}, "y": 42},
        ...     "b": {},
        ...     "c": {"z": 100},
        ... }
        >>> result_dict = remove_empty_nested_dicts(input_dict)
        >>> print(result_dict)
        {'a': {'y': 42}, 'c': {'z': 100}}
    """
    # Create a list to store keys for deletion
    keys_to_remove = []

    # Iterate through the outer dictionary
    for key, value in input_dict.items():
        if isinstance(value, dict):
            # Recursively remove empty nested dictionaries
            modified_value = remove_empty_nested_dicts(value)
            if not modified_value:
                keys_to_remove.append(key)
            else:
                input_dict[key] = modified_value

    # Remove empty dictionaries
    for key in keys_to_remove:
        print("Removing empty session:", key)
        del input_dict[key]

    return input_dict


def remove_empy_repetitions(input_dict):

    keys_to_remove = set()

    # These are the empy repetitions that should be removed from the dictionary
    for p in input_dict.keys():
        for n in input_dict[p].keys():
            for s in input_dict[p][n].keys():
                for r in input_dict[p][n][s].keys():
                    for v in input_dict[p][n][s][r][1].keys():
                        if len(input_dict[p][n][s][r][1][v]) != 101:
                            print("Removing empty repetition:", p, n, s, r)
                            keys_to_remove.add((p, n, s, r))

    # Remove the collected keys
    for key_tuple in keys_to_remove:
        p, n, s, r = key_tuple
        del input_dict[p][n][s][r]

    return input_dict


def remove_empty(input_dict):

    input_dict = remove_empty_nested_dicts(input_dict)
    input_dict = remove_empy_repetitions(input_dict)

    return input_dict


def count_total_reps(nordict):
    # Create a dictionary to store the total rep counts for each participant
    total_rep_counts = {}

    # Iterate through the nested dictionary
    for p in nordict.keys():
        total_rep_count = 0  # Initialize the count for this participant
        for n in nordict[p].keys():
            for s in nordict[p][n].keys():
                for r in nordict[p][n][s].keys():
                    print(p, n, s, r)
                    total_rep_count += 1

        # Store the total count for this participant
        total_rep_counts[p] = total_rep_count

    # Now, total_rep_counts is a dictionary where keys are participants and values are their total rep counts
    print(total_rep_counts)


# save discrete variables data frame
def nht_disc_todf(my_dict):

    import pandas as pd

    datadf = pd.DataFrame({(p, n, s, r, v): my_dict[p][n][s][r][0][v]
                           for p in my_dict.keys()
                           for n in my_dict[p].keys()
                           for s in my_dict[p][n].keys()
                           for r in my_dict[p][n][s].keys()
                           for v in my_dict[p][n][s][r][0].keys()}, index=[0]).stack(level=[0, 1, 2, 3, 4]).to_frame()

    datadf.reset_index(inplace=True)
    datadf.columns = ['timepoint', 'par',
                      'trses', 'set', 'rep', 'var', 'value']

    # create 'tr_group' variable
    NH, IK, CO = alocate_training_group()
    datadf.loc[datadf['par'].isin(NH), 'tr_group'] = 'NH'
    datadf.loc[datadf['par'].isin(IK), 'tr_group'] = 'IK'
    datadf.loc[datadf['par'].isin(CO), 'tr_group'] = 'CO'

    colu = ['par', 'trses', 'set', 'rep', 'var', 'tr_group']
    datadf['all_labels'] = datadf[colu].apply(lambda x: ' '.join(x), axis=1)
    datadf.reset_index(inplace=True)

    return datadf


# # Remove extra sets from the training data frame
# training_df_clean = training_df[
#     ~((training_df['trses'] == "tr_1") & (training_df['set'].isin(["set_4", "set_5"])) |
#       (training_df['trses'] == "tr_5") & (training_df['set'] == "set_5") |
#       (training_df['trses'] == "tr_4") & (training_df['set'] == "set_5") |
#       (training_df['trses'] == "tr_10") & (training_df['set'].isin(["set_6", "set_7"])) |
#       (training_df['trses'] == "tr_15") & (training_df['set'] == "set_7"))
# ]
