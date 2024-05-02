def check_result_file(pathtodata, res_file):
    """This function loads raw data if `res_file` exists or creates 
    a new dictionary and saves it. 

    Arguments:
        path_to_folders {[type]} -- [description]
        res_file {[type]} -- [description]

    Returns:
        dict -- Dictionary containing raw data of the specific `res_file`
    """
    import jhamon.pathutils as pathutils
    from jhamon.saveload import save_obj
    from pathlib import Path

    my_file = Path(pathtodata / "_RESULTS_TRAINING" / res_file).exists()

    if not my_file:
        print('File does not exist! Computing paths and getting data...')

        ################## TRAINING DATA ########################################
        if res_file == 'nht_results.pkl':
            from jhamon_training.data.nordic import dame_nht_data
            nht_paths = pathutils.dame_nht_paths(path_to_folders=pathtodata)
            my_dict = dame_nht_data(nht_paths)

        if res_file == 'ikt_results.pkl':
            from jhamon_training.data.ik import dame_ik_data
            ik_paths = pathutils.dame_ik_paths(path_to_folders=pathtodata)
            my_dict = dame_ik_data(ik_paths)

        save_obj(obj=my_dict, path=pathtodata / '_RESULTS_TRAINING' / res_file)

        return my_dict

    else:
        print('File results exists! Loading it...')

        from jhamon.saveload import load_obj
        raw_data = load_obj(
            path=(pathtodata / '_RESULTS_TRAINING'), name=res_file)

        return raw_data
