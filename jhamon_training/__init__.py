from typing import Optional, Union, List


def check_result_file(
    pathtodata,
    results_dir,
    res_file,
    participant_id: Optional[Union[str, List[str]]] = None,
):
    """This function loads raw data if `res_file` exists in results_dir
    or creates a new dictionary from pathtodata and saves it to results_dir.

    Arguments:
        pathtodata: Path to the directory containing raw input data.
        results_dir: Path to the directory where results are stored/saved.
        res_file: The name of the result file (e.g., 'nht_results.pkl').
        participant_id (Optional[Union[str, List[str]]]): The specific participant ID(s)
                                                        to process. Can be a single
                                                        string or a list of strings.
                                                        If None, process all participants.
                                                        Defaults to None.

    Returns:
        dict -- Dictionary containing data of the specific `res_file`
    """
    import jhamon_training.pathutils as pathutils
    from jhamon_training.saveload import save_obj
    from pathlib import Path

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)  # Ensure results dir exists
    result_file_path = results_path / res_file

    if not result_file_path.exists():
        print(
            f"Result file {result_file_path} does not exist! Computing paths and getting data..."
        )

        # Ensure pathtodata is Path object for consistency
        pathtodata = Path(pathtodata)

        ################## TRAINING DATA ########################################
        if res_file == "nht_results.pkl":
            from jhamon_training.data.nordic import dame_nht_data

            # Use pathtodata for raw data paths and pass participant_id
            nht_paths = pathutils.dame_nht_paths(
                path_to_folders=pathtodata, participant_id=participant_id
            )
            # Pass participant_id to dame_nht_data
            my_dict = dame_nht_data(nht_paths, participant_id=participant_id)

        elif res_file == "ikt_results.pkl":
            from jhamon_training.data.ik import dame_ik_data

            # Use pathtodata for raw data paths and pass participant_id
            ik_paths = pathutils.dame_ik_paths(
                path_to_folders=pathtodata, participant_id=participant_id
            )
            my_dict = dame_ik_data(ik_paths)

        else:
            raise ValueError(f"Unknown res_file specified: {res_file}")

        # Save to the results directory
        save_obj(obj=my_dict, path=result_file_path)

        return my_dict

    else:
        print(f"Result file {result_file_path} exists! Loading it...")

        from jhamon_training.saveload import load_obj

        # Load from the results directory
        # Assuming load_obj needs directory and filename separately
        # Adjust if load_obj takes a full path
        raw_data = load_obj(path=results_path, name=res_file)

        return raw_data
