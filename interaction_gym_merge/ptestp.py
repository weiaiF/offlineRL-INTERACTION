import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="Name of the scenario (to identify map and folder for track "
                        "files)", nargs="?")
    parser.add_argument("track_file_number", type=int, help="Number of the track file (int)", default=0, nargs="?")
    args = parser.parse_args()
    print(args.scenario_name)
    print(args.track_file_number)