import argparse
import os

import wget

AVAIL_DATASET = {
    "Caltech": (
        "Caltech Pedestrians",
        ["https://data.caltech.edu/records/f6rph-90m20/files/data_and_labels.zip?download=1"],
    ),
    "ADAED": (
        "Avenue Dataset for Abnormal Event Detection",
        [
            "https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip",
            "https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/ground_truth_demo.zip",
        ],
    ),
    "UCSDPed": (
        "UCSD Anomaly Detection Dataset",
        ["http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"],
    ),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download test/train dataset")
    parser.add_argument("-d", "--data", type=str, help='Dataset to be downloaded(split by ",")')
    parser.add_argument("-l", "--list", action="store_true", help="List all availble dataset")
    parser.add_argument("--out", type=str, help="The directory for downloaded dataset")
    args = parser.parse_args()

    if args.list:
        print("Available dataset: ")
        for name in AVAIL_DATASET.keys():
            print(name, " : ", AVAIL_DATASET[name][0])
    else:
        dataset_str: str = args.data
        datasets = dataset_str.split(",")

        for dataset in datasets:
            ret = AVAIL_DATASET.get(dataset)
            if ret is None:
                print("WARN: unknown dataset: ", dataset)
                continue

            fullname = ret[0]
            urls = ret[1]
            print(f"Fetch dataset {dataset}({fullname}) from URL {urls}")

            out_dir = os.path.join(args.out, dataset)
            if os.path.exists(out_dir) and not os.path.isdir(out_dir):
                print("ERROR: failed create dir: ", out_dir)
            elif not os.path.exists(out_dir):
                os.mkdir(out_dir)

            for url in urls:
                wget.download(url, out_dir)
