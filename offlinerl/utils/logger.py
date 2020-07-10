import logging
import shutil
import os
import pathlib
import numpy as np
from tabulate import tabulate
import sys
import json
import csv


class Logger():
    def __init__(self, experiment_id, env_name, seed, params, log_dir="./log"):
        self.logger = logging.getLogger(
            "{}_{}_{}".format(experiment_id, env_name, str(seed)))
        self.logger.handlers = []
        self.logger.propagate = False
        sh = logging.StreamHandler(sys.stdout)
        format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
        formatter = logging.Formatter(format)
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)
        self.logger.addHandler(sh)
        self.logger.setLevel(logging.INFO)

        work_dir = os.path.join(log_dir, experiment_id, env_name, str(seed))
        self.work_dir = work_dir
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        pathlib.Path(work_dir).mkdir(parents=True, exist_ok=True)

        self.csv_file_path = os.path.join(work_dir, 'log.csv')

        self.update_count = 0
        self.stored_infos = {}

        with open(os.path.join(work_dir, 'params.json'), 'w') as output_param:
            json.dump(params, output_param, indent=2)

        self.logger.info("Experiment Name:{}".format(experiment_id))
        self.logger.info(json.dumps(params, indent=2))

    def log(self, info):
        self.logger.info(info)

    def add_update_info(self, infos):
        for info in infos:
            if info not in self.stored_infos:
                self.stored_infos[info] = []
            self.stored_infos[info].append(infos[info])

        self.update_count += 1

    def add_eval_info(
            self, update_counts, total_time, infos, csv_write=True):
        if csv_write:
            if update_counts == 0:
                csv_titles = ["Updates", "Time Consumed"]
            csv_values = [update_counts, total_time]

        self.logger.info("Updates:{}".format(update_counts))
        self.logger.info("Time Consumed:{}s".format(total_time))

        tabulate_list = [["Name", "Value"]]
        for info in infos:
            tabulate_list.append([info, "{:.5f}".format(infos[info])])
            if csv_write:
                if update_counts == 0:
                    csv_titles += [info]
                csv_values += ["{:.5f}".format(infos[info])]
        tabulate_list.append([])

        method_list = [np.mean, np.std, np.max, np.min]
        name_list = ["Mean", "Std", "Max", "Min"]
        tabulate_list.append(["Name"] + name_list)

        for info in self.stored_infos:
            temp_list = [info]
            for name, method in zip(name_list, method_list):
                processed_info = method(self.stored_infos[info])
                temp_list.append("{:.5f}".format(processed_info))
                if csv_write:
                    if update_counts == 0:
                        csv_titles += ["{}_{}".format(info, name)]
                    csv_values += ["{:.5f}".format(processed_info)]

            tabulate_list.append(temp_list)

        # clear
        self.stored_infos = {}
        if csv_write:
            with open(self.csv_file_path, 'a') as f:
                self.csv_writer = csv.writer(f)
                if update_counts == 0:
                    self.csv_writer.writerow(csv_titles)
                self.csv_writer.writerow(csv_values)
        print(tabulate(tabulate_list))
