#
import os
from warnings import warn
from argparse import ArgumentParser
import re

from run_controller import get_run_params

IDS = {
"K1": [
    'C101',
    'C102',
    'C103',
    'C104',
    'C105',
    'C106',
    'C107',
    'C108',
    'C109',
    'C201',
    'C202',
    'C203',
    'C204',
    'C205',
    'C206',
    'C207',
    'C208',
    'R101',
    'R102',
    'R103',
    'R104',
    'R105',
    'R106',
    'R107',
    'R108',
    'R109',
    'R110',
    'R111',
    'R112',
    'R201',
    'R202',
    'R203',
    'R204',
    'R205',
    'R206',
    'R207',
    'R208',
    'R209',
    'R210',
    'R211',
    'RC101',
    'RC102',
    'RC103',
    'RC104',
    'RC105',
    'RC106',
    'RC107',
    'RC108',
    'RC201',
    'RC202',
    'RC203',
    'RC204',
    'RC205',
    'RC206',
    'RC207',
    'RC208'
],
"K2": [
    'C1_2_10',
    'C1_2_1',
    'C1_2_2',
    'C1_2_3',
    'C1_2_4',
    'C1_2_5',
    'C1_2_6',
    'C1_2_7',
    'C1_2_8',
    'C1_2_9',
    'C2_2_10',
    'C2_2_1',
    'C2_2_2',
    'C2_2_3',
    'C2_2_4',
    'C2_2_5',
    'C2_2_6',
    'C2_2_7',
    'C2_2_8',
    'C2_2_9',
    'R1_2_10',
    'R1_2_1',
    'R1_2_2',
    'R1_2_3',
    'R1_2_4',
    'R1_2_5',
    'R1_2_6',
    'R1_2_7',
    'R1_2_8',
    'R1_2_9',
    'R2_2_10',
    'R2_2_1',
    'R2_2_2',
    'R2_2_3',
    'R2_2_4',
    'R2_2_5',
    'R2_2_6',
    'R2_2_7',
    'R2_2_8',
    'R2_2_9',
    'RC1_2_10',
    'RC1_2_1',
    'RC1_2_2',
    'RC1_2_3',
    'RC1_2_4',
    'RC1_2_5',
    'RC1_2_6',
    'RC1_2_7',
    'RC1_2_8',
    'RC1_2_9',
    'RC2_2_10',
    'RC2_2_1',
    'RC2_2_2',
    'RC2_2_3',
    'RC2_2_4',
    'RC2_2_5',
    'RC2_2_6',
    'RC2_2_7',
    'RC2_2_8',
    'RC2_2_9',
],
"K4": [
    'C1_4_10',
    'C1_4_1',
    'C1_4_2',
    'C1_4_3',
    'C1_4_4',
    'C1_4_5',
    'C1_4_6',
    'C1_4_7',
    'C1_4_8',
    'C1_4_9',
    'C2_4_10',
    'C2_4_1',
    'C2_4_2',
    'C2_4_3',
    'C2_4_4',
    'C2_4_5',
    'C2_4_6',
    'C2_4_7',
    'C2_4_8',
    'C2_4_9',
    'R1_4_10',
    'R1_4_1',
    'R1_4_2',
    'R1_4_3',
    'R1_4_4',
    'R1_4_5',
    'R1_4_6',
    'R1_4_7',
    'R1_4_8',
    'R1_4_9',
    'R2_4_10',
    'R2_4_1',
    'R2_4_2',
    'R2_4_3',
    'R2_4_4',
    'R2_4_5',
    'R2_4_6',
    'R2_4_7',
    'R2_4_8',
    'R2_4_9',
    'RC1_4_10',
    'RC1_4_1',
    'RC1_4_2',
    'RC1_4_3',
    'RC1_4_4',
    'RC1_4_5',
    'RC1_4_6',
    'RC1_4_7',
    'RC1_4_8',
    'RC1_4_9',
    'RC2_4_10',
    'RC2_4_1',
    'RC2_4_2',
    'RC2_4_3',
    'RC2_4_4',
    'RC2_4_5',
    'RC2_4_6',
    'RC2_4_7',
    'RC2_4_8',
    'RC2_4_9'
],
"K6": [
    'C1_6_10',
    'C1_6_1',
    'C1_6_2',
    'C1_6_3',
    'C1_6_4',
    'C1_6_5',
    'C1_6_6',
    'C1_6_7',
    'C1_6_8',
    'C1_6_9',
    'C2_6_10',
    'C2_6_1',
    'C2_6_2',
    'C2_6_3',
    'C2_6_4',
    'C2_6_5',
    'C2_6_6',
    'C2_6_7',
    'C2_6_8',
    'C2_6_9',
    'R1_6_10',
    'R1_6_1',
    'R1_6_2',
    'R1_6_3',
    'R1_6_4',
    'R1_6_5',
    'R1_6_6',
    'R1_6_7',
    'R1_6_8',
    'R1_6_9',
    'R2_6_10',
    'R2_6_1',
    'R2_6_2',
    'R2_6_3',
    'R2_6_4',
    'R2_6_5',
    'R2_6_6',
    'R2_6_7',
    'R2_6_8',
    'R2_6_9',
    'RC1_6_10',
    'RC1_6_1',
    'RC1_6_2',
    'RC1_6_3',
    'RC1_6_4',
    'RC1_6_5',
    'RC1_6_6',
    'RC1_6_7',
    'RC1_6_8',
    'RC1_6_9',
    'RC2_6_10',
    'RC2_6_1',
    'RC2_6_2',
    'RC2_6_3',
    'RC2_6_4',
    'RC2_6_5',
    'RC2_6_6',
    'RC2_6_7',
    'RC2_6_8',
    'RC2_6_9'
],
"K8": [
    'C1_8_10',
    'C1_8_1',
    'C1_8_2',
    'C1_8_3',
    'C1_8_4',
    'C1_8_5',
    'C1_8_6',
    'C1_8_7',
    'C1_8_8',
    'C1_8_9',
    'C2_8_10',
    'C2_8_1',
    'C2_8_2',
    'C2_8_3',
    'C2_8_4',
    'C2_8_5',
    'C2_8_6',
    'C2_8_7',
    'C2_8_8',
    'C2_8_9',
    'R1_8_10',
    'R1_8_1',
    'R1_8_2',
    'R1_8_3',
    'R1_8_4',
    'R1_8_5',
    'R1_8_6',
    'R1_8_7',
    'R1_8_8',
    'R1_8_9',
    'R2_8_10',
    'R2_8_1',
    'R2_8_2',
    'R2_8_3',
    'R2_8_4',
    'R2_8_5',
    'R2_8_6',
    'R2_8_7',
    'R2_8_8',
    'R2_8_9',
    'RC1_8_10',
    'RC1_8_1',
    'RC1_8_2',
    'RC1_8_3',
    'RC1_8_4',
    'RC1_8_5',
    'RC1_8_6',
    'RC1_8_7',
    'RC1_8_8',
    'RC1_8_9',
    'RC2_8_10',
    'RC2_8_1',
    'RC2_8_2',
    'RC2_8_3',
    'RC2_8_4',
    'RC2_8_5',
    'RC2_8_6',
    'RC2_8_7',
    'RC2_8_8',
    'RC2_8_9'
],
"K10": [
    'C1_10_10',
    'C1_10_1',
    'C1_10_2',
    'C1_10_3',
    'C1_10_4',
    'C1_10_5',
    'C1_10_6',
    'C1_10_7',
    'C1_10_8',
    'C1_10_9',
    'C2_10_10',
    'C2_10_1',
    'C2_10_2',
    'C2_10_3',
    'C2_10_4',
    'C2_10_5',
    'C2_10_6',
    'C2_10_7',
    'C2_10_8',
    'C2_10_9',
    'R1_10_10',
    'R1_10_1',
    'R1_10_2',
    'R1_10_3',
    'R1_10_4',
    'R1_10_5',
    'R1_10_6',
    'R1_10_7',
    'R1_10_8',
    'R1_10_9',
    'R2_10_10',
    'R2_10_1',
    'R2_10_2',
    'R2_10_3',
    'R2_10_4',
    'R2_10_5',
    'R2_10_6',
    'R2_10_7',
    'R2_10_8',
    'R2_10_9',
    'RC1_10_10',
    'RC1_10_1',
    'RC1_10_2',
    'RC1_10_3',
    'RC1_10_4',
    'RC1_10_5',
    'RC1_10_6',
    'RC1_10_7',
    'RC1_10_8',
    'RC1_10_9',
    'RC2_10_10',
    'RC2_10_1',
    'RC2_10_2',
    'RC2_10_3',
    'RC2_10_4',
    'RC2_10_5',
    'RC2_10_6',
    'RC2_10_7',
    'RC2_10_8',
    'RC2_10_9'
],
}

TYPES = ["R1", "R2", "C1", "C2", "RC1", "RC2"]


def write_bash_script_line(cfg):
    """Write line to bash script to run controller on specified instance."""
    return (
        f"\necho 'running controller for: {cfg['id']}...'"
        f"\n{cfg['ctrl_pth']} "
        f"JAMPR "
        f"{cfg['data_pth']} "
        f"{cfg['cpu_mark']} "
        f"{cfg['time_limit']} "
        f"{cfg['BKS']} "
        f"{cfg['optimal']} "
        f"{cfg['solver_pth']}"
        f"\nwait"
    )


def get_args():
    """Read in cmd arguments"""
    parser = ArgumentParser(description='Run DIMACS controller for specified instance')
    parser.add_argument('--version', '-v', action='version', version='%(prog)s 0.1')
    parser.add_argument('--ctrl_pth', '-c', type=str, default='~/DIMACS_challenge/VRPTWController/build/VRPTWController')
    parser.add_argument('--data_dir', '-d', type=str, default='~/DIMACS_challenge/VRPTWController/Instances/')
    parser.add_argument('--batch_id', '-b', type=str, default='K1')
    parser.add_argument('--type_id', nargs='+', help='instance types as list (e.g. --type_id R1 C1 C2 ...).')
    parser.add_argument('--solver_pth', '-s', type=str, default='./solver')
    parser.add_argument('--cpu_mark', type=int, default=2743)
    parser.add_argument('--time_limit_override', type=int, default=None)
    parser.add_argument('--dry_run', action='store_true', help='only write bash file without executing.')
    args = vars(parser.parse_args())  # parse to dict
    return args


def execute_bash():
    """Create and run bash script for DIMACS controller for full specified batch."""
    args = get_args()
    cwd = os.getcwd()
    path = os.path.join(cwd, f"challenge/bash_scripts/")
    os.makedirs(path, exist_ok=True)

    batch_id = args['batch_id'].upper()
    # select batch (problem graph size)
    instance_batch = IDS[batch_id]
    # select instances
    type_ids = [str(i).upper() for i in args['type_id']]

    instance_ids = []
    for inst_id in instance_batch:
        for t in type_ids:
            assert t in TYPES, f"type id must be one of {TYPES}"
            if re.search(f"^{t}", inst_id) is not None:
                instance_ids.append(inst_id)

    source = "Solomon" if batch_id == "K1" else "Homberger"
    t_id_str = '-'.join(type_ids)

    fpath = os.path.join(path, f"run_controller_{source}_{batch_id}_{t_id_str}.sh")
    if os.path.exists(fpath):
        print(f"Bash file exists already: {fpath} \nOverwrite file? (y/n)")
        a = input()
        if a != 'y':
            print('Could not write to configuration file.')
            return

    script = f"#!/usr/bin/env bash"     # header
    for instance_id in instance_ids:

        data_pth = os.path.join(args['data_dir'], source, (instance_id+f".txt"))
        # get params
        time_limit, bks, opt = get_run_params(instance_id, source)

        if args['time_limit_override'] is not None:
            warn(f"overwriting time limit: {time_limit} -> {args['time_limit_override']}")
            time_limit = args['time_limit_override']

        # assemble cfg
        run_cfg = {
            'id': f"{batch_id}_{t_id_str}_{instance_id}",
            'ctrl_pth': args['ctrl_pth'],
            'data_pth': data_pth,
            'cpu_mark': args['cpu_mark'],
            'time_limit': time_limit,
            'BKS': bks,
            'optimal': opt,
            'solver_pth': args['solver_pth']    # must be executable -> "sudo chmod +x solver"
        }

        # add to script file
        script += write_bash_script_line(run_cfg)

    script += f"\necho 'job finished.'"     # last line
    with open(fpath, "w") as file:
        file.write(script)

    # execute script
    if not args['dry_run']:
        print(f"Executing script {fpath}...")
        os.system(F"sh {fpath}")
    print(f" done.")


if __name__ == "__main__":
    execute_bash()
