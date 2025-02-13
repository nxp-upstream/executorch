# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess


def run_cmd(cmd: str, error_msg: str, logger, cwd=None):
    c_process = subprocess.run(cmd, capture_output=True, cwd=cwd, shell=True)
    logger.info(c_process.stdout.decode("utf-8"))
    logger.info(c_process.stderr.decode("utf-8"))

    if c_process.returncode != 0:
        logger.fatal(c_process.stderr)
        logger.fatal(error_msg)
        raise Exception('Error running command: "%s", rc: %d' % (cmd, c_process.returncode))
