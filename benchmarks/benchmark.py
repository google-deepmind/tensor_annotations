# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run pytype on Acme and see how long it takes.

Automatically repeats the test a number of times and report the average and
minimum/maximum.

When run normally, reports the time to type-check without using our stubs.
To check the time *with* the stubs, install the stubs as in the main README.md
and then run `export TYPESHED_HOME="$HOME/typeshed"` before launching
the script.
"""

import datetime
import logging
import os
import shutil
import subprocess
import tempfile

from absl import app
from absl import flags


_NUM_RUNS = flags.DEFINE_integer('num_runs', default=3,
                                 help='Number of times to repeat test')


def main(_):
  with tempfile.TemporaryDirectory() as d:
    os.chdir(d)

    # ===== Download Acme =====

    subprocess.run(['git', 'clone', 'https://github.com/deepmind/acme'],
                   check=True)
    os.chdir('acme')
    subprocess.run(['git', 'checkout', '4da30b8'], check=True)
    os.chdir(d)
    check_dir = os.path.join('acme', 'acme', 'agents', 'tf')

    # ===== Time how long it takes to run pytype =====
    times = []
    for run_num in range(_NUM_RUNS.value):
      logging.info('Test %d/%d', 1 + run_num, _NUM_RUNS.value)
      t1 = datetime.datetime.now()
      subprocess.run(['pytype', check_dir,
                      # Ignore dependencies. (I've tried installing dependencies
                      # to fix this, but it still chokes on trfl and reverb,
                      # so giving up for now.)
                      '--disable', 'import-error'],
                     check=True)
      t2 = datetime.datetime.now()
      shutil.rmtree('.pytype')  # Remove pytype cache
      delta = t2 - t1
      times.append(delta)
      logging.info('Test %d/%d: %d seconds',
                   1 + run_num, _NUM_RUNS.value, delta.total_seconds())

  # ===== Print statistics =====
  mean = sum(times, datetime.timedelta()).total_seconds() / _NUM_RUNS.value
  logging.info('Average: %d seconds', mean)
  logging.info('Minimum: %d seconds', min(times).total_seconds())
  logging.info('Maximum: %d seconds', max(times).total_seconds())
  logging.info('All times: %r', times)


if __name__ == '__main__':
  app.run(main)
