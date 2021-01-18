# Benchmarks

This directory contains a script to run a simple benchmark:
how long does it take to run pytype on a large machine learning
codebase with and without our custom stubs?

We use DeepMind's [Acme](https://github.com/deepmind/acme) framework for this test. Currently we only run on the TensorFlow agents code.
This comprises about 8k lines of code, of which 400 involve
a call to a TensorFlow library function.


## Results

On a 12-core 3.7 GHz Xeon:

* Without custom stubs: 78 seconds
* With custom stubs: 110 seconds (+41%)
