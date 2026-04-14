# Custom examples workflow

```
git clone -b DISPEED git@github.com:ThePebblesFr/dory.git
cd dory/
git submodule update --remote --init dory/dory_examples
git submodule update --remote --init dory/Hardware_targets/PULP/Backend_Kernels/pulp-nn
git submodule update --remote --init dory/Hardware_targets/PULP/Backend_Kernels/pulp-nn-mixed
python3 -m pip install -e .
```

```
python3 -m venv envDory
source envDory/bin/activate
```

Run the first cell of the notebook with envDory. Then, run the second cell to make sure everything is properly installed.

Run the whole notebook. The only cell to change is the 5th cell to target the correct ONNX files.