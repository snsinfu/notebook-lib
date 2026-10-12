# lib

A personal library of random Jupyter notebook utilities. It is deliberately
not a package and meant to be used by symlinking `lib/` into a notebook dir.

```sh
git submodule add https://github.com/snsinfu/notebook-lib.git submodules/github.com/snsinfu/notebook-lib
ln -s submodules/github.com/snsinfu/notebook-lib/lib
```

Then,

```python
from lib.files import GTF_FORMAT
from lib.density import kernel_regression
from lib.plot import set_axis, attach_colorbar, use_log_ticks
```
