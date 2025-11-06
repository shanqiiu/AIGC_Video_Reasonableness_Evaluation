# IO 模块冲突问题解决方案

## 问题描述

错误信息：
```
Fatal Python error: init_sys_streams: can't initialize sys standard streams
Python runtime state: core initialized
AttributeError: module 'io' has no attribute 'open'
```

## 根本原因

项目中存在 `src/io/` 目录，与 Python 标准库的 `io` 模块发生命名冲突。

当 Python 解释器初始化时，如果在 `sys.path` 中（特别是在工作目录或项目根目录）有一个名为 `io` 的模块或包，Python 会优先导入这个本地模块而不是标准库的 `io` 模块。

由于本地的 `io` 模块（`src/io/__init__.py`）是空的或不完整，它没有标准库 `io` 模块的 `open` 属性，导致 Python 解释器在初始化标准流时失败。

## 解决方案

### 方案 1：重命名目录（推荐）

将 `src/io/` 目录重命名为其他名称，避免与标准库冲突。

**建议重命名：**
- `src/io/` → `src/video_io/`
- `src/io/` → `src/video_utils/`
- `src/io/` → `src/video_loader/`

**操作步骤：**
1. 重命名目录
2. 更新 `src/io/__init__.py` 中的导入（如果有）
3. 更新所有引用 `src.io` 的代码（如果有）

### 方案 2：修改 sys.path 顺序（不推荐）

在代码中确保标准库优先导入，但这可能不够可靠。

```python
import sys
import os

# 确保标准库路径在项目路径之前
stdlib_paths = [p for p in sys.path if 'site-packages' in p or 'lib' in p]
project_paths = [p for p in sys.path if p not in stdlib_paths]
sys.path = stdlib_paths + project_paths
```

### 方案 3：使用绝对导入（临时方案）

如果无法重命名目录，可以在需要使用标准库 `io` 的地方使用绝对导入：

```python
import builtins
import sys

# 保存标准库 io 模块
_stdlib_io = None
if 'io' in sys.modules:
    # 如果已经导入了本地 io，需要重新导入标准库 io
    del sys.modules['io']
    import io as _stdlib_io
else:
    import io as _stdlib_io
```

但这会导致代码复杂且不可靠。

## 推荐操作

**最佳实践：重命名目录**

1. 检查是否有代码引用 `src.io`：
   ```bash
   grep -r "from.*io import\|import.*io\|from src.io\|import src.io" --include="*.py" .
   ```

2. 重命名目录：
   ```bash
   # Windows
   ren src\io src\video_io
   
   # Linux/Mac
   mv src/io src/video_io
   ```

3. 更新所有引用（如果有）：
   - 将 `from src.io import` 改为 `from src.video_io import`
   - 将 `import src.io` 改为 `import src.video_io`

4. 更新 `__init__.py`（如果有导入）：
   - 更新包内的相对导入

## 验证

重命名后，运行程序应该不再出现 `AttributeError: module 'io' has no attribute 'open'` 错误。

## 注意事项

1. **避免使用 Python 标准库名称作为模块名**：
   - `io`, `sys`, `os`, `json`, `csv`, `random`, `math`, `string`, `collections`, `itertools`, `functools`, `types`, `typing`, `dataclasses`, `enum`, `pathlib`, `datetime`, `time`, `copy`, `pickle`, `hashlib`, `base64`, `urllib`, `http`, `socket`, `threading`, `multiprocessing`, `subprocess`, `asyncio`, `logging`, `warnings`, `traceback`, `inspect`, `pdb`, `unittest`, `doctest`, `pytest`, `setuptools`, `distutils`, `pkgutil`, `importlib`, `imp`, `marshal`, `gc`, `weakref`, `abc`, `contextlib`, `functools`, `operator`, `collections`, `heapq`, `bisect`, `array`, `struct`, `mmap`, `select`, `selectors`, `queue`, `sched`, `timeit`, `profile`, `cProfile`, `pstats`, `trace`, `tracemalloc`, `faulthandler`, `pdb`, `doctest`, `unittest`, `test`, `lib2to3`, `distutils`, `setuptools`, `pkgutil`, `importlib`, `imp`, `marshal`, `gc`, `weakref`, `abc`, `contextlib`, `functools`, `operator`, `collections`, `heapq`, `bisect`, `array`, `struct`, `mmap`, `select`, `selectors`, `queue`, `sched`, `timeit`, `profile`, `cProfile`, `pstats`, `trace`, `tracemalloc`, `faulthandler`

2. **如果必须使用这些名称**：
   - 使用下划线后缀：`io_`, `sys_`, `os_`
   - 使用更具体的名称：`video_io`, `file_utils`, `system_utils`

3. **检查现有代码**：
   - 在创建新模块前，检查是否与标准库冲突
   - 使用 IDE 的自动完成功能可以帮助识别冲突

