"""包入口 ``smart_research_agent.vectorstore`` 的形状契约（M6-D3）.

本文件只回答"这个包**长什么形状**"，不回答"检索结果对不对"——后者由
``test_vectorstore_flat`` / ``_faiss`` / ``_chroma`` 等后端测试覆盖。
四件事各挡一类回归：

```text
可选后端不进导入时路径    干净的进程里 import 本包，sys.modules 不该多出 faiss_backend / numpy
延迟导出真的取得到        from ... import FaissVectorStore / ChromaVectorStore，且取用之后才落地
未登记符号是 AttributeError "名字不在包里"与"依赖没装"必须分家（后者是 BackendUnavailable）
__all__ 逐名可解析        导出表里写了一个错名字，会在 getattr 那一刻原形毕露
```

两条与"延迟"有关的断言**必须在子进程里做**：同一个测试进程里，
``numpy``（以及 ``faiss_backend`` 模块本身）早就被别的测试导进来了，
``'numpy' in sys.modules`` 在那里恒为 True，断言会变成一句空话。
一个新进程 + 只 import 本包，才等价于"用户第一次拿到这个包"。
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import smart_research_agent.vectorstore as vectorstore
from smart_research_agent.vectorstore.errors import BackendUnavailable

#: 仓库根（``tests/`` 的上一级）：子进程靠它拼 ``PYTHONPATH``。
REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_in_fresh_process(script: str) -> str:
    """在**干净子进程**里执行 ``script``，返回去掉首尾空白的 stdout.

    只继承环境变量、加上仓库根到 ``PYTHONPATH``，不设置任何"提前 import"
    的钩子——这样进程内的 ``sys.modules`` 就是从零开始的那一份。
    """
    env = {
        **os.environ,
        "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    completed = subprocess.run(
        [sys.executable or "python", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return completed.stdout.strip()


def test_importing_the_package_does_not_load_optional_backends() -> None:
    """只 ``import`` 本包：``faiss_backend`` 与 ``numpy`` 都不该进 ``sys.modules``.

    这就是 ``__init__`` 里"包的导入成本"那一段的可执行版本：``flat`` 是
    零依赖的参照实现，它的可用性不该被另外两个后端（numpy / chromadb）
    拖累——否则"只想用 flat"的人会因为缺一个可选依赖而在 import 阶段崩掉。
    """
    output = _run_in_fresh_process(
        "import sys, smart_research_agent.vectorstore as v; "
        "print('smart_research_agent.vectorstore.faiss_backend' in sys.modules, "
        "'numpy' in sys.modules)"
    )
    assert output == "False False"


def test_lazy_export_is_available_and_loads_the_backend_on_first_access() -> None:
    """延迟导出：取用**之前**后端不在 ``sys.modules``，取用**之后**才在.

    一段脚本里同时说清三件事：

    - ``from smart_research_agent.vectorstore import FaissVectorStore,
      ChromaVectorStore`` 两种写法都取得到（PEP 562 的模块级 ``__getattr__``
      对 from-import 同样生效）；
    - 拿到的是**类**，且 ``__name__`` 与取的名字一致（不是被换成了工厂函数）；
    - ``faiss_backend`` 是"取用那一刻"才进 ``sys.modules`` 的——这就是"延迟"。
    """
    script = (
        "import sys\n"
        "import smart_research_agent.vectorstore as vectorstore\n"
        "before = 'smart_research_agent.vectorstore.faiss_backend' in sys.modules\n"
        "from smart_research_agent.vectorstore import "
        "ChromaVectorStore, FaissVectorStore\n"
        "after = 'smart_research_agent.vectorstore.faiss_backend' in sys.modules\n"
        "print(before, after, isinstance(FaissVectorStore, type), "
        "isinstance(ChromaVectorStore, type), FaissVectorStore.__name__, "
        "ChromaVectorStore.__name__)"
    )

    assert (
        _run_in_fresh_process(script)
        == "False True True True FaissVectorStore ChromaVectorStore"
    )


def test_unknown_symbol_raises_attribute_error_not_backend_unavailable() -> None:
    """未登记的符号必须是 ``AttributeError``，不能是 ``BackendUnavailable``.

    两者的下一步动作相反：前者说明调用方拼错了名字（改代码），后者说明
    要装依赖或换后端（改环境）。更关键的是 ``hasattr``——它靠捕获
    ``AttributeError`` 工作；若这里抛 ``BackendUnavailable``，
    "探测这个名字在不在"会从返回 False 变成一次崩溃。
    """
    with pytest.raises(AttributeError) as excinfo:
        getattr(vectorstore, "NoSuchVectorStore")

    assert not isinstance(excinfo.value, BackendUnavailable)
    assert "NoSuchVectorStore" in str(excinfo.value)
    # 明确写死这条：这两个异常族在类型系统里就不该有交集。
    assert not issubclass(BackendUnavailable, AttributeError)

    with pytest.raises(AttributeError):
        vectorstore.__getattr__("not_a_backend")

    # 延迟导出的名字要出现在 dir() 里（IDE 补全靠它），且这一步不触发导入。
    assert "FaissVectorStore" in dir(vectorstore)


def test_every_name_in_dunder_all_is_resolvable() -> None:
    """``__all__`` 里每个名字都能被 ``getattr`` 取到（导出表的自查）.

    这条一次性挡住"导出表里写了个错名字"：一个拼错的 ``__all__`` 条目
    在 import 阶段不会报错，只会在使用者 ``from ... import *`` 或按名字
    取值时炸掉，而报错点在使用者那边、根因在包这边。
    """
    names = list(vectorstore.__all__)

    assert names, "__all__ 不该为空"
    assert len(names) == len(set(names)), "__all__ 里有重复条目"

    missing = [name for name in names if not hasattr(vectorstore, name)]
    assert missing == [], f"__all__ 里这些名字取不到：{missing}"


def test_lazy_export_table_points_at_real_symbols() -> None:
    """``LAZY_EXPORTS`` 的每条登记都指向真实符号，且都在 ``__all__`` 里.

    比上面那条更严一点：不仅要求取得到，还要求**取到的东西就是它声称的
    那个子模块里的那个对象**。把 ``FaissVectorStore`` 错填成
    ``chroma_backend``，前一条测试照样通过（名字能取到），
    但取回来的是一个不相干的类——只有逐个核对归属才拦得住。
    """
    for name, module_name in vectorstore.LAZY_EXPORTS.items():
        assert name in vectorstore.__all__, (
            f"{name!r} 登记在 LAZY_EXPORTS 里，却不在 __all__ 中"
        )
        module = importlib.import_module(f"{vectorstore.__name__}.{module_name}")
        assert getattr(vectorstore, name) is getattr(module, name), (
            f"{name!r} 应当来自 {module_name}"
        )
