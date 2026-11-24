# CATLASS Python 算子接入

`npu_a5_catlass` 在运行时把 `@tla.kernel` 编译成设备代码，没有预编译产物。`mm` / `matmul` 是参考实现，移植自上游 `cann/catlass!1815`。

## 接入一个算子

| 文件 | 改动 |
| --- | --- |
| `functions/<op>.py` | 校验参数后调用 `load_impl("<op>", implementation)` |
| `functions/__init__.py` | 导入并加入 `__all__` |
| `functions/_dispatch.py` | `("npu.a5", "catlass"): "mojo_opset.kernels.npu_a5_catlass"` |
| `kernels/torch/__init__.py` | 注册 `"<op>": "<module>"`，保证 `implementation="torch"` 可解析 |
| `config/npu_a5.yaml` | 按字母序加 `<op>: catlass` |

绑定写在架构级，与 `npu_a5_native` 一致：950PR 与 950DT 同属 A5 代际，`OPS` 同时服务两者。

```
mojo_opset/kernels/npu_a5_catlass/
    __init__.py    绑定表；不得在导入期加载工具链
    _toolchain.py  工具链懒加载、目标校验、缓存环境变量
    _tiling.py     tile 选择
    _kernels.py    设备端 kernel，只允许模块级 @tla.kernel
    _host.py       编译缓存、tensor 绑定、launch
```

## 四个容易踩的点

**工具链不能在导入期加载。** provider 是懒加载的，但目标选中 `catlass` 后 dispatcher 会导入本包。若 `__init__.py` 直接 `import catlass`，没装该包的机器上会失败，且报错位置离算子调用很远。`_toolchain.toolchain()` 带 `lru_cache`，首次使用时才导入。

**fake tensor 的 dtype 必须跟随真实输出。** 照搬 L0C 累加器的 `Float32` 而真实输出是 fp16，编译会成功但结果是 nan。另外 `make_fake_tensor` 的 stride 树必须与 shape 树同构，2 维 shape 要传 2 元 stride。

**编译不能发生在图捕获期。** 编译要写文件、加载二进制。`preload("<op>", implementation="catlass")` 完成与形状无关的部分，`register_fake` 让追踪不接触真实 kernel。冷启动 `torch.compile(..., fullgraph=True)` 前仍需一次 eager 调用。

**kernel 约束不是 Python 约束。** `tla.vector()` 必须存在且 `tla.copy` 嵌套其中；kernel 必须是模块级 `@tla.kernel` 且函数体内联，委托给普通 helper 会丢失前端发射状态；GM→UB 拷贝需要显式同步，否则向量单元读到未初始化的 UB。

## 编译缓存

用 `@lru_cache` 包住取句柄的函数，与本仓其他 provider 写法一致：

```python
@lru_cache(maxsize=64)
def _compiled(m_dim, n_dim, k_dim, batch, dtype, aic_num):
    ...
    return api.tla.compile(_kernels.batchinvariant_gemm, *fakes, tiling, options=f"--npu-arch {npu_arch()}")
```

key 要覆盖所有影响产物的维度。底层编译入口每次调用都先跑前端 lowering 再查自己的内存缓存，没有外层缓存时重复调用仍需数十毫秒。

## BC 模板前置步骤

首次使用报 `BC not compiled or version mismatch`。`catlass.bc_compile` 每个缓存目录编译一次，共 25 个 bitcode 模板，约 9 秒。它不是安装包的一部分，也无法从算子代码懒触发，必须写进安装步骤：

```sh
python -m pip install 'mojo-opset[catlass]'
python -m catlass.bc_compile
```

模板落在缓存目录的 `bc` 下（默认 `~/.cache/catlass/bc`）。换缓存目录或升级 `ascend-catlass-dsl` 后要重跑。

## 实测

Ascend 950PR，`torch 2.13.0+cpu`，`torch_npu 2.13.0`，`ascend-catlass-dsl 2.2.1`。

| 项目 | 结果 |
| --- | --- |
| `mm` fp16 / bf16 / fp32 | 误差 0.00098 / 0.00781 / 0.0，无 nan |
| `mm` fp32 精度（16³ 到 128×96×80） | 最大绝对误差 0.0 |
| `matmul` 六类形状 | 形状与数值全匹配 `torch.matmul` |
| 边界 `k=0` / `m=0` | `(16,16)` / `(0,16)`，与 `torch` 一致 |

fp32 逐位精确，因为 kernel 沿完整 K 顺序归约，无原子加、无动态 split-K。
