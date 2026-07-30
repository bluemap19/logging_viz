# cache_logging.py 设计文档

## 1. 概述

**文件名**：`cache_logging.py`  
**核心类**：`EnhancedWellLogCache`  
**设计目标**：为测井数据可视化系统提供高性能 LRU 缓存，支持数据压缩和命中率统计  
**适用数据**：常规测井 DataFrame、FMI 图像 List[np.ndarray]、NMR 谱数据 List[np.ndarray]、岩心实验数据（复用常规测井缓存）

---

## 2. 类设计

### 2.1 CacheConfig — 缓存配置

```python
@dataclass
class CacheConfig:
    max_entries: int = 10              # 最大缓存条目数
    max_memory_mb: float = 500.0       # 最大内存占用(MB)
    compression_enabled: bool = True   # 是否启用 zlib 压缩
    max_entry_size_mb: float = 50.0    # 单条数据最大大小(MB)，超过则跳过缓存
```

### 2.2 EnhancedWellLogCache — 缓存核心类

#### 2.2.1 初始化

```python
def __init__(self, config: CacheConfig = None)
```

**初始化内容**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `_data_cache` | `OrderedDict[tuple, DataFrame]` | 常规测井数据缓存 |
| `_fmi_cache` | `OrderedDict[tuple, List[np.ndarray]]` | FMI 图像缓存 |
| `_nmr_cache` | `OrderedDict[tuple, List[np.ndarray]]` | NMR 谱数据缓存 |
| `_fmi_compression_stats` | `dict` | FMI 压缩统计 |
| `_nmr_compression_stats` | `dict` | NMR 压缩统计 |
| `_core_stats` | `dict` | **新增**：岩心访问统计 |
| `stats` | `dict` | 全局命中率统计 |

#### 2.2.2 三层缓存设计

```
┌─────────────────────────────────────────────────────────┐
│                   EnhancedWellLogCache                    │
├──────────────────┬──────────────────┬───────────────────┤
│   _data_cache    │   _fmi_cache     │   _nmr_cache      │
│  (OrderedDict)   │  (OrderedDict)   │  (OrderedDict)     │
│                  │                  │                   │
│  key: depth_range│  key: depth_range│  key: depth_range │
│  value: DataFrame│ value: [np.array] │ value: [np.array] │
│  无压缩          │  zlib 压缩      │  zlib 压缩        │
│  LRU 自动淘汰   │  LRU 自动淘汰   │  LRU 自动淘汰    │
└──────────────────┴──────────────────┴───────────────────┘
```

#### 2.2.3 LRU 淘汰策略

当缓存条目数达到 `max_entries` 或内存超过 `max_memory_mb` 时，自动淘汰最旧条目：

```python
# 淘汰策略（以 data_cache 为例）
while (len(self._data_cache) >= self.config.max_entries
       or current_memory > self.config.max_memory_mb):
    oldest_key = next(iter(self._data_cache))
    del self._data_cache[oldest_key]
```

#### 2.2.4 zlib 压缩机制

**目标数据**：FMI 图像和 NMR 谱（浮点数组，数据冗余高，压缩效果好）

**压缩流程**：

```
原始数组 (np.ndarray)
    ↓ pickle.dumps()
原始字节流
    ↓ zlib.compress(level=6)
压缩字节流
    ↓ zlib.decompress()
压缩字节流
    ↓ pickle.loads()
还原数组
```

**压缩比参考**：

| 数据类型 | 典型压缩比 |
|----------|-----------|
| FMI 电阻率图像 | 3-8x |
| NMR T2 谱 | 2-4x |
| 常规测井曲线 | 1.5-3x（不推荐压缩，CPU 开销大于 IO 收益） |

#### 2.2.5 岩心数据缓存策略

**设计决策**：岩心数据不单独存储，复用常规测井 DataFrame 缓存。

**理由**：

- 岩心列与常规曲线列共存于同一 DataFrame，单独缓存会增加管理复杂度
- 岩心数据访问通过 `get_visible_logging_data()` 获取，自然命中 DataFrame 缓存
- 通过 `record_core_access()` 单独统计岩心访问次数

---

## 3. 核心方法

### 3.1 常规数据缓存

| 方法 | 签名 | 说明 |
|------|------|------|
| `get_data` | `(depth_range: tuple) -> Optional[DataFrame]` | 获取缓存，已命中则移到末尾（LRU 更新） |
| `set_data` | `(depth_range: tuple, data: DataFrame) -> None` | 设置缓存，自动淘汰最旧条目 |

### 3.2 FMI 图像缓存

| 方法 | 签名 | 说明 |
|------|------|------|
| `get_fmi_data` | `(depth_range: tuple) -> Optional[List[np.ndarray]]` | 获取 FMI 缓存 |
| `set_fmi_data` | `(depth_range: tuple, images: List[np.ndarray]) -> None` | 设置 FMI 缓存，自动压缩 |

### 3.3 NMR 谱缓存

| 方法 | 签名 | 说明 |
|------|------|------|
| `get_nmr_data` | `(depth_range: tuple) -> Optional[List[np.ndarray]]` | 获取 NMR 缓存 |
| `set_nmr_data` | `(depth_range: tuple, nmr_arrays: List[np.ndarray]) -> None` | 设置 NMR 缓存，自动压缩 |

### 3.4 岩心数据访问

| 方法 | 签名 | 说明 |
|------|------|------|
| `record_core_access` | `() -> None` | 记录一次岩心数据访问（复用 DataFrame 缓存） |

### 3.5 统计与维护

| 方法 | 签名 | 返回 |
|------|------|------|
| `get_cache_stats` | `() -> dict` | 命中率、命中次数、未命中次数 |
| `get_memory_usage` | `() -> dict` | data_mb、fmi_mb、nmr_mb、total_mb |
| `clear_cache` | `() -> None` | 清空所有缓存 |

---

## 4. 使用示例

```python
from cache_logging import EnhancedWellLogCache, CacheConfig

# 自定义配置
config = CacheConfig(
    max_entries=15,
    max_memory_mb=1024.0,
    compression_enabled=True,
    max_entry_size_mb=100.0
)

cache = EnhancedWellLogCache(config)

# 常规测井数据
df = pd.DataFrame({'DEPTH': [...], 'GR': [...]})
cache.set_data((1000.0, 2000.0), df)
result = cache.get_data((1000.0, 2000.0))

# FMI 图像
images = [np.random.rand(500, 200)]
cache.set_fmi_data((1000.0, 1500.0), images)
fmi = cache.get_fmi_data((1000.0, 1500.0))

# 统计
stats = cache.get_cache_stats()
print(f"FMI 命中率: {stats['fmi_hits'] / (stats['fmi_hits'] + stats['fmi_misses']):.1%}")
print(f"内存占用: {cache.get_memory_usage()['total_mb']:.1f} MB")
```

---

## 5. 设计亮点

1. **无第三方依赖**：纯 Python 标准库实现（`pickle`、`zlib`、`OrderedDict`）
2. **零拷贝友好**：DataFrame 缓存直接引用，不复制数据
3. **原子性保障**：压缩/解压在 try-except 中执行，解压失败时回退到未压缩格式
4. **可配置性**：通过 `CacheConfig` 灵活调整缓存策略
5. **线程安全注意**：当前实现非线程安全，高并发场景需外部加锁
