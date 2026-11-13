# XFS inode 时间字段技术核查

核查对象：`dissect.xfs`（基线 `abac18b`，`dissect-util 3.24`）。
核查目的：把 XFS inode 的四类时间纳入入侵时间线前，确认 bigtime 与旧 inode 的时间编码不会混用。

结论先行：

- `di_atime`、`di_mtime`、`di_ctime` 在所有 inode 版本中都存在；`di_crtime` 只存在于 v3 inode。
- 是否使用 bigtime 编码由 **v3 inode 的 `di_flags2 & 0x8`** 决定，不是由文件系统特性位单独决定。已启用 bigtime 的文件系统上仍可存在 legacy 编码的旧 inode。
- 两种编码的 64 位位型完全不同：legacy 是「有符号 32 位秒 | 32 位纳秒」，bigtime 是「从 1901-12-13 起算的无符号 64 位纳秒计数器」。同一串位按另一种格式解释会得到貌似合理、但错误数十年的时间。
- 当前实现 `dissect/xfs/xfs.py:556` 的 legacy 分支没有对有符号秒做符号扩展，1970 年以前的负秒值会被解成 2038/2106 年。这是本次核查发现的实际缺陷。
- 时间字段只能说明「内核记录该 inode 的某个时间点被更新」，不能单独证明用户执行过某个操作。详见第 5 节。

## 1. 四类时间与 inode 版本的关系

| 字段 | 含义（内核语义） | v1 | v2 | v3 | 备注 |
|---|---|---|---|---|---|
| `di_atime` | 最后访问时间 | 有 | 有 | 有 | 受 `noatime`/`relatime`/`lazytime` 影响 |
| `di_mtime` | 文件数据最后修改时间 | 有 | 有 | 有 | 写入数据、截断等会更新 |
| `di_ctime` | inode 元数据变更时间 | 有 | 有 | 有 | 不是「创建时间」；权限、属主、链接数、扩展属性等变更都会更新 |
| `di_crtime` | inode 创建时间（birth time） | 无 | 无 | 有 | 仅 v3；字段存在但值可能为 0（未记录） |

版本判定依据：

- `c_xfs.py:333` 的 `xfs_dinode` 结构从 `di_next_unlinked`（偏移 96..100）往后是 v3 扩展区。
- `di_crtime` 位于偏移 144（`c_xfs.py:411`），不在 v1/v2 的 inode core 内。v1/v2 inode 的该偏移属于数据叉（data fork），其内容可能是文件数据、目录项或扩展记录。
- 因此：**v1/v2 inode 没有创建时间字段**。在 v1/v2 上读取偏移 144 得到的 8 字节不是时间，不能填进时间线。
- 仓库现有样本的 `sb_inodesize` 为 512；v2 inode 的 core 只到 100 字节，144 字节处必然落在数据叉。

### 1.1 版本与标志的判定顺序

正确顺序是：

1. 读 `di_version`。只有 `di_version >= 3` 才存在 `di_flags2`（偏移 120）和 `di_crtime`（偏移 144）。
2. 对 v3，读 `di_flags2`，检查 bit 3（`XFS_DIFLAG2_BIGTIME`，值 `0x8`）。
3. 用超级块 `sb_features_incompat` 的 bit 3（`XFS_SB_FEAT_INCOMPAT_BIGTIME`）做交叉校验，而不是把它当作唯一开关。

超级块特性位表示「这个文件系统允许/曾经启用 bigtime」，不等于「每个 inode 都是 bigtime」。启用该特性前创建的 inode 会保持 legacy 编码，只有被重新写入时内核才会按当前格式重写时间字段。反过来，某个 v3 inode 带 bigtime 标志而超级块没有该特性，属于元数据矛盾：应报异常或标记为不可信，而不是猜一种解释。

仓库当前实现 `_has_bigtime()`（`xfs.py:264`）只检查 `di_version >= 3 and di_flags2 & 0x8`，不检查超级块特性；这能避免把 v1/v2 误判为 bigtime，但无法发现上述矛盾。`crtime` 属性（`xfs.py:353`）没有版本门，v1/v2 上会返回数据叉字节。

## 2. 两种位编码

### 2.1 Legacy（`di_version < 3`，或 v3 且未置 bigtime 位）

磁盘上是 big-endian 64 位值，布局等同于：

```
 63              32 31               0
+-----------------+------------------+
| int32  t_sec    | uint32 t_nsec    |
+-----------------+------------------+
```

- `t_sec` 是**有符号** 32 位秒，纪元为 1970-01-01T00:00:00Z。
- `t_nsec` 是 0..999,999,999 的纳秒。
- 可表示范围：`1901-12-13T20:45:52Z` 至 `2038-01-19T03:14:07.999999999Z`。
- 关键陷阱：`0x80000000` 在高位表示 `-2147483648` 秒，即 1901 年；若按无符号解释会变成 2038-01-19T03:14:08Z。两者相邻但方向相反。
- 另一个陷阱：负时间的纳秒仍是正数。`1969-12-31T23:59:59.5Z` 编码为 `sec=-1, nsec=500000000`，不是 `sec=0, nsec=-500000000`。

仓库 `_parse_ts` 的 legacy 分支是 `ts >> 32`，Python 的 `int` 不会自动截断为 32 位有符号，所以高位 `>= 0x80000000` 时得到 2147483648..4294967295 的正秒，而不是负秒。后果见 3.1。

### 2.2 Bigtime（v3 且 `di_flags2 & 0x8`）

磁盘上是一个无符号 64 位计数器，单位为纳秒，纪元为 1901-12-13T20:45:52Z（即 Unix 纪元减 `2**31` 秒）：

```
unix_ns = raw - 2**31 * 10**9
```

- 没有「秒/纳秒」字段分割。把高 32 位当秒、低 32 位当纳秒会得到完全错误的结果。
- 可表示范围：`1901-12-13T20:45:52Z` 至 `2486-07-02T20:20:25.709551615Z`（`raw = 2**64 - 1`）。
- 1970-01-01T00:00:00Z 对应 `raw = 0x1dcd650000000000`（2147483648000000000），不是 0。
- 2038 年边界对 bigtime 没有特殊意义；`raw` 在 `2**31 * 10**9` 附近连续。

### 2.3 同一串位的两种解释（原始位串对照）

下表由 `tools/xfs_time_audit.py` 生成。`legacy` 与 `bigtime` 两列是对**同一 raw 值**分别按两种格式解码的结果；`basis` 说明该向量在契约检查中应采用的解释。

| 用例 | raw (hex) | 按 legacy 解释 | 按 bigtime 解释 | basis |
|---|---|---|---|---|
| legacy_epoch | `0x0000000000000000` | 1970-01-01T00:00:00.000000000Z | 1901-12-13T20:45:52.000000000Z | 同一 raw，纪元不同 |
| legacy_min_1901 | `0x8000000000000000` | 1901-12-13T20:45:52.000000000Z | 2194-03-23T20:33:08.854775808Z | 高位为负秒 |
| legacy_-1s | `0xffffffff00000000` | 1969-12-31T23:59:59.000000000Z | 2486-07-02T20:20:21.414584320Z | 高位为负秒 |
| legacy_-0.5s | `0xffffffff1dcd6500` | 1969-12-31T23:59:59.500000000Z | 2486-07-02T20:20:21.914584320Z | 负秒 + 正纳秒 |
| legacy_-1ns | `0xffffffff3b9ac9ff` | 1969-12-31T23:59:59.999999999Z | 2486-07-02T20:20:22.414584319Z | 最小负值 |
| legacy_max_2038 | `0x7fffffff3b9ac9ff` | 2038-01-19T03:14:07.999999999Z | 2194-03-23T20:33:05.559808511Z | legacy 上限 |
| legacy_nsec_123456789 | `0x00000000075bcd15` | 1970-01-01T00:00:00.123456789Z | 1901-12-13T20:45:52.123456789Z | 纳秒只在整数结果中完整 |
| legacy_nsec_out_of_range | `0x000000003b9aca00` | 非法（nsec = 1,000,000,000） | 1901-12-13T20:45:53.000000000Z | 低字不是合法纳秒 |
| bigtime_1901 | `0x0000000000000000` | 1970-01-01T00:00:00.000000000Z | 1901-12-13T20:45:52.000000000Z | bigtime 纪元 |
| bigtime_1969-12-31T23:59:59.5 | `0x1dcd64ffe2329b00` | 非法（nsec 越界） | 1969-12-31T23:59:59.500000000Z | 负 Unix 时间 |
| bigtime_1970 | `0x1dcd650000000000` | 1985-11-05T00:53:20.000000000Z | 1970-01-01T00:00:00.000000000Z | 高位不是秒 |
| bigtime_2038-1ns | `0x3b9ac9ffffffffff` | 非法（nsec 越界） | 2038-01-19T03:14:07.999999999Z | 2038 前 1ns |
| bigtime_2038+1s | `0x3b9aca0000000000` | 2001-09-09T01:46:40.000000000Z | 2038-01-19T03:14:08.000000000Z | 2038 后 1s |
| bigtime_nsec_123456789 | `0x1dcd65013161bf15` | 1985-11-05T00:53:21.828489493Z | 1970-01-01T00:00:05.123456789Z | 连续计数器 |
| bigtime_max_2486 | `0xffffffffffffffff` | 非法（nsec 越界） | 2486-07-02T20:20:25.709551615Z | uint64 上限 |

注意：表中 `legacy` 列在若干 bigtime 值上显示「非法」，这是参考解码器对 `t_nsec >= 10**9` 的显式拒绝；它展示了「用错格式」的典型特征，但不能反过来作为判定格式的充分证据（存在低位恰好小于 10**9 的巧合，见 3.3）。

## 3. 核查发现

### 3.1 Legacy 负秒没有符号扩展（缺陷）

`dissect/xfs/xfs.py:556-562`：

```python
def _parse_ts(ts: int, is_bigtime: bool) -> int:
    if is_bigtime:
        sec, nsec = divmod(ts, 1000000000)
        sec -= 1 << 31
    else:
        sec, nsec = ts >> 32, ts & 0xFFFFFFFF
    return (sec * 1000000000) + nsec
```

`ts >> 32` 对 `0x80000000..0xFFFFFFFF` 得到 2147483648..4294967295，而不是 -2147483648..-1。实测差异：

| raw | 正确 unix_ns | 当前实现 unix_ns | 显示差异 |
|---|---|---|---|
| `0x8000000000000000` | -2147483648000000000 (1901-12-13) | 2147483648000000000 (2038-01-19) | 约 136 年，方向相反 |
| `0xffffffff1dcd6500` | -500000000 (1969-12-31T23:59:59.5Z) | 4294967295500000000 (2106-02-07T06:28:15.5Z) | 约 136 年 |

影响面：1970 年以前写入的 legacy 时间（旧文件系统、从其他系统迁移/恢复的文件）会跳到 2038/2106 年。legacy 编码本身无法表示 2038 年之后的时间，遇到超范围值内核会截断或拒绝，因此这条缺陷主要影响负秒区间。取证时间线上，这类值会从「1901/1969 年」跳成「2038/2106 年」，顺序完全反转。

`2038-01-19T03:14:07.999999999Z`（`0x7fffffff3b9ac9ff`）在边界内，当前实现与参考实现一致；问题只在负秒区间。

### 3.2 v1/v2 的 `crtime` 是数据叉内容

`INode.crtime_ns`（`xfs.py:353`）无条件读取 `di_crtime`，没有 `di_version >= 3` 判断。合成 v2 inode 验证：偏移 144 写入 `0x1122334455667788`，`crtime_ns` 会把它当 legacy 时间解成 `287454021432778632` ns，即 `1979-02-10T00:20:21.432778632Z`（低字 1,432,778,632 已超过 10**9，按当前实现还会发生秒进位）；该值是文件数据，不是创建时间。

同时，v2 inode 的偏移 120 也不存在 `di_flags2`。如果数据叉在这 8 字节恰好是 `0x8`，任何「先读 flags2 再判断」的代码都可能误判，但 `_has_bigtime()` 的 `di_version >= 3` 门挡住了这一路径。结论：v1/v2 上四类时间只能报告三类（atime/mtime/ctime），crtime 必须显式标记为「不存在」，不能报 0，也不能报该偏移的值。

### 3.3 标志与版本/超级块矛盾

三种矛盾及建议处置：

- **v3 有 bigtime 位、超级块无该特性**：无法确定该 inode 的时间字段是按哪种格式写入的。当前 `_has_bigtime()` 会按 bigtime 解，可能把 legacy 字节解释成 2000 年以后。应标记为不可信并保留原始位串，不应静默选一种。
- **v3 无 bigtime 位、超级块有该特性**：这是正常情况（旧 inode 未重写），必须按 legacy 解。若按超级块特性统一按 bigtime 解，会把 2022 年解成 2126 年。仓库 `xfs_bigtime.bin.gz` 样本中的 root 是 bigtime，但完全可能存在 legacy inode，不能以文件系统为单位一刀切。
- **v1/v2 偏移 120 恰好像 bigtime 位**：必须忽略，因为 v1/v2 没有 `di_flags2`。

### 3.4 同一 raw 值的巧合可解性

`0x1dcd650000000001` 是 bigtime 的 `1970-01-01T00:00:00.000000001Z`。它的低 32 位是 1（小于 10**9），若按 legacy 解则得到 `1985-11-05T00:53:20.000000001Z`；两种解释都不会触发「纳秒越界」。相反，`0x35210145b9cf25a5`（仓库 bigtime 样本 `file` 的 crtime，低位 3,117,360,549）按 legacy 解时低位越界，但这只是巧合。

结论：**不能只靠纳秒字段是否越界来判断格式**，必须依赖 inode 版本和标志。两条路径都可能产生貌似合法的时间：legacy 样本 crtime `0x6262ba890fc4dd80` 按 bigtime 解为 `2126-08-09T11:56:53.272427904Z`；bigtime 的 1970 年值按 legacy 解为 `1985-11-05T00:53:20Z`。单看结果无法区分。

## 4. 可运行的最小对照

脚本：`tools/xfs_time_audit.py`。它不依赖 `dissect.xfs` 的解码函数，独立实现参考解码器，并做五组检查：

1. **向量表**：打印每个 raw 位串在两种格式下的解释，覆盖 2038 边界、负时间、纳秒精度、uint64 上限。
2. **契约检查**：对每个向量断言 `decode_timestamp()` 的 `unix_ns` 或错误码，覆盖 v1/v2、v3 legacy、v3 bigtime、标志矛盾。
3. **合成 inode**：验证 v2 的偏移 120/144 属于数据叉，并指出 v1/v2 无 crtime。
4. **表示方案对比**：同一 `unix_ns` 分别用整数纳秒和 `datetime` 表示，量化 `datetime` 的微秒截断/浮点舍入。
5. **库差异探针**：把当前 `_parse_ts` 与参考实现逐项对比，标出 `MISMATCH`。

运行：

```bash
.venv/bin/python tools/xfs_time_audit.py
# 跳过样本镜像读取：
.venv/bin/python tools/xfs_time_audit.py --no-samples
```

关键输出（节选，完整输出见脚本）：

```
legacy_min_1901   raw=0x8000000000000000 library_ns= 2147483648000000000 reference_ns=-2147483648000000000 MISMATCH
legacy_-0.5s      raw=0xffffffff1dcd6500 library_ns= 4294967295500000000 reference_ns=          -500000000 MISMATCH
legacy_max_2038   raw=0x7fffffff3b9ac9ff library_ns= 2147483647999999999 reference_ns= 2147483647999999999 OK
bigtime_1970      raw=0x1dcd650000000000 library_ns=                    0 reference_ns=                    0 OK
bigtime_max_2486  raw=0xffffffffffffffff library_ns= 16299260425709551615 reference_ns= 16299260425709551615 OK
```

样本镜像的原始位串（`tests/_data/xfs.bin.gz` 与 `xfs_bigtime.bin.gz`）：

| 样本 | inode | `di_version` | `di_flags2` | 字段 | raw | 解释 |
|---|---|---|---|---|---|---|
| legacy | root | 3 | `0x0` | crtime | `0x6262ba890fc4dd80` | sec=1650637449, nsec=264560000 → 2022-04-22T14:24:09.264560000Z |
| legacy | root | 3 | `0x0` | atime | `0x6262bac816141f0d` | sec=1650637512, nsec=370417421 → 2022-04-22T14:25:12.370417421Z |
| bigtime | root | 3 | `0x8` | crtime | `0x3521012cec2546d8` | counter=3828342450693687000 → 2023-04-07T09:13:22.693687000Z |
| bigtime | file | 3 | `0x8` | crtime | `0x35210145b9cf25a5` | counter=3828342557223364005 → 2023-04-07T09:15:09.223364005Z |

两套样本的超级块 `sb_features_incompat` 分别为 `0x3`（无 bigtime）和 `0xb`（含 bigtime），与各自的 `di_flags2` 一致。

### 4.1 两个表示方案的直接比较

| 维度 | 整数纳秒（`*_ns`） | `datetime` |
|---|---|---|
| 精度 | 保留完整 9 位纳秒 | 只到微秒，纳秒被截断或舍入 |
| 排序 | 任意整数比较，负数也正确 | 可比较，但必须先有正确转换 |
| 范围 | Python int 无上限；但落库若用 int64 纳秒，上限为 `2262-04-11T23:47:16.854775807Z` | 依赖平台；`fromtimestamp` 在部分平台对负时间/极远时间会异常 |
| 序列化 | JSON 整数在 2^53 以上（约 104 天纳秒）会被 JS 等消费方丢精度 | ISO 8601 字符串可读，但已丢失纳秒 |
| 取证建议 | **作为时间线的规范值**，用于排序、差值和存储 | 仅用于展示；从整数派生，不要反向用 `datetime` 重算 |

`dissect.util.ts.from_unix_ns` 的实现是 `float(ts) * 1e-9` 再 `datetime.fromtimestamp`。浮点在 2022 年附近的双精度分辨率约为 0.24 µs，在 2486 年约为 2 µs。实测：

```
unix_ns=1650637512370417421  floor=...12.370417  float=...12.370418  delta=1µs
unix_ns=-1                   floor=1969-12-31T23:59:59.999999  float=1970-01-01T00:00:00.000000  delta=1µs
unix_ns=16299260425709551615 floor=...25.709551  float=...25.709553  delta=2µs
```

`datetime` 的微秒字段也无法表示 `...999999999` 与下一秒的区别：legacy 上限 `2038-01-19T03:14:07.999999999Z` 经 `from_unix_ns` 变成 `2038-01-19T03:14:08Z`，跨过了 2038 边界。若时间线以 `datetime` 排序，这会与真正的 2038-01-19T03:14:08Z 并列，造成边界事件顺序不确定。

## 5. 不能据此推断用户操作顺序的情形

以下限制与编码无关，是 XFS 时间语义本身的边界：

- **`ctime` 不是创建时间。** 它是 inode 元数据变更时间。`chmod`、`chown`、`link`、`unlink`、扩展属性修改都会推进它；创建后没有任何元数据变更时，`ctime` 可能等于 `crtime`，但二者语义不同。不要把 `ctime` 填进「创建时间」列。
- **`crtime` 不是采集时间，也不是「文件首次出现在磁盘上的时间」。** 它是 inode 创建时间，仅在 v3 且内核支持时存在；复制、恢复、解包会生成新的 `crtime`。取证采集时间来自采集工具/镜像元数据，不是 inode 字段。
- **`atime` 不能证明用户「打开/读取」了文件。** `relatime`（多数发行版默认）只在 atime 早于 mtime/ctime 或超过 24 小时时更新；`noatime` 完全不更新；`lazytime` 延迟写回。atime 缺失或未变不等于没有访问。
- **`mtime` 只说明数据被写过，不能区分写入者。** 应用层可能保留/伪造 mtime（`touch -d`、`os.utime`、解包工具），`mtime` 可以早于 `crtime`。若看到 `mtime < crtime`，应视为时间被显式设置或时钟异常，而不是「先修改后创建」。
- **同一秒内的事件无法用这些字段排序。** 纳秒字段虽在，但文件系统日志、页缓存回写、`lazytime` 都会让落盘时间晚于实际操作时间；不同 inode 的纳秒也不保证同一时钟源下严格有序。
- **时钟回拨/虚拟化快照会破坏单调性。** 挂起恢复、NTP 校正、快照回滚都可让后发生的事件拥有更早的时间戳。不能仅凭时间戳排序推断因果。
- **未挂载/只读镜像上的时间不会更新。** 分析镜像时看到的是最后一次写入磁盘的值，不是采集时刻的实时状态。
- `crtime` 为 0 或缺失不是「1970 年创建」。v1/v2 没有该字段；部分工具写 0 表示未知。应报告「不可用」，而不是把 0 当纪元。注意 bigtime 下 `raw = 0` 对应 1901-12-13，也不是 1970；「0 等于未知」只在 legacy 的 `sec=0, nsec=0` 或工具约定下成立。
- **不能把「文件创建时间」「元数据修改时间」「取证采集时间」三者混为一列。** 本报告建议时间线至少保留四个独立列：`crtime`（若可用）、`mtime`、`ctime`、`atime`，另加采集时间（来自镜像/工具，非 inode）。

## 6. 建议的落库字段

对每个 inode 的每个时间字段，建议同时保留：

```text
field_name        atime | mtime | ctime | crtime
raw_u64           原始 64 位无符号值（十进制与 0x 十六进制）
encoding          legacy | bigtime | absent | inconsistent
inode_version     di_version
flags2            di_flags2（仅 v3；v1/v2 为 null）
fs_has_bigtime    超级块特性位（仅用于交叉校验）
unix_ns           有符号 64 位纳秒（可空；不可信时为 null）
display_utc       ISO 8601（仅展示，从 unix_ns 派生）
```

规则：

- `encoding=absent` 仅用于 v1/v2 的 `crtime`；`unix_ns` 必须为 null。
- `encoding=inconsistent` 用于标志与版本/超级块矛盾；保留 `raw_u64`，`unix_ns` 置 null，并在时间线中标注不可用。
- 排序、差值、窗口过滤一律用 `unix_ns`（int），不要用 `display_utc`。
- JSON 输出若必须跨语言，`unix_ns` 以字符串传输，避免 2^53 精度丢失。

## 7. 修复建议（按优先级）

1. **`_parse_ts` 的 legacy 分支补符号扩展**：

   ```python
   import struct

   sec, nsec = struct.unpack(">iI", int(ts).to_bytes(8, "big"))
   if nsec >= 1_000_000_000:
       raise ValueError("invalid legacy nanosecond field")
   return sec * 1_000_000_000 + nsec
   ```

   已用 `0x8000000000000000`、`0xffffffff1dcd6500`、`0x7fffffff3b9ac9ff` 验证，结果与参考实现一致。

2. **`crtime` 增加版本门**：`di_version < 3` 时应抛出明确异常（或返回 `None`），不要读偏移 144。
3. **`_has_bigtime()` 增加超级块交叉校验**：`di_version >= 3`、`di_flags2 & 0x8` 且 `sb_features_incompat & 0x8` 三者同时满足才返回 `True`；只有 inode 标志而超级块无特性时，应暴露为不一致状态而非静默按 bigtime 解。
4. **时间线落库保留 `unix_ns` 整数**，`datetime` 仅在展示层从整数派生。

## 8. 复现与基线

```bash
export NO_INTERNET=1 TZ=UTC
.venv/bin/python -m pytest -q -o addopts= tests   # 7 passed
.venv/bin/python tools/xfs_time_audit.py          # all assertions passed
```

核查基于仓库基线 `abac18b` 与 `dissect-util 3.24`。若后续升级 `dissect-util`，需重新确认 `from_unix_ns` 的浮点实现是否变化；`_parse_ts` 的符号扩展缺陷需在库内修复后，本报告的 MISMATCH 才应消失。
