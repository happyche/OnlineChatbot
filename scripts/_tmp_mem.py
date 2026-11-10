"""临时：实测 Chroma 在目标规模下的进程内存占用。"""
import ctypes, ctypes.wintypes as wt
import random, shutil, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.GetCurrentProcess.restype = wt.HANDLE
_k32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
_k32.K32GetProcessMemoryInfo.restype = wt.BOOL
_h = _k32.GetCurrentProcess()


def _counters():
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    if not _k32.K32GetProcessMemoryInfo(_h, ctypes.byref(c), c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return c


def rss_mb():
    return _counters().WorkingSetSize / 1024 / 1024


def peak_mb():
    return _counters().PeakWorkingSetSize / 1024 / 1024


prev = rss_mb()


def step(label):
    global prev
    cur = rss_mb()
    print(f"{label:<34} {cur:8.1f} MB   (+{cur - prev:6.1f})")
    prev = cur


print(f"{'裸 Python 解释器':<34} {prev:8.1f} MB")

import chromadb
step("import chromadb")

N, DIM = 13000, 512
tmp = Path(tempfile.mkdtemp(prefix="mem-"))
client = chromadb.PersistentClient(path=str(tmp))
step("PersistentClient 打开")

col = client.create_collection(
    name="memtest", metadata={"hnsw:space": "cosine"}, embedding_function=None
)
rnd = random.Random(0)
docs = []
for f in sorted(Path("uploads").glob("*.md")):
    t = f.read_text(encoding="utf-8")
    docs += [t[i : i + 320] for i in range(0, len(t), 320)]
docs = [d for d in docs if d.strip()]
corpus = [docs[i % len(docs)] for i in range(N)]

for i in range(0, N, 2000):
    n = min(2000, N - i)
    col.add(
        ids=[f"c{j}" for j in range(i, i + n)],
        embeddings=[[rnd.random() for _ in range(DIM)] for _ in range(n)],
        documents=corpus[i : i + n],
        metadatas=[{"source": f"doc{j % 200}.md", "heading": "章节"} for j in range(i, i + n)],
    )
step(f"写入 {N} 条 x {DIM} 维")

q = [rnd.random() for _ in range(DIM)]
for _ in range(200):
    col.query(query_embeddings=[q], n_results=20, include=["documents", "distances"])
step("跑 200 次检索后")

from retrieval import BM25Index
idx = BM25Index("jieba")
idx.build([f"c{i}" for i in range(N)], corpus)
step("再叠加 BM25 索引")

print(f"\n理论向量数据量: {N * DIM * 4 / 1024 / 1024:.1f} MB")
print(f"进程峰值 (Peak Working Set): {peak_mb():.1f} MB")

shutil.rmtree(tmp, ignore_errors=True)
