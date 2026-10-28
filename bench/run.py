import shutil
import subprocess
import sys, os, signal, sys, time, statistics, uuid, json
import httpx, argparse, socket
import asyncio, random
import numpy as np
import pynvml
import vllm
from pathlib import Path
from datetime import datetime
from openai import AsyncOpenAI
from collections import defaultdict


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


class VLLMService:
    def __init__(self, model: str, port: int = 8000,
                 host: str = "127.0.0.1", ready_timeout: int = 300,
                 extra_args: str = None):
        self.model = model
        self.port = port
        self.host = host
        self.ready_timeout = ready_timeout
        self.extra_args = extra_args
        self.log_f = None
        self.proc = None

    async def wait_ready(self):  # *** httpx.get()和time.sleep都是阻塞事件循环 ***
        url = f"http://{self.host}:{self.port}/health" # f string会自动转变量成str不用再加
        deadline = time.perf_counter() + self.ready_timeout
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None: # 进程退出 status_code是0或非0 vs if self.proc.poll()
                raise RuntimeError(
                    f"vllm exited with code={self.proc.returncode}, see{self.log_f.name}"
                )
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    r = await client.get(url)  # 探测次数10-50次合理，加载时间30-120s
                if r.status_code == 200:
                    return 
            except httpx.HTTPError:
                pass
            await asyncio.sleep(3) # 30s-60s重连
        raise RuntimeError(
            f"service is not ready, timeout is {self.ready_timeout}, see{self.log_f.name}"
        )

    def start(self):
        args = ["vllm", "serve", self.model,
                "--port", str(self.port), "--host", self.host]
        # cmd指令只涉及字符串没有int float
        if self.extra_args:
            args += self.extra_args
        # 优先 uv（带 .env 隔离环境），否则回退到当前解释器的 vLLM 模块
        if shutil.which("uv") and (_ROOT / ".env").exists():
            cmd = ["uv", "run", "--env-file", ".env"] + args
        else:
            cmd = [sys.executable, "-m",
                   "vllm.entrypoints.openai.api_server"] + args[2:]
        # 日志固定落在脚本目录，供采集脚本按 tag 归档
        self.log_f = open(_HERE / "vllm.log", mode="w")
        self.proc = subprocess.Popen(
            cmd, stdout=self.log_f, stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(_ROOT),
        )   
        # start_new_session子进程创建新的会话并成为组长，如果跟随父进程则用killpg只会删除父进程，
        # 子进程无法释放，方便后续os.killpg(r.pid)

    def stop(self):
        if self.proc.poll() is not None:
            if self.log_f:
                self.log_f.close()
            return
        try:
            pgid = os.getpgid(self.proc.pid) # 上面加入了start_new_session参数让
                                             # 子进程重新开了一组并且恰好包含所有子孙进程
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=20) # vLLM 释放 NCCL/CUDA context 通常几秒十几秒
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self.proc.wait()
        except ProcessLookupError: # 在拿到pgid后结束了报错系统调用级异常
            pass
        finally:
            if self.log_f:
                self.log_f.close()

    async def __aenter__(self):
        self.start()
        try:
            await self.wait_ready()
        except BaseException: # 任何失败都要清理
            self.stop();
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.stop()
        return False  # None or False让异常继续传递 Ture会覆盖异常


async def warmup(client, args, n_warm=3):
    prompt = make_prompts(args)[0]
    for i in range(n_warm):
        await stream_one(client=client, req_id=-1, 
                         prompt_ids=prompt, args=args)

def make_prompts(args):
    """
    共享前缀模式（由 prefix_len / prefix_ratio 控制）
    1、每条 prompt 长度恒等于 S
    2、prefix_len 必须是 block_size 的整数倍，否则最后一块不命中
    3、前 n_shared = round(N * prefix_ratio) 条共用同一前缀，其余完全独立
    说明：legacy 内核无 seed 形参，沿用固定种子 123 以保持历史可比。
    """
    rng = random.Random(123)
    prefix_rng = random.Random(args.prefix_seed)
    if args.prefix_len is None or args.prefix_len % args.block_size != 0:
        raise ValueError(f"prefix_len isn't expected config: {args.prefix_len}")
    if args.prefix_ratio is None or not (0 < args.prefix_ratio <= 1):
        raise ValueError(f"prefix_ratio isn't between (0, 1]: {args.prefix_ratio}")
    n_shared = round(args.N * args.prefix_ratio)
    shared_prefix = [prefix_rng.randrange(1000, args.vocab_size)
                     for _ in range(args.prefix_len)]
    shared_requests = [
        shared_prefix + [rng.randrange(1000, args.vocab_size)
                         for _ in range(args.S - args.prefix_len)]
        for _ in range(n_shared)
    ]
    independent_requests = [
        [rng.randrange(1000, args.vocab_size) for _ in range(args.S)]
        for _ in range(args.N - n_shared)
    ]
    return shared_requests + independent_requests

def aggregate(results, args, wall_time: float): 
    # status_dict是否是BaseException的实例
    ok_record = [status_dict for status_dict in results if not isinstance(status_dict, BaseException)]
    err_record = [repr(err) for err in results if isinstance(err, BaseException)]
    # python异常对象返回的是error类的实例化，需要换成用户向的str()或开发者向的repr()才方便记录进json文件
    
    # itl_mean = statistics.mean([statistics.mean(status["itls"]) for status in ok_record 
    #                             if status.get("itls") is not None])
    # 按照两个维度做两次P会造成误差，在大量数据上误差会被放大，需要变成一维列表
    def PCT(data, p): # 返回的是np.float64 json.dumpS时报错 
                      # TypeError: Object of type float64 is not json serializable
        if data is None:
            return
        if not data:
            return None
        return  float(np.percentile(data, p)) # 源头处作类型转换
    ttft_data = [status["ttft"] for status in ok_record # 需要有None保护
                 if status.get("ttft") is not None]
    itl_data = [x for s in ok_record for x in s.get("itls", [])] # 若列表不存在用空列表代替，
                                                              # 防止for x in None
    tpot_data = [(s["window"][1] - s["window"][0]) / 
                 (s["n_out"] - 1) for s in ok_record 
                 if s["n_out"] > 1 and s["window"]]
                # TPOT分母为N-1，加判别条件否则除以0会有IndexError
    # steady_window_s = min(s["window"][1] for s in ok_record 
    #                       if s.get("window")) - max(s["window"][0] for s in ok_record if s.get("window"))
    # 多个并发时稳态窗口不能简单用交集计算
    latency = {
        "ttft_mean": statistics.mean(ttft_data) if ttft_data else None,
        "ttft_p50": PCT(ttft_data, 50),
        "ttft_p95": PCT(ttft_data, 95),
        "itl_mean": statistics.mean(itl_data) if itl_data else None,
        "itl_p95": PCT(itl_data, 95),
        "tpot_mean": statistics.mean(tpot_data) if tpot_data else None,
    }
    system = {
        "wall_time": wall_time,
        "token_per_s": sum([s["n_out"] for s in ok_record]) / wall_time, 
                                                            # NameError: name 'system' is not defined
        "request_per_s": len(ok_record) / wall_time,
        "speedup": sum([s["t_end"]-s["t_start"] for s in ok_record]) / wall_time, 
    }
    workload = {
        "N": args.N,
        "B": args.B,
        "S_in": args.S,
        "S_out": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "vocab_size": args.vocab_size,
    }
    return {
        "error": err_record,
        "workload": workload,
        "system": system,
        "latency": latency,
        "raw": ok_record,
    }

async def stream_one(client: AsyncOpenAI, req_id: int, 
                     prompt_ids: list, args):
    status = {
        "req_id": req_id,
        "ttft": None,
        "itls": [],
        "window": [],
        "t_start": 0,
        "t_end": 0,
        "t_create": 0,
        "n_out": 0,
    }
    t0 = time.perf_counter()
    stream = await client.completions.create( # 不await拿到的是协程对象而非异步迭代器
        model=args.model,
        prompt=prompt_ids,
        stream=True,
        temperature=args.temperature,
        top_p=args.top_p, 
        max_tokens=args.max_tokens,
        extra_body={ # 压测设置要求prompt拉满
            "ignore_eos": True, # 忽略结束符
            "min_tokens": args.max_tokens, # 最少生成token数
            "add_special_tokens": False, # 避免prompt被特殊token修饰
        },
    )
    status["t_create"] = time.perf_counter()
    ft = True; new_time = 0; n_out = 0; time_ft = None
    async for chunk in stream:
        text = chunk.choices[0].text
        if text:
            now = time.perf_counter()
            if ft:
                time_ft = now; ft = False
                status["ttft"] = now - t0
            else:
                status["itls"].append(now - new_time)
            n_out += 1
            new_time = now
    status["t_end"] = time.perf_counter()
    status["n_out"] = n_out
    status["t_start"] = t0
    if time_ft is not None:
        status["window"] = [time_ft, new_time]
    return status


async def run_benchmark(client, args):
    prompts = make_prompts(args)
    sem = asyncio.Semaphore(args.B)
    async def one(req_id, prompt):
        async with sem:
            return await stream_one(client=client, req_id=req_id,
                                    prompt_ids=prompt, args=args)
    begin_time = time.perf_counter()
    tasks = [one(rid, prompt) for rid, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks, return_exceptions=True) # return_exceptions=True时
                                                                   # 失败请求会返回BaseException
    end_time = time.perf_counter()
    return aggregate(results, args, end_time-begin_time)

def dump_results(record, args):
    timestamp = datetime.now().isoformat() 
    run_id = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}_{str(uuid.uuid4())[:8]}"
    
    pynvml.nvmlInit()
    device_status = []
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i) # 通过设备索引获取GPU设备的句柄
        name = pynvml.nvmlDeviceGetName(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle) 
        util = pynvml.nvmlDeviceGetUtilizationRates(handle) # c_nvmlUtilization_t 对象 需要转换成可序列化对象
        status = {
            "name": name, 
            "mem_total_mib": mem.total // 1024**2,
            "mem_used_mib": mem.used // 1024**2,
            "util_gpu_pct": util.gpu,
            "util_mem_pct": util.memory,
        }
        device_status.append(status)

    vllm_version = vllm.__version__
    python_version = sys.version
    hostname = socket.gethostname()
    run_meta = {
        "run_id": run_id,
        "device_status": device_status,
        "vllm_version": vllm_version,
        "python_version": python_version,
        "timestamp": timestamp,
        "hostname": hostname,
        "record": record,
        "quality": "reference-only",
        "quality_reason": "shared GPU",
    }
    out_dir = Path(args.path)
    out_dir.mkdir(parents=True, exist_ok=True) # 递归创建目录 目录存在时忽略错误
    with open(f"{args.path}/{run_id}.json", mode="w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False, default=float) # 缩近和防止中文乱码

    
async def main():
    # perf_counter()获得的是墙上的时钟时间可读性差不能跨设备比较
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8000) # argparse.SUPPRESS会让属性不存在
    p.add_argument("--ready_timeout", type=int, default=300)
    p.add_argument("--S", type=int, default=1024)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.55)
    p.add_argument("--path", type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    
    args = p.parse_args()

    svc = VLLMService(args.model, args.port,
                      ready_timeout=args.ready_timeout,
                      extra_args=["--max-num-seqs", str(args.B),
                                "--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://127.0.0.1:{args.port}/v1",
    )
    async with svc:
        await warmup(client, args=args)  # async 函数是惰性，只有被await或者asyncio
                                # 调用产生协程才会放入事件循环中执行
                                # warmup被创建coroutine对象 最后报错RuntimeWarning
        record = await run_benchmark(client, args=args)
        dump_results(record, args) # 落盘操作是纯I/O行为会阻塞协程事件循环，阻塞行为：
                                   # 1、是否是同步API；
                                   # 2、是否会长时间占用线程（磁盘、网络、锁、大计算）；
                                   # 3、中间有无await让出点；

if __name__ == "__main__":
    asyncio.run(main())
