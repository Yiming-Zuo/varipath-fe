from __future__ import annotations

import copy
import fcntl
import json
from pathlib import Path

import numpy as np
import torch


def _atomic_torch(path,value):
    pending=path.with_name(path.name+".pending")
    torch.save(value,pending)
    pending.replace(path)


def _state(model):
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}


def _same_state(first,second):
    return first.keys()==second.keys() and all(torch.equal(first[key].cpu(),second[key].cpu()) for key in first)


def _header(path,shape):
    array=np.load(path,mmap_mode="r",allow_pickle=False)
    if array.shape!=tuple(shape) or array.dtype!=np.dtype("float64") or not array.flags.c_contiguous:
        raise ValueError("冻结中心轨迹的 shape、dtype 或存储顺序与元数据不同")
    offset=int(array.offset)
    expected=offset+int(np.prod(shape,dtype=object))*8
    if path.stat().st_size!=expected:
        raise ValueError("冻结中心轨迹文件没有完整的预定长度")
    del array
    return offset


class DiskCenters:

    def __init__(self,folder,*,run_root=None,expected_metadata=None,cache_steps=128):
        self.folder=Path(folder).resolve()
        self.run_root=Path(run_root).resolve() if run_root is not None else self.folder.parent
        if not self.folder.is_relative_to(self.run_root):
            raise ValueError("冻结中心轨迹必须位于运行根目录内")
        if type(cache_steps) is not int or cache_steps<1:
            raise ValueError("中心读取缓存间隔须为正整数")
        self.cache_steps=cache_steps
        self.metadata=json.loads((self.folder/"metadata.json").read_text())
        if expected_metadata is not None and self.metadata!=expected_metadata:
            raise ValueError("冻结中心 provider 的不可变元数据发生变化")
        if self.metadata.get("format_version")!=1 or self.metadata.get("dtype")!="float64":
            raise ValueError("冻结中心 provider 格式不受支持")
        self.shape=tuple(self.metadata["shape"])
        if len(self.shape)!=3 or any(type(value) is not int or value<1 for value in self.shape):
            raise ValueError("冻结中心轨迹须有正整数 [steps,centers,dim] 形状")
        self.path=self.folder/"centers.npy"
        self.offset=_header(self.path,self.shape)
        progress=torch.load(self.folder/"progress.pt",map_location="cpu",weights_only=True)
        if progress.get("metadata")!=self.metadata or progress.get("next_step")!=self.shape[0] or progress.get("complete") is not True:
            raise ValueError("冻结中心轨迹尚未完整生成，不能用部分前缀训练")
        if progress["point"].shape!=self.shape[1:] or not torch.isfinite(progress["point"]).all():
            raise ValueError("冻结中心终点检查点无效")
        self._source_state=progress["source_state"]
        self._initial_centers=progress["initial_centers"]
        self._cache=None
        self._cache_first=-1

    def __getstate__(self):
        raise TypeError("DiskCenters 不能序列化为模型对象；请保存 descriptor 和 source_state")

    def descriptor(self):
        try:
            relative=self.folder.relative_to(self.run_root)
        except ValueError as exc:
            raise ValueError("中心轨迹必须位于其运行根目录内") from exc
        return dict(kind="frozen-disk-centers",format_version=1,directory=relative.as_posix(),metadata=copy.deepcopy(self.metadata))

    @classmethod
    def from_descriptor(cls,descriptor,run_root,*,expected_source_state=None,cache_steps=128):
        if descriptor.get("kind")!="frozen-disk-centers" or descriptor.get("format_version")!=1:
            raise ValueError("冻结中心 descriptor 格式错误")
        relative=Path(descriptor["directory"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("中心 descriptor 只能引用运行目录内的相对路径")
        result=cls(Path(run_root)/relative,run_root=run_root,expected_metadata=descriptor["metadata"],cache_steps=cache_steps)
        if expected_source_state is not None and not _same_state(result._source_state,expected_source_state):
            raise ValueError("冻结中心的源预训练模型参数与检查点不同")
        return result

    def source_state(self):
        return {key:value.clone() for key,value in self._source_state.items()}

    def validate(self):
        current=DiskCenters(self.folder,run_root=self.run_root,expected_metadata=self.metadata,cache_steps=self.cache_steps)
        if not _same_state(current._source_state,self._source_state) or not torch.equal(current._initial_centers,self._initial_centers):
            raise ValueError("已打开的冻结中心源预训练模型或初始中心发生变化")

    def at(self,step,*,device=None):
        if type(step) is not int or not 0<=step<self.shape[0]:
            raise IndexError("中心轨迹步编号超出固定网格")
        if self._cache is None or not self._cache_first<=step<self._cache_first+len(self._cache):
            first=(step//self.cache_steps)*self.cache_steps
            stop=min(first+self.cache_steps,self.shape[0])
            stride=self.shape[1]*self.shape[2]
            with self.path.open("rb") as stream:
                stream.seek(self.offset+first*stride*8)
                values=np.fromfile(stream,dtype=np.float64,count=(stop-first)*stride)
            if values.size!=(stop-first)*stride or not np.isfinite(values).all():
                raise FloatingPointError("冻结中心轨迹块缺失或含非有限值")
            self._cache=values.reshape(stop-first,*self.shape[1:])
            self._cache_first=first

        return torch.from_numpy(self._cache[step-self._cache_first].copy()).to(device=device)


def freeze_centers(pretrain,initial_centers,folder,*,run_root=None,chunk_steps=128,
                   resume=False,check_budget=None):

    if type(chunk_steps) is not int or chunk_steps<1:
        raise ValueError("中心生成检查点间隔须为正整数")
    initial=torch.as_tensor(initial_centers,dtype=torch.float64)
    pretrain._check_batch(initial)
    if not len(initial):
        raise ValueError("移动中心不能为空")
    if getattr(pretrain,"center_provider_descriptor",lambda:None)() is not None or getattr(pretrain,"time_centers",None) is not None:
        raise ValueError("移动中心预训练图不能再依赖另一条磁盘中心轨迹")
    folder=Path(folder)
    metadata=dict(format_version=1,dtype="float64",shape=[pretrain.n_steps,len(initial),pretrain.dim],
                  spline_count=pretrain.spline_count,chunk_steps=chunk_steps,
                  convention="center[k] is the frozen pretraining map state before actual step k")
    source_state=_state(pretrain)
    if not resume:
        folder.mkdir(parents=True,exist_ok=False)
    elif not folder.is_dir():
        raise FileNotFoundError("恢复移动中心需要原目录和检查点")
    with (folder/".writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resume:
            if json.loads((folder/"metadata.json").read_text())!=metadata:
                raise ValueError("中心生成配置或不可变元数据发生变化")
            progress=torch.load(folder/"progress.pt",map_location="cpu",weights_only=True)
            if progress.get("metadata")!=metadata or not _same_state(progress["source_state"],source_state) or not torch.equal(progress["initial_centers"],initial.detach().cpu()):
                raise ValueError("中心生成源预训练模型或初始中心发生变化")
            _header(folder/"centers.npy",metadata["shape"])
            first=int(progress["next_step"])
            if not 0<=first<=pretrain.n_steps or progress["point"].shape!=initial.shape or not torch.isfinite(progress["point"]).all():
                raise ValueError("移动中心恢复检查点的进度或坐标无效")
            point=progress["point"].to(initial)
            if first==pretrain.n_steps:
                return DiskCenters(folder,run_root=run_root)
        else:
            (folder/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
            mapped=np.lib.format.open_memmap(folder/"centers.npy",mode="w+",dtype=np.float64,shape=tuple(metadata["shape"]))
            mapped.flush()
            del mapped
            first,point=0,initial.detach().clone()
            progress=dict(metadata=metadata,next_step=0,complete=False,point=point.cpu(),
                          source_state=source_state,initial_centers=initial.detach().cpu().clone())
            _atomic_torch(folder/"progress.pt",progress)
        with torch.no_grad():
            for start in range(first,pretrain.n_steps,chunk_steps):
                if check_budget is not None:
                    check_budget()
                stop=min(start+chunk_steps,pretrain.n_steps)
                mapped=np.lib.format.open_memmap(folder/"centers.npy",mode="r+")
                try:
                    for k in range(start,stop):
                        mapped[k]=point.detach().cpu().numpy()
                        point=pretrain.step(point,k)[0].detach()
                    mapped.flush()
                finally:
                    del mapped
                progress.update(next_step=stop,complete=stop==pretrain.n_steps,point=point.cpu())
                _atomic_torch(folder/"progress.pt",progress)
                if check_budget is not None:
                    check_budget()
    return DiskCenters(folder,run_root=run_root)
