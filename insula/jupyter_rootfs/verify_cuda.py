#!/usr/bin/env python3
"""Integrated CUDA checks for the GPU insula: toolchain, compilers, and torch interop.

Run inside the Jupyter insula (insula/jupyter_rootfs/verify_cuda.sh does that):
    python insula/jupyter_rootfs/verify_cuda.py -v
Layers, each failing independently so the first red test names the broken layer:
  1. toolchain  nvcc/ptxas/g++ present, versions consistent with torch's CUDA, CUDA_HOME set
  2. compile    nvcc -> cubin, nvcc -> ptx, NVRTC -> ptx, and a bad source is rejected
  3. launch     compiled kernels run on torch tensors (torch helper and cuda.bindings driver API)
  4. interop    __cuda_array_interface__, DLPack, numpy, streams share memory with torch tensors
"""
import ctypes
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

VECTOR_ADD = r"""
extern "C" __global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
extern "C" __global__ void fill_index(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)i;
}
"""
NAMES = ["vector_add", "fill_index"]
CUDA_BIN = Path("/usr/local/cuda/bin")


def sm():
    return "".join(map(str, torch.cuda.get_device_capability()))


def nvcc(*args, cwd=None):
    return subprocess.run(["nvcc", *args], capture_output=True, text=True, cwd=cwd)


class Workdir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.source = self.tmp / "vector_add.cu"
        self.source.write_text(VECTOR_ADD)

    def build(self, kind):
        """kind: cubin | ptx via nvcc, or nrtc via NVRTC. Returns bytes the driver can load."""
        if kind == "nvrtc":
            from torch.cuda._utils import _nvrtc_compile
            return _nvrtc_compile(VECTOR_ADD, "vector_add")[0]
        out = self.tmp / f"vector_add.{kind}"
        result = nvcc(f"-arch=sm_{sm()}", f"-{kind}", str(self.source), "-o", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        return out.read_bytes()

    def load(self, image):
        from torch.cuda._utils import _cuda_load_module
        return _cuda_load_module(image, NAMES)


class ToolchainTests(unittest.TestCase):
    def test_gpu_visible_to_torch(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertGreaterEqual(torch.cuda.device_count(), 1)

    def test_tools_resolve_under_cuda_bin_not_symlinks_elsewhere(self):
        for tool in ("nvcc", "ptxas", "g++"):
            self.assertIsNotNone(shutil.which(tool), f"{tool} missing from PATH")
        self.assertEqual(Path(shutil.which("nvcc")).parent, CUDA_BIN)
        self.assertTrue((CUDA_BIN / "ptxas").is_file())

    def test_nvcc_version_matches_torch_cuda(self):
        text = nvcc("--version").stdout
        release = re.search(r"release (\d+\.\d+)", text).group(1)
        self.assertEqual(release, torch.version.cuda, text)

    def test_ptxas_and_nvcc_are_the_same_patch_release(self):
        """A newer cicc/nvvm emits PTX an older ptxas rejects; the pinned set must agree."""
        nvcc_build = re.search(r"V(\d+\.\d+\.\d+)", nvcc("--version").stdout).group(1)
        ptxas_build = re.search(r"V(\d+\.\d+\.\d+)",
                                subprocess.run(["ptxas", "--version"], capture_output=True, text=True).stdout).group(1)
        self.assertEqual(nvcc_build, ptxas_build)

    def test_cuda_home_and_headers(self):
        from torch.utils import cpp_extension
        self.assertEqual(os.environ.get("CUDA_HOME"), "/usr/local/cuda")
        self.assertEqual(cpp_extension.CUDA_HOME, "/usr/local/cuda")
        for header in ("cuda_runtime.h", "cuda_fp16.h"):
            self.assertTrue((Path(os.environ["CUDA_HOME"]) / "include" / header).is_file(), header)

    def test_device_arch_is_one_nvcc_can_generate_code_for(self):
        self.assertIn(f"sm_{sm()}", nvcc("--list-gpu-code").stdout.split())


class CompileTests(Workdir):
    def test_nvcc_builds_a_cubin_for_this_device(self):
        image = self.build("cubin")
        self.assertTrue(image.startswith(b"\x7fELF"), "cubin is an ELF image")

    def test_nvcc_builds_ptx_that_ptxas_accepts(self):
        ptx = self.build("ptx")
        self.assertIn(b".visible .entry vector_add", ptx)
        result = subprocess.run(["ptxas", f"-arch=sm_{sm()}", str(self.tmp / "vector_add.ptx"), "-o", str(self.tmp / "x.cubin")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_nvrtc_builds_a_loadable_image_for_this_device(self):
        """torch's helper returns a cubin on current GPUs and PTX otherwise; both name the kernel."""
        image = self.build("nvrtc")
        self.assertTrue(image.startswith(b"\x7fELF") or b".entry vector_add" in image)
        self.assertIn("vector_add", self.load(image))

    def test_nvcc_rejects_broken_source_with_a_located_error(self):
        broken = self.tmp / "broken.cu"
        broken.write_text("__global__ void k(float* x) { x[0] = undefined_name; }\n")
        result = nvcc(f"-arch=sm_{sm()}", "-cubin", str(broken), "-o", str(self.tmp / "b.cubin"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("undefined_name", result.stderr)

    def test_host_compiler_builds_with_nvcc_into_an_executable(self):
        main = self.tmp / "main.cu"
        main.write_text('#include <cstdio>\n__global__ void k(int* x) { *x = 42; }\n'
                        'int main() { int* d; int h = 0; cudaMalloc(&d, 4); k<<<1,1>>>(d);\n'
                        'cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); printf("%d\\n", h); return h != 42; }\n')
        exe = self.tmp / "main"
        build = nvcc(f"-arch=sm_{sm()}", str(main), "-o", str(exe))
        self.assertEqual(build.returncode, 0, build.stderr)
        run = subprocess.run([str(exe)], capture_output=True, text=True)
        self.assertEqual((run.returncode, run.stdout.strip()), (0, "42"), run.stderr)


class LaunchTests(Workdir):
    def check_vector_add(self, kernels, n=1000):
        a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
        out = torch.empty_like(a)
        kernels["vector_add"](grid=((n + 255) // 256, 1, 1), block=(256, 1, 1), args=[a, b, out, n])
        torch.cuda.synchronize()
        torch.testing.assert_close(out, a + b)

    def test_torch_launches_nvcc_cubin_on_tensors(self):
        self.check_vector_add(self.load(self.build("cubin")))

    def test_ptx_jit_follows_the_driver_vs_toolkit_version_rule(self):
        """PTX must be JIT-compiled by a driver at least as new as the toolkit that wrote it.

        Cubins for the real architecture (tests above) are exempt via minor-version compatibility.
        """
        from cuda.bindings import driver
        torch.zeros(1, device="cuda")
        driver_version = driver.cuDriverGetVersion()[1]
        major, minor = map(int, torch.version.cuda.split("."))
        ptx = self.build("ptx")
        if driver_version >= major * 1000 + minor * 10:
            self.check_vector_add(self.load(ptx))
        else:
            with self.assertRaisesRegex(RuntimeError, "unsupported toolchain"):
                self.load(ptx)

    def test_torch_launches_nvrtc_ptx(self):
        self.check_vector_add(self.load(self.build("nvrtc")))

    def test_driver_api_launch_uses_torch_pointers(self):
        """Public cuda.bindings path: no torch helper, raw pointers from __cuda_array_interface__."""
        from cuda.bindings import driver
        image = self.build("cubin")
        torch.zeros(1, device="cuda")  # make torch's primary context current on this thread

        def ok(result):
            self.assertEqual(result[0], driver.CUresult.CUDA_SUCCESS, result)
            return result[1] if len(result) > 1 else None

        module = ok(driver.cuModuleLoadData(image))
        function = ok(driver.cuModuleGetFunction(module, b"vector_add"))
        n = 513
        a, b, out = torch.randn(n, device="cuda"), torch.randn(n, device="cuda"), torch.zeros(n, device="cuda")
        pointers = [t.__cuda_array_interface__["data"][0] for t in (a, b, out)]
        stream = torch.cuda.current_stream().cuda_stream
        ok(driver.cuLaunchKernel(function, 3, 1, 1, 256, 1, 1, 0, stream,
                                 (tuple(pointers) + (n,), (ctypes.c_void_p,) * 3 + (ctypes.c_int,)), 0))
        torch.cuda.synchronize()
        torch.testing.assert_close(out, a + b)
        ok(driver.cuModuleUnload(module))

    def test_kernel_writes_into_a_noncontiguous_view_through_its_pointer(self):
        kernels = self.load(self.build("cubin"))
        base = torch.zeros(10, device="cuda")
        view = base[2:7]  # contiguous slice of a larger allocation: offset pointer, length 5
        kernels["fill_index"](grid=(1, 1, 1), block=(32, 1, 1), args=[view, 5])
        torch.cuda.synchronize()
        self.assertEqual(base.tolist(), [0, 0, 0, 1, 2, 3, 4, 0, 0, 0])

    def test_launch_on_a_side_stream_is_ordered_by_torch_synchronization(self):
        kernels = self.load(self.build("cubin"))
        out = torch.zeros(4096, device="cuda")
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            kernels["fill_index"](grid=(16, 1, 1), block=(256, 1, 1), args=[out, 4096], stream=side)
        torch.cuda.current_stream().wait_stream(side)
        self.assertEqual(out[-1].item(), 4095.0)


class InteropTests(Workdir):
    def test_cuda_array_interface_describes_the_tensor(self):
        t = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
        cai = t.__cuda_array_interface__
        self.assertEqual((cai["shape"], cai["typestr"]), ((3, 4), "<f4"))
        self.assertEqual(cai["data"][0], t.data_ptr())
        self.assertIn(cai["strides"], (None, (16, 4)))

    def test_cuda_array_interface_reports_view_offsets_and_strides(self):
        t = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
        column = t[:, 1]
        cai = column.__cuda_array_interface__
        self.assertEqual(cai["shape"], (3,))
        self.assertEqual(cai["strides"], (16,))
        self.assertEqual(cai["data"][0], t.data_ptr() + 4)

    def test_dlpack_round_trip_shares_device_memory(self):
        t = torch.arange(6, dtype=torch.float32, device="cuda")
        again = torch.from_dlpack(t)
        self.assertEqual(again.data_ptr(), t.data_ptr())
        again[0] = 99.0
        self.assertEqual(t[0].item(), 99.0)

    def test_dlpack_preserves_dtype_and_zero_copy_for_half(self):
        t = torch.ones(8, dtype=torch.float16, device="cuda")
        again = torch.from_dlpack(t.__dlpack__())
        self.assertEqual((again.dtype, again.data_ptr()), (torch.float16, t.data_ptr()))

    def test_numpy_shares_cpu_memory_and_round_trips_through_the_gpu(self):
        host = np.arange(8, dtype=np.float32)
        shared = torch.from_numpy(host)
        host[0] = 7.0
        self.assertEqual(shared[0].item(), 7.0)
        kernels = self.load(self.build("cubin"))
        on_gpu = shared.cuda()
        out = torch.empty_like(on_gpu)
        kernels["vector_add"](grid=(1, 1, 1), block=(32, 1, 1), args=[on_gpu, on_gpu, out, 8])
        np.testing.assert_allclose(out.cpu().numpy(), host * 2)

    def test_numpy_refuses_implicit_gpu_conversion(self):
        with self.assertRaises(TypeError):
            np.asarray(torch.zeros(2, device="cuda"))

    def test_pinned_host_memory_is_usable_by_a_kernel_argument(self):
        pinned = torch.arange(4, dtype=torch.float32).pin_memory()
        self.assertTrue(pinned.is_pinned())
        kernels = self.load(self.build("cubin"))
        out = torch.empty(4, device="cuda")
        device_copy = pinned.to("cuda", non_blocking=True)
        kernels["vector_add"](grid=(1, 1, 1), block=(32, 1, 1), args=[device_copy, device_copy, out, 4])
        self.assertEqual(out.tolist(), [0.0, 2.0, 4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
