"""GPU softmax references (Triton and CUDA) and the CPU production reference."""
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import TestCase, skipUnless

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "softmax_gpu"))

HAS_CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# (dtype, atol/rtol): half dtypes differ from the float32 oracle by rounding only.
TOLERANCES = ((torch.float32, 1e-6), (torch.float16, 2e-3), (torch.bfloat16, 2e-2))
CASES = (((5, 7), -1), ((3, 4, 9), 1), ((2, 3), 0), ((4, 10000), -1), ((3, 70000), -1))


class Contract:
    """Shared checks; subclasses set `softmax`."""

    def softmax(self, x, dim=-1):
        raise NotImplementedError

    def test_matches_framework_forward_and_backward(self):
        for dtype, tol in TOLERANCES:
            for shape, dim in CASES:
                with self.subTest(dtype=dtype, shape=shape, dim=dim):
                    x = (torch.randn(shape, device="cuda") * 30).to(dtype).requires_grad_()
                    ref = x.detach().clone().requires_grad_()
                    g = torch.randn_like(x)
                    y, y_ref = self.softmax(x, dim), torch.softmax(ref.float(), dim).to(dtype)
                    (y * g).sum().backward()
                    (y_ref * g).sum().backward()
                    self.assertEqual(y.dtype, dtype)
                    torch.testing.assert_close(y, y_ref, atol=tol, rtol=tol)
                    torch.testing.assert_close(x.grad.float(), ref.grad.float(), atol=10 * tol, rtol=10 * tol)

    def test_extreme_logits_and_long_half_row(self):
        y = self.softmax(torch.tensor([1000., 1001., -1000.], device="cuda"))
        torch.testing.assert_close(y, torch.tensor([0.2689414, 0.7310586, 0.], device="cuda"))
        half = self.softmax(torch.zeros(70000, dtype=torch.float16, device="cuda"))
        torch.testing.assert_close(half.float().sum(), torch.tensor(1., device="cuda"), atol=2e-2, rtol=0)

    def test_noncontiguous_input_and_input_unchanged(self):
        x = torch.randn(8, 6, device="cuda").T
        before = x.clone()
        torch.testing.assert_close(self.softmax(x, 1), torch.softmax(x, 1))
        self.assertTrue(torch.equal(x, before))

    def test_scalar_and_cpu_and_dtype_rejection(self):
        self.assertEqual(self.softmax(torch.tensor(3., device="cuda")).item(), 1.)
        with self.assertRaises(ValueError):
            self.softmax(torch.zeros(3))
        with self.assertRaises(TypeError):
            self.softmax(torch.zeros(3, dtype=torch.float64, device="cuda"))


@skipUnless(HAS_CUDA and HAS_TRITON, "needs CUDA and triton")
class TritonSoftmaxTests(Contract, TestCase):
    def softmax(self, x, dim=-1):
        from softmax_triton import softmax_triton
        return softmax_triton(x, dim)


@skipUnless(HAS_CUDA, "needs CUDA")
class CudaSoftmaxTests(Contract, TestCase):
    def softmax(self, x, dim=-1):
        from softmax_cuda import softmax_cuda
        return softmax_cuda(x, dim)


class ProductionReferenceTests(TestCase):
    """The CPU softmax_production in the eval script, beyond what evaluate() checks."""

    @classmethod
    def setUpClass(cls):
        cls.softmax = staticmethod(runpy.run_path(str(ROOT / "dojo" / "0002_softmax_eval.py"), run_name="not_main")["softmax_production"])

    def test_gradcheck_and_half_backward(self):
        for dim in (0, 1, -1):
            x = torch.randn(3, 5, 7, dtype=torch.float64, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda t: self.softmax(t, dim), (x,)))
        x = (torch.randn(4, 9) * 20).to(torch.bfloat16).requires_grad_()
        ref = x.detach().clone().requires_grad_()
        g = torch.randn_like(x)
        (self.softmax(x) * g).sum().backward()
        (torch.softmax(ref.float(), -1).to(torch.bfloat16) * g).sum().backward()
        torch.testing.assert_close(x.grad, ref.grad, atol=5e-2, rtol=5e-2)

    def test_rejects_integers_and_handles_scalars(self):
        with self.assertRaises(TypeError):
            self.softmax(torch.arange(3))
        self.assertEqual(self.softmax(torch.tensor(2.)).item(), 1.)


class BackwardExerciseTests(TestCase):
    """The lesson's softmax_backward formula agrees with autograd."""

    def test_reference_formula_matches_autograd(self):
        x = (torch.randn(4, 6, dtype=torch.float64) * 500).T.requires_grad_()
        g = torch.randn_like(x)
        for dim in (0, 1, -1):
            ref = torch.autograd.grad(torch.softmax(x, dim), x, g)[0]
            y = torch.softmax(x.detach(), dim)
            torch.testing.assert_close(y * (g - (g * y).sum(dim, keepdim=True)), ref)


class LessonGpuExerciseTests(TestCase):
    """The lesson's GPU self-check accepts both worked references and is import-safe."""

    @classmethod
    def setUpClass(cls):
        source = (ROOT / "dojo" / "0002_softmax.py").read_text()
        function = "def check_softmax_gpu" + source.split("def check_softmax_gpu", 1)[1].split("\ncheck_softmax_gpu()")[0]
        namespace = {"torch": torch}
        exec(compile(function, "0002_softmax.py", "exec"), namespace)
        cls.check = staticmethod(namespace["check_softmax_gpu"])

    @skipUnless(HAS_CUDA, "needs CUDA")
    def test_check_accepts_cuda_reference_and_rejects_wrong_answer(self):
        from softmax_cuda import softmax_cuda
        self.assertIn("matches", self.check(softmax_cuda))
        with self.assertRaises(AssertionError):
            self.check(lambda x, dim=-1: torch.softmax(x, dim) + 1e-2)

    @skipUnless(HAS_CUDA and HAS_TRITON, "needs CUDA and triton")
    def test_check_accepts_triton_reference(self):
        from softmax_triton import softmax_triton
        self.assertIn("matches", self.check(softmax_triton))


@skipUnless(HAS_CUDA, "needs CUDA")
class CudaWorkoutScriptTests(TestCase):
    """experiments/softmax_gpu/workout.py runs end to end with each compiler."""

    def run_workout(self, compiler):
        script = ROOT / "experiments" / "softmax_gpu" / "workout.py"
        result = subprocess.run([sys.executable, str(script), "--compiler", compiler, "--rows", "64", "--cols", "1000"],
                                capture_output=True, text=True, cwd=ROOT, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for step in ("[1/5]", "[2/5]", "[3/5]", "[4/5]", "[5/5]"):
            self.assertIn(step, result.stdout)

    def test_nvrtc(self):
        self.run_workout("nvrtc")

    @skipUnless(shutil.which("nvcc"), "needs nvcc (rebuild the insula rootfs)")
    def test_nvcc(self):
        self.run_workout("nvcc")
