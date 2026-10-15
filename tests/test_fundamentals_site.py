"""The teaching page's math must agree with the dojo's hand-checkable answers."""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless

PAGE = Path(__file__).resolve().parents[1] / "site" / "fundamentals" / "index.html"
HARNESS = r"""
const FM = require(process.argv[2]);
const close = (a, b, e = 1e-9) => Math.abs(a - b) < e;
const out = {};
// 0125: uniform attention, causal rows, empty rows
const S = [[0, 0], [0, 0]];
out.mask_free = FM.attentionWeights(S, [true, true], false, true).weights;
out.mask_causal = FM.attentionWeights(S, [true, true], true, true).weights;
out.mask_empty_safe = FM.attentionWeights(S, [false, true], false, true).weights;
out.mask_empty_naive = FM.attentionWeights(S, [false, true], false, false).weights[0].map(Number.isNaN);
// 0126: identity at temperature 1, identical embeddings
out.contrastive_identity = FM.contrastive([0, Math.PI / 2], [0, Math.PI / 2], 1, [true, true]).loss;
out.contrastive_same = FM.contrastive([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 0.5, [true, true, true, true, true]).loss;
out.contrastive_one = FM.contrastive([1, 2], [1, 2], 1, [false, true]).loss;
out.contrastive_wrong = FM.contrastive([0, Math.PI], [Math.PI, 0], 0.05, [true, true]).loss;
// 0127: hand-checkable convolution and shape formula
const X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
out.conv_diag = FM.conv2d(X, [[1, 0], [0, -1]], 1, 0, false).out;
out.conv_asym_noflip = FM.conv2d([[0, 1], [2, 3]], [[1, 2], [3, 4]], 1, 0, false).out;
out.conv_asym_flip = FM.conv2d([[0, 1], [2, 3]], [[1, 2], [3, 4]], 1, 0, true).out;
out.conv_size = FM.outSize(8, 3, 2, 1);
out.conv_too_big = FM.conv2d(X, [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], 1, 0, false).error === true;
// 0128
out.iou_seventh = FM.iou([0, 0, 2, 2], [1, 1, 3, 3]).iou;
out.iou_far = FM.iou([0, 0, 1, 1], [2, 2, 3, 3]).iou;
out.iou_zero_area = FM.iou([1, 1, 1, 1], [1, 1, 1, 1]).iou;
// 0129
const d = 1;
out.huber_loss = (FM.huber(-2, d) + FM.huber(0.5, d)) / 2;
out.huber_grad = [FM.huberGrad(-2, d) / 2, FM.huberGrad(0.5, d) / 2];
out.huber_select_nan = Number.isNaN(FM.maskedHuber([1.5, -0.4, NaN], [true, true, false], 1, 'select'));
out.huber_multiply_nan = Number.isNaN(FM.maskedHuber([1.5, -0.4, NaN], [true, true, false], 1, 'multiply'));
// 0130: worked voxel example
const pts = [0.1, 0.9, 1.0, 2.0].map((x, i) => ({ x, y: 0, f: [2, 4, 8, 99][i] }));
const v = FM.voxel(pts, [0, 0], 1, [2, 1]);
out.voxel = v.occupied;
out.voxel_negative = FM.voxel([{ x: -0.1, y: 0, f: 1 }], [0, 0], 1, [2, 1]).occupied.length;
// 0131
out.ego_translation = FM.egoTransform([2, 0], { x: 10, y: 0, th: 0 }, { x: 7, y: 0, th: 0 }).current;
out.ego_rotation = FM.egoTransform([1, 0], { x: 0, y: 0, th: Math.PI / 2 }, { x: 0, y: 0, th: 0 }).current;
console.log(JSON.stringify(out));
"""


def extract_math(html: str) -> str:
    return re.search(r'<script id="math">(.*?)</script>', html, re.S).group(1)


@skipUnless(shutil.which("node"), "node is required to run the page math")
class FundamentalsSiteTests(TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            math_js = Path(tmp) / "math.js"
            harness = Path(tmp) / "harness.js"
            math_js.write_text(extract_math(PAGE.read_text()))
            harness.write_text(HARNESS)
            result = subprocess.run(["node", str(harness), str(math_js)], capture_output=True, text=True, check=True)
        cls.r = json.loads(result.stdout)

    def test_masks(self):
        self.assertEqual(self.r["mask_free"], [[0.5, 0.5], [0.5, 0.5]])
        self.assertEqual(self.r["mask_causal"], [[1, 0], [0.5, 0.5]])
        self.assertEqual(self.r["mask_empty_safe"], [[0, 0], [0, 1]])
        self.assertEqual(self.r["mask_empty_naive"], [True, True])

    def test_contrastive(self):
        self.assertAlmostEqual(self.r["contrastive_identity"], 0.3132616875, places=9)
        self.assertAlmostEqual(self.r["contrastive_same"], 1.6094379124341003, places=9)
        self.assertEqual(self.r["contrastive_one"], 0)
        self.assertGreater(self.r["contrastive_wrong"], 0.69314718056)  # log(2) is not a ceiling

    def test_conv(self):
        self.assertEqual(self.r["conv_diag"], [[-4, -4], [-4, -4]])
        self.assertEqual(self.r["conv_asym_noflip"], [[20]])
        self.assertEqual(self.r["conv_asym_flip"], [[10]])
        self.assertEqual(self.r["conv_size"], 4)
        self.assertTrue(self.r["conv_too_big"])

    def test_iou(self):
        self.assertAlmostEqual(self.r["iou_seventh"], 1 / 7)
        self.assertEqual(self.r["iou_far"], 0)
        self.assertEqual(self.r["iou_zero_area"], 0)

    def test_huber(self):
        self.assertAlmostEqual(self.r["huber_loss"], 0.8125)
        self.assertEqual(self.r["huber_grad"], [-0.5, 0.25])
        self.assertFalse(self.r["huber_select_nan"])
        self.assertTrue(self.r["huber_multiply_nan"])

    def test_voxel(self):
        self.assertEqual([(o["c"], o["n"], o["mean"]) for o in self.r["voxel"]], [([0, 0], 2, 3), ([1, 0], 1, 8)])
        self.assertEqual(self.r["voxel_negative"], 0)

    def test_ego(self):
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(self.r["ego_translation"], [5, 0])))
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(self.r["ego_rotation"], [0, 1])))

    def test_page_has_no_external_runtime_dependencies(self):
        html = PAGE.read_text()
        self.assertNotRegex(html, r'<script[^>]+src=')
        self.assertNotRegex(html, r'<link[^>]+href=["\']https?:')
