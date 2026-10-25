"""Tests for src/external_eval (torch-free parts). Run from the repo root:

    PYTHONPATH=src python -m unittest tests.test_external_eval -v        (works without pytest)
    PYTHONPATH=src python -m pytest tests/test_external_eval.py -q       (if pytest is installed)

Tests on the real datasets are skipped if the data is not at ./external (set EXTERNAL_DIR to change).
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from external_eval import IGNORE  # noqa: E402
from external_eval.datasets import (CWDB_PALETTE, cwdb_items, dfutissue_items, load_cwdb_mask,  # noqa: E402
                                    load_dfutissue_mask, load_image_rgb, snap_to_palette)
from external_eval.evaluate import evaluate_native, zero_shot_variants  # noqa: E402
from external_eval.mapping import (apply_policy, cwdb_common4, dfutissue_common3,  # noqa: E402
                                   metrics_from_confusion, native_confusion)

EXT = Path(os.environ.get("EXTERNAL_DIR", "external"))
DFU = EXT / "DFUTissueSegNet" / "DFUTissue"
CWDB = EXT / "recogna-datasets" / "ComplexWoundDB"


def brute_conf(gt, pred, n_gt, n_pred):
    c = np.zeros((n_gt, n_pred), dtype=np.int64)
    for g, p in zip(gt.ravel(), pred.ravel()):
        if g != IGNORE and g < n_gt:
            c[g, p] += 1
    return c


class TestMetrics(unittest.TestCase):
    def test_native_confusion_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        gt = rng.integers(0, 5, (20, 30)).astype(np.uint8)
        gt[rng.random(gt.shape) < 0.1] = 255
        pred = rng.integers(0, 6, gt.shape).astype(np.uint8)
        np.testing.assert_array_equal(native_confusion(gt, pred, 5, 6), brute_conf(gt, pred, 5, 6))

    def test_perfect_prediction_is_100(self):
        conf = np.diag([50, 20, 10, 5])
        m = metrics_from_confusion(conf, ["a", "b", "c", "d"])
        self.assertAlmostEqual(m["mean_iou"], 100.0)
        self.assertAlmostEqual(m["mean_dice"], 100.0)

    def test_hand_example(self):
        # class0: TP 8 FN 2 FP 1 -> IoU 8/11, Dice 16/19 ; class1: TP 4 FN 1 FP 2 -> IoU 4/7, Dice 8/11
        conf = np.array([[8, 2], [1, 4]])
        m = metrics_from_confusion(conf, ["bg", "x"])
        self.assertAlmostEqual(m["classes"]["bg"]["iou"], 8 / 11 * 100)
        self.assertAlmostEqual(m["classes"]["x"]["dice"], 8 / 11 * 100)

    def test_absent_class_counts_zero_in_mean(self):
        conf = np.diag([10, 0, 0])
        m = metrics_from_confusion(conf, ["a", "b", "c"])
        self.assertAlmostEqual(m["mean_iou"], 100.0 / 3)  # TiSage-style: absent classes count as 0

    def test_unmapped_prediction_is_error_without_false_positive(self):
        # 2 classes + unmapped column. gt class1 has 10 px: 6 right, 4 predicted as unmapped.
        conf = np.array([[20, 0, 0], [0, 6, 4]])
        m = metrics_from_confusion(conf, ["bg", "x"])
        self.assertAlmostEqual(m["classes"]["x"]["iou"], 6 / 10 * 100)   # FN includes unmapped, FP = 0
        self.assertAlmostEqual(m["classes"]["bg"]["iou"], 100.0)         # no false positives for bg
        self.assertEqual(m["unmapped_pixels"], 4)


class TestMapping(unittest.TestCase):
    def test_dfutissue_policies(self):
        # native N[gt (bg, fibrin, gran, callus), our pred (bg, epi, slough, gran, necr, other)]
        n = np.zeros((4, 6), dtype=np.int64)
        n[0, 0] = 100       # bg -> bg
        n[1, 2] = 10        # fibrin -> slough  (correct)
        n[2, 3] = 20        # granulation -> granulation (correct)
        n[2, 1] = 5         # granulation -> epithelial (unmapped)
        n[3, 3] = 7         # callus -> granulation
        sp = dfutissue_common3()
        strict_ign = apply_policy(n, sp, "callus_ignore", "strict")
        self.assertEqual(strict_ign.sum(), 100 + 10 + 20 + 5)            # callus pixels dropped
        self.assertEqual(strict_ign[2, 3], 5)                            # unmapped column
        self.assertEqual(strict_ign[2, 2], 20)
        len_bg = apply_policy(n, sp, "callus_background", "lenient")
        self.assertEqual(len_bg[0, 2], 7)                                # callus (as bg) predicted granulation
        self.assertEqual(len_bg[2, 0], 5)                                # epithelial folded into background

    def test_cwdb_policy(self):
        n = np.zeros((5, 6), dtype=np.int64)   # native gt: 0 bg, 1 granulation, 2 fibrinoid, 3 necrosis, 4 hematoma
        n[1, 3] = 9                            # granulation -> granulation
        n[2, 2] = 4                            # fibrinoid -> slough
        n[3, 4] = 3                            # necrosis -> necrotic
        n[4, 3] = 50                           # hematoma: ignored
        conf = apply_policy(n, cwdb_common4(), "default", "strict")
        self.assertEqual(conf.sum(), 16)
        self.assertEqual(conf[2, 2], 9)        # common ids: 0 bg, 1 slough, 2 granulation, 3 necrotic
        self.assertEqual(conf[1, 1], 4)
        self.assertEqual(conf[3, 3], 3)

    def test_bad_policy_name(self):
        with self.assertRaises(ValueError):
            apply_policy(np.zeros((4, 6), dtype=np.int64), dfutissue_common3(), "nope", "strict")


class TestSnap(unittest.TestCase):
    def test_exact_and_blended_colours(self):
        img = np.zeros((2, 3, 3), dtype=np.uint8)
        img[0, 0] = (255, 255, 255)            # white -> 0
        img[0, 1] = (237, 28, 36)              # red -> 1
        img[0, 2] = (250, 40, 50)              # blended red -> 1
        img[1, 0] = (255, 242, 0)              # yellow -> 2
        img[1, 1] = (3, 2, 1)                  # near black -> 3
        img[1, 2] = (160, 40, 140)             # near purple -> 4
        out = snap_to_palette(img, CWDB_PALETTE)
        np.testing.assert_array_equal(out, [[0, 1, 1], [2, 3, 4]])

    def test_far_colour_ignored_with_max_dist(self):
        img = np.full((1, 1, 3), (120, 200, 255), dtype=np.uint8)  # light blue: far from every palette colour
        self.assertEqual(int(snap_to_palette(img, CWDB_PALETTE, max_dist=40)[0, 0]), IGNORE)


class TestEvaluateLoop(unittest.TestCase):
    def test_evaluate_native_with_fake_predictor(self):
        from PIL import Image
        import tempfile
        from external_eval.datasets import Item

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gt = np.zeros((10, 12), dtype=np.uint8)
            gt[2:6, 3:9] = 2
            Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)).save(d / "a.png")
            Image.fromarray(gt).save(d / "m.png")
            it = Item("a", d / "a.png", d / "m.png")
            native, rows = evaluate_native([it], lambda i: load_dfutissue_mask(i.mask_path),
                                           lambda img: gt.copy(), n_gt=4, n_pred=4)
            self.assertEqual(int(native.sum()), 120)
            self.assertEqual(int(native[2, 2]), int((gt == 2).sum()))
            with self.assertRaises(ValueError):                      # size mismatch must not pass silently
                evaluate_native([it], lambda i: load_dfutissue_mask(i.mask_path),
                                lambda img: np.zeros((5, 5), np.uint8), 4, 4)


@unittest.skipUnless(DFU.is_dir(), f"DFUTissue not found at {DFU}")
class TestDFUTissueReal(unittest.TestCase):
    def test_split_sizes_and_disjoint(self):
        tr, va, te = (dfutissue_items(DFU, s) for s in ("train", "val", "test"))
        self.assertEqual((len(tr), len(va), len(te)), (78, 16, 16))
        self.assertEqual(len(dfutissue_items(DFU, "trainval")), 94)
        self.assertEqual(len(dfutissue_items(DFU, "all")), 110)
        ids = [i.id for i in tr] + [i.id for i in va] + [i.id for i in te]
        self.assertEqual(len(set(ids)), 110)

    def test_mask_values_and_sizes(self):
        for it in dfutissue_items(DFU, "all"):
            m = load_dfutissue_mask(it.mask_path)
            self.assertTrue(set(np.unique(m)).issubset({0, 1, 2, 3}), it.id)
            self.assertEqual(load_image_rgb(it.image_path).size, (m.shape[1], m.shape[0]), it.id)

    def test_oracle_prediction_scores_100(self):
        # our-class prediction that perfectly mirrors the ground truth: bg->0, fibrin->Slough(2), gran->Granulation(3),
        # callus->Background(0). Every policy that keeps the mapped classes must give 100% on them.
        lut = np.array([0, 2, 3, 0], dtype=np.uint8)
        items = dfutissue_items(DFU, "test")
        native, _ = evaluate_native(items, lambda i: load_dfutissue_mask(i.mask_path),
                                    lambda img, it=iter(items): lut[load_dfutissue_mask(next(it).mask_path)], 4, 6)
        v = zero_shot_variants(native, dfutissue_common3())
        for k in ("callus_ignore|strict", "callus_ignore|lenient"):
            m = v[k]["metrics"]
            for c in ("background", "fibrin/slough", "granulation"):
                self.assertAlmostEqual(m["classes"][c]["iou"], 100.0, msg=k)
        # callus treated as background but predicted as background too -> also perfect
        self.assertAlmostEqual(v["callus_background|strict"]["metrics"]["mean_iou"], 100.0)


@unittest.skipUnless(CWDB.is_dir(), f"ComplexWoundDB not found at {CWDB}")
class TestCWDBReal(unittest.TestCase):
    def test_27_items_and_clean_expert4(self):
        items = cwdb_items(CWDB, 4)
        self.assertEqual(len(items), 27)
        seen = set()
        for it in items:
            m = load_cwdb_mask(it.mask_path)
            seen |= set(np.unique(m).tolist())
            self.assertEqual(load_image_rgb(it.image_path).size, (m.shape[1], m.shape[0]), it.id)
            self.assertNotIn(IGNORE, np.unique(m), f"{it.id}: pixels farther than 40 from every palette colour")
        self.assertTrue(seen.issubset({0, 1, 2, 3, 4}))

    def test_yellow2_ignore_changes_only_fibrinoid(self):
        it = [i for i in cwdb_items(CWDB, 4) if i.id == "4"][0]   # image 4 contains only the second yellow
        a, b = load_cwdb_mask(it.mask_path, "fibrinoid"), load_cwdb_mask(it.mask_path, "ignore")
        self.assertGreater(int((a == 2).sum()), 0)
        self.assertEqual(int((b == 2).sum()), 0)
        self.assertEqual(int((b == IGNORE).sum()), int((a == 2).sum()))

    def test_oracle_prediction_scores_100(self):
        lut = np.zeros(256, dtype=np.uint8)
        lut[1], lut[2], lut[3], lut[4] = 3, 2, 4, 0   # native gran->Granulation(3), fibrinoid->Slough(2), necrosis->Necrotic(4)
        items = cwdb_items(CWDB, 4)
        native, _ = evaluate_native(items, lambda i: load_cwdb_mask(i.mask_path),
                                    lambda img, it=iter(items): lut[load_cwdb_mask(next(it).mask_path)], 5, 6)
        m = zero_shot_variants(native, cwdb_common4())["default|strict"]["metrics"]
        self.assertAlmostEqual(m["mean_iou"], 100.0)


if __name__ == "__main__":
    unittest.main()
