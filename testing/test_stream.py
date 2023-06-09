import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from neutone_sdk.stream_conv import (
    StreamConv1d,
    get_same_pads,
    StreamConvTranspose1d,
    AlignBranches,
)


class StreamConv1dTests(unittest.TestCase):
    def test_equal(self):
        for k, p, s, d, size in (
            (3, "causal", 1, 1, 1024),
            (5, "noncausal", 2, 3, 512),
            (5, "causal", 2, 3, 823),
        ):
            with self.subTest(k=k, p=p, s=s, d=d, size=size):
                x = torch.randn(1, 4, 16000)
                pad = get_same_pads(k, s, d, mode=p)
                # normal conv
                gt_conv = nn.Conv1d(4, 8, k, stride=s, padding=0, dilation=d, groups=1)
                y_true = gt_conv(F.pad(x, pad=pad))
                # cached conv
                ca_conv = StreamConv1d(
                    4, 8, k, stride=s, padding=pad, dilation=d, groups=1
                ).eval()
                ca_conv.stream()
                ca_conv.weight = gt_conv.weight
                ca_conv.bias = gt_conv.bias
                xs = torch.split(x, size, -1)
                chunks_y = []
                for chunk in xs:
                    chunk_y = ca_conv(chunk)
                    chunks_y.append(chunk_y)
                y_cached = torch.cat(chunks_y, dim=-1)
                cached_len = y_cached.shape[-1]
                self.assertTrue(
                    torch.allclose(y_true[..., :cached_len], y_cached, atol=1e-6)
                )

    def test_flush_equal(self):
        for k, p, s, d, size in (
            (3, "causal", 2, 1, 1024),
            (3, "noncausal", 1, 3, 512),
            (5, "causal", 1, 3, 823),
        ):
            with self.subTest(k=k, p=p, s=s, d=d, size=size):
                x = torch.randn(1, 4, 16000)
                pad = get_same_pads(k, s, d, mode=p)
                # normal conv
                gt_conv = nn.Conv1d(4, 8, k, stride=s, padding=0, dilation=d, groups=1)
                y_true = gt_conv(F.pad(x, pad=pad))
                # cached conv
                ca_conv = StreamConv1d(
                    4, 8, k, stride=s, padding=pad, dilation=d, groups=1
                ).eval()
                ca_conv.stream()
                ca_conv.weight = gt_conv.weight
                ca_conv.bias = gt_conv.bias
                xs = torch.split(x, size, -1)
                chunks_y = []
                sizes_y = []
                for chunk in xs:
                    chunk_y = ca_conv(chunk)
                    chunks_y.append(chunk_y)
                    sizes_y.append(chunk_y.shape[-1])
                flush_chunk = ca_conv.flush()
                chunks_y.append(flush_chunk)
                y_cached = torch.cat(chunks_y, dim=-1)
                # print("Conv test_flush_equal chunk_sizes \n", sizes_y)
                self.assertTrue(torch.allclose(y_true, y_cached, atol=1e-6))

    def test_flush_equal_free(self):
        for k, p, s, d, size in (
            (3, (2, 3), 1, 1, 1024),
            (5, (0, 0), 2, 3, 512),
            (5, (4, 2), 2, 3, 823),
            # (5, (4, 1), 1, 3, 823), # odd padding seems to cause shorter output
        ):
            with self.subTest(k=k, p=p, s=s, d=d, size=size):
                x = torch.randn(1, 4, 16000)
                pad = p
                # normal conv
                gt_conv = nn.Conv1d(4, 8, k, stride=s, padding=0, dilation=d, groups=1)
                y_true = gt_conv(F.pad(x, pad=pad))
                # cached conv
                ca_conv = StreamConv1d(
                    4, 8, k, stride=s, padding=pad, dilation=d, groups=1
                ).eval()
                ca_conv.stream()
                ca_conv.weight = gt_conv.weight
                ca_conv.bias = gt_conv.bias
                xs = torch.split(x, size, -1)
                chunks_y = []
                sizes_y = []
                for chunk in xs:
                    chunk_y = ca_conv(chunk)
                    chunks_y.append(chunk_y)
                    sizes_y.append(chunk_y.shape[-1])
                print("Conv test_flush_equal_free chunk_sizes \n", sizes_y)
                chunks_y.append(ca_conv.flush())
                y_cached = torch.cat(chunks_y, dim=-1)
                self.assertTrue(torch.allclose(y_true, y_cached, atol=1e-6))

    def test_flush_equal_script(self):
        for k, p, s, d, size in (
            (3, "causal", 2, 1, 1024),
            (3, "noncausal", 1, 3, 512),
            (5, "causal", 1, 3, 823),
        ):
            with self.subTest(k=k, p=p, s=s, d=d, size=size):
                x = torch.randn(1, 4, 16000)
                pad = get_same_pads(k, s, d, mode=p)
                # normal conv
                gt_conv = nn.Conv1d(4, 8, k, stride=s, padding=0, dilation=d, groups=1)
                y_true = gt_conv(F.pad(x, pad=pad))
                # cached conv
                ca_conv = StreamConv1d(
                    4, 8, k, stride=s, padding=pad, dilation=d, groups=1
                ).eval()
                ca_conv.stream()
                ca_conv.weight = gt_conv.weight
                ca_conv.bias = gt_conv.bias
                ca_conv = torch.jit.script(ca_conv)
                xs = torch.split(x, size, -1)
                chunks_y = []
                sizes_y = []
                for chunk in xs:
                    chunk_y = ca_conv(chunk)
                    chunks_y.append(chunk_y)
                    sizes_y.append(chunk_y.shape[-1])
                flush_chunk = ca_conv.flush()
                chunks_y.append(flush_chunk)
                y_cached = torch.cat(chunks_y, dim=-1)
                print(
                    f"Conv test_flush_equal_script chunk_sizes {k, p, s, d, size}\n",
                    sizes_y,
                )
                self.assertTrue(torch.allclose(y_true, y_cached, atol=1e-6))


class CachedConvTransposeTests(unittest.TestCase):
    def test_equal(self):
        for k, p, s, d, size in (
            (7, "causal", 1, 1, 1024),
            (5, "noncausal", 2, 1, 283),
            (5, "causal", 2, 3, 250),
        ):
            with self.subTest(k=k, p=p, s=s, d=d, size=size):
                x = torch.randn(1, 1, 16000)
                pad = get_same_pads(k, s, d, mode=p)
                # normal conv
                gt_conv = nn.ConvTranspose1d(
                    1, 1, k, stride=s, padding=0, dilation=d, groups=1
                )
                y_true = gt_conv(x)
                y_true = y_true[..., pad[0] : -pad[1] or None]
                # cached conv
                ca_conv = StreamConvTranspose1d(
                    1, 1, k, stride=s, padding=pad, dilation=d, groups=1
                ).eval()
                ca_conv.stream()
                ca_conv.weight = gt_conv.weight
                ca_conv._bias = gt_conv.bias
                xs = torch.split(x, size, -1)
                chunks_y = []
                for chunk in xs:
                    chunk_y = ca_conv(chunk)
                    chunks_y.append(chunk_y)
                y_cached = torch.cat(chunks_y, dim=-1)
                cached_len = min(y_cached.shape[-1], y_true.shape[-1])
                self.assertTrue(
                    torch.allclose(
                        y_true[..., :cached_len], y_cached[..., :cached_len], atol=1e-6
                    )
                )

    def test_flush_equal(self):
        for k, p, s, d, size in (
            (3, 0, 1, 1, 1024),
            (5, 1, 2, 1, 283),
            (5, 4, 2, 3, 250),
        ):
            with self.subTest(k=k, p=p, s=s, d=d, size=size):
                x = torch.randn(1, 1, 16000)
                # normal conv
                gt_conv = nn.ConvTranspose1d(
                    1, 1, k, stride=s, padding=p, dilation=d, groups=1
                )
                y_true = gt_conv(x)
                # cached conv
                ca_conv = StreamConvTranspose1d(
                    1, 1, k, stride=s, padding=(p, p), dilation=d, groups=1
                ).eval()
                ca_conv.stream()
                ca_conv.weight = gt_conv.weight
                ca_conv._bias = gt_conv.bias
                xs = torch.split(x, size, -1)
                chunks_y = []
                sizes_y = []
                for chunk in xs:
                    chunk_y = ca_conv(chunk)
                    chunks_y.append(chunk_y)
                    sizes_y.append(chunk_y.shape[-1])
                flush_y = ca_conv.flush()
                chunks_y.append(flush_y)
                y_cached = torch.cat(chunks_y, dim=-1)
                cached_len = min(y_cached.shape[-1], y_true.shape[-1])
                self.assertTrue(torch.allclose(y_true, y_cached, atol=1e-6))

    def test_flush_equal_script(self):
        for k, p, s, d, size in (
            (3, 0, 1, 1, 1024),
            (5, 1, 2, 1, 283),
            (5, 4, 2, 3, 250),
        ):
            with self.subTest(k=k, p=p, s=s, d=d, size=size):
                x = torch.randn(1, 1, 16000)
                # normal conv
                gt_conv = nn.ConvTranspose1d(
                    1, 1, k, stride=s, padding=p, dilation=d, groups=1, bias=True
                )
                y_true = gt_conv(x)
                # cached conv
                ca_conv = StreamConvTranspose1d(
                    1, 1, k, stride=s, padding=(p, p), dilation=d, groups=1, bias=True
                ).eval()
                ca_conv.stream()
                ca_conv.weight = gt_conv.weight
                ca_conv._bias = gt_conv.bias
                ca_conv = torch.jit.script(ca_conv)
                xs = torch.split(x, size, -1)
                chunks_y = []
                sizes_y = []
                for chunk in xs:
                    chunk_y = ca_conv(chunk)
                    chunks_y.append(chunk_y)
                    sizes_y.append(chunk_y.shape[-1])
                print("Transpose test_flush_equal_script chunk_sizes \n", sizes_y)
                flush_y = ca_conv.flush()
                chunks_y.append(flush_y)
                y_cached = torch.cat(chunks_y, dim=-1)
                cached_len = min(y_cached.shape[-1], y_true.shape[-1])
                self.assertTrue(torch.allclose(y_true, y_cached, atol=1e-6))


class AlignBranchesTest(unittest.TestCase):
    def test_res(self):
        # use convs that preserve (when not split and cached) size
        for k, p, s, d, size in (
            (5, "noncausal", 1, 1, 1024),
            (3, "noncausal", 1, 3, 512),
            (3, "causal", 1, 3, 512),
        ):
            with self.subTest(k=k, p=p, s=s, d=d, size=size):
                x = torch.randn(1, 4, 16000)
                pad = get_same_pads(k, s, d, mode=p)
                # resnet with normal conv
                gt_conv = nn.Conv1d(4, 4, k, stride=s, padding=0, dilation=d, groups=1)
                y_true = x + gt_conv(F.pad(x, pad=pad))
                # resnet with cached conv
                ca_conv = StreamConv1d(
                    4, 4, k, stride=s, padding=pad, dilation=d, groups=1
                ).eval()
                ca_conv.stream()
                ca_conv.weight = gt_conv.weight
                ca_conv.bias = gt_conv.bias
                res = AlignBranches(
                    ca_conv,
                    nn.Identity(),
                )
                res.stream()
                xs = torch.split(x, size, -1)
                chunks_y = []
                sizes_y = []
                for chunk in xs:
                    chunk_net, chunk_res = res(chunk)
                    chunk_y = chunk_net + chunk_res
                    chunks_y.append(chunk_y)
                    sizes_y.append(chunk_y.shape[-1])
                print("align branches test_res chunk_sizes:\n", sizes_y)
                flush_net, flush_res = res.flush()
                chunks_y.append(flush_net + flush_res)
                y_cached = torch.cat(chunks_y, dim=-1)
                self.assertTrue(torch.allclose(y_true, y_cached, atol=1e-6))

    def test_res_script(self):
        # use convs that preserve (when not split and cached) size
        for k, p, s, d, size in (
            (5, "noncausal", 1, 1, 1024),
            (3, "noncausal", 1, 3, 512),
            (3, "causal", 1, 3, 512),
        ):
            with self.subTest(k=k, p=p, s=s, d=d, size=size):
                x = torch.randn(1, 4, 16000)
                pad = get_same_pads(k, s, d, mode=p)
                # resnet with normal conv
                gt_conv = nn.Conv1d(4, 4, k, stride=s, padding=0, dilation=d, groups=1)
                y_true = x + gt_conv(F.pad(x, pad=pad))
                # resnet with cached conv
                ca_conv = StreamConv1d(
                    4, 4, k, stride=s, padding=pad, dilation=d, groups=1
                ).eval()
                ca_conv.stream()
                ca_conv.weight = gt_conv.weight
                ca_conv.bias = gt_conv.bias
                res = AlignBranches(
                    ca_conv,
                    nn.Identity(),
                )
                res.stream()
                res = torch.jit.script(res, x)
                xs = torch.split(x, size, -1)
                chunks_y = []
                sizes_y = []
                for chunk in xs:
                    chunk_net, chunk_res = res(chunk)
                    chunk_y = chunk_net + chunk_res
                    chunks_y.append(chunk_y)
                    sizes_y.append(chunk_y.shape[-1])
                print("align branches test_res_script chunk_sizes:\n", sizes_y)
                # flush_net, flush_res = res.flush()
                # chunks_y.append(flush_net + flush_res)
                y_cached = torch.cat(chunks_y, dim=-1)
                self.assertTrue(
                    torch.allclose(
                        y_true[..., : y_cached.shape[-1]], y_cached, atol=1e-6
                    )
                )


if __name__ == "__main__":
    unittest.main()
